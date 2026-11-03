"""
helpsteer2.py -- Evaluate a model on HelpSteer2-style data using an
LLM-as-a-judge win rate, per-attribute weighted scoring, and
generation time.

This script:

1. Loads a locally downloaded JSON file where every example has "id",
   "prompt", "response" (the human reference response), and optionally
   "human_scores" (the HelpSteer2 attribute ratings), e.g.:

        [
          {
            "id": "validation-000000",
            "prompt": "explain master slave replication nsql",
            "response": "...",
            "human_scores": {
              "helpfulness": 4,
              "correctness": 4,
              "coherence": 4,
              "complexity": 3,
              "verbosity": 2
            }
          },
          ...
        ]

2. Loads a locally downloaded generation model and generates a
   candidate response for every prompt, timing how long generation
   takes.
3. Loads a separate judge model (FlowJudge) and, for every example:
     a. Asks it to compare the candidate response to the human
        reference response and say which is better (pairwise
        Yes/No judgement).
     b. Asks it to independently score the candidate response on the
        same five weighted HelpSteer2 attributes used in the human
        annotations -- helpfulness, correctness, coherence,
        complexity, verbosity -- each on a 0-4 scale.
4. Reports these headline metrics:
     - llm_judge_win_rate: % of examples where the judge preferred the
       generation model's candidate answer over the human reference.
     - avg_llm_<attribute>: the judge's average score for the
       candidate answers on each of the five weighted attributes.
     - generation_time_seconds: total wall-clock time spent generating
       candidate responses.
5. Saves all results (metrics + per-example detail, including the
   original human_scores and the judge's per-attribute llm_scores) as
   JSON.

Designed for use with evaluate-job.slurm.

Example:

python helpsteer2.py \
    --generation-model /home/xrai/models/Meta-Llama-3.1-8B-Instruct \
    --judge-model /home/xrai/models/flowjudge \
    --dataset-path /home/xrai/datasets/helpsteer2/helpsteer2.json
"""


import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ---------------------------------------------------------------------
# Prompt formatting helper
# ---------------------------------------------------------------------


def format_for_judge(tokenizer, user_content):
    """
    Format a judge instruction using the tokenizer's chat template
    when one is available, so an instruction-tuned judge model sees
    the prompt in the format it was actually trained on rather than
    as raw continuation text. Instruct models prompted as plain text
    (no chat markers) are far more likely to ignore output-format
    instructions and instead continue/echo the input, which is what
    produces garbled, unparsable judge output.

    Falls back to the raw string when the tokenizer has no chat
    template configured (e.g. a base, non-instruct model).
    """

    if getattr(tokenizer, "chat_template", None):

        messages = [{"role": "user", "content": user_content}]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return user_content


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_GENERATION_MODEL = "/home/xrai/models/Meta-Llama-3.1-8B-Intstruct"

DEFAULT_JUDGE_MODEL = "/home/xrai/models/flowjudge"

DEFAULT_DATASET = "/home/xrai/datasets/helpsteer2/helpsteer2.json"

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "evaluation_results",
)

# The five weighted HelpSteer2 attributes, each scored 0-4.
SCORE_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

# ---------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate a model on HelpSteer2 with LLM-as-a-judge, weighted attribute scoring, and generation time"
    )

    parser.add_argument(
        "--generation-model",
        default=os.environ.get(
            "GENERATION_MODEL",
            DEFAULT_GENERATION_MODEL,
        ),
        help="Path to the model being evaluated.",
    )

    parser.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL",
            DEFAULT_JUDGE_MODEL,
        ),
        help="Path to the FlowJudge model used for both the pairwise "
             "comparison and the weighted attribute scoring.",
    )

    parser.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH",
            DEFAULT_DATASET,
        ),
        help="Path to the helpsteer2.json file (a JSON array of "
             "id/prompt/response/human_scores records).",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON results.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for the generation model's response.",
    )

    parser.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=10,
        help="Max new tokens for the judge model's Yes/No verdict.",
    )

    parser.add_argument(
        "--score-max-new-tokens",
        type=int,
        default=60,
        help="Max new tokens for the judge model's weighted attribute "
             "scoring output.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate only the first N examples.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Load Dataset -- HelpSteer2
# ---------------------------------------------------------------------


def load_helpsteer2(dataset_path):
    """
    Load a JSON file containing a list of records shaped like:

        {
            "id": "...",
            "prompt": "...",
            "response": "...",
            "human_scores": {
                "helpfulness": ...,
                "correctness": ...,
                "coherence": ...,
                "complexity": ...,
                "verbosity": ...
            }
        }

    Returns a list of dicts (one per example).
    """

    print("=" * 60)
    print("Loading HelpSteer2 dataset")
    print("=" * 60)

    print(f"Loading: {dataset_path}")

    with open(dataset_path, "r") as f:
        examples = json.load(f)

    required_fields = ("prompt", "response")

    for i, example in enumerate(examples):

        missing = [
            field for field in required_fields
            if field not in example
        ]

        if missing:
            raise ValueError(
                f"Example {i} is missing required field(s) "
                f"{missing}: {example}"
            )

        example.setdefault("id", str(i))
        example.setdefault("human_scores", {})

    print(f"Loaded {len(examples)} examples.")

    return examples


# ---------------------------------------------------------------------
# Load Generation Model
# ---------------------------------------------------------------------


def load_generation_model(model_path, device):

    print("=" * 60)
    print("Loading generation model")
    print("=" * 60)

    print(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------
# Load Judge Model
# ---------------------------------------------------------------------


def load_judge_model(model_path, device):

    print("=" * 60)
    print("Loading judge model")
    print("=" * 60)

    print(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------


def generate_answers(
    prompts,
    tokenizer,
    model,
    device,
    batch_size,
    max_new_tokens,
):

    generated_answers = []

    num_batches = (
        len(prompts) + batch_size - 1
    ) // batch_size

    print("=" * 60)
    print("Generating Responses")
    print("=" * 60)

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(prompts),
        batch_size,
    ):

        batch_prompts = prompts[
            batch_start:
            batch_start + batch_size
        ]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Only decode the newly generated tokens, not the prompt
        # (model.generate on a causal LM returns prompt + completion).
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        batch_answers = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        generated_answers.extend(batch_answers)

        batch_num = batch_start // batch_size + 1

        if (
            batch_num % 10 == 0
            or batch_num == num_batches
        ):

            elapsed = time.perf_counter() - start

            print(
                f"Batch {batch_num}/{num_batches}"
                f" completed "
                f"({elapsed:.1f}s)"
            )

    total = time.perf_counter() - start

    print(f"\nGeneration completed in {total:.2f}s")

    return generated_answers, total


# ---------------------------------------------------------------------
# LLM-as-a-Judge Prompt -- pairwise comparison
# ---------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are grading an AI assistant's answer.

Prompt:
{prompt}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

You are an LLM judge. Compare the candidate answer to the reference
answer with regard to the prompt and determine which answer is better,
considering helpfulness, correctness, coherence, complexity, and
verbosity.
If the candidate answer is better, respond with "Yes". If it is not,
respond with "No".
Respond with only the single word "Yes" or "No" and nothing else. Do
not repeat or quote any part of the prompt, reference answer, or
candidate answer. Do not provide any explanation or additional text.
Judgement:"""


def judge_batch(
    prompts,
    references,
    candidates,
    tokenizer,
    model,
    device,
    max_new_tokens,
):

    prompt_texts = [

        format_for_judge(
            tokenizer,
            JUDGE_PROMPT_TEMPLATE.format(
                prompt=p,
                reference=r,
                candidate=c,
            ),
        )

        for p, r, c in zip(
            prompts,
            references,
            candidates,
        )

    ]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.inference_mode():

        # NOTE: repetition_penalty / no_repeat_ngram_size were removed
        # here. The judge prompt itself contains words like "Yes" /
        # "No" and other short repeated tokens, and n-gram/repetition
        # penalties applied on top of a prompt containing the exact
        # tokens we want the model to output can push the model away
        # from the correct, simple answer. For a short constrained
        # Yes/No verdict, plain greedy decoding is more reliable.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

    verdicts = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return verdicts


def run_judge(
    prompts,
    references,
    generated_answers,
    tokenizer,
    model,
    device,
    batch_size,
    max_new_tokens,
):

    print("=" * 60)
    print("Running LLM-as-a-Judge (pairwise)")
    print("=" * 60)

    all_verdicts = []

    num_batches = (
        len(generated_answers) + batch_size - 1
    ) // batch_size

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(generated_answers),
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            len(generated_answers),
        )

        verdicts = judge_batch(

            prompts[batch_start:batch_end],

            references[batch_start:batch_end],

            generated_answers[batch_start:batch_end],

            tokenizer,
            model,
            device,
            max_new_tokens,

        )

        all_verdicts.extend(verdicts)

        batch_num = batch_start // batch_size + 1

        if (
            batch_num % 10 == 0
            or batch_num == num_batches
        ):

            elapsed = time.perf_counter() - start

            print(
                f"Judged batch "
                f"{batch_num}/{num_batches} "
                f"({elapsed:.1f}s)"
            )

    total = time.perf_counter() - start

    print(f"\nJudging completed in {total:.2f}s")

    return all_verdicts, total


def parse_judgements(verdicts):

    scores = []
    unparsed_count = 0

    for verdict in verdicts:

        match = re.search(
            r"\b(yes|no)\b",
            verdict,
            flags=re.IGNORECASE,
        )

        if match:

            if match.group(1).lower() == "yes":

                scores.append(1)

            else:

                scores.append(0)

        else:

            # The judge didn't produce a parseable Yes/No (e.g. it
            # rambled or echoed part of the prompt instead of
            # following the format instruction). This is currently
            # counted as a loss for win-rate purposes, but track it
            # separately so a spike here is visible rather than
            # silently deflating the win rate.
            unparsed_count += 1
            scores.append(0)

    percentage = (
        sum(scores) / len(scores) * 100
        if scores
        else 0.0
    )

    print("=" * 60)
    print("Judge Results")
    print("=" * 60)

    print(
        f"LLM-judge win rate: "
        f"{percentage:.2f}%"
    )

    if unparsed_count:

        print(
            f"WARNING: {unparsed_count}/{len(verdicts)} pairwise "
            f"verdicts had no parseable Yes/No and were counted as "
            f"losses -- inspect judge_verdict in the output for "
            f"these examples."
        )

    return scores, percentage, unparsed_count


# ---------------------------------------------------------------------
# LLM-as-a-Judge Prompt -- weighted attribute scoring
# ---------------------------------------------------------------------

SCORE_PROMPT_TEMPLATE = """
You are grading an AI assistant's answer to a prompt using the
following five weighted attributes. Score each attribute from 0
(worst) to 4 (best):

- helpfulness: how well the answer addresses the prompt and satisfies
  the user's intent
- correctness: whether the answer is factually accurate and free of
  errors
- coherence: how clear, well organized, and easy to follow the answer
  is
- complexity: the level of domain expertise required to have written
  the answer
- verbosity: how much detail the answer includes relative to what the
  prompt actually needs

Prompt:
{prompt}

Reference Answer:
{reference}

Answer to grade:
{candidate}

Use the reference answer as a guide for what a strong response looks
like (e.g. what facts are correct, and a reasonable level of detail),
but score the answer to grade on its own merits against the five
attributes above -- do not penalize it merely for differing from the
reference.

Respond with ONLY a JSON object in exactly this format and nothing
else, with integer scores from 0 to 4 filled in for each placeholder
below. Do not repeat or quote the prompt, reference answer, or answer
to grade. Do not leave the placeholders as-is -- they are not valid
scores, you must replace them with your actual integer judgement:
{{"helpfulness": "<0-4>", "correctness": "<0-4>", "coherence": "<0-4>", "complexity": "<0-4>", "verbosity": "<0-4>"}}
"""


RETRY_SCORE_PROMPT_TEMPLATE = """
You previously failed to follow the required output format. Try
again.

Score the following answer on five attributes, each from 0 (worst)
to 4 (best): helpfulness, correctness, coherence, complexity,
verbosity.

Prompt:
{prompt}

Reference Answer:
{reference}

Answer to grade:
{candidate}

Output ONLY the JSON object below, with the scores filled in as
integers. Nothing else -- no words, no headings, no repeated prompt
text:
{{"helpfulness": "<0-4>", "correctness": "<0-4>", "coherence": "<0-4>", "complexity": "<0-4>", "verbosity": "<0-4>"}}
"""


def _clean_attribute_value(value):
    """
    Normalize a single raw attribute value (as parsed from JSON, or
    matched via regex) into a valid integer score in [0, 4], or None
    if it can't be trusted.

    Handles both numeric JSON values (3) and the quoted numeric
    strings the prompt template actually asks for ("3"), and rejects
    anything outside the valid 0-4 range instead of silently
    accepting it -- a single wildly out-of-range value (e.g. a
    garbled "-435") should not be able to distort the averaged
    metric.
    """

    if isinstance(value, bool):
        # bool is a subclass of int in Python -- explicitly exclude
        # it so True/False are never mistaken for scores.
        return None

    if isinstance(value, (int, float)):
        ivalue = int(round(value))

    elif isinstance(value, str):

        match = re.search(r"-?\d+", value.strip())

        if not match:
            return None

        ivalue = int(match.group(0))

    else:
        return None

    if 0 <= ivalue <= 4:
        return ivalue

    return None


def parse_attribute_scores(text):
    """
    Extract the five weighted attribute scores from the judge's raw
    output. Tries to parse a JSON object first; falls back to
    per-attribute regex extraction for any attribute the JSON pass
    didn't recover. Any attribute that can't be recovered, or whose
    value falls outside the valid 0-4 range, is left as None.
    """

    scores = {attr: None for attr in SCORE_ATTRIBUTES}

    # Try to find and parse a JSON object in the output.
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if json_match:

        try:

            parsed = json.loads(json_match.group(0))

            for attr in SCORE_ATTRIBUTES:

                if attr in parsed:
                    scores[attr] = _clean_attribute_value(parsed[attr])

        except (json.JSONDecodeError, AttributeError):

            pass

    # Fall back to a per-attribute regex scan for anything JSON
    # parsing didn't recover (missing key, malformed JSON, or an
    # out-of-range value that got rejected above). Matches both bare
    # numbers ("helpfulness": 3) and quoted numeric strings
    # ("helpfulness": "3"), including negative numbers so
    # out-of-range values are still detected and rejected rather than
    # left silently unmatched.
    for attr in SCORE_ATTRIBUTES:

        if scores[attr] is not None:
            continue

        match = re.search(
            rf'"?{attr}"?\s*[:=]\s*"?(-?\d+)"?',
            text,
            flags=re.IGNORECASE,
        )

        if match:
            scores[attr] = _clean_attribute_value(match.group(1))

    return scores


def score_batch(
    prompts,
    references,
    candidates,
    tokenizer,
    model,
    device,
    max_new_tokens,
    template=SCORE_PROMPT_TEMPLATE,
):

    prompt_texts = [

        format_for_judge(
            tokenizer,
            template.format(
                prompt=p,
                reference=r,
                candidate=c,
            ),
        )

        for p, r, c in zip(prompts, references, candidates)

    ]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.inference_mode():

        # NOTE: repetition_penalty / no_repeat_ngram_size were removed
        # here. The scoring prompt template contains the exact
        # attribute names ("helpfulness", "correctness", etc.) that
        # the model is required to repeat back verbatim in its JSON
        # answer, plus repetitive JSON syntax (quotes, colons,
        # commas). no_repeat_ngram_size bans repeating any n-gram
        # that already appeared in the prompt -- including the
        # prompt itself -- which was forcing the model to mangle
        # attribute names and JSON syntax instead of reproducing
        # them correctly. This was the primary cause of the garbled,
        # unparsable judge output (e.g. "helpfulnes",
        # "correctess", junk tokens instead of valid JSON).
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

    raw_outputs = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return raw_outputs


def run_scoring(
    prompts,
    references,
    generated_answers,
    tokenizer,
    model,
    device,
    batch_size,
    max_new_tokens,
):

    print("=" * 60)
    print("Running LLM-as-a-Judge (weighted attribute scoring)")
    print("=" * 60)

    all_raw_outputs = []
    all_scores = []

    num_batches = (
        len(generated_answers) + batch_size - 1
    ) // batch_size

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(generated_answers),
        batch_size,
    ):

        batch_end = min(
            batch_start + batch_size,
            len(generated_answers),
        )

        raw_outputs = score_batch(

            prompts[batch_start:batch_end],

            references[batch_start:batch_end],

            generated_answers[batch_start:batch_end],

            tokenizer,
            model,
            device,
            max_new_tokens,

        )

        for raw in raw_outputs:

            all_raw_outputs.append(raw)
            all_scores.append(parse_attribute_scores(raw))

        batch_num = batch_start // batch_size + 1

        if (
            batch_num % 10 == 0
            or batch_num == num_batches
        ):

            elapsed = time.perf_counter() - start

            print(
                f"Scored batch "
                f"{batch_num}/{num_batches} "
                f"({elapsed:.1f}s)"
            )

    total = time.perf_counter() - start

    print(f"\nScoring completed in {total:.2f}s")

    # -------------------------------------------------
    # Retry pass: re-ask, with a stricter prompt, for any example
    # where the judge's first attempt didn't yield ANY parseable
    # attribute score (all five came back None). This recovers cases
    # where the judge ignored the format instructions and echoed the
    # prompt template or rambled instead of returning JSON.
    # -------------------------------------------------

    failed_indices = [
        i for i, s in enumerate(all_scores)
        if all(s.get(attr) is None for attr in SCORE_ATTRIBUTES)
    ]

    if failed_indices:

        print(
            f"\n{len(failed_indices)} example(s) had no parseable "
            f"attribute scores on the first attempt -- retrying with "
            f"a stricter prompt."
        )

        retry_prompts = [prompts[i] for i in failed_indices]
        retry_references = [references[i] for i in failed_indices]
        retry_candidates = [generated_answers[i] for i in failed_indices]

        retry_start = time.perf_counter()

        for batch_start in range(0, len(failed_indices), batch_size):

            batch_end = min(
                batch_start + batch_size,
                len(failed_indices),
            )

            retry_raw_outputs = score_batch(
                retry_prompts[batch_start:batch_end],
                retry_references[batch_start:batch_end],
                retry_candidates[batch_start:batch_end],
                tokenizer,
                model,
                device,
                max_new_tokens,
                template=RETRY_SCORE_PROMPT_TEMPLATE,
            )

            for offset, raw in enumerate(retry_raw_outputs):

                original_index = failed_indices[batch_start + offset]
                retry_scores = parse_attribute_scores(raw)

                all_raw_outputs[original_index] = raw
                all_scores[original_index] = retry_scores

        retry_total = time.perf_counter() - retry_start

        still_failed = sum(
            1 for i in failed_indices
            if all(
                all_scores[i].get(attr) is None
                for attr in SCORE_ATTRIBUTES
            )
        )

        print(
            f"Retry completed in {retry_total:.2f}s -- "
            f"recovered {len(failed_indices) - still_failed}/"
            f"{len(failed_indices)} example(s) "
            f"({still_failed} still unparseable)."
        )

        total += retry_total

    return all_scores, all_raw_outputs, total


def summarize_attribute_scores(all_scores):
    """
    Compute the mean of each weighted attribute across all examples,
    ignoring unparsed (None) values for that attribute. Also returns
    the count of unparsed values per attribute.
    """

    averages = {}
    unparsed_counts = {}

    for attr in SCORE_ATTRIBUTES:

        values = [
            s[attr] for s in all_scores
            if s.get(attr) is not None
        ]

        unparsed_counts[attr] = len(all_scores) - len(values)

        averages[attr] = (
            sum(values) / len(values)
            if values
            else None
        )

    print("=" * 60)
    print("Weighted Attribute Score Results")
    print("=" * 60)

    for attr in SCORE_ATTRIBUTES:

        avg = averages[attr]
        unparsed = unparsed_counts[attr]

        if avg is not None:
            print(f"{attr:12s}: {avg:.2f} (unparsed: {unparsed})")
        else:
            print(f"{attr:12s}: N/A (unparsed: {unparsed})")

    return averages, unparsed_counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("HelpSteer2 Evaluation")
    print("=" * 60)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load dataset
    # -------------------------------------------------

    examples = load_helpsteer2(
        args.dataset_path
    )

    if args.max_examples:

        examples = examples[:args.max_examples]

        print(
            f"Limited evaluation to "
            f"{args.max_examples} examples"
        )

    prompts = [ex["prompt"] for ex in examples]
    references = [ex["response"] for ex in examples]

    # -------------------------------------------------
    # Load generation model and generate candidate answers
    # -------------------------------------------------

    gen_tokenizer, gen_model = load_generation_model(
        args.generation_model,
        device
    )

    generated_answers, generation_time = generate_answers(
        prompts,
        gen_tokenizer,
        gen_model,
        device,
        args.batch_size,
        args.max_new_tokens,
    )

    del gen_model
    del gen_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Load judge model (FlowJudge) once, run both judge passes
    # -------------------------------------------------

    judge_tokenizer, judge_model = load_judge_model(
        args.judge_model,
        device
    )

    verdicts, judge_time = run_judge(
        prompts,
        references,
        generated_answers,
        judge_tokenizer,
        judge_model,
        device,
        args.batch_size,
        args.judge_max_new_tokens,
    )

    scores, win_rate, unparsed_pairwise_count = parse_judgements(
        verdicts
    )

    attribute_scores, score_raw_outputs, scoring_time = run_scoring(
        prompts,
        references,
        generated_answers,
        judge_tokenizer,
        judge_model,
        device,
        args.batch_size,
        args.score_max_new_tokens,
    )

    attribute_averages, attribute_unparsed_counts = summarize_attribute_scores(
        attribute_scores
    )

    del judge_model
    del judge_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    metrics = {

        "llm_judge_win_rate": win_rate,

        "unparsed_pairwise_verdicts": unparsed_pairwise_count,

        "generation_time_seconds": generation_time,

        "judge_time_seconds": judge_time,

        "scoring_time_seconds": scoring_time,

        "num_examples": len(examples),

    }

    for attr in SCORE_ATTRIBUTES:

        metrics[f"avg_llm_{attr}"] = attribute_averages[attr]
        metrics[f"unparsed_llm_{attr}"] = attribute_unparsed_counts[attr]

    results = {

        "timestamp": timestamp,

        "generation_model":
            args.generation_model,

        "judge_model":
            args.judge_model,

        "dataset":
            args.dataset_path,

        "metrics": metrics,

        "examples": [

            {
                "id": ex["id"],
                "prompt": ex["prompt"],
                "reference_response": ex["response"],
                "human_scores": ex["human_scores"],
                "generated_response": g,
                "judge_verdict": v,
                "score": s,
                "llm_scores": llm_score,
                "llm_score_raw_output": raw,
            }

            for ex, g, v, s, llm_score, raw in zip(
                examples,
                generated_answers,
                verdicts,
                scores,
                attribute_scores,
                score_raw_outputs,
            )

        ]

    }

    output_file = (
        output_dir /
        f"helpsteer2_eval_{timestamp}.json"
    )

    with open(output_file, "w") as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("=" * 60)
    print("Evaluation Finished")
    print("=" * 60)

    print(
        f"Results saved to: {output_file}"
    )


# Required to actually run the script
if __name__ == "__main__":
    main()