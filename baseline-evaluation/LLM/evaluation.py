"""
LLM/evaluate.py -- Evaluate FLAN-T5 on the UltraFeedback dataset.

This script:

1. Loads a locally downloaded UltraFeedback dataset.
2. Loads a locally downloaded FLAN-T5 model.
3. Generates responses for every prompt.
4. Computes BLEU.
5. Uses FLAN-T5-Large as an LLM judge.
6. Saves all results as JSON.

Designed for use with evaluate-job.slurm.

Example:

python evaluate.py \
    --generation-model /home/xrai/models/flan-t5-base \
    --judge-model /home/xrai/models/flan-t5-large \
    --dataset-path /home/xrai/datasets/ultrafeedback_binarized
"""


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from sacrebleu import corpus_bleu
import torch

from collections import Counter

from rouge_score import rouge_scorer
from bert_score import score as bertscore
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_GENERATION_MODEL = "/home/xrai/models/Qwen3-32B"

DEFAULT_JUDGE_MODEL = "/home/xrai/models/prometheus-7b-v2.0"

DEFAULT_DATASET = "/home/xrai/datasets/UltraFeedbackB"

DEFAULT_EMBEDDING_MODEL = "/home/xrai/models/all-MiniLM-L6-v2"


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "evaluation_results",
)

# ---------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate FLAN-T5 on UltraFeedback"
    )

    parser.add_argument(
        "--generation-model",
        default=os.environ.get(
            "GENERATION_MODEL",
            DEFAULT_GENERATION_MODEL,
        ),
        help="Path to FLAN-T5 generation model.",
    )

    parser.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL",
            DEFAULT_JUDGE_MODEL,
        ),
        help="Path to FLAN-T5 judge model.",
    )

    parser.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH",
            DEFAULT_DATASET,
        ),
        help="Path to UltraFeedback dataset.",
    )

    parser.add_argument(
        "--embedding-model",
        default=os.environ.get(
            "EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        ),
        help="Path to sentence embedding model.",
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
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate only the first N examples.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------


def load_ultrafeedback(dataset_path):

    print("=" * 60)
    print("Loading UltraFeedback test_prefs dataset")
    print("=" * 60)

    prefs_file = (
        f"{dataset_path}/data/"
        "test_prefs-00000-of-00001.parquet"
    )

    print(f"Loading: {prefs_file}")

    dataset = load_dataset(
        "parquet",
        data_files={
            "test_prefs": prefs_file
        }
    )

    # Select the test preference split
    subset = dataset["test_prefs"]

    questions = subset["prompt"]

    references = [
        example[-1]["content"]
        for example in subset["chosen"]
    ]

    print(f"Loaded {len(questions)} prompts.")

    return questions, references


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
        local_files_only=True,
    )

    model.to(device)

    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------
# normalise text
# ---------------------------------------------------------------------


def normalize_text(text):
    """
    Lowercase, remove punctuation and extra whitespace.
    """

    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------


def generate_answers(
    questions,
    tokenizer,
    model,
    device,
    batch_size,
    max_new_tokens,
):

    generated_answers = []

    num_batches = (
        len(questions) + batch_size - 1
    ) // batch_size

    print("=" * 60)
    print("Generating Responses")
    print("=" * 60)

    start = time.perf_counter()

    for batch_start in range(
        0,
        len(questions),
        batch_size,
    ):

        batch_questions = questions[
            batch_start:
            batch_start + batch_size
        ]

        inputs = tokenizer(
            batch_questions,
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

        batch_answers = tokenizer.batch_decode(
            outputs,
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
#  Evaluation
# ---------------------------------------------------------------------


def evaluate_bleu(predictions, references):

    print("=" * 60)
    print("Computing BLEU")
    print("=" * 60)

    # sacreBLEU expects a list of reference lists
    bleu = corpus_bleu(predictions, [references])

    print(f"BLEU Score: {bleu.score:.4f}")

    return {
        "bleu": bleu.score
    }

def evaluate_rouge(predictions, references):

    print("=" * 60)
    print("Computing ROUGE")
    print("=" * 60)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rouge1 = []
    rouge2 = []
    rougeL = []

    for pred, ref in zip(predictions, references):

        scores = scorer.score(ref, pred)

        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

    results = {
        "rouge1": sum(rouge1) / len(rouge1),
        "rouge2": sum(rouge2) / len(rouge2),
        "rougeL": sum(rougeL) / len(rougeL),
    }

    print(results)

    return results

def evaluate_cosine(predictions, references, embedding_model):

    print("=" * 60)
    print("Computing Cosine Similarity")
    print("=" * 60)

    pred_embeddings = embedding_model.encode(
        predictions,
        convert_to_numpy=True,
        batch_size=32,
    )

    ref_embeddings = embedding_model.encode(
        references,
        convert_to_numpy=True,
        batch_size=32,
    )

    similarities = []

    for pred, ref in zip(pred_embeddings, ref_embeddings):
        sim = cosine_similarity(
            pred.reshape(1, -1),
            ref.reshape(1, -1),
        )[0][0]

        similarities.append(sim)

    mean_similarity = np.mean(similarities)

    print(f"Mean cosine similarity: {mean_similarity:.4f}")

    return {
        "cosine_similarity": float(mean_similarity)
    }


def evaluate_f1(predictions, references):

    print("=" * 60)
    print("Computing Precision / Recall / F1")
    print("=" * 60)

    precisions = []
    recalls = []
    f1s = []

    for pred, ref in zip(predictions, references):

        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }

def evaluate_token_metrics(predictions, references):

    print("=" * 60)
    print("Computing Token Precision / Recall / F1")
    print("=" * 60)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, ref in zip(predictions, references):

        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()

        common = Counter(pred_tokens) & Counter(ref_tokens)
        overlap = sum(common.values())

        if overlap == 0:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    metrics = {

        "precision":
            sum(precision_scores) / len(precision_scores),

        "recall":
            sum(recall_scores) / len(recall_scores),

        "f1":
            sum(f1_scores) / len(f1_scores),

    }

    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")

    return metrics


# ---------------------------------------------------------------------
# LLM-as-a-Judge Prompt
# ---------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """
You are grading an AI assistant's answer.

Question:
{question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

you are a LLM judge, compare the candidate answer to the reference answer with regards to the question and determine which answer is better based on helpfulness, harmlessness and honesty. 
If the candidate answer is better, respond with "Yes". If it is not, respond with "No".
Respond with only "Yes" or "No" and nothing else. Do not provide any explanation or additional text.
Judgement:
"""


# ---------------------------------------------------------------------
# Judge One Batch
# ---------------------------------------------------------------------


def judge_batch(
    questions,
    references,
    candidates,
    tokenizer,
    model,
    device,
):

    prompts = [

        JUDGE_PROMPT_TEMPLATE.format(
            question=q,
            reference=r,
            candidate=c,
        )

        for q, r, c in zip(
            questions,
            references,
            candidates,
        )

    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    verdicts = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    return verdicts


# ---------------------------------------------------------------------
# Judge Entire Dataset
# ---------------------------------------------------------------------


def run_judge(
    questions,
    references,
    generated_answers,
    tokenizer,
    model,
    device,
    batch_size,
):

    print("=" * 60)
    print("Running LLM-as-a-Judge")
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

            questions[batch_start:batch_end],

            references[batch_start:batch_end],

            generated_answers[batch_start:batch_end],

            tokenizer,
            model,
            device,

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


# ---------------------------------------------------------------------
# Parse Judge Responses
# ---------------------------------------------------------------------


def parse_judgements(verdicts):

    scores = []

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
        f"Win rate: "
        f"{percentage:.2f}%"
    )

    return scores, percentage


# ---------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("UltraFeedback Evaluation")
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

    questions, references = load_ultrafeedback(
        args.dataset_path
    )

    if args.max_examples:

        questions = questions[:args.max_examples]
        references = references[:args.max_examples]

        print(
            f"Limited evaluation to "
            f"{args.max_examples} examples"
        )

    # -------------------------------------------------
    # Load generation model
    # -------------------------------------------------

    gen_tokenizer, gen_model = load_generation_model(
        args.generation_model,
        device
    )



    # -------------------------------------------------
    # Generate answers
    # -------------------------------------------------

    generated_answers, generation_time = generate_answers(
        questions,
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
    # BLEU
    # -------------------------------------------------

    bleu_results = evaluate_bleu(
        generated_answers,
        references
    )

    rouge = evaluate_rouge(
        generated_answers,
        references
    )


    print("=" * 60)
    print("Loading sentence embedding model")
    print("=" * 60)

    embedding_model = SentenceTransformer(
    "/home/xrai/models/all-MiniLM-L6-v2"
    )

    token_metrics = evaluate_token_metrics(
        generated_answers,
        references
    )

    classification = evaluate_f1(
        generated_answers, 
        references
    )

    cosine_results = evaluate_cosine(
    generated_answers,
        references,
        embedding_model
    )
    # -------------------------------------------------
    # Load judge model
    # -------------------------------------------------

    del embedding_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    judge_tokenizer, judge_model = load_judge_model(
        args.judge_model,
        device
    )

    # -------------------------------------------------
    # Run judge
    # -------------------------------------------------

    verdicts, judge_time = run_judge(
        questions,
        references,
        generated_answers,
        judge_tokenizer,
        judge_model,
        device,
        args.batch_size,
    )

    scores, truth_score = parse_judgements(
        verdicts
    )


    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    results = {

        "timestamp": timestamp,

        "generation_model":
            args.generation_model,

        "judge_model":
            args.judge_model,

        "dataset":
            args.dataset_path,

        "metrics": {

            "bleu":
                bleu_results["bleu"],

            "win-rate":
                truth_score,

            "generation_time_seconds":
                generation_time,

            "judge_time_seconds":
                judge_time,


            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],

            "cosine_similarity":
                cosine_results["cosine_similarity"],

            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],

            "precision":
                classification["precision"],

            "recall":
                classification["recall"],

            "f1":
                classification["f1"],

            "token_precision":token_metrics["precision"],
            "token_recall":token_metrics["recall"],  
            "token_f1":token_metrics["f1"]
        },

        "examples": [

            {
                "question": q,
                "reference": r,
                "generated": g,
                "judge": v,
            }

            for q, r, g, v in zip(
                questions,
                references,
                generated_answers,
                verdicts,
            )

        ]

    }


    output_file = (
        output_dir /
        f"ultrafeedback_eval_{timestamp}.json"
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