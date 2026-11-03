"""
steering_eval_truthfulqa_generation.py -- Compare a base generation
model against CAA-steered and MAT-Steer-steered variants on
TruthfulQA's GENERATION task (truthful / informative / win rate vs.
best_answer), reusing evaluate_truthfulqa_generation.py's dataset
loading, generation, and judging code UNCHANGED.

Unlike steering_eval_truthfulqa.py (the MC1/MC2 version), this one
does NOT need a different hook design. evaluate_truthfulqa_generation
.generate_answers is structurally identical to helpsteer2.py's
generate_answers -- same batched model.generate(), same
do_sample=False, same left-padded tokenizer -- so it uses real
KV-cached autoregressive generation, exactly what SteeringHook/
AttnOProjSteeringHook (defined below) were built for (always edit
position -1, which is correct specifically because KV-caching makes
each post-prompt forward call exactly one new token).

(Contrast with the MC1/MC2 script: TruthfulQA's multiple-choice task
scores each choice via ONE forward pass over the whole "context +
continuation" string, with no autoregressive stepping -- that's why
THAT script needed range-steering hooks instead, steering every
position from the question boundary onward rather than just the
last one. Generation tasks and log-likelihood-scoring tasks need
different hook semantics; this file is the generation case.)

Metrics are unchanged from evaluate_truthfulqa_generation.py --
truthful_percent, informative_percent,
truthful_and_informative_percent, llm_judge_win_rate,
generation_time_seconds -- computed identically per variant
(baseline/caa/mat_steer), all judged by ONE judge-model load.

NOTE ON SELF-CONTAINMENT: this file used to import
get_decoder_layers/get_attn_o_proj/build_caa_vector/train_mat_steer/
make_caa_edit_fn/make_mat_steer_edit_fn/SteeringHook/
AttnOProjSteeringHook/MATSteerModule from steeredhelpsteer2.py. All
of that (model-architecture / steering-mechanism code, which is
dataset-agnostic) is now defined directly in this file instead, the
same way steering_eval_ultrafeedback.py is self-contained rather
than importing steering_eval.py. Nothing about the steering
mechanics themselves changed -- same hook sites, same CAA diff-mean
construction, same MAT-Steer loss/training loop.

As before, CAA/MAT-Steer can be fit two ways, controlled by
--fit-source:

  - --fit-source helpsteer2 (default, backward compatible): fit on a
    HelpSteer2-shaped --fit-dataset-path (via human_scores) --
    unrelated to TruthfulQA, positive/negative examples come from
    DIFFERENT prompts entirely (the unpaired DiffMean construction).

  - --fit-source truthfulqa: fits directly from TruthfulQA's OWN
    best_answer vs. its first incorrect_answers entry, via
    build_fit_examples_from_truthfulqa_generation() below -- exactly
    ONE pair per question, not every correct/incorrect combination
    the dataset happens to list. This is actually a BETTER match to
    CAA's original methodology than the helpsteer2 path: the two
    answers share the same question, giving a genuinely paired
    per-prompt contrastive example (closer to nrimsky/CAA's original
    A/B multiple-choice setup) -- best_answer scored 4, the first
    incorrect_answers entry scored 0, on a single "helpfulness" axis.
    Since there's no per-attribute breakdown here (just correct vs.
    incorrect), --steer-attributes is forced to ["helpfulness"] in
    this mode. Requires either --fit-dataset-path (a SEPARATE
    TruthfulQA generation-shaped file, disjoint from --dataset-path)
    or, if that isn't given, --dataset-path itself gets shuffled and
    split (--holdout-frac/--split-seed) into a fitting portion and a
    disjoint evaluation portion -- so fitting never touches the exact
    questions being evaluated on either way.

--load-caa-vector/--load-mat-steer reuse a vector already fit in a
previous run (either --fit-source), so you don't need to refit if you
already have one.

Example (fitting from TruthfulQA's own correct/incorrect answers,
splitting one file into fit/eval portions):

python steering_eval_truthfulqa_generation.py \
    --generation-model /home/xrai/models/Meta-Llama-3.1-8B-Instruct \
    --judge-model /home/xrai/models/flowjudge \
    --dataset-path /home/xrai/datasets/truthfulqa/truthfulqa_generation.json \
    --fit-source truthfulqa

Example (fitting from HelpSteer2, or reusing a saved vector -- the
original behavior):

python steering_eval_truthfulqa_generation.py \
    --generation-model /home/xrai/models/Meta-Llama-3.1-8B-Instruct \
    --judge-model /home/xrai/models/flowjudge \
    --dataset-path /home/xrai/datasets/truthfulqa/truthfulqa_generation.json \
    --load-caa-vector ./steering_vectors/caa_vector_20260911_143000.pt \
    --load-mat-steer ./steering_vectors/mat_steer_20260911_143000.pt
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import truthfulqa_generation as tqg


# ---------------------------------------------------------------------
# Dataset-agnostic constants that used to live in steeredhelpsteer2.py
# ---------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "./steering_eval_results"

# HelpSteer2's published per-response attribute schema. Only used for
# --fit-source helpsteer2's --steer-attributes choices; --fit-source
# truthfulqa always forces ["helpfulness"] regardless of this list.
SCORE_ATTRIBUTES = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


def load_helpsteer2(path):
    """
    Loads a HelpSteer2-shaped dataset file: a JSON list (or JSONL) of
    records each with at least {"id", "prompt", "response",
    "human_scores"}, where human_scores maps attribute name -> 0-4
    rating. Only used for --fit-source helpsteer2.
    """

    with open(path) as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


# ---------------------------------------------------------------------
# Model-architecture helpers (dataset-agnostic; same hook sites used
# by every steering_eval_*.py variant)
# ---------------------------------------------------------------------

def get_decoder_layers(model):
    """
    Returns the list/ModuleList of transformer decoder layers for a
    Llama/Mistral/Qwen2-style causal LM (model.model.layers). Adjust
    this if you're steering a different architecture.
    """

    return model.model.layers


def get_attn_o_proj(model, layer_idx):
    """
    Returns the self-attention output-projection module of a decoder
    layer (model.model.layers[i].self_attn.o_proj) -- MAT-Steer's
    intervention site, matching the official MAT-Steer repo
    (interveners.py: "model.layers[{layer}].self_attn.o_proj.input")
    and the paper's own definition (Section 2.1) of the per-layer
    activation as the output of the self-attention layer BEFORE it's
    mixed back into the residual stream by o_proj. CAA steers the
    decoder layer's full residual-stream output instead (see
    SteeringHook below).
    """

    return get_decoder_layers(model)[layer_idx].self_attn.o_proj


# ---------------------------------------------------------------------
# Steering hooks
# ---------------------------------------------------------------------

class SteeringHook:
    """
    CAA's intervention site: registers a forward hook on a single
    decoder layer that edits its OUTPUT hidden states (the residual
    stream) in place. Always edits only the LAST token position of
    whatever gets passed through the hook -- this covers both the
    single new token of each post-prompt generation step under
    KV-caching, AND (during the one multi-token forward pass over the
    prompt) the prompt's own final token.
    """

    def __init__(self, layer, edit_fn):

        self.edit_fn = edit_fn
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):

        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] = self.edit_fn(hidden_states[:, -1, :])

        if is_tuple:
            return (hidden_states,) + output[1:]

        return hidden_states

    def remove(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class AttnOProjSteeringHook:
    """
    MAT-Steer's intervention site: registers a forward PRE-hook on the
    self-attention output projection (self_attn.o_proj) that edits its
    INPUT -- the concatenated multi-head attention output -- before
    o_proj mixes it back into the residual stream. Same
    last-token-always semantics as SteeringHook.
    """

    def __init__(self, o_proj_module, edit_fn):

        self.edit_fn = edit_fn
        self.handle = o_proj_module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):

        attn_output = args[0].clone()
        attn_output[:, -1, :] = self.edit_fn(attn_output[:, -1, :])

        return (attn_output,) + args[1:]

    def remove(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# ---------------------------------------------------------------------
# Activation collection for CAA / MAT-Steer fitting. Both fit-sources
# (--fit-source helpsteer2 and --fit-source truthfulqa) hand
# build_caa_vector / train_mat_steer the SAME flat shape --
# {"prompt", "response", "human_scores"} -- so a single pair of
# collection functions covers both.
# ---------------------------------------------------------------------

def collect_layer_activations(
    examples,
    tokenizer,
    model,
    device,
    layer_idx,
    batch_size,
    max_length=512,
):
    """
    Encodes `prompt + response` for every (flat) example and returns
    the layer_idx residual-stream activation at the LAST token
    position for each one, as a single [N, hidden_dim] tensor (on
    CPU, float32).

    Because the tokenizer pads on the left, the last real token is
    always at sequence position -1 regardless of each example's
    length.
    """

    layer = get_decoder_layers(model)[layer_idx]

    captured = {}

    def _capture_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured["activations"] = hidden_states[:, -1, :].detach()

    handle = layer.register_forward_hook(_capture_hook)

    all_activations = []

    try:

        for batch_start in range(0, len(examples), batch_size):

            batch = examples[batch_start: batch_start + batch_size]

            texts = [
                ex["prompt"] + "\n\n" + ex["response"]
                for ex in batch
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.inference_mode():
                model(**inputs)

            all_activations.append(
                captured["activations"].to(torch.float32).cpu()
            )

    finally:
        handle.remove()

    if not all_activations:
        return torch.empty(0)

    return torch.cat(all_activations, dim=0)


def collect_attn_input_activations(
    examples,
    tokenizer,
    model,
    device,
    layer_idx,
    batch_size,
    max_length=512,
):
    """
    Same as collect_layer_activations, but captures activations at
    MAT-Steer's actual site: the INPUT to self_attn.o_proj (the
    concatenated multi-head attention output, before it's projected
    back into the residual stream) -- NOT the decoder layer's output.
    """

    o_proj = get_attn_o_proj(model, layer_idx)

    captured = {}

    def _capture_pre_hook(module, args):
        captured["activations"] = args[0][:, -1, :].detach()

    handle = o_proj.register_forward_pre_hook(_capture_pre_hook)

    all_activations = []

    try:

        for batch_start in range(0, len(examples), batch_size):

            batch = examples[batch_start: batch_start + batch_size]

            texts = [
                ex["prompt"] + "\n\n" + ex["response"]
                for ex in batch
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.inference_mode():
                model(**inputs)

            all_activations.append(
                captured["activations"].to(torch.float32).cpu()
            )

    finally:
        handle.remove()

    if not all_activations:
        return torch.empty(0)

    return torch.cat(all_activations, dim=0)


def split_by_attribute_label(examples, attribute, positive_threshold):
    """
    Splits (flat) examples into (positive_examples, negative_examples)
    for a single attribute, based on human_scores[attribute], skipping
    any example missing that score. Pooled and unpaired in the
    --fit-source helpsteer2 case; for --fit-source truthfulqa, every
    question contributes exactly one positive (best_answer, score 4)
    and one negative (first incorrect_answers entry, score 0) row, so
    the split just recovers those same pairs by threshold.
    """

    positive, negative = [], []

    for ex in examples:

        score = ex.get("human_scores", {}).get(attribute)

        if score is None:
            continue

        if score >= positive_threshold:
            positive.append(ex)
        else:
            negative.append(ex)

    return positive, negative


# ---------------------------------------------------------------------
# CAA -- Contrastive Activation Addition
# ---------------------------------------------------------------------

def build_caa_vector(
    fit_examples,
    tokenizer,
    model,
    device,
    layer_idx,
    attributes,
    positive_threshold,
    batch_size,
    normalize_per_attribute=False,
):
    """
    For each attribute, computes the DiffMean vector (mean positive
    activation - mean negative activation) at layer_idx, then
    combines across the requested attributes to get a single steering
    direction.
    """

    print("=" * 60)
    print("Building CAA steering vector")
    print("=" * 60)

    start = time.perf_counter()

    per_attribute = {}
    combined = None

    for attribute in attributes:

        positive_ex, negative_ex = split_by_attribute_label(
            fit_examples, attribute, positive_threshold,
        )

        print(
            f"[{attribute}] {len(positive_ex)} positive / "
            f"{len(negative_ex)} negative fitting examples"
        )

        if not positive_ex or not negative_ex:
            print(
                f"[{attribute}] skipped -- need at least one "
                f"positive and one negative example."
            )
            continue

        pos_acts = collect_layer_activations(
            positive_ex, tokenizer, model, device, layer_idx, batch_size,
        )

        neg_acts = collect_layer_activations(
            negative_ex, tokenizer, model, device, layer_idx, batch_size,
        )

        diff = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)

        norm = diff.norm().item()

        per_attribute[attribute] = {
            "num_positive": len(positive_ex),
            "num_negative": len(negative_ex),
            "raw_diff_norm": norm,
        }

        if norm == 0:
            continue

        contribution = (diff / norm) if normalize_per_attribute else diff

        combined = contribution if combined is None else combined + contribution

    elapsed = time.perf_counter() - start

    if combined is None:
        raise ValueError(
            "Could not build a CAA vector for any of the requested "
            "attributes -- check --positive-threshold and that the "
            "fitting split has human_scores for these attributes."
        )

    print(f"CAA vector built in {elapsed:.2f}s")

    return combined, per_attribute, elapsed


def make_caa_edit_fn(vector, multiplier, device, dtype):
    """
    `device`/`dtype` are just the fallback used to precompute the
    scaled vector once. The actual add always re-casts to
    hidden_states.device/.dtype at call time, because with
    device_map="auto" (multi-GPU sharding) the layer being steered
    may not live on `device` at all.
    """

    steer_cpu = (vector * multiplier).to(dtype=torch.float32).cpu()
    cache = {}

    def edit_fn(hidden_states):
        key = (hidden_states.device, hidden_states.dtype)
        if key not in cache:
            cache[key] = steer_cpu.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        return hidden_states + cache[key]

    return edit_fn


# ---------------------------------------------------------------------
# MAT-Steer -- Multi-Attribute Targeted Steering
# ---------------------------------------------------------------------

class MATSteerModule(nn.Module):
    """
    Per-attribute gated steering vectors, following Nguyen et al.
    2025 (arXiv:2502.12446), Section 3:

      G_t(a)      = sigmoid(w_t . a + b_t)                    (Eq. 3)
      f(a)        = a + sum_t G_t(a) * theta_t                (Eq. 4)
      normalized  = f(a) * ||a|| / ||f(a)||                   (Eq. 9)
    """

    def __init__(self, hidden_dim, attributes, init_vectors=None):

        super().__init__()

        self.attributes = list(attributes)
        num_attrs = len(self.attributes)

        if init_vectors is not None:
            theta_init = torch.stack(
                [init_vectors[a] for a in self.attributes], dim=0
            )
        else:
            theta_init = 0.01 * torch.randn(num_attrs, hidden_dim)

        self.theta = nn.Parameter(theta_init.clone())
        self.w = nn.Parameter(0.01 * torch.randn(num_attrs, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_attrs))

    def gates(self, a):
        logits = torch.einsum("...d,td->...t", a, self.w) + self.b
        return torch.sigmoid(logits)

    def steer(self, a):
        """Applies Eq. 4 (unnormalized) -- used during training."""
        g = self.gates(a)
        delta = torch.einsum("...t,td->...d", g, self.theta)
        return a + delta, g

    def steer_normalized(self, a):
        """Applies Eq. 4 + Eq. 9 (norm-preserving) -- used at inference."""
        edited, g = self.steer(a)
        orig_norm = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        edited_norm = edited.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return edited * (orig_norm / edited_norm), g


def gaussian_mmd(x, y, sigma):
    """
    Squared MMD between samples x [N, d] and y [M, d] with a Gaussian
    kernel.
    """

    def kernel(a, b):
        dist_sq = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = kernel(x, x).mean()
    k_yy = kernel(y, y).mean()
    k_xy = kernel(x, y).mean()

    return k_xx + k_yy - 2 * k_xy


def train_mat_steer(
    fit_examples,
    tokenizer,
    model,
    device,
    layer_idx,
    attributes,
    positive_threshold,
    activation_batch_size,
    steps,
    lr,
    lambda_pos,
    lambda_sparse,
    lambda_ortho,
    mmd_sigma,
):
    """
    Collects per-attribute positive/negative activation sets once,
    then trains the MATSteerModule's gates + steering vectors against
    the full loss (paper Eq. 11):

        L = L_MMD + lambda_pos * L_pos
              + lambda_sparse * L_sparse + lambda_ortho * L_ortho

    Activations are collected at self_attn.o_proj's INPUT (see
    collect_attn_input_activations), not the decoder-layer
    residual-stream output CAA uses.
    """

    print("=" * 60)
    print("Collecting activations for MAT-Steer (self_attn.o_proj input)")
    print("=" * 60)

    pos_by_attr, neg_by_attr = {}, {}

    for attribute in attributes:

        positive_ex, negative_ex = split_by_attribute_label(
            fit_examples, attribute, positive_threshold,
        )

        print(
            f"[{attribute}] {len(positive_ex)} positive / "
            f"{len(negative_ex)} negative fitting examples"
        )

        if not positive_ex or not negative_ex:
            raise ValueError(
                f"MAT-Steer needs both positive and negative fitting "
                f"examples for attribute '{attribute}'; adjust "
                f"--holdout-frac, --positive-threshold, or drop this "
                f"attribute from --steer-attributes."
            )

        pos_by_attr[attribute] = collect_attn_input_activations(
            positive_ex, tokenizer, model, device, layer_idx,
            activation_batch_size,
        ).to(device)

        neg_by_attr[attribute] = collect_attn_input_activations(
            negative_ex, tokenizer, model, device, layer_idx,
            activation_batch_size,
        ).to(device)

    hidden_dim = next(iter(pos_by_attr.values())).shape[-1]

    init_vectors = {
        attribute: (
            pos_by_attr[attribute].mean(dim=0)
            - neg_by_attr[attribute].mean(dim=0)
        )
        for attribute in attributes
    }

    module = MATSteerModule(
        hidden_dim, attributes, init_vectors=init_vectors,
    ).to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    print("=" * 60)
    print("Training MAT-Steer gates + steering vectors")
    print("=" * 60)

    start = time.perf_counter()

    for step in range(steps):

        optimizer.zero_grad()

        mmd_loss = 0.0
        pos_loss = 0.0
        sparse_loss = 0.0

        for t_idx, attribute in enumerate(module.attributes):

            pos_acts = pos_by_attr[attribute]
            neg_acts = neg_by_attr[attribute]

            edited_neg, gates_neg = module.steer_normalized(neg_acts)
            mmd_loss = mmd_loss + gaussian_mmd(pos_acts, edited_neg, mmd_sigma)

            _, gates_pos = module.steer(pos_acts)
            pos_loss = pos_loss + (gates_pos[:, t_idx] ** 2).mean()

            sparse_loss = sparse_loss + gates_neg[:, t_idx].abs().mean()

        theta_norm = F.normalize(module.theta, dim=-1)
        cos_sim = theta_norm @ theta_norm.T
        off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
        ortho_loss = (off_diag ** 2).sum()

        total_loss = (
            mmd_loss
            + lambda_pos * pos_loss
            + lambda_sparse * sparse_loss
            + lambda_ortho * ortho_loss
        )

        total_loss.backward()
        optimizer.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(
                f"step {step:4d}/{steps} | total={total_loss.item():.4f} "
                f"mmd={float(mmd_loss):.4f} pos={float(pos_loss):.4f} "
                f"sparse={float(sparse_loss):.4f} "
                f"ortho={float(ortho_loss):.4f}"
            )

    elapsed = time.perf_counter() - start

    print(f"MAT-Steer training completed in {elapsed:.2f}s")

    module.eval()

    return module, elapsed


def make_mat_steer_edit_fn(module, device, dtype):
    """
    Same device-safety concern as make_caa_edit_fn: the steered layer
    may not be on `device` under multi-GPU sharding.
    """

    module_cpu = module.to("cpu")
    cache = {}

    def edit_fn(hidden_states):
        dev = hidden_states.device
        if dev not in cache:
            m = MATSteerModule(
                module_cpu.theta.shape[1], module_cpu.attributes,
            ).to(device=dev, dtype=torch.float32)
            m.load_state_dict(module_cpu.state_dict())
            m.eval()
            cache[dev] = m

        orig_dtype = hidden_states.dtype
        a = hidden_states.to(device=dev, dtype=torch.float32)
        with torch.no_grad():
            edited, _ = cache[dev].steer_normalized(a)
        return edited.to(dtype=orig_dtype)

    return edit_fn


# ---------------------------------------------------------------------
# Build HelpSteer2-shaped fit examples directly from TruthfulQA's own
# correct/incorrect answers (--fit-source truthfulqa)
# ---------------------------------------------------------------------

def build_fit_examples_from_truthfulqa_generation(examples, attribute_name="helpfulness"):
    """
    Converts TruthfulQA generation-task examples (question/
    best_answer/correct_answers/incorrect_answers) into the flat
    (prompt/response/human_scores) rows split_by_attribute_label /
    build_caa_vector / train_mat_steer already expect.

    Exactly ONE pair per question: best_answer (score 4) vs. the
    FIRST entry in incorrect_answers (score 0) -- not every
    correct_answer/incorrect_answer combination. A clear better/worse
    direction, not a graded 0-4 rating, so a non-binary score would be
    fabricating precision that isn't in the data. Both rows from one
    example share that example's OWN question as "prompt", so
    positive and negative are genuinely paired on the same prompt,
    not drawn independently from different questions.
    """

    fit_examples = []

    for idx, ex in enumerate(examples):

        question_prompt = tqg.GENERATION_PROMPT_TEMPLATE.format(
            question=ex["question"],
        )

        correct_answer = ex["best_answer"]
        incorrect_answer = ex["incorrect_answers"][0]

        fit_examples.append({
            "id": f"tqa-gen-fit-{idx}-pos",
            "prompt": question_prompt,
            "response": correct_answer,
            "human_scores": {attribute_name: 4},
        })

        fit_examples.append({
            "id": f"tqa-gen-fit-{idx}-neg",
            "prompt": question_prompt,
            "response": incorrect_answer,
            "human_scores": {attribute_name: 0},
        })

    return fit_examples


# ---------------------------------------------------------------------
# Per-variant generation + judging, reusing
# evaluate_truthfulqa_generation.py's functions unchanged
# ---------------------------------------------------------------------

def run_variant_generation(variant_name, hook_cls, hook_target, edit_fn,
                            examples, tokenizer, model, device, args):
    """
    Optionally attaches a steering hook (edit_fn=None means baseline),
    then runs evaluate_truthfulqa_generation.generate_answers exactly
    as-is -- no changes needed to that function itself, since steering
    is entirely handled by the external hook.
    """

    print("#" * 60)
    print(f"Generating: {variant_name}")
    print("#" * 60)

    hook = None
    if edit_fn is not None:
        hook = hook_cls(hook_target, edit_fn)

    try:
        generated_answers, generation_time = tqg.generate_answers(
            examples, tokenizer, model, device,
            args.batch_size, args.max_new_tokens,
        )
    finally:
        if hook is not None:
            hook.remove()

    return generated_answers, generation_time


def judge_variant(variant_name, examples, generated_answers,
                   judge_tokenizer, judge_model, device, args):
    """
    Runs all three of evaluate_truthfulqa_generation.py's judgement
    passes (truthful / informative / win_rate_vs_best_answer) and
    summarizes them -- unchanged logic, just called once per variant.
    """

    print("#" * 60)
    print(f"Judging: {variant_name}")
    print("#" * 60)

    truthful_verdicts, truthful_raw, truthful_time = tqg.run_yes_no_judgment(
        "truthful", tqg.build_truthful_prompt, examples, generated_answers,
        judge_tokenizer, judge_model, device,
        args.batch_size, args.judge_max_new_tokens,
    )

    informative_verdicts, informative_raw, informative_time = tqg.run_yes_no_judgment(
        "informative", tqg.build_informative_prompt, examples, generated_answers,
        judge_tokenizer, judge_model, device,
        args.batch_size, args.judge_max_new_tokens,
    )

    win_rate_verdicts, win_rate_raw, win_rate_time = tqg.run_yes_no_judgment(
        "win_rate_vs_best_answer", tqg.build_win_rate_prompt, examples,
        generated_answers, judge_tokenizer, judge_model, device,
        args.batch_size, args.judge_max_new_tokens,
    )

    truthful_scores, truthful_percent, truthful_unparsed = tqg.summarize_yes_no(
        truthful_verdicts, "Truthful"
    )
    informative_scores, informative_percent, informative_unparsed = tqg.summarize_yes_no(
        informative_verdicts, "Informative"
    )
    win_rate_scores, win_rate_percent, win_rate_unparsed = tqg.summarize_yes_no(
        win_rate_verdicts, "Win rate vs. best_answer"
    )

    truthful_and_informative_scores = [
        1 if (t and i) else 0
        for t, i in zip(truthful_scores, informative_scores)
    ]
    truthful_and_informative_percent = (
        sum(truthful_and_informative_scores) / len(truthful_and_informative_scores) * 100
        if truthful_and_informative_scores else 0.0
    )

    metrics = {
        "truthful_percent": truthful_percent,
        "informative_percent": informative_percent,
        "truthful_and_informative_percent": truthful_and_informative_percent,
        "llm_judge_win_rate": win_rate_percent,
        "truthful_judge_time_seconds": truthful_time,
        "informative_judge_time_seconds": informative_time,
        "win_rate_judge_time_seconds": win_rate_time,
        "num_unparsed_truthful": truthful_unparsed,
        "num_unparsed_informative": informative_unparsed,
        "num_unparsed_win_rate": win_rate_unparsed,
    }

    per_example = [
        {
            "id": ex["id"], "type": ex["type"], "category": ex["category"],
            "source": ex["source"], "question": ex["question"],
            "best_answer": ex["best_answer"],
            "correct_answers": ex["correct_answers"],
            "incorrect_answers": ex["incorrect_answers"],
            "generated_response": generated,
            "truthful": t_verdict, "truthful_raw_output": t_raw,
            "informative": i_verdict, "informative_raw_output": i_raw,
            "wins_vs_best_answer": w_verdict, "win_rate_raw_output": w_raw,
        }
        for ex, generated, t_verdict, t_raw, i_verdict, i_raw, w_verdict, w_raw in zip(
            examples, generated_answers,
            truthful_verdicts, truthful_raw,
            informative_verdicts, informative_raw,
            win_rate_verdicts, win_rate_raw,
        )
    ]

    return metrics, per_example


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Compare a base model against CAA-steered and "
                     "MAT-Steer-steered variants on TruthfulQA's "
                     "generation task (truthful/informative/win rate).",
    )

    parser.add_argument(
        "--generation-model",
        default=os.environ.get("GENERATION_MODEL", tqg.DEFAULT_GENERATION_MODEL),
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", tqg.DEFAULT_JUDGE_MODEL),
    )
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("DATASET_PATH", tqg.DEFAULT_DATASET),
        help="TruthfulQA generation-task JSON/JSONL -- used ONLY for "
             "evaluation, never for fitting.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-max-new-tokens", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=None)

    # -- fitting (only needed if not using --load-caa-vector /
    # --load-mat-steer) --

    parser.add_argument(
        "--fit-source",
        choices=["helpsteer2", "truthfulqa"],
        default="helpsteer2",
        help="Where to fit the CAA vector / MAT-Steer from. "
             "'helpsteer2' (default): --fit-dataset-path is a "
             "HelpSteer2-shaped file with human_scores, unrelated to "
             "TruthfulQA. 'truthfulqa': fits directly from "
             "TruthfulQA's own best_answer/correct_answers "
             "(positive) vs. incorrect_answers (negative) -- "
             "genuinely paired per-question contrastive examples, no "
             "external dataset needed. Forces --steer-attributes to "
             "just ['helpfulness'], since there's no per-attribute "
             "breakdown in correct/incorrect answers.",
    )

    parser.add_argument(
        "--fit-dataset-path",
        default=None,
        help="With --fit-source helpsteer2: a HelpSteer2-shaped JSON "
             "(human_scores), e.g. helpsteer2_train.json. With "
             "--fit-source truthfulqa: a SEPARATE TruthfulQA "
             "generation-shaped file (question/best_answer/"
             "correct_answers/incorrect_answers), disjoint from "
             "--dataset-path -- if omitted in truthfulqa mode, "
             "--dataset-path itself is shuffled and split "
             "(--holdout-frac/--split-seed) into a fitting portion "
             "and a disjoint evaluation portion instead. Not needed "
             "at all if using --load-caa-vector / --load-mat-steer.",
    )

    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Only used with --fit-source truthfulqa AND no "
             "--fit-dataset-path: fraction of --dataset-path held "
             "out for evaluation, the rest used for fitting (default "
             "0.2 -> 80/20 fit/eval split).",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the --fit-source truthfulqa internal shuffle "
             "split (see --holdout-frac).",
    )

    parser.add_argument(
        "--steer-attributes", nargs="+",
        default=["helpfulness", "correctness", "coherence"],
        choices=list(SCORE_ATTRIBUTES),
        help="Ignored (forced to ['helpfulness']) when --fit-source "
             "is truthfulqa.",
    )
    parser.add_argument("--steer-layer", type=int, default=14)
    parser.add_argument("--positive-threshold", type=int, default=3)
    parser.add_argument("--activation-batch-size", type=int, default=8)

    parser.add_argument("--caa-multiplier", type=float, default=1.0)
    parser.add_argument("--caa-normalize-per-attribute", action="store_true")

    parser.add_argument("--mat-steer-steps", type=int, default=300)
    parser.add_argument("--mat-steer-lr", type=float, default=1e-3)
    parser.add_argument("--mat-steer-lambda-pos", type=float, default=0.8)
    parser.add_argument("--mat-steer-lambda-sparse", type=float, default=0.8)
    parser.add_argument("--mat-steer-lambda-ortho", type=float, default=0.1)
    parser.add_argument("--mmd-sigma", type=float, default=2.0)

    parser.add_argument("--save-steering-dir", default=None)
    parser.add_argument("--load-caa-vector", default=None)
    parser.add_argument("--load-mat-steer", default=None)

    parser.add_argument("--skip-caa", action="store_true")
    parser.add_argument("--skip-mat-steer", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    has_vector_source = (
        args.load_caa_vector or args.load_mat_steer
        or args.fit_dataset_path
        or args.fit_source == "truthfulqa"  # can self-split --dataset-path
    )

    if not has_vector_source:
        raise ValueError(
            "Need one of: --fit-dataset-path, --fit-source truthfulqa "
            "(to split --dataset-path itself), or --load-caa-vector / "
            "--load-mat-steer."
        )

    if args.fit_source == "truthfulqa" and args.steer_attributes != ["helpfulness"]:
        print(
            "--fit-source truthfulqa only supports a single "
            "'helpfulness' axis (correct vs. incorrect answers carry "
            "no per-attribute breakdown) -- overriding "
            "--steer-attributes to ['helpfulness']."
        )
        args.steer_attributes = ["helpfulness"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(args.save_steering_dir) if args.save_steering_dir else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TruthfulQA Generation Steering Comparison: "
          "Baseline vs CAA vs MAT-Steer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load eval examples + fit examples
    # -------------------------------------------------

    if args.fit_source == "truthfulqa":

        if args.fit_dataset_path:

            # Two separate, already-disjoint TruthfulQA-generation
            # files: one to fit correct/incorrect contrasts from, one
            # to evaluate baseline/CAA/MAT-Steer generation on.
            fit_source_examples = tqg.load_truthfulqa_generation_examples(
                args.fit_dataset_path,
            )
            eval_examples = tqg.load_truthfulqa_generation_examples(
                args.dataset_path,
            )

            if args.max_examples:
                eval_examples = eval_examples[: args.max_examples]
                print(f"Limited evaluation to {args.max_examples} examples")

            print(
                f"Using separate fit/eval TruthfulQA files: "
                f"{len(fit_source_examples)} questions from "
                f"--fit-dataset-path for fitting, {len(eval_examples)} "
                f"questions from --dataset-path for evaluation."
            )

        else:

            # One file, internally shuffled and split.
            all_examples = tqg.load_truthfulqa_generation_examples(
                args.dataset_path,
            )

            if args.max_examples:
                all_examples = all_examples[: args.max_examples]
                print(f"Limited evaluation to {args.max_examples} examples")

            holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
            shuffled = all_examples.copy()
            random.Random(args.split_seed).shuffle(shuffled)

            fit_source_examples = (
                shuffled[:-holdout_n] if holdout_n < len(shuffled) else []
            )
            eval_examples = shuffled[-holdout_n:]

            print(
                f"Split --dataset-path internally: "
                f"{len(fit_source_examples)} questions for fitting "
                f"(correct vs. incorrect answers), {len(eval_examples)} "
                f"held out for evaluation."
            )

        fit_examples = build_fit_examples_from_truthfulqa_generation(
            fit_source_examples, attribute_name="helpfulness",
        )

        print(
            f"Built {len(fit_examples)} paired fit rows from "
            f"{len(fit_source_examples)} questions' correct/incorrect "
            f"answers."
        )

    else:  # --fit-source helpsteer2 (original behavior)

        eval_examples = tqg.load_truthfulqa_generation_examples(args.dataset_path)

        if args.max_examples:
            eval_examples = eval_examples[: args.max_examples]
            print(f"Limited evaluation to {args.max_examples} examples")

        fit_examples = (
            load_helpsteer2(args.fit_dataset_path)
            if args.fit_dataset_path else []
        )

    gen_tokenizer, gen_model = tqg.load_generation_model(
        args.generation_model, device,
    )

    layer = get_decoder_layers(gen_model)[args.steer_layer]
    o_proj = get_attn_o_proj(gen_model, args.steer_layer)
    model_dtype = next(gen_model.parameters()).dtype

    variants = {}  # name -> (generated_answers, generation_time)
    steering_meta = {}

    # --- Baseline ---
    variants["baseline"] = run_variant_generation(
        "baseline", SteeringHook, layer, None,
        eval_examples, gen_tokenizer, gen_model, device, args,
    )

    # --- CAA ---
    if not args.skip_caa:

        if args.load_caa_vector:
            print(f"Loading CAA vector from {args.load_caa_vector}")
            loaded = torch.load(args.load_caa_vector, map_location="cpu")
            caa_vector = loaded["vector"].to(torch.float32)
            caa_per_attribute = loaded.get("per_attribute", {})
        else:
            caa_vector, caa_per_attribute, _ = build_caa_vector(
                fit_examples, gen_tokenizer, gen_model, device,
                args.steer_layer, args.steer_attributes,
                args.positive_threshold, args.activation_batch_size,
                normalize_per_attribute=args.caa_normalize_per_attribute,
            )
            caa_save_path = save_dir / f"caa_vector_tqa_gen_{timestamp}.pt"
            torch.save(
                {"vector": caa_vector, "per_attribute": caa_per_attribute,
                 "attributes": args.steer_attributes, "layer": args.steer_layer},
                caa_save_path,
            )
            print(f"Saved CAA vector to {caa_save_path}")

        caa_edit_fn = make_caa_edit_fn(
            caa_vector, args.caa_multiplier, device, model_dtype,
        )

        variants["caa"] = run_variant_generation(
            "caa", SteeringHook, layer, caa_edit_fn,
            eval_examples, gen_tokenizer, gen_model, device, args,
        )

        steering_meta["caa"] = {
            "attributes": args.steer_attributes, "layer": args.steer_layer,
            "hook_site": "decoder_layer_output (residual stream)",
            "multiplier": args.caa_multiplier,
            "per_attribute": caa_per_attribute,
        }

    # --- MAT-Steer ---
    if not args.skip_mat_steer:

        if args.load_mat_steer:
            print(f"Loading MAT-Steer module from {args.load_mat_steer}")
            loaded = torch.load(args.load_mat_steer, map_location="cpu")
            mat_module = MATSteerModule(loaded["hidden_dim"], loaded["attributes"])
            mat_module.load_state_dict(loaded["state_dict"])
            mat_module.eval()
            mat_module = mat_module.to(device=device, dtype=torch.float32)
        else:
            mat_module, _ = train_mat_steer(
                fit_examples, gen_tokenizer, gen_model, device,
                args.steer_layer, args.steer_attributes,
                args.positive_threshold, args.activation_batch_size,
                args.mat_steer_steps, args.mat_steer_lr,
                args.mat_steer_lambda_pos, args.mat_steer_lambda_sparse,
                args.mat_steer_lambda_ortho, args.mmd_sigma,
            )
            mat_save_path = save_dir / f"mat_steer_tqa_gen_{timestamp}.pt"
            torch.save(
                {"state_dict": mat_module.state_dict(),
                 "attributes": mat_module.attributes,
                 "hidden_dim": mat_module.theta.shape[1],
                 "layer": args.steer_layer},
                mat_save_path,
            )
            print(f"Saved MAT-Steer module to {mat_save_path}")

        mat_edit_fn = make_mat_steer_edit_fn(mat_module, device, model_dtype)

        variants["mat_steer"] = run_variant_generation(
            "mat_steer", AttnOProjSteeringHook, o_proj, mat_edit_fn,
            eval_examples, gen_tokenizer, gen_model, device, args,
        )

        steering_meta["mat_steer"] = {
            "attributes": args.steer_attributes, "layer": args.steer_layer,
            "hook_site": "self_attn.o_proj input (pre-projection "
                         "attention output)",
        }

    del gen_model
    del gen_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Judge every variant with ONE judge-model load
    # -------------------------------------------------

    judge_tokenizer, judge_model = tqg.load_judge_model(
        args.judge_model, device,
    )

    variant_results = {}

    for name, (generated_answers, generation_time) in variants.items():

        metrics, per_example = judge_variant(
            name, eval_examples, generated_answers,
            judge_tokenizer, judge_model, device, args,
        )

        metrics["generation_time_seconds"] = generation_time
        metrics["num_examples"] = len(eval_examples)

        variant_results[name] = {"metrics": metrics, "examples": per_example}

    del judge_model
    del judge_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Comparison table + save
    # -------------------------------------------------

    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'variant':12s} {'truthful':>9s} {'informat.':>10s} "
          f"{'T&I':>7s} {'win_rate':>9s} {'gen_s':>8s}")

    for name, result in variant_results.items():
        m = result["metrics"]
        print(
            f"{name:12s} {m['truthful_percent']:8.2f}% "
            f"{m['informative_percent']:9.2f}% "
            f"{m['truthful_and_informative_percent']:6.2f}% "
            f"{m['llm_judge_win_rate']:8.2f}% "
            f"{m['generation_time_seconds']:8.2f}"
        )

    results = {
        "timestamp": timestamp,
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
        "dataset": args.dataset_path,
        "fit_source": args.fit_source,
        "fit_dataset": args.fit_dataset_path,
        "num_fit_examples": len(fit_examples),
        "steer_attributes": args.steer_attributes,
        "steer_layer": args.steer_layer,
        "steering_meta": steering_meta,
        "variants": variant_results,
    }

    output_file = output_dir / f"steering_eval_truthfulqa_generation_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Evaluation Finished")
    print("=" * 60)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()