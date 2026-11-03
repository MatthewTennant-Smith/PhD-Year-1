"""
steering_eval_ultrafeedback.py -- Compare a base generation model
against the SAME model under two activation-steering interventions
(CAA and MAT-Steer), evaluated with ultrafeedback.py's OWN
LLM-as-a-judge pipeline: win rate against ALL human-reference
completions for each instruction, weighted per-attribute scoring
(helpfulness / honesty / instruction_following / truthfulness), and
generation time.

This is a port of steering_eval.py (which drove the same CAA /
MAT-Steer comparison against helpsteer2.py's single-reference
pipeline) onto ultrafeedback.py instead. All of the steering-specific
code (hooks, vector fitting, MAT-Steer module/training) is unchanged
-- it's model-architecture code, not dataset code. What changed:

  * The dataset loader/shape. helpsteer2 examples are flat
    (id/prompt/response/human_scores): one response, one score dict,
    directly usable to fit a CAA/MAT-Steer contrastive vector.
    ultrafeedback examples are nested (id/prompt/"references": a
    LIST of completions, each with its own response + human_scores).
    flatten_examples_for_fitting() below flattens each instruction's
    reference completions into individual (prompt, response,
    human_scores) records -- but ONLY for the steering-vector-fitting
    split. The evaluation split is left in ultrafeedback's native
    nested shape and run through ultrafeedback.py's own functions
    untouched, so eval numbers mean exactly what they mean when you
    run ultrafeedback.py directly.

  * The judge/scoring wiring. helpsteer2.run_judge takes
    (prompts, references, generated_answers) -- one reference per
    example, pairwise. ultrafeedback.run_judge instead takes a flat
    list of (candidate vs. every reference completion) comparisons,
    built via ultrafeedback.build_comparisons(), and
    ultrafeedback.parse_judgements() returns (scores, win_rate) --
    two values, not three (no unparsed-pairwise-verdict count).
    judge_and_score_variant() below calls ultrafeedback's actual
    functions in the order its own main() does, so the metrics this
    script reports for baseline/CAA/MAT-Steer are IDENTICAL in kind
    and computation to the metrics ultrafeedback.py reports for a
    single unsteered model: llm_judge_win_rate (pooled across every
    candidate-vs-reference comparison), win_rate_by_reference_model,
    avg_llm_<attribute> / unparsed_llm_<attribute> for all four
    weighted attributes, plus generation/judge/scoring timings and
    per-example win rates. Nothing about how those numbers are
    computed was changed -- only which variant (baseline / CAA /
    MAT-Steer) is being fed through the same computation.

  * SCORE_ATTRIBUTES / --steer-attributes choices now come from
    ultrafeedback.SCORE_ATTRIBUTES ("helpfulness", "honesty",
    "instruction_following", "truthfulness") instead of helpsteer2's
    set. Default is the same "helpfulness" + 2 others pattern as the
    original script, here "helpfulness", "honesty",
    "instruction_following" -- pass --steer-attributes with
    "truthfulness" included if you want all four.

  * --positive-threshold: UltraFeedback's four weighted attributes
    are rated 1-5 (ultrafeedback.SCORE_MIN/SCORE_MAX), NOT HelpSteer's
    0-4 scale that the original >=3-is-positive rule was calibrated
    to. The flag/default (>=3 positive) is left unchanged here so the
    CLI surface matches the original script, but on a 1-5 scale that
    threshold includes the middle score as "positive," which may not
    give a very separated contrastive split. Consider passing
    --positive-threshold 4 to require a clearly-positive completion
    for the "positive" side.

Everything else (hooks, CAA vector construction, MAT-Steer module and
training, argument names, --save-steering-dir / --load-caa-vector /
--load-mat-steer, output JSON shape for steering_meta) mirrors the
original steering_eval.py as closely as the dataset differences allow.

Example:

python steering_eval_ultrafeedback.py \
    --generation-model /home/xrai/models/Meta-Llama-3.1-8B-Instruct \
    --judge-model /home/xrai/models/flowjudge \
    --dataset-path /home/xrai/datasets/ultrafeedback/ultrafeedback.json \
    --steer-attributes helpfulness honesty instruction_following \
    --steer-layer 14 \
    --caa-multiplier 4.0 \
    --mat-steer-steps 300
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

# Reuses dataset loading, generation, judging (vs-all-references),
# weighted attribute scoring, and per-example/per-model win-rate
# breakdowns exactly as defined in the original evaluation script.
import ultrafeedback as uf


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Compare a base model against CAA-steered and "
                     "MAT-Steer-steered variants on UltraFeedback, "
                     "using ultrafeedback.py's judge/scoring/timing "
                     "pipeline (win rate vs ALL reference "
                     "completions, weighted attribute scoring)."
    )

    parser.add_argument(
        "--generation-model",
        default=os.environ.get(
            "GENERATION_MODEL",
            uf.DEFAULT_GENERATION_MODEL,
        ),
    )

    parser.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL",
            uf.DEFAULT_JUDGE_MODEL,
        ),
    )

    parser.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH",
            uf.DEFAULT_DATASET,
        ),
        help="Dataset used for EVALUATION (baseline/CAA/MAT-Steer "
             "judging+scoring, via ultrafeedback.py's own vs-all-"
             "references pipeline). If --fit-dataset-path is not "
             "given, this same file is also split (--holdout-frac / "
             "--split-seed) at the instruction level to provide the "
             "fitting portion.",
    )

    parser.add_argument(
        "--fit-dataset-path",
        default=None,
        help="Optional separate dataset used ONLY to fit the CAA "
             "vector / train MAT-Steer (e.g. a pre-made "
             "ultrafeedback_train.json). When given, --dataset-path "
             "is used ENTIRELY for evaluation (no internal split, "
             "--holdout-frac/--split-seed are ignored) and every "
             "reference completion in this file is flattened into a "
             "fitting example -- i.e. you're supplying already-"
             "disjoint train/test files rather than having this "
             "script split one file itself.",
    )

    parser.add_argument(
        "--output-dir",
        default=uf.DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-max-new-tokens", type=int, default=10)
    parser.add_argument("--score-max-new-tokens", type=int, default=60)
    parser.add_argument("--max-examples", type=int, default=None)

    # -- steering-specific args --

    parser.add_argument(
        "--steer-attributes",
        nargs="+",
        default=["helpfulness", "honesty", "instruction_following"],
        choices=list(uf.SCORE_ATTRIBUTES),
        help="Which UltraFeedback weighted attributes to build "
             "steering vectors for (both CAA and MAT-Steer). Passing "
             "more than one for CAA sums their diff-of-means vectors "
             "into a single steering direction (unit-normalized "
             "first if --caa-normalize-per-attribute); MAT-Steer "
             "keeps one vector + gate per attribute, as in the "
             "paper.",
    )

    parser.add_argument(
        "--steer-layer",
        type=int,
        default=14,
        help="Decoder layer index (0-indexed, index into "
             "model.model.layers) at which activations are collected "
             "and steering vectors are applied. The CAA paper finds "
             "13-15 works best for Llama 2 7B/13B; used as the "
             "default here too, but sweep it for other models.",
    )

    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of INSTRUCTIONS (not individual reference "
             "completions) held out for evaluation only -- default "
             "0.2 gives an 80/20 fit/eval split at the instruction "
             "level. All reference completions for a fit-side "
             "instruction get flattened into fitting examples; none "
             "of an eval-side instruction's completions are used for "
             "fitting.",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for shuffling the dataset (by instruction) before "
             "the fit/eval split. The split shuffles rather than "
             "taking a contiguous slice, since a dataset file ordered "
             "by source/topic would otherwise put systematically "
             "different examples in the fitting vs. eval split. "
             "Fixed seed -> reproducible split across runs.",
    )

    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=3,
        help="Only used when --pairing-strategy=threshold. A "
             "flattened (prompt, response) fitting example's "
             "human_scores[attr] >= this is 'positive', below it is "
             "'negative'. NOTE: UltraFeedback's four weighted "
             "attributes are rated 1-5 (unlike HelpSteer's 0-4 scale "
             "this default was originally calibrated to) -- consider "
             "--positive-threshold 4 for a more clearly separated "
             "contrastive split.",
    )

    parser.add_argument(
        "--pairing-strategy",
        choices=["threshold", "extremes"],
        default="threshold",
        help="How positive/negative fitting examples are chosen. "
             "'threshold' (default): pool every reference completion "
             "across every fitting instruction and bucket each one "
             "independently by human_scores[attr] >= "
             "--positive-threshold -- unpaired, matches the ODESteer/"
             "CAA-literature DiffMean convention this script defaults "
             "to elsewhere. 'extremes': for each fitting instruction, "
             "pair its HIGHEST-scoring completion (positive) against "
             "its LOWEST-scoring completion (negative) for that "
             "attribute, holding the prompt fixed -- closer to the "
             "original CAA paper's paired A/B construction, and "
             "removes prompt-content variation from the diff-mean "
             "direction. Uses fewer, but cleaner, contrastive pairs; "
             "see --pairing-min-margin.",
    )

    parser.add_argument(
        "--pairing-min-margin",
        type=int,
        default=1,
        help="Only used when --pairing-strategy=extremes. An "
             "instruction's highest/lowest completions for an "
             "attribute are only used as a pair if their score gap is "
             ">= this. Filters out instructions whose completions are "
             "near-tied on that attribute, which would otherwise add "
             "noise rather than signal to the contrastive direction.",
    )

    parser.add_argument(
        "--activation-batch-size",
        type=int,
        default=8,
        help="Batch size used only for the activation-collection "
             "forward passes that build steering vectors.",
    )

    parser.add_argument(
        "--save-steering-dir",
        default=None,
        help="If set, saves the fitted CAA vector (as a .pt dict with "
             "the vector + metadata) and/or the trained MAT-Steer "
             "module (state_dict + metadata) into this directory, "
             "timestamped, so you can re-run generation/judging later "
             "against the SAME fitted steering direction without "
             "refitting -- e.g. to sweep --caa-multiplier or "
             "re-evaluate on a different --dataset-path. Defaults to "
             "--output-dir if steering methods are being fit and this "
             "isn't set.",
    )

    parser.add_argument(
        "--load-caa-vector",
        default=None,
        help="Path to a .pt file previously saved via "
             "--save-steering-dir. Skips fitting entirely (no "
             "activation collection, no need for --steer-attributes' "
             "positive/negative examples) and runs generation+judging "
             "directly with the loaded vector. --caa-multiplier still "
             "applies on top of the loaded (raw) vector.",
    )

    parser.add_argument(
        "--load-mat-steer",
        default=None,
        help="Path to a .pt file previously saved via "
             "--save-steering-dir. Skips training entirely and runs "
             "generation+judging directly with the loaded module.",
    )

    parser.add_argument(
        "--caa-multiplier",
        type=float,
        default=1.0,
        help="Scalar strength ('T') applied to the CAA steering "
             "vector during generation. When combining multiple "
             "attributes (--steer-attributes with >1 entry) the "
             "summed vector's magnitude is larger, so you'll likely "
             "want to sweep this down, or pass "
             "--caa-normalize-per-attribute to keep per-attribute "
             "magnitude comparable regardless of how many attributes "
             "you combine.",
    )

    parser.add_argument(
        "--caa-normalize-per-attribute",
        action="store_true",
        help="Unit-normalize each per-attribute DiffMean vector "
             "before summing across --steer-attributes, so every "
             "attribute contributes equally regardless of its raw "
             "activation-difference magnitude. Off by default, which "
             "instead sums the RAW DiffMean vectors.",
    )

    parser.add_argument(
        "--mat-steer-steps",
        type=int,
        default=300,
        help="Training steps for MAT-Steer's gates + vectors.",
    )

    parser.add_argument(
        "--mat-steer-lr",
        type=float,
        default=1e-3,
        help="Matches the official MAT-Steer repo's README default "
             "(python steering.py ... --lr 0.001).",
    )

    parser.add_argument(
        "--mat-steer-lambda-pos",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--mat-steer-lambda-sparse",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--mat-steer-lambda-ortho",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--mmd-sigma",
        type=float,
        default=2.0,
        help="Gaussian kernel bandwidth for the MMD alignment loss. "
             "The paper's sigma=2 was tuned for their activation "
             "scale/model; residual-stream norms vary a lot across "
             "model families and layers, so check the printed MMD "
             "loss trace and adjust if it saturates at ~0 or explodes.",
    )

    parser.add_argument(
        "--skip-caa",
        action="store_true",
    )

    parser.add_argument(
        "--skip-mat-steer",
        action="store_true",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Model-architecture helpers (unchanged from steering_eval.py --
# dataset-agnostic)
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
    layer (model.model.layers[i].self_attn.o_proj). This is the
    intervention site the official MAT-Steer repo uses
    (interveners.py: "model.layers[{layer}].self_attn.o_proj.input"),
    matching the paper's own definition (Section 2.1) of the
    per-layer activation as "the output of the self-attention layer"
    -- i.e. the concatenated multi-head attention output BEFORE it's
    mixed back into the residual stream by o_proj, not the decoder
    block's full residual-stream output. CAA steers the latter (see
    SteeringHook below); MAT-Steer steers this instead.
    """

    return get_decoder_layers(model)[layer_idx].self_attn.o_proj


# ---------------------------------------------------------------------
# Steering hooks (unchanged from steering_eval.py)
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
# Dataset flattening -- ONLY used for the steering-vector-fitting
# split. UltraFeedback nests multiple reference completions per
# instruction ("references": [...]); CAA/MAT-Steer fitting needs
# individual (prompt, response, human_scores) records to build
# contrastive positive/negative sets, so each instruction's
# completions are unpacked into one fitting example per completion.
# The evaluation split is NEVER flattened -- it stays in
# ultrafeedback.py's native nested shape and goes straight into
# ultrafeedback.py's own functions unchanged.
# ---------------------------------------------------------------------

def flatten_examples_for_fitting(examples):
    """
    Turns a list of ultrafeedback-shaped examples (each with "prompt"
    and a "references" list of completions) into a flat list of
    {"prompt", "response", "human_scores"} records, one per reference
    completion, suitable for split_by_attribute_label() /
    collect_layer_activations() / collect_attn_input_activations()
    below (all of which expect exactly that shape).
    """

    flattened = []

    for ex in examples:

        for reference in ex["references"]:

            flattened.append({
                "prompt": ex["prompt"],
                "response": reference["response"],
                "human_scores": reference["human_scores"],
                "source_instruction_id": ex["id"],
                "reference_model": reference.get("model", "unknown"),
            })

    return flattened


# ---------------------------------------------------------------------
# Positive/negative example selection for CAA / MAT-Steer fitting.
# Two strategies -- see --pairing-strategy's help text:
#
#   * threshold (default): pooled, unpaired -- every reference
#     completion across every fitting instruction is bucketed
#     independently by human_scores[attr] >= --positive-threshold.
#     Operates on the FLATTENED fit examples.
#
#   * extremes: paired, per-instruction -- for each fitting
#     instruction, its highest-scoring completion for the attribute
#     is paired against its lowest-scoring one, holding the prompt
#     fixed. Closer to the original CAA paper's paired A/B
#     construction. Operates on the NESTED (un-flattened) fit
#     instructions, since it needs to see all of one instruction's
#     completions together to pick its extremes.
# ---------------------------------------------------------------------

def split_by_attribute_label(examples, attribute, positive_threshold):
    """
    Splits (flattened) examples into (positive_examples,
    negative_examples) for a single attribute, based on
    human_scores[attribute], skipping any example missing that score.
    Pooled and unpaired: two records in the resulting lists may come
    from completely different instructions.
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


def select_extreme_pairs(fit_top_examples, attribute, min_margin):
    """
    Per-instruction paired contrastive selection: for each fitting
    instruction, picks the reference completion with the HIGHEST
    human_scores[attribute] as the positive example and the one with
    the LOWEST as the negative example, holding the prompt fixed
    across the pair. This is what the original CAA paper's paired A/B
    prompts approximate for UltraFeedback's multi-completion
    structure (same prompt, only the answer differs) -- unlike
    split_by_attribute_label's pooled, unpaired threshold split,
    where a "positive" example for one instruction gets averaged
    together with a "positive" example from an unrelated instruction.

    Skips any instruction with fewer than 2 completions carrying a
    score for this attribute, or where the highest/lowest score gap
    is below min_margin (near-tied completions on that attribute add
    noise rather than a meaningful contrastive signal).

    Returns (positive_examples, negative_examples) in the same
    {"prompt", "response", "human_scores", ...} shape
    flatten_examples_for_fitting() produces, so build_caa_vector /
    train_mat_steer don't need to know which strategy selected them.
    """

    positive, negative = [], []

    for ex in fit_top_examples:

        scored_refs = [
            (ref["human_scores"].get(attribute), ref)
            for ref in ex["references"]
            if ref["human_scores"].get(attribute) is not None
        ]

        if len(scored_refs) < 2:
            continue

        scored_refs.sort(key=lambda pair: pair[0])

        min_score, min_ref = scored_refs[0]
        max_score, max_ref = scored_refs[-1]

        if max_score - min_score < min_margin:
            continue

        positive.append({
            "prompt": ex["prompt"],
            "response": max_ref["response"],
            "human_scores": max_ref["human_scores"],
            "source_instruction_id": ex["id"],
            "reference_model": max_ref.get("model", "unknown"),
        })

        negative.append({
            "prompt": ex["prompt"],
            "response": min_ref["response"],
            "human_scores": min_ref["human_scores"],
            "source_instruction_id": ex["id"],
            "reference_model": min_ref.get("model", "unknown"),
        })

    return positive, negative


def get_positive_negative_examples(fit_top_examples, fit_examples_flat, attribute, args):
    """
    Dispatches to the selection strategy named by --pairing-strategy.
    build_caa_vector / train_mat_steer call this instead of either
    selection function directly, so they don't need to know which
    strategy is active.
    """

    if args.pairing_strategy == "extremes":
        return select_extreme_pairs(
            fit_top_examples, attribute, args.pairing_min_margin,
        )

    return split_by_attribute_label(
        fit_examples_flat, attribute, args.positive_threshold,
    )


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
    Encodes `prompt + response` for every (flattened) example and
    returns the layer_idx residual-stream activation at the LAST
    token position for each one, as a single [N, hidden_dim] tensor
    (on CPU, float32).

    Because the tokenizer pads on the left (set in
    ultrafeedback.load_generation_model), the last real token is
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


# ---------------------------------------------------------------------
# CAA -- Contrastive Activation Addition (unchanged from
# steering_eval.py)
# ---------------------------------------------------------------------

def build_caa_vector(
    fit_top_examples,
    fit_examples_flat,
    tokenizer,
    model,
    device,
    layer_idx,
    attributes,
    args,
    batch_size,
    normalize_per_attribute=False,
):
    """
    For each attribute, computes the DiffMean vector (mean positive
    activation - mean negative activation) at layer_idx, then
    combines across the requested attributes to get a single steering
    direction. Positive/negative examples come from
    get_positive_negative_examples(), which selects them per
    --pairing-strategy (pooled threshold split, or per-instruction
    highest-vs-lowest pairing).
    """

    print("=" * 60)
    print("Building CAA steering vector")
    print("=" * 60)

    start = time.perf_counter()

    per_attribute = {}
    combined = None

    for attribute in attributes:

        positive_ex, negative_ex = get_positive_negative_examples(
            fit_top_examples, fit_examples_flat, attribute, args,
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
            "pairing_strategy": args.pairing_strategy,
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
# MAT-Steer -- Multi-Attribute Targeted Steering (unchanged from
# steering_eval.py)
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
    fit_top_examples,
    fit_examples_flat,
    tokenizer,
    model,
    device,
    layer_idx,
    attributes,
    args,
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

    Positive/negative examples come from get_positive_negative_examples()
    (pooled threshold split, or per-instruction highest-vs-lowest
    pairing, per --pairing-strategy). Activations are collected at
    self_attn.o_proj's INPUT (see collect_attn_input_activations), not
    the decoder-layer residual-stream output CAA uses.
    """

    print("=" * 60)
    print("Collecting activations for MAT-Steer (self_attn.o_proj input)")
    print("=" * 60)

    pos_by_attr, neg_by_attr = {}, {}

    for attribute in attributes:

        positive_ex, negative_ex = get_positive_negative_examples(
            fit_top_examples, fit_examples_flat, attribute, args,
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
# Running one full variant (baseline / CAA / MAT-Steer) through
# ultrafeedback.py's OWN generation + judge + scoring pipeline
# ---------------------------------------------------------------------

def run_variant(
    variant_name,
    edit_fn,
    hook_cls,
    hook_target,
    prompts,
    gen_tokenizer,
    gen_model,
    device,
    args,
):
    """
    Optionally attaches a steering hook (edit_fn=None means baseline,
    unsteered generation), then runs ultrafeedback.generate_answers --
    identical signature/behavior to the helpsteer2 version this
    replaces, so nothing about generation itself changed.
    """

    print("#" * 60)
    print(f"Generating: {variant_name}")
    print("#" * 60)

    hook = None
    if edit_fn is not None:
        hook = hook_cls(hook_target, edit_fn)

    try:
        generated_answers, generation_time = uf.generate_answers(
            prompts,
            gen_tokenizer,
            gen_model,
            device,
            args.batch_size,
            args.max_new_tokens,
        )
    finally:
        if hook is not None:
            hook.remove()

    return generated_answers, generation_time


def judge_and_score_variant(
    variant_name,
    eval_examples,
    eval_prompts,
    generated_answers,
    judge_tokenizer,
    judge_model,
    device,
    args,
):
    """
    Runs exactly ultrafeedback.py's own evaluation pipeline (the same
    calls its main() makes, in the same order) for one variant's
    generated answers -- so the metrics computed here are identical
    in kind and computation to running ultrafeedback.py directly on
    a single model:

      1. build_comparisons(): candidate vs. EVERY reference
         completion for each instruction.
      2. run_judge() + parse_judgements(): pooled pairwise win rate
         across all of those comparisons.
      3. summarize_per_example_win_rates() /
         summarize_win_rate_by_model(): the same two win-rate
         breakdowns ultrafeedback.py reports.
      4. run_scoring() + summarize_attribute_scores(): the four
         weighted attributes (helpfulness/honesty/
         instruction_following/truthfulness), independent of the
         references.
    """

    print("#" * 60)
    print(f"Judging: {variant_name}")
    print("#" * 60)

    comparisons = uf.build_comparisons(eval_examples, generated_answers)

    verdicts, judge_time = uf.run_judge(
        comparisons, judge_tokenizer, judge_model, device,
        args.batch_size, args.judge_max_new_tokens,
    )

    comparison_scores, win_rate = uf.parse_judgements(verdicts)

    per_example_win_rates = uf.summarize_per_example_win_rates(
        eval_examples, comparisons, comparison_scores,
    )

    win_rate_by_model = uf.summarize_win_rate_by_model(
        comparisons, comparison_scores,
    )

    attribute_scores, score_raw_outputs, scoring_time = uf.run_scoring(
        eval_prompts, generated_answers, judge_tokenizer, judge_model,
        device, args.batch_size, args.score_max_new_tokens,
    )

    attribute_averages, attribute_unparsed = uf.summarize_attribute_scores(
        attribute_scores
    )

    metrics = {
        "llm_judge_win_rate": win_rate,
        "win_rate_by_reference_model": win_rate_by_model,
        "judge_time_seconds": judge_time,
        "scoring_time_seconds": scoring_time,
        "num_comparisons": len(comparisons),
    }

    for attr in uf.SCORE_ATTRIBUTES:
        metrics[f"avg_llm_{attr}"] = attribute_averages[attr]
        metrics[f"unparsed_llm_{attr}"] = attribute_unparsed[attr]

    # Re-group the flat comparison verdicts back per example, exactly
    # as ultrafeedback.py's own main() does, for the per-example
    # output records below.
    verdicts_by_example = [[] for _ in eval_examples]

    for comparison, verdict, score in zip(
        comparisons, verdicts, comparison_scores
    ):
        verdicts_by_example[comparison["example_index"]].append({
            "reference_model": comparison["reference_model"],
            "judge_verdict": verdict,
            "score": score,
        })

    return (
        metrics,
        verdicts_by_example,
        per_example_win_rates,
        attribute_scores,
        score_raw_outputs,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path(args.save_steering_dir) if args.save_steering_dir else output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UltraFeedback Steering Comparison: Baseline vs CAA vs MAT-Steer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load fit/eval examples -- either from one file (internally
    # split, by instruction, via --holdout-frac/--split-seed) or from
    # two separate, already-disjoint files (--fit-dataset-path given).
    # Only the FIT side ever gets flattened (one record per reference
    # completion); the EVAL side stays in ultrafeedback's native
    # nested shape.
    # -------------------------------------------------

    eval_examples = uf.load_ultrafeedback(args.dataset_path)

    if args.max_examples:
        eval_examples = eval_examples[: args.max_examples]
        print(f"Limited evaluation to {args.max_examples} examples")

    if args.fit_dataset_path:

        fit_top_examples = uf.load_ultrafeedback(args.fit_dataset_path)
        fit_examples = flatten_examples_for_fitting(fit_top_examples)

        print(
            f"Using separate fit/eval files: {len(fit_top_examples)} "
            f"instructions ({len(fit_examples)} flattened reference "
            f"completions) from --fit-dataset-path for fitting "
            f"steering vectors, {len(eval_examples)} instructions "
            f"from --dataset-path (used in full, no internal split) "
            f"for evaluation."
        )

    else:

        holdout_n = max(1, int(len(eval_examples) * args.holdout_frac))

        shuffled = eval_examples.copy()
        random.Random(args.split_seed).shuffle(shuffled)

        fit_top_examples = (
            shuffled[:-holdout_n] if holdout_n < len(shuffled) else []
        )
        eval_examples = shuffled[-holdout_n:]

        fit_examples = flatten_examples_for_fitting(fit_top_examples)

        print(
            f"Split --dataset-path internally by instruction: "
            f"{len(fit_top_examples)} instructions "
            f"({len(fit_examples)} flattened reference completions) "
            f"for fitting steering vectors, {len(eval_examples)} "
            f"instructions held out for evaluation."
        )

    eval_prompts = [ex["prompt"] for ex in eval_examples]

    # -------------------------------------------------
    # Load generation model once; build steering vectors on it, then
    # generate baseline + each steered variant with it
    # -------------------------------------------------

    gen_tokenizer, gen_model = uf.load_generation_model(
        args.generation_model, device,
    )

    layer = get_decoder_layers(gen_model)[args.steer_layer]
    o_proj = get_attn_o_proj(gen_model, args.steer_layer)
    model_dtype = next(gen_model.parameters()).dtype

    variants = {}  # name -> (generated_answers, generation_time)
    steering_meta = {}

    # --- Baseline (unsteered) ---
    variants["baseline"] = run_variant(
        "baseline", None, SteeringHook, layer,
        eval_prompts,
        gen_tokenizer, gen_model, device, args,
    )

    # --- CAA (steers the decoder layer's residual-stream output) ---
    if not args.skip_caa:

        if args.load_caa_vector:

            print(f"Loading CAA vector from {args.load_caa_vector} "
                  f"(skipping fitting)")

            loaded = torch.load(args.load_caa_vector, map_location="cpu")
            caa_vector = loaded["vector"].to(torch.float32)
            caa_per_attribute = loaded.get("per_attribute", {})
            caa_fit_time = 0.0
            caa_source = {"loaded_from": str(args.load_caa_vector)}

        else:

            caa_vector, caa_per_attribute, caa_fit_time = build_caa_vector(
                fit_top_examples, fit_examples, gen_tokenizer, gen_model,
                device, args.steer_layer, args.steer_attributes,
                args, args.activation_batch_size,
                normalize_per_attribute=args.caa_normalize_per_attribute,
            )

            caa_save_path = save_dir / f"caa_vector_{timestamp}.pt"
            torch.save(
                {
                    "vector": caa_vector,
                    "per_attribute": caa_per_attribute,
                    "attributes": args.steer_attributes,
                    "layer": args.steer_layer,
                    "positive_threshold": args.positive_threshold,
                    "normalize_per_attribute": args.caa_normalize_per_attribute,
                    "generation_model": args.generation_model,
                },
                caa_save_path,
            )
            print(f"Saved CAA vector to {caa_save_path}")
            caa_source = {"saved_to": str(caa_save_path)}

        caa_edit_fn = make_caa_edit_fn(
            caa_vector, args.caa_multiplier, device, model_dtype,
        )

        variants["caa"] = run_variant(
            "caa", caa_edit_fn, SteeringHook, layer,
            eval_prompts,
            gen_tokenizer, gen_model, device, args,
        )

        steering_meta["caa"] = {
            "attributes": args.steer_attributes,
            "layer": args.steer_layer,
            "hook_site": "decoder_layer_output (residual stream)",
            "multiplier": args.caa_multiplier,
            "normalize_per_attribute": args.caa_normalize_per_attribute,
            "vector_fit_time_seconds": caa_fit_time,
            "per_attribute": caa_per_attribute,
            **caa_source,
        }

    # --- MAT-Steer (steers self_attn.o_proj's input) ---
    if not args.skip_mat_steer:

        if args.load_mat_steer:

            print(f"Loading MAT-Steer module from {args.load_mat_steer} "
                  f"(skipping training)")

            loaded = torch.load(args.load_mat_steer, map_location="cpu")
            mat_module = MATSteerModule(
                loaded["hidden_dim"], loaded["attributes"],
            )
            mat_module.load_state_dict(loaded["state_dict"])
            mat_module.eval()
            mat_module = mat_module.to(device=device, dtype=torch.float32)
            mat_fit_time = 0.0
            mat_source = {"loaded_from": str(args.load_mat_steer)}

        else:

            mat_module, mat_fit_time = train_mat_steer(
                fit_top_examples, fit_examples, gen_tokenizer, gen_model,
                device, args.steer_layer, args.steer_attributes,
                args, args.activation_batch_size,
                args.mat_steer_steps, args.mat_steer_lr,
                args.mat_steer_lambda_pos, args.mat_steer_lambda_sparse,
                args.mat_steer_lambda_ortho, args.mmd_sigma,
            )

            mat_save_path = save_dir / f"mat_steer_{timestamp}.pt"
            torch.save(
                {
                    "state_dict": mat_module.state_dict(),
                    "attributes": mat_module.attributes,
                    "hidden_dim": mat_module.theta.shape[1],
                    "layer": args.steer_layer,
                    "positive_threshold": args.positive_threshold,
                    "training_steps": args.mat_steer_steps,
                    "lr": args.mat_steer_lr,
                    "lambda_pos": args.mat_steer_lambda_pos,
                    "lambda_sparse": args.mat_steer_lambda_sparse,
                    "lambda_ortho": args.mat_steer_lambda_ortho,
                    "mmd_sigma": args.mmd_sigma,
                    "generation_model": args.generation_model,
                },
                mat_save_path,
            )
            print(f"Saved MAT-Steer module to {mat_save_path}")
            mat_source = {"saved_to": str(mat_save_path)}

        mat_edit_fn = make_mat_steer_edit_fn(mat_module, device, model_dtype)

        variants["mat_steer"] = run_variant(
            "mat_steer", mat_edit_fn, AttnOProjSteeringHook, o_proj,
            eval_prompts,
            gen_tokenizer, gen_model, device, args,
        )

        steering_meta["mat_steer"] = {
            "attributes": args.steer_attributes,
            "layer": args.steer_layer,
            "hook_site": "self_attn.o_proj input (pre-projection "
                         "attention output)",
            "training_steps": args.mat_steer_steps,
            "lr": args.mat_steer_lr,
            "lambda_pos": args.mat_steer_lambda_pos,
            "lambda_sparse": args.mat_steer_lambda_sparse,
            "lambda_ortho": args.mat_steer_lambda_ortho,
            "mmd_sigma": args.mmd_sigma,
            "vector_fit_time_seconds": mat_fit_time,
            "warm_started_from": "own per-attribute DiffMean "
                                  "(self_attn.o_proj-input space)",
            **mat_source,
        }

    del gen_model
    del gen_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Judge + score every variant with a single judge-model load,
    # using ultrafeedback.py's own vs-all-references pipeline
    # -------------------------------------------------

    judge_tokenizer, judge_model = uf.load_judge_model(
        args.judge_model, device,
    )

    variant_results = {}

    for name, (generated_answers, generation_time) in variants.items():

        (
            metrics,
            verdicts_by_example,
            per_example_win_rates,
            attribute_scores,
            score_raw_outputs,
        ) = judge_and_score_variant(
            name, eval_examples, eval_prompts, generated_answers,
            judge_tokenizer, judge_model, device, args,
        )

        metrics["generation_time_seconds"] = generation_time
        metrics["num_examples"] = len(eval_examples)

        variant_results[name] = {
            "metrics": metrics,
            "examples": [
                {
                    "id": ex["id"],
                    "source": ex["source"],
                    "instruction": ex["prompt"],
                    "reference_completions": ex["references"],
                    "generated_response": g,
                    "judge_results": verdicts_for_example,
                    "example_win_rate": per_example["win_rate"],
                    "llm_scores": llm_score,
                    "llm_score_raw_output": raw,
                }
                for ex, g, verdicts_for_example, per_example, llm_score, raw in zip(
                    eval_examples, generated_answers, verdicts_by_example,
                    per_example_win_rates, attribute_scores, score_raw_outputs,
                )
            ],
        }

    del judge_model
    del judge_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------
    # Print a compact comparison table
    # -------------------------------------------------

    print("=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    header = f"{'variant':12s} {'win_rate':>9s} " + " ".join(
        f"{a[:4]:>6s}" for a in uf.SCORE_ATTRIBUTES
    ) + f" {'gen_s':>8s}"
    print(header)

    for name, result in variant_results.items():
        m = result["metrics"]
        row = f"{name:12s} {m['llm_judge_win_rate']:8.2f}% "
        row += " ".join(
            f"{(m[f'avg_llm_{a}'] or 0.0):6.2f}" for a in uf.SCORE_ATTRIBUTES
        )
        row += f" {m['generation_time_seconds']:8.2f}"
        print(row)

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    results = {
        "timestamp": timestamp,
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
        "dataset": args.dataset_path,
        "fit_dataset": args.fit_dataset_path,  # None if internally split
        "steer_attributes": args.steer_attributes,
        "steer_layer": args.steer_layer,
        "holdout_frac": args.holdout_frac if not args.fit_dataset_path else None,
        "split_seed": args.split_seed if not args.fit_dataset_path else None,
        "positive_threshold": args.positive_threshold,
        "pairing_strategy": args.pairing_strategy,
        "pairing_min_margin": (
            args.pairing_min_margin
            if args.pairing_strategy == "extremes" else None
        ),
        "num_fit_instructions": len(fit_top_examples),
        "num_fit_examples": len(fit_examples),
        "num_eval_examples": len(eval_examples),
        "steering_meta": steering_meta,
        "variants": variant_results,
    }

    output_file = output_dir / f"steering_eval_ultrafeedback_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Evaluation Finished")
    print("=" * 60)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
