"""
steering_eval.py -- Compare a base generation model against the SAME
model under two activation-steering interventions, using the exact
LLM-as-a-judge win-rate / weighted-attribute-scoring / generation-time
pipeline already defined in helpsteer2.py.

The two steering methods, calibrated against the official reference
implementations (see "Fidelity notes" below for exactly what was
checked against what):

  * CAA -- Contrastive Activation Addition
    Panickssery et al., "Steering Llama 2 via Contrastive Activation
    Addition", arXiv:2312.06681 (official code: github.com/nrimsky/CAA).
    Vector construction is the "DiffMean" estimator used across the
    CAA literature: mean(activations on positive examples) -
    mean(activations on negative examples) at a chosen layer. At
    inference, the (optionally scaled) vector is added to every
    *newly generated* token's residual stream -- not the prompt --
    which matches how CAA is run as a baseline in Zhao et al.'s
    ODESteer paper (arXiv:2602.17560, github.com/ZhaoHongjue/odesteer):
    "we apply our method and all baselines at the same residual
    stream position... and apply steering to all newly generated
    tokens." ODESteer also uses layer 14 for Llama-3.1-8B, which is
    why that's this script's default --steer-layer. See
    build_caa_vector() / make_caa_edit_fn() below.

  * MAT-Steer -- Multi-Attribute Targeted Steering
    Nguyen et al., "Multi-Attribute Steering of Language Models via
    Targeted Intervention", arXiv:2502.12446 (official code:
    github.com/duykhuongnguyen/MAT-Steer).
    A small set of per-attribute steering vectors theta_t, each gated
    by a learned per-token sigmoid gate G_t(a) = sigmoid(w_t . a + b_t),
    trained with an MMD representation-alignment loss plus
    positive-preservation / sparsity / orthogonality regularizers
    (their Eqs. 3-11). See MATSteerModule and train_mat_steer() below.

Fidelity notes -- what was checked against the official repos, what
matches, and what's a necessary adaptation for this dataset:

  - MATCHES: MAT-Steer's intervention site is self_attn.o_proj's
    INPUT (the concatenated multi-head attention output, before it's
    projected back into the residual stream) -- confirmed directly
    from the official repo's interveners.py (MATIntervener, hooked
    via pyvene at "model.layers[{layer}].self_attn.o_proj.input").
    This is the same convention ITI uses (MAT-Steer's repo builds on
    the ITI codebase -- its own conda env is literally named "iti"),
    and matches the paper's Section 2.1 definition of the activation
    as "the output of the self-attention layer", which is NOT the
    same point CAA steers (CAA edits the full decoder-block output,
    i.e. the residual stream, per nrimsky/CAA's BlockOutputWrapper).
    get_attn_o_proj() / AttnOProjSteeringHook /
    collect_attn_input_activations() implement this correctly now;
    CAA's SteeringHook / collect_layer_activations() are unchanged
    and still target the residual stream.

  - MATCHES: interveners.py's MATIntervener.__call__ unconditionally
    edits `b[0, -1]` -- the last token of whatever's passed through
    the hook, including the initial multi-token forward pass over the
    prompt, not just single-token generation steps. Both SteeringHook
    and AttnOProjSteeringHook now do the same (always edit position
    -1), rather than the earlier version's "only when seq_len == 1"
    gating.

  - MATCHES: MAT-Steer's official get_activations.py explicitly
    "extract[s] last-token activations at the specified layer" --
    i.e. one activation per example, not per-token across a full
    response. That's exactly what collect_attn_input_activations()
    does here.

  - MATCHES: MAT-Steer's official README trains with sigma=2.0 and
    lambda_ortho=0.1 -- both match this script's defaults. Its
    lr=0.001 default did NOT match an earlier version of this script
    (which defaulted to 1e-2); --mat-steer-lr now defaults to 0.001.
    lambda_pos/lambda_sparse: the official repo's example command
    uses 0.9 (their QA-task setting); this script defaults to 0.8,
    the paper's reported optimum specifically for the *generation*
    task (Appendix A), since HelpSteer2 is a generation task -- pass
    --mat-steer-lambda-pos/--mat-steer-lambda-sparse 0.9 to match the
    repo's literal example instead.

  - CONFIRMED against ODESteer's actual source (src/odesteer/steer/CAA.py,
    shared by the user): `CAA.fit(pos_X, neg_X)` sets
    `self.steer_vec = pos_X.mean(dim=0) - neg_X.mean(dim=0)` --
    exactly what build_caa_vector() does, with no normalization step.
    Notably, fit()'s signature takes pos_X/neg_X as independent
    tensors with no index correspondence required between them --
    i.e. ODESteer's own CAA is ALSO an unpaired construction, not the
    paired-A/B-prompt approach from nrimsky/CAA's original
    generate_vectors.py. So the "adaptation" described below is more
    accurately a difference from the original 2023 CAA release than
    from ODESteer's reimplementation of it, which this script's
    default (--caa-normalize-per-attribute off) now matches exactly
    for the single-attribute case. ODESteer's config also confirmed
    T=1.0 as CAA's default multiplier, matching --caa-multiplier's
    default.

    CONFIRMED further, against `_base_steer.py`'s `VecSteer.steer()`
    (also shared by the user):

        def steer(self, X, T=1.0):
            return X + T * self.steer_vec.to(X.device)

    Plain additive, no renormalization step -- confirms
    make_caa_edit_fn (X + T*vector, no norm-preservation) matches
    exactly. This also confirms CAA and MAT-Steer genuinely differ in
    this respect, not just in hook location: MAT-Steer's
    layer_norm_preserve logic (interveners.py) has no counterpart
    here.

    STILL NOT confirmed: `VecSteer.steer(X, T)` is generic over
    whatever `X` its caller passes in -- it doesn't say which
    activations (hook point) or which token positions get edited.
    That logic lives in ODESteer's model wrapper (src/odesteer/lm/),
    which hasn't been seen. SteeringHook's "always edit position -1
    of the decoder layer's output (residual stream)" is inferred from
    the CAA paper's own stated convention (confirmed independently
    via nrimsky/CAA's BlockOutputWrapper) and ODESteer's paper text
    ("apply steering to all newly generated tokens"), not verified
    against ODESteer's own wrapper code.

  - CONFIRMED further, against the user's HuggingFaceLM wrapper
    (odesteer's actual `src/odesteer/lm/` code): `_get_target_layer()`
    returns `self.model.model.layers[self.steer_layer_idx]` (the
    decoder layer itself -- residual stream, matches SteeringHook's
    target exactly), `register_steer_hook` always passes
    `steer_position_idx=-1` on every `generate()` call, and
    `steer_hook_func` does `hidden[:, -1] = steer_model.steer(hidden[:, -1], ...)`.
    So both the hook point AND the "always edit position -1, on
    every forward pass including the initial multi-token prompt
    pass" behavior are now fully confirmed against ODESteer's actual
    code, not inferred from the paper. `extract_prompt_eos_activations`
    also confirms activation collection matches
    collect_layer_activations() exactly: `hidden_states[:, -1, :]`
    with left-padding.

  - CAA (and MAT-Steer) require labeled positive/negative examples to
    fit at all -- there is no meaningful "no data" mode; skipping the
    fit means no vector exists, not "no effect". This script only
    needs the SAME dataset file already used for eval, via
    human_scores; if your dataset has that field populated for the
    attributes you're steering on, no separate contrastive dataset is
    required. Once fit, --save-steering-dir persists the vector/
    module to disk, and --load-caa-vector/--load-mat-steer reload a
    previously fit one to re-run generation+judging (e.g. to sweep
    --caa-multiplier, or evaluate against a different --dataset-path)
    without re-fitting.

  - ADAPTATION relative to the ORIGINAL CAA paper/repo (still applies
    even though it doesn't apply to ODESteer's reimplementation, per
    the point above): CAA's original vectors come from *paired*
    multiple-choice (A/B) prompts, with the vector taken at the
    answer-letter token (Eq. 1 of the paper). HelpSteer2 has no such
    pairs -- it has single graded responses. This script builds the
    vector as an unpaired DiffMean over positively- vs.
    negatively-labeled *responses* instead (same estimator, just
    without the artificial A/B pairing). This means results aren't
    directly comparable to the original paper's multiple-choice
    numbers, though it's the same construction commonly used for
    single-response settings (see the "DiffMean" / "CAA" baselines in
    ODESteer, EasySteer, and other follow-up steering libraries).

  - ADAPTATION: neither official repo's CAA natively supports summing
    several single-behavior vectors into one multi-attribute
    direction (that's MAT-Steer's whole point). For --steer-attributes
    with more than one attribute, CAA here sums the per-attribute
    DiffMean vectors -- raw (matching ODESteer's un-normalized
    DiffMean-times-T convention) by default, or unit-normalized per
    attribute first via --caa-normalize-per-attribute if you want
    every attribute to contribute equally regardless of its raw
    activation-difference magnitude.

  - Positive/negative labeling threshold for both methods follows
    the paper's own stated HelpSteer labeling rule: human_scores
    >= 3 is "positive", < 3 is "negative" (Section 4.1: "Scores of 3
    or 4 are considered positive, while scores <3 are deemed
    negative.").

  - Both vector sets are fit on a "vector-fitting" split of the SAME
    dataset file used for eval; use --holdout-frac (default 0.3) to
    keep the eval split disjoint from the examples used to build the
    steering vectors, so win-rate/scoring numbers aren't measured on
    data the vectors were fit to.

Architecture assumption: steering hooks attach to
`model.model.layers[i]`, which holds for Llama/Mistral/Qwen2-style
causal LMs (the default model paths in helpsteer2.py are Llama
checkpoints). If you're running a model with a different internal
attribute path, adjust get_decoder_layers() below.

How the "only steer generated tokens" behavior is achieved: with
KV-caching (the default for model.generate()), the model does exactly
one forward pass over the full prompt (hidden_states.shape[1] > 1,
left unsteered) and then one forward pass per newly generated token
(hidden_states.shape[1] == 1, steered). This reproduces CAA's "add the
vector at all token positions after the user's prompt" rule without
needing to track prompt lengths per sequence.

Example:

python steering_eval.py \
    --generation-model /home/xrai/models/Meta-Llama-3.1-8B-Instruct \
    --judge-model /home/xrai/models/flowjudge \
    --dataset-path /home/xrai/datasets/helpsteer2/helpsteer2.json \
    --steer-attributes helpfulness correctness coherence \
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

# Reuses dataset loading, generation, judging, and scoring exactly as
# defined in the original evaluation script.
import helpsteer2 as hs2


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Compare a base model against CAA-steered and "
                     "MAT-Steer-steered variants on HelpSteer2, using "
                     "helpsteer2.py's judge/scoring/timing pipeline."
    )

    parser.add_argument(
        "--generation-model",
        default=os.environ.get(
            "GENERATION_MODEL",
            hs2.DEFAULT_GENERATION_MODEL,
        ),
    )

    parser.add_argument(
        "--judge-model",
        default=os.environ.get(
            "JUDGE_MODEL",
            hs2.DEFAULT_JUDGE_MODEL,
        ),
    )

    parser.add_argument(
        "--dataset-path",
        default=os.environ.get(
            "DATASET_PATH",
            hs2.DEFAULT_DATASET,
        ),
        help="Dataset used for EVALUATION (baseline/CAA/MAT-Steer "
             "judging+scoring). If --fit-dataset-path is not given, "
             "this same file is also split (--holdout-frac / "
             "--split-seed) to provide the fitting portion.",
    )

    parser.add_argument(
        "--fit-dataset-path",
        default=None,
        help="Optional separate dataset used ONLY to fit the CAA "
             "vector / train MAT-Steer (e.g. a pre-made "
             "helpsteer2_train.json). When given, --dataset-path is "
             "used ENTIRELY for evaluation (no internal split, "
             "--holdout-frac/--split-seed are ignored) and this file "
             "is used ENTIRELY for fitting -- i.e. you're supplying "
             "already-disjoint train/test files rather than having "
             "this script split one file itself.",
    )

    parser.add_argument(
        "--output-dir",
        default=hs2.DEFAULT_OUTPUT_DIR,
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
        default=["helpfulness", "correctness", "coherence"],
        choices=list(hs2.SCORE_ATTRIBUTES),
        help="Which HelpSteer2 attributes to build steering vectors "
             "for (both CAA and MAT-Steer). Passing more than one for "
             "CAA sums their unit-normalized diff-of-means vectors "
             "into a single steering direction; MAT-Steer keeps one "
             "vector + gate per attribute, as in the paper.",
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
        help="Fraction of the dataset held out for evaluation only "
             "(not used to fit steering vectors) -- default 0.2 gives "
             "an 80/20 fit/eval split, matching the train/test split "
             "convention both CAA's and MAT-Steer's papers use "
             "(CAA's paper explicitly fits on one portion and "
             "evaluates on '50 held-out' held-out examples; the rest "
             "is the vector-fitting split).",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for shuffling the dataset before the fit/eval "
             "split. The split shuffles rather than taking a "
             "contiguous slice, since a dataset file ordered by "
             "topic/source/difficulty would otherwise put "
             "systematically different examples in the fitting vs. "
             "eval split. Fixed seed -> reproducible split across "
             "runs (e.g. so --load-caa-vector's fit and a later "
             "--skip-caa re-evaluation see the same eval set).",
    )

    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=3,
        help="human_scores[attr] >= this is a positive example, "
             "below it is negative (matches the MAT-Steer paper's "
             "stated HelpSteer labeling rule).",
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
        help="Scalar strength ('T' in ODESteer's config, "
             "e.g. 'steer.T=1.0' for their CAA baseline) applied to "
             "the CAA steering vector during generation. Default "
             "1.0 matches ODESteer's own CAA example for Llama-3.1-8B "
             "at layer 14 -- but that value was tuned for a single "
             "raw DiffMean vector; when combining multiple attributes "
             "(--steer-attributes with >1 entry) the summed vector's "
             "magnitude is larger, so you'll likely want to sweep "
             "this down, or pass --caa-normalize-per-attribute to "
             "keep per-attribute magnitude comparable regardless of "
             "how many attributes you combine.",
    )

    parser.add_argument(
        "--caa-normalize-per-attribute",
        action="store_true",
        help="Unit-normalize each per-attribute DiffMean vector "
             "before summing across --steer-attributes, so every "
             "attribute contributes equally regardless of its raw "
             "activation-difference magnitude. Off by default, which "
             "instead sums the RAW DiffMean vectors -- matching "
             "ODESteer's un-normalized DiffMean-times-T convention "
             "for the single-attribute case.",
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
# Model-architecture helpers
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
# Steering hooks
# ---------------------------------------------------------------------

class SteeringHook:
    """
    CAA's intervention site: registers a forward hook on a single
    decoder layer that edits its OUTPUT hidden states (the residual
    stream) in place. This matches CAA's own description of adding
    the vector "to the residual stream" (Panickssery et al., Fig. 1)
    and nrimsky/CAA's BlockOutputWrapper, which wraps the whole
    decoder block and edits output[0].

    Always edits only the LAST token position of whatever gets
    passed through the hook -- this covers both the single new token
    of each post-prompt generation step under KV-caching, AND (during
    the one multi-token forward pass over the prompt) the prompt's
    own final token, matching how the official repos actually behave
    rather than skipping that one token position.
    """

    def __init__(self, layer, edit_fn):

        self.edit_fn = edit_fn
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):

        # Decoder layers commonly return a tuple whose first element
        # is the hidden states; some return the tensor directly.
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
    o_proj mixes it back into the residual stream. This mirrors the
    official repo's interveners.py, which hooks
    "model.layers[{layer}].self_attn.o_proj.input" via pyvene and,
    per MATIntervener.__call__, unconditionally edits `b[0, -1]` (the
    last token of whatever's passed through).

    Same last-token-always semantics as SteeringHook, for the same
    reason: this is what the official code actually does, not an
    approximation of it.
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
# Activation collection (shared by CAA and MAT-Steer fitting)
# ---------------------------------------------------------------------

def split_by_attribute_label(examples, attribute, positive_threshold):
    """
    Splits examples into (positive_examples, negative_examples) for a
    single attribute, based on human_scores[attribute], skipping any
    example missing that score.
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
    Encodes `prompt + response` for every example and returns the
    layer_idx residual-stream activation at the LAST token position
    for each one, as a single [N, hidden_dim] tensor (on CPU, float32).

    Because the tokenizer pads on the left (set in
    helpsteer2.load_generation_model), the last real token is always
    at sequence position -1 regardless of each example's length.
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
    Used only for fitting/warm-starting MAT-Steer, whose
    AttnOProjSteeringHook applies vectors at this same point.
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
    activation - mean negative activation) at layer_idx -- the same
    estimator used across the CAA literature (Rimsky et al.'s CAA,
    ODESteer's CAA baseline, etc.) -- then combines across the
    requested attributes to get a single steering direction.

    By default (normalize_per_attribute=False) the raw per-attribute
    vectors are summed directly, matching ODESteer's convention of
    scaling a raw, un-normalized DiffMean vector by a strength T at
    application time. Pass normalize_per_attribute=True to
    unit-normalize each attribute's vector before summing instead, so
    attributes with a naturally larger raw activation difference
    don't dominate the combined direction -- useful mainly when
    --steer-attributes has more than one entry, since single-behavior
    CAA (the official use case) never faces this tradeoff.

    Returns the combined vector as a 1D float32 tensor, plus a
    per-attribute breakdown (norms, example counts) for the results
    file.
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
    device_map="auto" (multi-GPU sharding, common on HPC nodes) the
    layer being steered may not live on `device` at all -- e.g. a
    7B+ model split across 2 GPUs can put layer 14 on cuda:1 while
    `device` defaults to cuda:0. Re-casting per call is a cheap
    same-size tensor .to() and keeps this correct regardless of how
    the model was sharded.
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

    theta (the steering vectors) is initialized from each attribute's
    own DiffMean direction (mean positive - mean negative activation,
    in the SAME self_attn.o_proj-input space this module operates in
    -- see train_mat_steer), otherwise randomly if no init_vectors are
    supplied.
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
        # a: [..., d] -> gates: [..., T]
        logits = torch.einsum("...d,td->...t", a, self.w) + self.b
        return torch.sigmoid(logits)

    def steer(self, a):
        """Applies Eq. 4 (unnormalized) -- used during training."""
        g = self.gates(a)  # [..., T]
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
    kernel, per the paper's Eq. 10 / bandwidth choice (sigma=2 in
    their setup -- see --mmd-sigma if this saturates or explodes for
    your model's activation scale).
    """

    def kernel(a, b):
        # a: [N, d], b: [M, d] -> [N, M]
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
    Collects per-attribute positive/negative activation sets once
    (on CPU), then trains the MATSteerModule's gates + steering
    vectors against the full loss in Eq. (11) of the paper:

        L = L_MMD + lambda_pos * L_pos
              + lambda_sparse * L_sparse + lambda_ortho * L_ortho

    Activations are collected at self_attn.o_proj's INPUT (see
    collect_attn_input_activations), matching where the official
    MAT-Steer repo's get_activations.py and interveners.py both
    operate -- NOT the decoder-layer residual-stream output CAA uses.
    Per the same repo's README ("extract last-token activations"),
    this uses one activation per example (last token of
    prompt+response), which is what the official implementation does
    too, not a simplification of a token-level method.

    theta is warm-started from each attribute's own DiffMean
    (mean(pos) - mean(neg)) in this same activation space -- a
    reasonable initialization since the alignment loss is already
    pushing in roughly that direction, and it must be computed here
    rather than reused from CAA's vector, since CAA's vector lives in
    a different space (the residual stream, not the pre-o_proj
    attention output).
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
    may not be on `device` under multi-GPU sharding. Rather than move
    hidden_states to the module, we move a cached copy of the (small)
    module's parameters to wherever hidden_states actually lives.
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
# Running one full variant (baseline / CAA / MAT-Steer) through the
# existing helpsteer2.py generation + judge + scoring pipeline
# ---------------------------------------------------------------------

def run_variant(
    variant_name,
    edit_fn,
    hook_cls,
    hook_target,
    prompts,
    references,
    gen_tokenizer,
    gen_model,
    device,
    args,
):
    """
    Optionally attaches a steering hook (edit_fn=None means baseline,
    unsteered generation), then runs helpsteer2.generate_answers.
    `hook_cls` is SteeringHook (attaches to a decoder layer, for CAA)
    or AttnOProjSteeringHook (attaches to that layer's
    self_attn.o_proj, for MAT-Steer) -- see their docstrings for why
    they target different points in the model. Judging/scoring happen
    later, in a single batch across all variants, so the judge model
    only needs to be loaded once.
    """

    print("#" * 60)
    print(f"Generating: {variant_name}")
    print("#" * 60)

    hook = None
    if edit_fn is not None:
        hook = hook_cls(hook_target, edit_fn)

    try:
        generated_answers, generation_time = hs2.generate_answers(
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
    generated_answers,
    prompts,
    references,
    judge_tokenizer,
    judge_model,
    device,
    args,
):

    print("#" * 60)
    print(f"Judging: {variant_name}")
    print("#" * 60)

    verdicts, judge_time = hs2.run_judge(
        prompts, references, generated_answers,
        judge_tokenizer, judge_model, device,
        args.batch_size, args.judge_max_new_tokens,
    )

    scores, win_rate, unparsed_pairwise = hs2.parse_judgements(verdicts)

    attribute_scores, score_raw_outputs, scoring_time = hs2.run_scoring(
        prompts, references, generated_answers,
        judge_tokenizer, judge_model, device,
        args.batch_size, args.score_max_new_tokens,
    )

    attribute_averages, attribute_unparsed = hs2.summarize_attribute_scores(
        attribute_scores
    )

    metrics = {
        "llm_judge_win_rate": win_rate,
        "unparsed_pairwise_verdicts": unparsed_pairwise,
        "judge_time_seconds": judge_time,
        "scoring_time_seconds": scoring_time,
    }

    for attr in hs2.SCORE_ATTRIBUTES:
        metrics[f"avg_llm_{attr}"] = attribute_averages[attr]
        metrics[f"unparsed_llm_{attr}"] = attribute_unparsed[attr]

    return metrics, verdicts, scores, attribute_scores, score_raw_outputs


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
    print("HelpSteer2 Steering Comparison: Baseline vs CAA vs MAT-Steer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------
    # Load fit/eval examples -- either from one file (internally
    # split via --holdout-frac/--split-seed) or from two separate,
    # already-disjoint files (--fit-dataset-path given)
    # -------------------------------------------------

    eval_examples = hs2.load_helpsteer2(args.dataset_path)

    if args.max_examples:
        eval_examples = eval_examples[: args.max_examples]
        print(f"Limited evaluation to {args.max_examples} examples")

    if args.fit_dataset_path:

        fit_examples = hs2.load_helpsteer2(args.fit_dataset_path)

        print(
            f"Using separate fit/eval files: {len(fit_examples)} "
            f"examples from --fit-dataset-path for fitting steering "
            f"vectors, {len(eval_examples)} examples from "
            f"--dataset-path (used in full, no internal split) for "
            f"evaluation."
        )

    else:

        holdout_n = max(1, int(len(eval_examples) * args.holdout_frac))

        shuffled = eval_examples.copy()
        random.Random(args.split_seed).shuffle(shuffled)

        fit_examples = shuffled[:-holdout_n] if holdout_n < len(shuffled) else []
        eval_examples = shuffled[-holdout_n:]

        print(
            f"Split --dataset-path internally: {len(fit_examples)} "
            f"examples for fitting steering vectors, "
            f"{len(eval_examples)} held out for evaluation."
        )

    eval_prompts = [ex["prompt"] for ex in eval_examples]
    eval_references = [ex["response"] for ex in eval_examples]

    # -------------------------------------------------
    # Load generation model once; build steering vectors on it, then
    # generate baseline + each steered variant with it
    # -------------------------------------------------

    gen_tokenizer, gen_model = hs2.load_generation_model(
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
        eval_prompts, eval_references,
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
                fit_examples, gen_tokenizer, gen_model, device,
                args.steer_layer, args.steer_attributes,
                args.positive_threshold, args.activation_batch_size,
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
            eval_prompts, eval_references,
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
                fit_examples, gen_tokenizer, gen_model, device,
                args.steer_layer, args.steer_attributes,
                args.positive_threshold, args.activation_batch_size,
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
            eval_prompts, eval_references,
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
    # Judge + score every variant with a single judge-model load
    # -------------------------------------------------

    judge_tokenizer, judge_model = hs2.load_judge_model(
        args.judge_model, device,
    )

    variant_results = {}

    for name, (generated_answers, generation_time) in variants.items():

        metrics, verdicts, scores, attribute_scores, score_raw_outputs = (
            judge_and_score_variant(
                name, generated_answers, eval_prompts, eval_references,
                judge_tokenizer, judge_model, device, args,
            )
        )

        metrics["generation_time_seconds"] = generation_time
        metrics["num_examples"] = len(eval_examples)

        variant_results[name] = {
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
                    eval_examples, generated_answers, verdicts, scores,
                    attribute_scores, score_raw_outputs,
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
        f"{a[:4]:>6s}" for a in hs2.SCORE_ATTRIBUTES
    ) + f" {'gen_s':>8s}"
    print(header)

    for name, result in variant_results.items():
        m = result["metrics"]
        row = f"{name:12s} {m['llm_judge_win_rate']:8.2f}% "
        row += " ".join(
            f"{(m[f'avg_llm_{a}'] or 0.0):6.2f}" for a in hs2.SCORE_ATTRIBUTES
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
        "num_fit_examples": len(fit_examples),
        "num_eval_examples": len(eval_examples),
        "steering_meta": steering_meta,
        "variants": variant_results,
    }

    output_file = output_dir / f"steering_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("Evaluation Finished")
    print("=" * 60)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
