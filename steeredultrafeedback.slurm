#!/bin/bash
#SBATCH --job-name=steered_ultrafeedback_eval_layer12
#SBATCH --output=logs/steered_ultrafeedback_eval_%j.out
#SBATCH --error=logs/steered_ultrafeedback_eval_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# Steered UltraFeedback Evaluation Job (baseline vs CAA vs MAT-Steer)
#
# Same generation/judge/scoring pipeline as an unsteered UltraFeedback
# evaluation job, run THREE times against the same held-out eval set:
# once unsteered (baseline), once with a CAA steering vector added,
# once with a trained MAT-Steer module. The CAA vector / MAT-Steer
# module are fit on FIT_DATASET_PATH (a separate train split), never
# on DATASET_PATH (the eval set), so metrics aren't measured on data
# the steering vectors were fit to.
#
# Because each UltraFeedback instruction carries several human-scored
# reference completions (not one, like HelpSteer2), fitting supports
# two ways of picking positive/negative examples -- see
# PAIRING_STRATEGY below.
#
# Metrics (per variant -- baseline/caa/mat_steer), computed by
# ultrafeedback.py's own vs-all-references pipeline:
#   - generation_time_seconds: wall-clock time to generate candidate
#     responses for every eval-set instruction.
#   - llm_judge_win_rate: % of ALL candidate-vs-reference comparisons
#     (candidate vs. EVERY reference completion for that instruction,
#     across every eval instruction) where the FlowJudge judge model
#     preferred that variant's candidate answer.
#   - win_rate_by_reference_model: the same win rate broken out per
#     reference model being compared against.
#   - avg_llm_<attribute>: FlowJudge's average score (1-5) for that
#     variant's candidate answers on each of the four weighted
#     UltraFeedback attributes (helpfulness, honesty,
#     instruction_following, truthfulness).
#
# Time budget: this does more work per job than an unsteered
# baseline job -- activation collection for CAA + a MAT-Steer
# training loop happen before any of the three generation/judging
# passes -- so --time is set higher. Tune down once you've seen one
# full run's actual wall-clock time in the log.
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch steeredultrafeedback.slurm
#
#   # or override paths/steering settings on the command line, e.g.:
#   FIT_DATASET_PATH=/home/xrai/datasets/ultrafeedback/ultrafeedback_train.json \
#   DATASET_PATH=/home/xrai/datasets/ultrafeedback/ultrafeedback_test.json \
#   STEER_ATTRIBUTES="helpfulness honesty" \
#   PAIRING_STRATEGY=extremes \
#   PAIRING_MIN_MARGIN=2 \
#   sbatch steeredultrafeedback.slurm
#
# =============================================================================

echo "========================================================="
echo "Steered UltraFeedback Evaluation"
echo "========================================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "GPU(s)    : $SLURM_GPUS_ON_NODE"
echo "Started   : $(date)"
echo "========================================================="

# -------------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------------

source ~/.bashrc
conda activate keele-llm

# Hugging Face cache

export HF_HOME="/home/xrai/.cache/huggingface"

# Offline mode

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

export GENERATION_MODEL="${GENERATION_MODEL:-/home/xrai/models/Llama-3.1-8B-Instruct}"

export JUDGE_MODEL="${JUDGE_MODEL:-/home/xrai/models/flowjudge}"

# Evaluation set -- baseline/CAA/MAT-Steer are all judged+scored
# against this file only (vs every reference completion in it).
export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/ultrafeedback/ultrafeedback_test.json}"

# Fitting set -- used ONLY to build the CAA vector / train MAT-Steer.
# If left unset, steering_eval_ultrafeedback.py falls back to
# internally splitting DATASET_PATH itself, by instruction
# (--holdout-frac/--split-seed) -- but since you have a real train
# split now, set this.
export FIT_DATASET_PATH="${FIT_DATASET_PATH:-/home/xrai/datasets/ultrafeedback/ultrafeedback_train.json}"

# -------------------------------------------------------------------------
# Steering settings
# -------------------------------------------------------------------------

STEER_ATTRIBUTES="${STEER_ATTRIBUTES:-helpfulness honesty}"
STEER_LAYER="${STEER_LAYER:-15}"
CAA_MULTIPLIER="${CAA_MULTIPLIER:-0.8}"
MAT_STEER_STEPS="${MAT_STEER_STEPS:-100}"

# How positive/negative fitting examples are chosen -- see
# steering_eval_ultrafeedback.py's --pairing-strategy help text.
# "threshold" (default): pool every reference completion across every
#   fitting instruction and bucket each one independently by
#   human_scores[attr] >= POSITIVE_THRESHOLD.
# "extremes": pair each fitting instruction's highest-scoring
#   completion against its lowest-scoring one, holding the prompt
#   fixed -- closer to the original CAA paper's paired construction.
#   PAIRING_MIN_MARGIN only applies to this strategy.
PAIRING_STRATEGY="${PAIRING_STRATEGY:-extremes}"
POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-3}"
PAIRING_MIN_MARGIN="${PAIRING_MIN_MARGIN:-1}"

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}"

OUTPUT_DIR="${REPO_DIR}/data/evaluation_results"

echo ""
echo "Generation model    : $GENERATION_MODEL"
echo "Judge model         : $JUDGE_MODEL"
echo "Eval dataset        : $DATASET_PATH"
echo "Fit dataset         : $FIT_DATASET_PATH"
echo "Steer attributes    : $STEER_ATTRIBUTES"
echo "Steer layer         : $STEER_LAYER"
echo "CAA multiplier      : $CAA_MULTIPLIER"
echo "MAT-Steer steps     : $MAT_STEER_STEPS"
echo "Pairing strategy    : $PAIRING_STRATEGY"
echo "Positive threshold  : $POSITIVE_THRESHOLD"
echo "Pairing min margin  : $PAIRING_MIN_MARGIN"
echo "Output              : $OUTPUT_DIR"
echo ""

# -------------------------------------------------------------------------
# GPU Diagnostics
# -------------------------------------------------------------------------

echo "GPU Information"

nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

echo ""

# -------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------

if [[ ! -d "$GENERATION_MODEL" ]]; then
    echo "ERROR: Generation model not found."
    exit 1
fi

if [[ ! -d "$JUDGE_MODEL" ]]; then
    echo "ERROR: Judge model not found."
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: Evaluation dataset file not found: $DATASET_PATH"
    exit 1
fi

if [[ ! -f "$FIT_DATASET_PATH" ]]; then
    echo "ERROR: Fit (train) dataset file not found: $FIT_DATASET_PATH"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/ultrafeedback.py" ]]; then
    echo "ERROR: ultrafeedback.py not found in ${SCRIPT_DIR} -- steering_eval_ultrafeedback.py imports it."
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/steeredultrafeedback.py" ]]; then
    echo "ERROR: steeredultrafeedback.py not found in ${SCRIPT_DIR}."
    exit 1
fi

mkdir -p "${REPO_DIR}/logs"
mkdir -p "$OUTPUT_DIR"

echo "Pre-flight checks passed."
echo ""

# -------------------------------------------------------------------------
# Run Evaluation
# -------------------------------------------------------------------------

echo "========================================================="
echo "Starting Steered Evaluation"
echo "========================================================="

python "${SCRIPT_DIR}/steeredultrafeedback.py" \
    --generation-model "$GENERATION_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --fit-dataset-path "$FIT_DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --max-new-tokens 256 \
    --judge-max-new-tokens 10 \
    --score-max-new-tokens 60 \
    --steer-attributes $STEER_ATTRIBUTES \
    --steer-layer "$STEER_LAYER" \
    --caa-multiplier "$CAA_MULTIPLIER" \
    --mat-steer-steps "$MAT_STEER_STEPS" \
    --pairing-strategy "$PAIRING_STRATEGY" \
    --positive-threshold "$POSITIVE_THRESHOLD" \
    --pairing-min-margin "$PAIRING_MIN_MARGIN"

EXIT_CODE=$?

echo ""
echo "========================================================="
echo "Finished"
echo "Exit code : $EXIT_CODE"
echo "Finished  : $(date)"
echo "========================================================="

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Evaluation failed."
    exit $EXIT_CODE
fi

LATEST=$(ls -t "$OUTPUT_DIR"/steering_eval_ultrafeedback_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi