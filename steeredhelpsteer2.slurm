#!/bin/bash
#SBATCH --job-name=steered_helpsteer2_eval_layer12
#SBATCH --output=logs/steered_helpsteer2_eval_%j.out
#SBATCH --error=logs/steered_helpsteer2_eval_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# Steered HelpSteer2 Evaluation Job (baseline vs CAA vs MAT-Steer)
#
# Same generation/judge/scoring pipeline as evaluate-job.slurm, run
# THREE times against the same held-out eval set: once unsteered
# (baseline), once with a CAA steering vector added, once with a
# trained MAT-Steer module. The CAA vector / MAT-Steer module are
# fit on FIT_DATASET_PATH (a separate train split), never on
# DATASET_PATH (the eval set), so metrics aren't measured on data the
# steering vectors were fit to.
#
# Metrics (per variant -- baseline/caa/mat_steer):
#   - generation_time_seconds: wall-clock time to generate candidate
#     responses for every eval-set prompt.
#   - llm_judge_win_rate: % of examples where the FlowJudge judge model
#     preferred that variant's candidate answer over the human
#     reference answer in DATASET_PATH.
#   - avg_llm_<attribute>: FlowJudge's average score (0-4) for that
#     variant's candidate answers on each of the five weighted
#     HelpSteer2 attributes (helpfulness, correctness, coherence,
#     complexity, verbosity).
#
# Time budget: this does more work per job than evaluate-job.slurm --
# activation collection for CAA + a MAT-Steer training loop happen
# before any of the three generation/judging passes -- so --time is
# set higher than the plain baseline job. Tune down once you've seen
# one full run's actual wall-clock time in the log.
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch steeredhelpsteer2.slurm
#
#   # or override paths/steering settings on the command line, e.g.:
#   FIT_DATASET_PATH=/home/xrai/datasets/helpsteer2/helpsteer2_train.json \
#   DATASET_PATH=/home/xrai/datasets/helpsteer2/helpsteer2_test.json \
#   STEER_ATTRIBUTES="helpfulness correctness" \
#   sbatch steeredhelpsteer2.slurm
#
# =============================================================================

echo "========================================================="
echo "Steered HelpSteer2 Evaluation"
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
# against this file only.
export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/helpsteer2/helpsteer2_test.json}"

# Fitting set -- used ONLY to build the CAA vector / train MAT-Steer.
# If left unset, steering_eval.py falls back to internally splitting
# DATASET_PATH itself (--holdout-frac/--split-seed) instead -- but
# since you have a real train split now, set this.
export FIT_DATASET_PATH="${FIT_DATASET_PATH:-/home/xrai/datasets/helpsteer2/helpsteer2_train.json}"

# -------------------------------------------------------------------------
# Steering settings
# -------------------------------------------------------------------------

STEER_ATTRIBUTES="${STEER_ATTRIBUTES:-helpfulness}"
STEER_LAYER="${STEER_LAYER:-13}"
CAA_MULTIPLIER="${CAA_MULTIPLIER:-1.2}"
MAT_STEER_STEPS="${MAT_STEER_STEPS:-400}"

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}"

OUTPUT_DIR="${REPO_DIR}/data/evaluation_results"

echo ""
echo "Generation model  : $GENERATION_MODEL"
echo "Judge model       : $JUDGE_MODEL"
echo "Eval dataset      : $DATASET_PATH"
echo "Fit dataset       : $FIT_DATASET_PATH"
echo "Steer attributes  : $STEER_ATTRIBUTES"
echo "Steer layer       : $STEER_LAYER"
echo "CAA multiplier    : $CAA_MULTIPLIER"
echo "MAT-Steer steps   : $MAT_STEER_STEPS"
echo "Output            : $OUTPUT_DIR"
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

if [[ ! -f "${SCRIPT_DIR}/helpsteer2.py" ]]; then
    echo "ERROR: helpsteer2.py not found in ${SCRIPT_DIR} -- steering_eval.py imports it."
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

python "${SCRIPT_DIR}/steeredhelpsteer2.py" \
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
    --mat-steer-steps "$MAT_STEER_STEPS"

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

LATEST=$(ls -t "$OUTPUT_DIR"/steering_eval_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi
