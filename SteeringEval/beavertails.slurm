#!/bin/bash
#SBATCH --job-name=beavertails_is_safe_eval
#SBATCH --output=logs/beavertails_is_safe_eval_%j.out
#SBATCH --error=logs/beavertails_is_safe_eval_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# BeaverTails is_safe Prediction Accuracy Evaluation Job
#
# The generation model is shown a prompt plus its response and asked
# to classify the response as "safe" or "unsafe". Accuracy (percent
# correct) = fraction of examples where the predicted label matches
# the dataset's ground-truth "is_safe" field.
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch evaluate-beavertails-job.slurm
#
#   # or override paths on the command line, e.g.:
#   DATASET_PATH=/home/xrai/datasets/beavertails.jsonl sbatch evaluate-beavertails-job.slurm
#
# =============================================================================

echo "========================================================="
echo "BeaverTails is_safe Prediction Accuracy Evaluation"
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

export GENERATION_MODEL="${GENERATION_MODEL:-/home/xrai/models/Meta-Llama-3.1-8B-Intstruct}"

export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/beavertails/beavertails_test.json}"

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}"

OUTPUT_DIR="${REPO_DIR}/data/evaluation_results"

echo ""
echo "Generation model : $GENERATION_MODEL"
echo "Dataset          : $DATASET_PATH"
echo "Output           : $OUTPUT_DIR"
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

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: BeaverTails dataset file not found."
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
echo "Starting Evaluation"
echo "========================================================="

python "${SCRIPT_DIR}/zerobeavertails.py" \
    --generation-model "$GENERATION_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --max-new-tokens 5

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

LATEST=$(ls -t "$OUTPUT_DIR"/beavertails_test_is_safe_eval_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi
