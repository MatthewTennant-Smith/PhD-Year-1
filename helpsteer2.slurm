#!/bin/bash
#SBATCH --job-name=helpsteer2_eval
#SBATCH --output=logs/helpsteer2_eval_%j.out
#SBATCH --error=logs/helpsteer2_eval_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# HelpSteer2 Evaluation Job
#
# Metrics:
#   - generation_time_seconds: wall-clock time to generate candidate
#     responses for every prompt.
#   - llm_judge_win_rate: % of examples where the FlowJudge judge model
#     preferred the generation model's candidate answer over the human
#     reference answer in helpsteer2.json.
#   - avg_llm_<attribute>: FlowJudge's average score (0-4) for the
#     candidate answers on each of the five weighted HelpSteer2
#     attributes (helpfulness, correctness, coherence, complexity,
#     verbosity).
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch evaluate-job.slurm
#
#   # or override paths on the command line, e.g.:
#   DATASET_PATH=/home/xrai/datasets/helpsteer2/helpsteer2.json sbatch evaluate-job.slurm
#
# =============================================================================

echo "========================================================="
echo "HelpSteer2 Evaluation"
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

export JUDGE_MODEL="${JUDGE_MODEL:-/home/xrai/models/flowjudge}"

export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/helpsteer2/helpsteer2_test.json}"

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}"

OUTPUT_DIR="${REPO_DIR}/data/evaluation_results"

echo ""
echo "Generation model : $GENERATION_MODEL"
echo "Judge model      : $JUDGE_MODEL"
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

if [[ ! -d "$JUDGE_MODEL" ]]; then
    echo "ERROR: Judge model not found."
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "ERROR: HelpSteer2 dataset file not found."
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

python "${SCRIPT_DIR}/helpsteer2.py" \
    --generation-model "$GENERATION_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --max-new-tokens 256 \
    --judge-max-new-tokens 10 \
    --score-max-new-tokens 60

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

LATEST=$(ls -t "$OUTPUT_DIR"/helpsteer2_eval_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi
