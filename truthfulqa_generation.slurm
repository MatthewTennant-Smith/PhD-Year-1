#!/bin/bash
#SBATCH --job-name=truthfulqa_generation_eval
#SBATCH --output=logs/truthfulqa_generation_eval_%j.out
#SBATCH --error=logs/truthfulqa_generation_eval_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# TruthfulQA Generation-Task Evaluation Job
#
# The generation model free-generates one candidate answer per
# question. A separate judge model (FlowJudge) then rates each
# candidate with three independent Yes/No judgements:
#
#   - Truthful   -- is the candidate free of any false claim (given
#                   the dataset's own correct_answers/incorrect_
#                   answers as reference sets)?
#   - Informative -- does the candidate give substantive, relevant
#                   information (judged on the candidate alone)?
#   - Win rate    -- is the candidate at least as truthful and
#                   informative as the dataset's own best_answer?
#
# Headline metrics: truthful_percent, informative_percent,
# truthful_and_informative_percent, and llm_judge_win_rate.
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch evaluate-truthfulqa-generation-job.slurm
#
#   # or override paths on the command line, e.g.:
#   DATASET_PATH=/home/xrai/datasets/truthfulqa_generation.jsonl sbatch evaluate-truthfulqa-generation-job.slurm
#
# =============================================================================

echo "========================================================="
echo "TruthfulQA Generation-Task Evaluation"
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

export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/truthfulqa_generation/truthfulqa_generation.json}"

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
    echo "ERROR: TruthfulQA generation-task dataset file not found."
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

python "${SCRIPT_DIR}/truthfulqa_generation.py" \
    --generation-model "$GENERATION_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR"

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

LATEST=$(ls -t "$OUTPUT_DIR"/truthfulqa_generation_eval_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi
