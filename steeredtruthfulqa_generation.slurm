#!/bin/bash
#SBATCH --job-name=steered_truthfulqa_gen_eval
#SBATCH --output=logs/steered_truthfulqa_gen_eval_%j.out
#SBATCH --error=logs/steered_truthfulqa_gen_eval_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu

# =============================================================================
# Steered TruthfulQA Generation Evaluation Job (baseline vs CAA vs MAT-Steer)
#
# Same generate + FlowJudge pipeline as
# evaluate_truthfulqa_generation.py, run THREE times against the same
# TruthfulQA generation-task eval set: once unsteered (baseline), once
# with a CAA steering vector added, once with a trained MAT-Steer
# module.
#
# Steering vectors are fit ENTIRELY FROM TRUTHFULQA ITSELF
# (--fit-source truthfulqa), NOT from HelpSteer2 -- FIT_DATASET_PATH
# points at a separate TruthfulQA generation-shaped train split
# (question/best_answer/correct_answers/incorrect_answers), and
# build_fit_examples_from_truthfulqa_generation() turns each question
# into exactly one contrastive pair: best_answer (score 4) vs. the
# first incorrect_answers entry (score 0), sharing that question's own
# prompt -- the same "fit on your own held-out train split of the
# dataset you're evaluating" style used for the UltraFeedback setup
# (FIT_DATASET_PATH -> ultrafeedback_train.json there), just applied
# to TruthfulQA's own train split instead. Since correct/incorrect
# answers carry no per-attribute breakdown, --fit-source truthfulqa
# forces --steer-attributes to a single "helpfulness" axis.
#
# This job always fits FRESH vectors for both CAA and MAT-Steer (no
# LOAD_CAA_VECTOR/LOAD_MAT_STEER defaults) -- see steeredhelpsteer2.slurm
# if you want the reuse-a-saved-vector workflow instead.
#
# Metrics (per variant -- baseline/caa/mat_steer):
#   - truthful_percent / informative_percent /
#     truthful_and_informative_percent: FlowJudge's Yes/No verdicts
#     on the TruthfulQA paper's own truthful/informative definitions.
#   - llm_judge_win_rate: % of examples where FlowJudge judged the
#     candidate answer at least as truthful+informative as the
#     dataset's own best_answer reference.
#   - generation_time_seconds: wall-clock time to generate candidate
#     answers for every eval-set question.
#
# Time budget: activation collection for CAA + a MAT-Steer training
# loop happen before any of the three generate+judge passes, and each
# pass itself runs THREE separate judge passes (truthful/informative/
# win_rate) rather than helpsteer2's two (pairwise + attribute
# scoring) -- --time is set the same as steeredhelpsteer2.slurm as a
# starting point; tune it once you've seen one full run's actual
# wall-clock time in the log.
#
# Submit:
#
#   cd BaselineEvaluation
#   sbatch steeredtruthfulqageneration.slurm
#
#   # or override paths on the command line, e.g.:
#   FIT_DATASET_PATH=/home/xrai/datasets/truthfulqa/truthfulqa_generation_train.json \
#   DATASET_PATH=/home/xrai/datasets/truthfulqa/truthfulqa_generation.json \
#   sbatch steeredtruthfulqageneration.slurm
#
# =============================================================================

echo "========================================================="
echo "Steered TruthfulQA Generation Evaluation"
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

# Evaluation set -- baseline/CAA/MAT-Steer are all generated+judged
# against this file only.
export DATASET_PATH="${DATASET_PATH:-/home/xrai/datasets/truthfulqa_generation/truthfulqa_generation_test.json}"

# Fitting set -- a SEPARATE TruthfulQA generation-shaped train split
# (question/best_answer/correct_answers/incorrect_answers), disjoint
# from DATASET_PATH. Used ONLY to build the CAA vector / train
# MAT-Steer, via --fit-source truthfulqa (see below) -- NOT
# HelpSteer2, matching how the UltraFeedback setup fits from its own
# ultrafeedback_train.json rather than an unrelated dataset.
export FIT_DATASET_PATH="${FIT_DATASET_PATH:-/home/xrai/datasets/truthfulqa_generation/truthfulqa_generation_train.json}"

# -------------------------------------------------------------------------
# Steering settings
# -------------------------------------------------------------------------

# --fit-source truthfulqa only supports a single "helpfulness" axis
# (correct vs. incorrect carries no per-attribute breakdown) --
# steering_eval_truthfulqa_generation.py forces this itself even if
# you override it, but set it correctly here too so the echoed
# settings below aren't misleading.
STEER_ATTRIBUTES="${STEER_ATTRIBUTES:-helpfulness}"
STEER_LAYER="${STEER_LAYER:-15}"
CAA_MULTIPLIER="${CAA_MULTIPLIER:-1.2}"
MAT_STEER_STEPS="${MAT_STEER_STEPS:-400}"

REPO_DIR="$SLURM_SUBMIT_DIR"
SCRIPT_DIR="${REPO_DIR}"

OUTPUT_DIR="${REPO_DIR}/data/evaluation_results"

echo ""
echo "Generation model  : $GENERATION_MODEL"
echo "Judge model       : $JUDGE_MODEL"
echo "Eval dataset      : $DATASET_PATH"
echo "Fit dataset       : $FIT_DATASET_PATH (--fit-source truthfulqa)"
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

for required_file in  truthfulqa_generation.py steeredtruthfulqa_generation.py; do
    if [[ ! -f "${SCRIPT_DIR}/${required_file}" ]]; then
        echo "ERROR: ${required_file} not found in ${SCRIPT_DIR} -- it's imported by this pipeline."
        exit 1
    fi
done

mkdir -p "${REPO_DIR}/logs"
mkdir -p "$OUTPUT_DIR"

echo "Pre-flight checks passed."
echo ""

# -------------------------------------------------------------------------
# Run Evaluation
# -------------------------------------------------------------------------

echo "========================================================="
echo "Starting Steered TruthfulQA Generation Evaluation"
echo "========================================================="

python "${SCRIPT_DIR}/steeredtruthfulqa_generation.py" \
    --generation-model "$GENERATION_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --fit-source truthfulqa \
    --fit-dataset-path "$FIT_DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 4 \
    --max-new-tokens 256 \
    --judge-max-new-tokens 10 \
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

LATEST=$(ls -t "$OUTPUT_DIR"/steering_eval_truthfulqa_generation_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then

    echo ""
    echo "Latest results:"
    echo "$LATEST"

fi
