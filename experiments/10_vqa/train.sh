#!/bin/bash
# =============================================================================
# Unified EO1 VQA training script
#
# All training parameters are configurable via environment variables.
# Use a profile for common presets, or set variables directly.
#
# Usage (from EO1 repo root):
#   # Default: 1 GPU, multiview_local, 20 steps
#   bash experiments/10_vqa/train.sh
#
#   # Override individual vars
#   GPUS=2 DATASET=experiments/10_vqa/configs/data/vqa_robot_local.yaml bash experiments/10_vqa/train.sh
#
#   # Use a profile (auto-loads configs/profiles/<name>.env)
#   PROFILE=local_2gpu bash experiments/10_vqa/train.sh
#   PROFILE=prod bash experiments/10_vqa/train.sh
#
#   # Source profile manually + override
#   source experiments/10_vqa/configs/profiles/local_2gpu.env
#   DATASET=experiments/10_vqa/configs/data/multiview_chatml_s3.yaml bash experiments/10_vqa/train.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-load profile if PROFILE is set
if [ -n "${PROFILE:-}" ] && [ -f "${SCRIPT_DIR}/configs/profiles/${PROFILE}.env" ]; then
    echo "[train.sh] Loading profile: ${PROFILE}"
    set -a && source "${SCRIPT_DIR}/configs/profiles/${PROFILE}.env" && set +a
fi

# --------------- Training parameters (all overridable) ---------------

GPUS="${GPUS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
# Default max steps only when neither epochs nor max_steps is set (so epoch-based profiles are not overridden)
if [ -z "${MAX_STEPS}" ] && [ -z "${NUM_EPOCHS}" ]; then
    MAX_STEPS="20"
fi
DATASET="${DATASET:-experiments/10_vqa/configs/data/multiview_local.yaml}"
VLM_PATH="${VLM_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
PACK_DATASET="${PACK_DATASET:-False}"
MAX_PACKED_LENGTH="${MAX_PACKED_LENGTH:-16384}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-2e-5}"
VISION_LR="${VISION_LR:-2e-6}"
MERGER_LR="${MERGER_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
RUN_NAME="${RUN_NAME:-vqa_$(date +%m%d_%H%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
REPORT_TO="${REPORT_TO:-none}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --------------- Multi-node support (set by cloud entrypoint) ---------------

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
TOTAL_PROCS="${TOTAL_PROCS:-${GPUS}}"

# --------------- Print config ---------------

echo "============================================"
echo "  EO1 VQA Training"
echo "============================================"
echo "  Profile:       ${PROFILE:-none}"
echo "  GPUs:          ${GPUS}"
echo "  Nodes:         ${NNODES}"
echo "  Total procs:   ${TOTAL_PROCS}"
echo "  Dataset:       ${DATASET}"
echo "  Model:         ${VLM_PATH}"
echo "  Batch/GPU:     ${PER_DEVICE_BATCH_SIZE}"
echo "  Grad accum:    ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Eff. batch:    $((PER_DEVICE_BATCH_SIZE * TOTAL_PROCS * GRADIENT_ACCUMULATION_STEPS))"
if [ -n "${MAX_STEPS}" ]; then
    echo "  Max steps:     ${MAX_STEPS}"
else
    echo "  Epochs:        ${NUM_EPOCHS}"
fi
echo "  Attn impl:     ${ATTN_IMPL}"
echo "  Pack dataset:  ${PACK_DATASET}"
echo "  Max packed:    ${MAX_PACKED_LENGTH}"
echo "  Save strategy: ${SAVE_STRATEGY}"
echo "  Save total:    ${SAVE_TOTAL_LIMIT}"
echo "  Run name:      ${RUN_NAME}"
echo "============================================"

# --------------- Build accelerate args ---------------

ACCELERATE_ARGS=(
    --num_machines "${NNODES}"
    --machine_rank "${NODE_RANK}"
    --num_processes "${TOTAL_PROCS}"
)

if [ "${NNODES}" -gt 1 ] && [ -n "${MASTER_ADDR:-}" ]; then
    ACCEL_PORT="${ACCELERATE_MAIN_PORT:-$((MASTER_PORT + 1))}"
    ACCELERATE_ARGS+=(
        --main_process_ip "${MASTER_ADDR}"
        --main_process_port "${ACCEL_PORT}"
    )
fi

# --------------- Build training args ---------------

TRAIN_ARGS=(
    --vlm-name-or-path "${VLM_PATH}"
    --data-path "${DATASET}"
    --chunk-size "${CHUNK_SIZE}"
    --pack-dataset "${PACK_DATASET}"
    --max-packed-length "${MAX_PACKED_LENGTH}"
    --dataloader-num-workers "${NUM_WORKERS}"
    --train-mm-only True
    --train-lerobot-only False
    --freeze-vision-tower False
    --freeze-llm False
    --freeze-merger False
    --bf16 True
    --tf32 True
    --fp16 False
    --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${LR}"
    --merger-lr "${MERGER_LR}"
    --vision-lr "${VISION_LR}"
    --weight-decay 0.1
    --warmup-ratio "${WARMUP_RATIO}"
    --lr-scheduler-type cosine
    --gradient-checkpointing True
    --save-strategy "${SAVE_STRATEGY}"
    --save-steps "${SAVE_STEPS}"
    --save-total-limit "${SAVE_TOTAL_LIMIT}"
    --logging-steps "${LOGGING_STEPS}"
    --report-to "${REPORT_TO}"
    --run-name "${RUN_NAME}"
    --attn-implementation "${ATTN_IMPL}"
)

if [ -n "${MAX_STEPS}" ]; then
    TRAIN_ARGS+=(--max-steps "${MAX_STEPS}")
elif [ -n "${NUM_EPOCHS}" ]; then
    TRAIN_ARGS+=(--num-train-epochs "${NUM_EPOCHS}")
fi

if [ -n "${OUTPUT_DIR}" ]; then
    TRAIN_ARGS+=(--output-dir "${OUTPUT_DIR}")
elif [ -n "${OUTPUT_BASE}" ]; then
    TRAIN_ARGS+=(--output-base "${OUTPUT_BASE}")
fi

# --------------- Launch ---------------

accelerate launch "${ACCELERATE_ARGS[@]}" scripts/train.py "${TRAIN_ARGS[@]}" ${EXTRA_ARGS}
