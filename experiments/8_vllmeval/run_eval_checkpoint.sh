#!/bin/bash
# =============================================================================
# Run VLMEvalKit validation on trained checkpoint
#
# Uses checkpoint-final-126 and benchmarks: EOBench, ERQABench, RoboVQA
# Requires: eo conda env, benchmarks downloaded via download_benchmarks.sh
#
# Usage: bash experiments/8_vllmeval/run_eval_checkpoint.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EO1_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QWEN_TUNE_ROOT="$(cd "${EO1_ROOT}/../.." && pwd)"
VLMEVALKIT_ROOT="${QWEN_TUNE_ROOT}/VLMEvalKit"
CONFIG_PATH="${SCRIPT_DIR}/dataset-config-eval.json"
CHECKPOINT_DIR="${QWEN_TUNE_ROOT}/eo_experiment/EO1/outputs/2026-02-11/15-23-44-robot_vqa_debug_multi_node_n2_g1_lr2e-5_bs2"
BENCHMARKS_ROOT="${QWEN_TUNE_ROOT}/benchmarks"

echo "============================================"
echo "  EO1 Checkpoint Validation"
echo "============================================"
echo "  Checkpoint: ${CHECKPOINT_DIR}/checkpoint-final-126"
echo "  VLMEvalKit: ${VLMEVALKIT_ROOT}"
echo "  Config: ${CONFIG_PATH}"
echo "============================================"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: Checkpoint dir not found: ${CHECKPOINT_DIR}"
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi

if [[ ! -d "${VLMEVALKIT_ROOT}" ]]; then
    echo "ERROR: VLMEvalKit not found at ${VLMEVALKIT_ROOT}"
    echo "  Run: git clone https://github.com/DelinQu/VLMEvalKit ${VLMEVALKIT_ROOT}"
    exit 1
fi

# Activate eo conda env if available
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate eo 2>/dev/null || true
fi

export BENCHMARKS_ROOT="${BENCHMARKS_ROOT}"
export GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

cd "${VLMEVALKIT_ROOT}"
torchrun --nproc-per-node=${GPU_COUNT} run.py \
    --config "${CONFIG_PATH}" \
    --work-dir "${QWEN_TUNE_ROOT}/eo_experiment/EO1/eval_outputs" \
    --reuse

echo ""
echo "============================================"
echo "  Validation complete. Results in eval_outputs/"
echo "============================================"
