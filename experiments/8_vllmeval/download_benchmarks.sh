#!/bin/bash
# =============================================================================
# Download benchmark datasets for EO1 VLM validation
#
# Downloads EO-Bench, ERQABench, and RoboVQA from HuggingFace to qwen_tune/benchmarks/
# Use with eo conda env: conda activate eo && bash experiments/8_vllmeval/download_benchmarks.sh
#
# Estimated total: ~5-15 GB
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# EO1 is at qwen_tune/eo_experiment/EO1; qwen_tune is 2 levels up from EO1
EO1_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QWEN_TUNE_ROOT="$(cd "${EO1_ROOT}/../.." && pwd)"
BENCHMARKS_ROOT="${QWEN_TUNE_ROOT}/benchmarks"

echo "============================================"
echo "  EO1 Benchmark Dataset Download"
echo "============================================"
echo "  BENCHMARKS_ROOT: ${BENCHMARKS_ROOT}"
echo "============================================"

mkdir -p "${BENCHMARKS_ROOT}"
cd "${BENCHMARKS_ROOT}"

# EO-Bench: exists at IPEC-COMMUNITY/EO-Bench
# ERQABench, RoboVQA: may require alternate sources - add if available
for dataset in EO-Bench; do
    echo ""
    echo "Downloading ${dataset}..."
    if huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        "IPEC-COMMUNITY/${dataset}" \
        --local-dir "${BENCHMARKS_ROOT}/${dataset}"; then
        echo "  ${dataset} downloaded."
    else
        echo "  WARNING: Failed to download ${dataset}"
    fi
done

echo ""
echo "Note: ERQABench and RoboVQA are not yet on HuggingFace."
echo "  Add them manually to ${BENCHMARKS_ROOT} if you have them."

echo ""
echo "============================================"
echo "  Download complete. Benchmarks at: ${BENCHMARKS_ROOT}"
echo "============================================"
