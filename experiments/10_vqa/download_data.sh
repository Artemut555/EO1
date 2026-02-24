#!/bin/bash
# Download robotics VQA data from EO-Data1.5M.
# Set EXTRA_EO; data will go to $EXTRA_EO/data/vqa_robot/ (jsonl + images).
# See parent repo README and IPEC-COMMUNITY/EO-Data1.5M for dataset details.
set -e
EXTRA_EO="${EXTRA_EO:-$HOME/projects/extra/eo}"
OUT_DIR="${EXTRA_EO}/data/vqa_robot"
mkdir -p "${OUT_DIR}/jsonl" "${OUT_DIR}/images"
echo "Target: ${OUT_DIR}"
echo "Download EO-Data1.5M QA subsets (e.g. via HuggingFace) and place jsonl + images under ${OUT_DIR}"
