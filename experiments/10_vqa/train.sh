#!/bin/bash
# Robotics VQA fine-tuning. Set EXTRA_EO and use a generated data yaml (see README).
# Example: --vlm-name-or-path ${EXTRA_EO}/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct
#          --data-path ../../config/data-vqa-extra.yaml
set -e
GPUS="${GPUS:-2}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
EXTRA_EO="${EXTRA_EO:-$HOME/projects/extra/eo}"
DATASET="${DATASET:-experiments/10_vqa/data-vqa.yaml}"
VLM_PATH="${VLM_PATH:-${EXTRA_EO}/checkpoints/pretrained/Qwen2.5-VL-3B-Instruct}"

accelerate launch --num_machines 1 --machine_rank 0 --num_processes "${GPUS}" scripts/train.py \
  --vlm-name-or-path "${VLM_PATH}" \
  --data-path "${DATASET}" \
  --chunk-size 30 \
  --dataloader-num-workers 8 \
  --train-mm-only True \
  --train-lerobot-only False \
  --freeze-vision-tower False \
  --freeze-llm False \
  --freeze-merger False \
  --bf16 True \
  --num-train-epochs 2 \
  --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-5 \
  --merger-lr 2e-5 \
  --vision-lr 2e-6 \
  --weight-decay 0.1 \
  --warmup-ratio 0.03 \
  --lr-scheduler-type cosine \
  --gradient-checkpointing True \
  --save-strategy steps \
  --save-steps 2000 \
  --save-total-limit 3 \
  --report-to none \
  --attn-implementation flash_attention_2
