#!/bin/bash
# =============================================================================
# DEBUG: Quick end-to-end pipeline test for multiview ChatML data (~2 min)
# Runs 20 steps on a small local data slice (200 samples).
#
# Prerequisites:
#   cd EO1
#   python experiments/10_vqa/convert_chatml_to_eo.py --download-images --limit 200
#
# Usage:
#   cd EO1
#   bash experiments/10_vqa/train_debug_multiview.sh
# =============================================================================

GPUS=2
PER_DEVICE_BATCH_SIZE=4

ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS}"

dataset=experiments/10_vqa/data-multiview-local.yaml
run_name=debug_multiview_chatml_test

accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --vlm-name-or-path ../../pretrained/Qwen2.5-VL-3B-Instruct \
    --data-path ${dataset} \
    --chunk-size 30 \
    --dataloader-num-workers 4 \
    --train-mm-only True \
    --train-lerobot-only False \
    --freeze-vision-tower False \
    --freeze-llm False \
    --freeze-merger False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --max-steps 20 \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps 1 \
    --learning-rate 2e-5 \
    --merger-lr 2e-5 \
    --vision-lr 2e-6 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy steps \
    --save-steps 10 \
    --save-total-limit 1 \
    --logging-steps 1 \
    --report-to none \
    --run-name ${run_name} \
    --attn-implementation sdpa
