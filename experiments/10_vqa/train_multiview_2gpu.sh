#!/bin/bash
# =============================================================================
# Multiview local training with 2 GPUs and batched data.
# Uses data-multiview-local.yaml (local images under extra/eo/data/multiview).
#
# Usage (from EO1 repo root, with conda env eo):
#   conda activate eo
#   bash experiments/10_vqa/train_multiview_2gpu.sh
#
# Or: conda run -n eo bash experiments/10_vqa/train_multiview_2gpu.sh
# =============================================================================

GPUS=2
PER_DEVICE_BATCH_SIZE=4
# Effective batch = GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 2 * 4 * 1 = 8
GRADIENT_ACCUMULATION_STEPS=1

ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS}"

dataset=experiments/10_vqa/data-multiview-local.yaml
run_name=multiview_local_2gpu

accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --vlm-name-or-path Qwen/Qwen2.5-VL-3B-Instruct \
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
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
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
