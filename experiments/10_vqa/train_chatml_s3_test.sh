#!/bin/bash
# =============================================================================
# Test: ChatML format with S3 image loading on 2 GPUs (~2 min)
# Uses data-qa-multiview-chatml-s3-test.yaml (10 ChatML samples, S3 images).
#
# Prerequisites:
#   cd EO1
#   python experiments/10_vqa/download_qa_multiview_local.py --no-download-images
#
# Usage:
#   cd EO1
#   conda run -n eo bash experiments/10_vqa/train_chatml_s3_test.sh
# =============================================================================

GPUS=2
PER_DEVICE_BATCH_SIZE=4

ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS}"

dataset=experiments/10_vqa/data-qa-multiview-chatml-s3-test.yaml
run_name=test_chatml_s3

accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --vlm-name-or-path Qwen/Qwen2.5-VL-3B-Instruct \
    --data-path ${dataset} \
    --chunk-size 30 \
    --dataloader-num-workers 0 \
    --train-mm-only True \
    --train-lerobot-only False \
    --freeze-vision-tower False \
    --freeze-llm False \
    --freeze-merger False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --max-steps 10 \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps 1 \
    --learning-rate 2e-5 \
    --merger-lr 2e-5 \
    --vision-lr 2e-6 \
    --weight-decay 0.1 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy no \
    --logging-steps 1 \
    --report-to none \
    --run-name ${run_name} \
    --attn-implementation sdpa \
    --output-dir /tmp/test_chatml_s3
