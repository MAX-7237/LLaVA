#!/bin/bash

# Two-pass eval for TextVQA: baseline vs pruned (first image token removed)
# Uses only the first sample from dataset
export CUDA_VISIBLE_DEVICES=0

CKPT="llava-v1.5-7b"
SPLIT="llava_textvqa_val_v051_ocr"
OUT_DIR="./playground/data/eval/textvqa/answers"

mkdir -p $OUT_DIR

python -m llava.research.model_prune_vqa_loader \
    --model-path /data/users/airprofly/FastV/llava-v1.5-7b \
    --question-file /data/users/Actor/DualHeadPruningActor/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/users/Actor/DualHeadPruningActor/playground/data/eval/textvqa/train_images \
    --answers-file $OUT_DIR/${CKPT}_two_pass.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=== TextVQA Two-Pass Eval Results ==="
cat $OUT_DIR/${CKPT}_two_pass.jsonl
