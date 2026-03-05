#!/bin/bash

# Two-pass eval for VQAv2: baseline vs pruned (first image token removed)
# Uses only the first sample from dataset

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
OUT_DIR="./playground/data/eval/vqav2/answers/$SPLIT"

mkdir -p $OUT_DIR

python research/pruning_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/vqav2/test2015 \
    --answers-file $OUT_DIR/${CKPT}_two_pass.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=== VQAv2 Two-Pass Eval Results ==="
cat $OUT_DIR/${CKPT}_two_pass.jsonl
