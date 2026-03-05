#!/bin/bash

# Two-pass eval for GQA: baseline vs pruned (first image token removed)
# Uses only the first sample from dataset

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"
OUT_DIR="./playground/data/eval/gqa/answers/$SPLIT"

mkdir -p $OUT_DIR

python research/pruning_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
    --image-folder ./playground/data/eval/gqa/data/images \
    --answers-file $OUT_DIR/${CKPT}_two_pass.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=== GQA Two-Pass Eval Results ==="
cat $OUT_DIR/${CKPT}_two_pass.jsonl
