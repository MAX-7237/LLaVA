#!/bin/bash

# Two-pass eval for MM-Vet: baseline vs pruned (first image token removed)
# Uses only the first sample from dataset

CKPT="llava-v1.5-7b"
OUT_DIR="./playground/data/eval/mm-vet/answers"

mkdir -p $OUT_DIR

python research/pruning_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file $OUT_DIR/${CKPT}_two_pass.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=== MM-Vet Two-Pass Eval Results ==="
cat $OUT_DIR/${CKPT}_two_pass.jsonl
