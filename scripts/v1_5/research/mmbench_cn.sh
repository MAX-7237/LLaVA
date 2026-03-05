#!/bin/bash

# Two-pass eval for MMBench-CN: baseline vs pruned (first image token removed)
# Uses only the first sample from dataset

CKPT="llava-v1.5-7b"
SPLIT="mmbench_dev_cn_20231003"
OUT_DIR="./playground/data/eval/mmbench_cn/answers/$SPLIT"

mkdir -p $OUT_DIR

python research/pruning_eval.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file $OUT_DIR/${CKPT}_two_pass.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

echo "=== MMBench-CN Two-Pass Eval Results ==="
cat $OUT_DIR/${CKPT}_two_pass.jsonl
