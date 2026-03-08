#!/bin/bash
#==========================================
# Baseline LLM Fine-tuning Script
# 只使用 LLM 进行微调，不使用 Actor 剪枝
# 记录 lm_loss 随着 step 的变化情况
#==========================================

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

export NUM_IMAGE_TOKENS=576
export SYS_PROMPT_LEN=35
export GLOBAL_COMPUTE_ATTN_WEIGHTS=1
# List of layers to process
PRUNE_LAYER_LIST=(0)

EVAL_START_INDEX=0
EVAL_END_INDEX=0

# Model path
MODEL_PATH="/data/users/airprofly/FastV/llava-v1.5-7b"

# Data path
DATA_PATH="/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json"

# Image folder
IMAGE_FOLDER="/data/users/Actor/Attention_Actor/playground/data"

# Learning rate
LEARNING_RATE=1e-4

# Batch size
BATCH_SIZE=1

# Epochs
EPOCHS=1

#==========================================
echo "=========================================="
echo "Starting loop processing..."
echo "=========================================="

# Loop through each layer
for PRUNE_LAYER_INDEX in "${PRUNE_LAYER_LIST[@]}"
do
    export PRUNE_LAYER_INDEX
    export PRUNE_TOKEN_INDEX=-1
    export DID_PRUNE=False

    echo ""
    echo "=========================================="
    echo "Processing Layer: ${PRUNE_LAYER_INDEX}"
    echo "=========================================="

    # Output directory
    OUTPUT_DIR=f"/data/users/Actor/LLaVA_Prune/results_attn_weights/results_attn_weights_${PRUNE_LAYER_INDEX}"

    # Create training config file (only if it doesn't exist)
    if [ ! -f "${OUTPUT_DIR}/training_config.md" ]; then
        cat > "${OUTPUT_DIR}/training_config.md" << EOF
# Baseline LLM Fine-tuning Configuration

**Training Date:** $(date '+%Y-%m-%d %H:%M:%S')

## Model Configuration
- **Model Path:** \`${MODEL_PATH}\`
- **Vision Tower:** \`${MODEL_PATH}\`

## Data Configuration
- **Dataset Path:** \`${DATA_PATH}\`
- **Image Folder:** \`${IMAGE_FOLDER}\`

## Training Configuration
- **Output Directory:** \`${OUTPUT_DIR}\`
- **Learning Rate:** \`${LEARNING_RATE}\`
- **Batch Size:** \`${BATCH_SIZE}\`
- **Epochs:** \`${EPOCHS}\`

## Training Mode
- **Actor:** ❌ Disabled (Baseline)
- **LLM:** ❌ Frozen (只记录 loss，不训练)
- **Training:** 仅记录数据，不更新参数

## Visualization Configuration
- **Visualization Enabled:** \`true\`
- **Visualization Directory:** \`${OUTPUT_DIR}\`
- **Plots Save Steps:** \`${VISUALIZATION_PLOTS_SAVE_STEPS}\`
- **Checkpoints Save Steps:** \`${VISUALIZATION_CHECKPOINT_SAVE_STEPS}\`

## Additional Configuration
- **Precision:** bf16
- **Max Length:** 2048
- **Gradient Checkpointing:** true
- **Lazy Preprocess:** true
- **LoRA Enabled:** true

EOF
        echo "[Config] Training configuration saved to: ${OUTPUT_DIR}/training_config.md"
    fi

    echo "Starting training for layer ${PRUNE_LAYER_INDEX}..."
    echo ""

    python -m research.trainer_evaluate_2 \
        --model_name_or_path ${MODEL_PATH} \
        --vision_tower ${MODEL_PATH} \
        --version llava_v1 \
        --data_path ${DATA_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --output_dir ${OUTPUT_DIR} \
        --freeze_backbone \
        --bf16 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 3 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 0. \
        --warmup_steps 0 \
        --max_steps -1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 False \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --num_train_epochs ${EPOCHS} \
        --bits 16 \
        --eval_start_index ${EVAL_START_INDEX} \
        --eval_end_index ${EVAL_END_INDEX} \
        --report_to none

    echo "Finished processing layer ${PRUNE_LAYER_INDEX}"
done

echo "=========================================="
echo "All processing complete!"
echo "=========================================="
