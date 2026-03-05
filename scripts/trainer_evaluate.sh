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
export PRUNE_LAYER_INDEX=0
export PRUNE_TOKEN_INDEX=-1
export DID_PRUNE=False

# 模型路径
MODEL_PATH="/data/users/airprofly/FastV/llava-v1.5-7b"

# 数据路径
DATA_PATH="/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json"

# 图片路径
IMAGE_FOLDER="/data/users/Actor/Attention_Actor/playground/data"

# 输出目录
OUTPUT_DIR=f"/data/users/Actor/LLaVA/results_${PRUNE_LAYER_INDEX}"

# 学习率
LEARNING_RATE=1e-4

# 批次大小
BATCH_SIZE=1

# 训练轮数
EPOCHS=1

#==========================================
echo "=========================================="
echo "Baseline LLM Fine-tuning (No Actor)"
echo "=========================================="

# 创建训练配置记录文件
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
echo ""
echo "Starting training..."
echo ""

python -m llava.research.trainer_evaluate \
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
    --eval_start_index 0 \
    --eval_end_index 0 \
    --report_to none

    #倒数2。3参数，使用区间来挑选样本

echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "  - Checkpoints saved here"
echo "  - LM Loss visualization: ${OUTPUT_DIR}/actor_visualizations/"
echo "=========================================="