# Baseline LLM Fine-tuning Configuration

**Training Date:** 2026-03-05 10:06:39

## Model Configuration
- **Model Path:** `/data/users/airprofly/FastV/llava-v1.5-7b`
- **Vision Tower:** `/data/users/airprofly/FastV/llava-v1.5-7b`

## Data Configuration
- **Dataset Path:** `/data/users/Actor/Attention_Actor/playground/data/llava_v1_5_mix665k_10pct_cleaned.json`
- **Image Folder:** `/data/users/Actor/Attention_Actor/playground/data`

## Training Configuration
- **Output Directory:** `/data/users/Actor/LLaVA/results`
- **Learning Rate:** `1e-4`
- **Batch Size:** `1`
- **Epochs:** `1`

## Training Mode
- **Actor:** ❌ Disabled (Baseline)
- **LLM:** ❌ Frozen (只记录 loss，不训练)
- **Training:** 仅记录数据，不更新参数

## Visualization Configuration
- **Visualization Enabled:** `true`
- **Visualization Directory:** `/data/users/Actor/LLaVA/results`
- **Plots Save Steps:** ``
- **Checkpoints Save Steps:** ``

## Additional Configuration
- **Precision:** bf16
- **Max Length:** 2048
- **Gradient Checkpointing:** true
- **Lazy Preprocess:** true
- **LoRA Enabled:** true

