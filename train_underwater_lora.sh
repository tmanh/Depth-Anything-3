#!/usr/bin/env bash
set -euo pipefail

# Train Depth Anything 3 with LoRA for underwater domain adaptation
# using image-only consistency training (original vs simulated underwater).
#
# Examples:
#   bash train_underwater_lora.sh
#   bash train_underwater_lora.sh --data_root ../dataset --epochs 30 --batch_size 8
#
# Any extra CLI args are forwarded to train.py.

# -----------------------------
# Defaults (can be overridden by env vars)
# -----------------------------
DATA_ROOT="${DATA_ROOT:-../dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints_underwater}"
MODEL_NAME="${MODEL_NAME:-da3-large}"
PRETRAINED_PATH="${PRETRAINED_PATH:-ByteDance-Seed/Depth-Anything-3}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

IMG_HEIGHT="${IMG_HEIGHT:-518}"
IMG_WIDTH="${IMG_WIDTH:-518}"
TRAIN_RATIO="${TRAIN_RATIO:-0.98}"

CONSISTENCY_GRAD_WEIGHT="${CONSISTENCY_GRAD_WEIGHT:-0.5}"
CONSISTENCY_CONF_WEIGHT="${CONSISTENCY_CONF_WEIGHT:-0.1}"

# -----------------------------
# Basic checks
# -----------------------------
if [[ ! -f "train.py" ]]; then
  echo "Error: train.py not found. Run this script from the repository root."
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Error: data root not found: $DATA_ROOT"
  echo "Set DATA_ROOT or pass: --data_root /path/to/dataset"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# Launch training
# -----------------------------
echo "Starting underwater LoRA training"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  PRETRAINED_PATH=$PRETRAINED_PATH"
echo "  EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR"

torchrun --standalone --nproc_per_node=1 train.py \
  --training_mode underwater_consistency \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --pretrained_path "$PRETRAINED_PATH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --use_dora \
  --img_height "$IMG_HEIGHT" \
  --img_width "$IMG_WIDTH" \
  --train_ratio "$TRAIN_RATIO" \
  --consistency_grad_weight "$CONSISTENCY_GRAD_WEIGHT" \
  --consistency_conf_weight "$CONSISTENCY_CONF_WEIGHT" \
  "$@"
