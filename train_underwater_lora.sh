#!/usr/bin/env bash
set -euo pipefail

# Train Depth Anything 3 with LoRA for underwater domain adaptation
# using image-only consistency training (original vs simulated underwater).
#
# For depth-aware underwater simulation:
#   1. Preprocess: python preprocess_depth.py --data_root ../dataset --output_dir ../dataset_depths
#   2. Train with:  DEPTHS_ROOT=../dataset_depths bash train_underwater_lora.sh
#
# Examples:
#   bash train_underwater_lora.sh
#   bash train_underwater_lora.sh --data_root ../dataset --epochs 30 --batch_size 8
#   DEPTHS_ROOT=../dataset_depths bash train_underwater_lora.sh  # Use depth-aware simulation
#
# Any extra CLI args are forwarded to train.py.

# -----------------------------
# Defaults (can be overridden by env vars)
# -----------------------------
DATA_ROOT="${DATA_ROOT:-../dataset}"
DEPTHS_ROOT="${DEPTHS_ROOT:-../dataset_depths}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints_underwater}"
MODEL_NAME="${MODEL_NAME:-da3-large}"
PRETRAINED_PATH="${PRETRAINED_PATH:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
USE_DORA="${USE_DORA:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

IMG_HEIGHT="${IMG_HEIGHT:-476}"
IMG_WIDTH="${IMG_WIDTH:-476}"
TRAIN_RATIO="${TRAIN_RATIO:-0.98}"

CONSISTENCY_GRAD_WEIGHT="${CONSISTENCY_GRAD_WEIGHT:-0.5}"
CONSISTENCY_CONF_WEIGHT="${CONSISTENCY_CONF_WEIGHT:-0.1}"

# Optional: Path to precomputed depth maps (for depth-aware underwater simulation)
DEPTHS_ROOT="${DEPTHS_ROOT:-}"

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

# Resolve a valid default Hugging Face checkpoint unless user overrides PRETRAINED_PATH.
# Public DA3 checkpoints live under depth-anything/*, not ByteDance-Seed/*.
if [[ -z "$PRETRAINED_PATH" ]]; then
  case "$MODEL_NAME" in
    da3-large)
      PRETRAINED_PATH="depth-anything/DA3-LARGE-1.1"
      ;;
    da3-giant)
      PRETRAINED_PATH="depth-anything/DA3-GIANT-1.1"
      ;;
    da3-base)
      PRETRAINED_PATH="depth-anything/DA3-BASE"
      ;;
    da3-small)
      PRETRAINED_PATH="depth-anything/DA3-SMALL"
      ;;
    da3metric-large)
      PRETRAINED_PATH="depth-anything/DA3-LARGE-1.1"
      ;;
    da3mono-large)
      PRETRAINED_PATH="depth-anything/DA3-LARGE-1.1"
      ;;
    *)
      PRETRAINED_PATH="depth-anything/DA3-LARGE-1.1"
      ;;
  esac
fi

# -----------------------------
# Launch training
# -----------------------------
echo "Starting underwater LoRA training"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  PRETRAINED_PATH=$PRETRAINED_PATH"echo "  DEPTHS_ROOT=${DEPTHS_ROOT:-none (heuristic simulation)}"echo "  EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS LR=$LR"
echo "  NPROC_PER_NODE=$NPROC_PER_NODE"

TORCH_ARGS=()
if [[ "$USE_DORA" == "1" ]]; then
  TORCH_ARGS+=(--use_dora)
fi

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
  --training_mode underwater_consistency \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --pretrained_path "$PRETRAINED_PATH" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --lr "$LR" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --img_height "$IMG_HEIGHT" \
  --img_width "$IMG_WIDTH" \
  --train_ratio "$TRAIN_RATIO" \
  --consistency_grad_weight "$CONSISTENCY_GRAD_WEIGHT" \
  --consistency_conf_weight "$CONSISTENCY_CONF_WEIGHT" \
  $([ -n "$DEPTHS_ROOT" ] && echo "--depths_root $DEPTHS_ROOT") \
  "${TORCH_ARGS[@]}" \
  "$@"
