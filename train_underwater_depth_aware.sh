#!/bin/bash
# Workflow for depth-aware underwater simulation training
#
# This script demonstrates how to use the new depth-aware underwater simulation:
# 1. Preprocess: Compute depth maps for all dataset images
# 2. Train: Run training with depth-aware underwater effects

set -e

# Configuration
DATASET_ROOT="../dataset"
DEPTHS_ROOT="../dataset_depths"
OUTPUT_DIR="./checkpoints_underwater_depth_aware"
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
IMG_HEIGHT=490
IMG_WIDTH=490
NPROC_PER_NODE=2
PRETRAINED_PATH="depth-anything/DA3NESTED-GIANT-LARGE-1.1"
LORA_RANK=16
LORA_ALPHA=32
EPOCHS=20

ask_yes_no() {
    local prompt="$1"
    local reply
    while true; do
        read -r -p "$prompt (y/n) " reply
        case "${reply,,}" in
            y|yes) return 0 ;;
            n|no) return 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

echo "==================================================================="
echo "Step 1: Preprocess - Compute depth maps for all dataset images"
echo "==================================================================="
echo "This will:"
echo "  - Load DA3 model"
echo "  - Iterate through all images in $DATASET_ROOT"
echo "  - Compute depth predictions"
echo "  - Save as .npy files in $DEPTHS_ROOT (mirroring directory structure)"
echo ""
echo "Command:"
echo "  python preprocess_depth.py \\"
echo "    --data_root $DATASET_ROOT \\"
echo "    --output_dir $DEPTHS_ROOT \\"
echo "    --model_name da3-large \\"
echo "    --pretrained_path $PRETRAINED_PATH \\"
echo "    --batch_size 8 \\"
echo "    --img_height 518 \\"
echo "    --img_width 518 \\" 
echo "    --skip_existing"
echo ""
if ask_yes_no "Run preprocessing?"; then
    python preprocess_depth.py \
        --data_root "$DATASET_ROOT" \
        --output_dir "$DEPTHS_ROOT" \
        --model_name da3-large \
        --pretrained_path "$PRETRAINED_PATH" \
        --batch_size 8 \
        --img_height 518 \
        --img_width 518 \
        --skip_existing
fi

echo ""
echo "==================================================================="
echo "Step 2: Train - LoRA fine-tuning with depth-aware underwater effects"
echo "==================================================================="
echo "This will:"
echo "  - Load precomputed depth maps from $DEPTHS_ROOT"
echo "  - Use depths to drive underwater simulation parameters"
echo "  - Train model with DDP on 2 GPUs"
echo ""
echo "Command:"
echo "  torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py \\"
echo "    --data_root $DATASET_ROOT \\"
echo "    --depths_root $DEPTHS_ROOT \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --model_name da3-large \\"
echo "    --pretrained_path $PRETRAINED_PATH \\"
echo "    --training_mode underwater_consistency \\"
echo "    --lora_rank $LORA_RANK \\"
echo "    --lora_alpha $LORA_ALPHA \\"
echo "    --batch_size $BATCH_SIZE \\"
echo "    --grad_accum_steps $GRAD_ACCUM_STEPS \\"
echo "    --img_height $IMG_HEIGHT \\"
echo "    --img_width $IMG_WIDTH \\"
echo "    --epochs $EPOCHS"
echo ""
if ask_yes_no "Run training?"; then
    mkdir -p "$OUTPUT_DIR"
    export CUDA_VISIBLE_DEVICES=0,1
    export TORCH_CUDA_ARCH_LIST="89"
    
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py \
        --data_root "$DATASET_ROOT" \
        --depths_root "$DEPTHS_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --model_name da3-large \
        --pretrained_path "$PRETRAINED_PATH" \
        --training_mode underwater_consistency \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --grad_accum_steps $GRAD_ACCUM_STEPS \
        --img_height $IMG_HEIGHT \
        --img_width $IMG_WIDTH \
        --epochs $EPOCHS
    echo ""
    echo "==================================================================="
    echo "Training complete! Checkpoints saved to $OUTPUT_DIR"
    echo "==================================================================="
else
    echo ""
    echo "Training skipped by user."
fi
