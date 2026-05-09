#!/bin/bash
# Quick test of preprocessing on a small subset
# This will process only 1 batch (or your batch_size) to verify it works

cd /scratch/antruong/Depth-Anything-3

echo "Testing preprocessing with just 1 batch..."
python preprocess_depth.py \
    --data_root ../dataset \
    --output_dir ../dataset_depths_test \
    --model_name da3-large \
    --pretrained_path depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
    --batch_size 2 \
    --img_height 518 \
    --img_width 518 \
    2>&1 | head -50

echo ""
echo "Checking if any depth files were created..."
find ../dataset_depths_test -name "*.npy" 2>/dev/null | head -5

echo "Done!"
