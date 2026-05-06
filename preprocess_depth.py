"""
Preprocess script to compute and cache depth maps for all images in a dataset.

Usage:
    python preprocess_depth.py \
        --data_root ../dataset \
        --output_dir ../dataset_depths \
        --model_name da3-large \
        --batch_size 4 \
        --img_height 518 \
        --img_width 518 \
        --device cuda
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess and cache depth maps")
    p.add_argument("--data_root", required=True, help="Root directory of images to process")
    p.add_argument("--output_dir", required=True, help="Directory to save depth maps (.npy files)")
    p.add_argument("--model_name", default="da3-large",
                   choices=["da3-small", "da3-base", "da3-large", "da3-giant",
                            "da3metric-large", "da3mono-large"],
                   help="DA3 model preset")
    p.add_argument("--pretrained_path", default=None,
                   help="HuggingFace Hub model ID or local path to load weights from")
    p.add_argument("--img_height", type=int, default=518)
    p.add_argument("--img_width", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip images that already have depth files")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def get_all_image_paths(data_root: Path) -> list[Path]:
    """Recursively find all image files."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for ext in extensions:
        paths.extend(data_root.rglob(f"*{ext}"))
        paths.extend(data_root.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def get_depth_output_path(img_path: Path, data_root: Path, output_dir: Path) -> Path:
    """Compute the output path for a depth map, preserving directory structure."""
    # Relative path from data_root to img_path
    rel_path = img_path.relative_to(data_root)
    # Replace extension with .npy
    depth_rel_path = rel_path.with_suffix(".npy")
    # Full output path
    output_path = output_dir / depth_rel_path
    return output_path


def process_images(
    image_paths: list[Path],
    data_root: Path,
    output_dir: Path,
    model: DepthAnything3,
    device: torch.device,
    img_size: tuple[int, int],
    batch_size: int = 4,
    skip_existing: bool = False,
) -> None:
    """Process images in batches and save depth maps."""
    model.eval()
    img_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Group images into batches
    batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
    skipped = 0
    processed = 0

    with torch.no_grad():
        for batch_idx, batch_paths in enumerate(tqdm(batches, desc="Processing batches")):
            # Filter paths if skip_existing
            if skip_existing:
                batch_paths = [
                    p for p in batch_paths
                    if not get_depth_output_path(p, data_root, output_dir).exists()
                ]
                if not batch_paths:
                    skipped += len(batch_paths)
                    continue

            images_list = []
            valid_paths = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = img_transform(img)
                    images_list.append(img_tensor)
                    valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

            if not images_list:
                continue

            # Stack into batch
            batch_tensor = torch.stack(images_list).to(device)  # (B, 3, H, W)

            # Add temporal dimension (single frame)
            batch_5d = batch_tensor.unsqueeze(1)  # (B, 1, 3, H, W)

            # Inference
            try:
                outputs = model(batch_5d)
                depths = outputs["depth"]  # (B, 1, H, W)
                depths = depths.squeeze(1).cpu().numpy()  # (B, H, W)

                # Save each depth map
                for img_path, depth_map in zip(valid_paths, depths):
                    output_path = get_depth_output_path(img_path, data_root, output_dir)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(output_path), depth_map.astype(np.float32))
                    processed += 1

            except Exception as e:
                logger.error(f"Inference failed for batch {batch_idx}: {e}")

    logger.info(f"Processed {processed} images, skipped {skipped}")


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    logger.info(f"Loading model: {args.model_name}")
    if args.pretrained_path:
        model = DepthAnything3.from_pretrained(args.pretrained_path)
    else:
        model = DepthAnything3(model_name=args.model_name)

    model = model.to(device)

    logger.info(f"Collecting image paths from {data_root}")
    image_paths = get_all_image_paths(data_root)

    if not image_paths:
        logger.warning(f"No image files found under {data_root}")
        return

    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"Saving depth maps to {output_dir}")

    img_size = (args.img_height, args.img_width)
    process_images(
        image_paths,
        data_root,
        output_dir,
        model,
        device,
        img_size,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
