"""
LoRA fine-tuning for Depth Anything 3.

Uses HuggingFace PEFT (>=0.10) with LoRA (+ optional DoRA) targeting
the DinoV2 ViT backbone attention and feed-forward layers.

References:
  - LoRA:  https://arxiv.org/abs/2106.09685
  - DoRA:  https://arxiv.org/abs/2402.09353
  - PEFT:  https://github.com/huggingface/peft

Usage:
    python train.py \
        --model_name da3-large \
        --pretrained_path ByteDance-Seed/Depth-Anything-V3-Large \
        --data_root /path/to/depth_dataset \
        --output_dir ./checkpoints \
        --lora_rank 16 \
        --lora_alpha 32 \
        --use_dora \
        --epochs 20 \
        --batch_size 4 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from PIL import Image
import torchvision.transforms as T

# PEFT – install with: pip install peft>=0.10.0
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer

from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import load_config
from depth_anything_3.utils.logger import logger


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset  (replace / extend for your actual data format)
# ---------------------------------------------------------------------------

class DepthDataset(Dataset):
    """
    Minimal depth dataset.  Expects a directory with sub-directories:
        <data_root>/rgb/   – *.jpg / *.png images
        <data_root>/depth/ – same-name depth files (16-bit PNG in mm, or *.npy)

    Override _load_depth() to adapt to your format.
    """

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: tuple[int, int] = (518, 518),
        max_depth: float = 80.0,
    ):
        self.data_root = Path(data_root)
        self.img_size  = img_size
        self.max_depth = max_depth

        rgb_dir   = self.data_root / split / "rgb"
        depth_dir = self.data_root / split / "depth"

        assert rgb_dir.exists(),   f"RGB directory not found: {rgb_dir}"
        assert depth_dir.exists(), f"Depth directory not found: {depth_dir}"

        self.rgb_paths   = sorted(rgb_dir.glob("*.[jp][pn]g"))
        self.depth_paths = [depth_dir / (p.stem + ".png") for p in self.rgb_paths]
        # fall back to .npy depth
        self.depth_paths = [
            d if d.exists() else depth_dir / (p.stem + ".npy")
            for d, p in zip(self.depth_paths, self.rgb_paths)
        ]

        self.img_transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD),
        ])

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def _load_depth(self, path: Path) -> torch.Tensor:
        """Returns (H, W) float32 depth in metres."""
        if path.suffix == ".npy":
            depth = np.load(str(path)).astype(np.float32)
        else:
            depth = np.array(Image.open(str(path)), dtype=np.float32) / 1000.0  # mm → m
        return torch.from_numpy(depth)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth = self._load_depth(self.depth_paths[idx])

        img_tensor = self.img_transform(image)  # (3, H, W)

        # Resize depth to match img_size
        depth = depth.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth = F.interpolate(depth, size=self.img_size, mode="nearest").squeeze()  # (H,W)

        # Clamp and mark invalid pixels
        valid_mask = (depth > 0) & (depth < self.max_depth)
        depth = depth.clamp(0, self.max_depth)

        return {
            "image":      img_tensor,   # (3, H, W)
            "depth_gt":   depth,        # (H, W)
            "valid_mask": valid_mask,   # (H, W) bool
        }


class ColorImageDataset(Dataset):
    """
    Image-only dataset for folders like ../dataset where images are spread across
    nested subdirectories (for example sa_000023/*.jpg, coco2017/*.jpg, ...).

    Returns:
      {
        "image":      FloatTensor [3, H, W] normalized,
        "image_path": str path to source image
      }
    """

    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(
        self,
        data_root: str,
        split: str = "all",
        img_size: tuple[int, int] = (518, 518),
        train_ratio: float = 0.98,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.split = split

        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Image root directory not found: {self.data_root}")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")

        all_paths = []
        for ext in self.EXTENSIONS:
            all_paths.extend(self.data_root.rglob(f"*{ext}"))
            all_paths.extend(self.data_root.rglob(f"*{ext.upper()}"))
        all_paths = sorted(set(all_paths))

        if not all_paths:
            raise RuntimeError(f"No image files found under {self.data_root}")

        # Deterministic split by shuffled index (works for arbitrary folder layouts).
        rng = random.Random(seed)
        indices = list(range(len(all_paths)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_ids = set(indices[:split_idx])

        if split == "train":
            self.image_paths = [p for i, p in enumerate(all_paths) if i in train_ids]
        elif split == "val":
            self.image_paths = [p for i, p in enumerate(all_paths) if i not in train_ids]
        else:
            self.image_paths = all_paths

        if not self.image_paths:
            raise RuntimeError(
                f"Split '{split}' is empty under {self.data_root}. "
                "Adjust train_ratio or use split='all'."
            )

        self.img_transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD),
        ])

        logger.info(
            f"ColorImageDataset split={split}: {len(self.image_paths)} images from {self.data_root}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.img_transform(image)
        return {
            "image": img_tensor,
            "image_path": str(image_path),
        }


def build_color_image_dataloader(
    data_root: str,
    split: str = "all",
    img_size: tuple[int, int] = (518, 518),
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool | None = None,
    train_ratio: float = 0.98,
    seed: int = 42,
) -> DataLoader:
    """Convenience factory for an RGB-only dataloader."""
    dataset = ColorImageDataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        train_ratio=train_ratio,
        seed=seed,
    )

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def scale_invariant_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.85,
) -> torch.Tensor:
    """Scale-invariant log loss (eigen et al.)."""
    pred   = pred[mask].clamp(min=1e-3)
    target = target[mask].clamp(min=1e-3)
    d = torch.log(pred) - torch.log(target)
    return (d ** 2).mean() - alpha * (d.mean() ** 2)


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-scale gradient matching loss."""
    total = torch.tensor(0.0, device=pred.device)
    for scale in range(4):
        step = 2 ** scale
        p = pred[:, ::step, ::step]
        t = target[:, ::step, ::step]
        m = mask[:, ::step, ::step]

        dx_p = p[:, :, 1:] - p[:, :, :-1]
        dy_p = p[:, 1:, :] - p[:, :-1, :]
        dx_t = t[:, :, 1:] - t[:, :, :-1]
        dy_t = t[:, 1:, :] - t[:, :-1, :]

        mx = m[:, :, 1:] & m[:, :, :-1]
        my = m[:, 1:, :] & m[:, :-1, :]

        if mx.any():
            total = total + (dx_p[mx] - dx_t[mx]).abs().mean()
        if my.any():
            total = total + (dy_p[my] - dy_t[my]).abs().mean()
    return total


def depth_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor | None,
    depth_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_grad: float = 0.5,
) -> torch.Tensor:
    """Combined scale-invariant + gradient loss, optionally weighted by confidence."""
    si = scale_invariant_log_loss(pred_depth, depth_gt, valid_mask)
    grad = gradient_loss(pred_depth, depth_gt, valid_mask)
    loss = si + lambda_grad * grad

    # Laplacian log-likelihood confidence loss
    if pred_conf is not None and valid_mask.any():
        p = pred_depth[valid_mask].clamp(min=1e-3)
        t = depth_gt[valid_mask].clamp(min=1e-3)
        c = pred_conf[valid_mask].clamp(min=1e-6)
        conf_loss = (torch.abs(torch.log(p) - torch.log(t)) / c + torch.log(c)).mean()
        loss = loss + 0.1 * conf_loss

    return loss


# ---------------------------------------------------------------------------
# LoRA configuration helpers
# ---------------------------------------------------------------------------

def get_lora_target_modules(model: nn.Module) -> list[str]:
    """
    Automatically collect the LoRA target module names from the DinoV2 backbone.
    Targets:
      - qkv projections  (attn.qkv)
      - output projections (attn.proj)
      - MLP fc1 / fc2    (mlp.fc1, mlp.fc2)
    Skips modules outside the ViT backbone (DPT head, camera decoders).
    """
    target_keywords = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2",
                       "attn.q", "attn.k", "attn.v"]  # covers both MHA variants
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Only target modules inside the ViT backbone
        if "backbone" not in name and "pretrained" not in name:
            continue
        for kw in target_keywords:
            if kw in name:
                # PEFT expects the *leaf* module name relative to the model
                targets.append(name)
                break
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def build_lora_model(
    base_model: DepthAnything3,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_dora: bool = False,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """
    Wraps the inner DepthAnything3Net with PEFT LoRA.

    Strategy:
      - The outer DepthAnything3 wrapper handles pre/post-processing.
      - We apply LoRA only to the inner `base_model.model` (DepthAnything3Net).
      - We keep the DPT head and camera decoders fully trainable by default so
        that the model can adapt its outputs even with frozen backbone LoRA adapters.
    """
    if target_modules is None:
        target_modules = get_lora_target_modules(base_model.model)

    if not target_modules:
        raise ValueError(
            "No LoRA target modules found.  Check that the model backbone contains "
            "nn.Linear layers matching attn.qkv / attn.proj / mlp.fc1 / mlp.fc2."
        )

    logger.info(f"Applying LoRA to {len(target_modules)} modules.")
    logger.info(f"  rank={lora_rank}, alpha={lora_alpha}, dora={use_dora}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=use_dora,
        # rank-stabilised LoRA (rsLoRA): divides scale by sqrt(r) instead of r
        use_rslora=True,
        target_modules=target_modules,
        # task_type is not strictly required for custom models but keeps PEFT happy
        task_type=TaskType.FEATURE_EXTRACTION,
        # Init LoRA B to zero so the adapter starts as an identity transform
        init_lora_weights="gaussian",
    )

    # Apply PEFT to the *inner* network only
    base_model.model = get_peft_model(base_model.model, lora_config)
    base_model.model.print_trainable_parameters()

    return base_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        images     = batch["image"].to(device)        # (B, 3, H, W)
        depth_gt   = batch["depth_gt"].to(device)     # (B, H, W)
        valid_mask = batch["valid_mask"].to(device)   # (B, H, W)

        # DA3 expects (B, N, 3, H, W) – single-frame mode: N=1
        images_5d = images.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = model.model(images_5d)

            # output["depth"]: (B, N, H, W)  →  take view 0
            pred_depth = output["depth"][:, 0]         # (B, H, W)
            pred_conf  = output.get("depth_conf", None)
            if pred_conf is not None:
                pred_conf = pred_conf[:, 0]             # (B, H, W)

            # Resize prediction to match GT if needed
            if pred_depth.shape[-2:] != depth_gt.shape[-2:]:
                pred_depth = F.interpolate(
                    pred_depth.unsqueeze(1),
                    size=depth_gt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                if pred_conf is not None:
                    pred_conf = F.interpolate(
                        pred_conf.unsqueeze(1),
                        size=depth_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

            loss = depth_loss(pred_depth, pred_conf, depth_gt, valid_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch} | step {step}/{len(loader)} | loss {loss.item():.4f}"
            )

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    abs_rel_sum = 0.0
    delta1_sum  = 0.0
    count       = 0

    for batch in loader:
        images     = batch["image"].to(device).unsqueeze(1)
        depth_gt   = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = model.model(images)

        pred = output["depth"][:, 0]
        if pred.shape[-2:] != depth_gt.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), size=depth_gt.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        pred   = pred[valid_mask].clamp(min=1e-3)
        target = depth_gt[valid_mask].clamp(min=1e-3)

        abs_rel_sum += (torch.abs(pred - target) / target).mean().item()
        thresh = torch.max(pred / target, target / pred)
        delta1_sum += (thresh < 1.25).float().mean().item()
        count += 1

    return {
        "abs_rel": abs_rel_sum / max(count, 1),
        "delta1":  delta1_sum  / max(count, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for Depth Anything 3")

    # Model
    p.add_argument("--model_name",      default="da3-large",
                   choices=["da3-small", "da3-base", "da3-large", "da3-giant",
                             "da3metric-large", "da3mono-large"],
                   help="DA3 model preset.")
    p.add_argument("--pretrained_path", default=None,
                   help="HuggingFace Hub model ID or local path to load weights from.")

    # LoRA
    p.add_argument("--lora_rank",    type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_dora",     action="store_true",
                   help="Use DoRA (Weight-Decomposed LoRA) instead of standard LoRA.")
    p.add_argument("--freeze_head",  action="store_true",
                   help="Freeze the DPT depth head (train only LoRA adapters).")

    # Data
    p.add_argument("--data_root",  required=True, help="Root directory of the dataset.")
    p.add_argument("--img_height", type=int, default=518)
    p.add_argument("--img_width",  type=int, default=518)
    p.add_argument("--max_depth",  type=float, default=80.0)

    # Training
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-2,  help="Weight decay.")
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--num_workers",  type=int, default=4)

    # I/O
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--save_every", type=int, default=5,
                   help="Save checkpoint every N epochs.")
    p.add_argument("--resume",     default=None, help="Path to LoRA checkpoint to resume from.")
    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base model
    # ------------------------------------------------------------------
    if args.pretrained_path:
        logger.info(f"Loading pretrained weights from: {args.pretrained_path}")
        model = DepthAnything3.from_pretrained(args.pretrained_path)
    else:
        logger.info(f"Initialising model from config: {args.model_name}")
        model = DepthAnything3(model_name=args.model_name)

    model = model.to(device)

    # ------------------------------------------------------------------
    # 2. Apply LoRA (or resume adapter)
    # ------------------------------------------------------------------
    if args.resume:
        logger.info(f"Resuming LoRA adapter from: {args.resume}")
        model.model = PeftModel.from_pretrained(model.model, args.resume, is_trainable=True)
    else:
        model = build_lora_model(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_dora=args.use_dora,
        )

    # Optionally freeze the DPT head
    if args.freeze_head:
        for name, param in model.model.named_parameters():
            if "lora_" not in name and "base_layer" not in name:
                if "head" in name or "cam_dec" in name or "cam_enc" in name:
                    param.requires_grad_(False)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)")

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    img_size = (args.img_height, args.img_width)
    train_dataset = DepthDataset(args.data_root, split="train",
                                  img_size=img_size, max_depth=args.max_depth)
    val_dataset   = DepthDataset(args.data_root, split="val",
                                  img_size=img_size, max_depth=args.max_depth)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Optimiser + scheduler
    # ------------------------------------------------------------------
    # Separate LR for LoRA adapters vs. the trainable DPT head
    lora_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_" in n]
    head_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and "lora_" not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr,        "weight_decay": args.wd},
            {"params": head_params, "lr": args.lr * 0.1,  "weight_decay": args.wd},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        # cosine decay to 1% of base lr
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    best_abs_rel = float("inf")
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args.grad_clip
        )
        scheduler.step(global_step + len(train_loader))
        global_step += len(train_loader)

        metrics = validate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"abs_rel={metrics['abs_rel']:.4f} | delta1={metrics['delta1']:.4f}"
        )

        # Save best
        if metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = metrics["abs_rel"]
            save_path = output_dir / "best_lora"
            _save_checkpoint(model, optimizer, scaler, epoch, metrics, save_path, args)
            logger.info(f"  → new best saved to {save_path}")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            save_path = output_dir / f"epoch_{epoch:04d}"
            _save_checkpoint(model, optimizer, scaler, epoch, metrics, save_path, args)

    logger.info("Training complete.")


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    metrics: dict,
    save_path: Path,
    args: argparse.Namespace,
) -> None:
    """Save LoRA adapter weights + training state."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Save only the LoRA adapter (tiny – no full model weights)
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(save_path / "lora_adapter"))

    # Save training state for resumption
    torch.save(
        {
            "epoch":     epoch,
            "metrics":   metrics,
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "args":      vars(args),
        },
        save_path / "training_state.pt",
    )
    logger.info(f"Checkpoint saved → {save_path}")


if __name__ == "__main__":
    main()
