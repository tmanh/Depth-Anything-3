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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# PEFT – install with: pip install peft>=0.10.0
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer

from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import load_config
from depth_anything_3.utils.logger import logger


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_scalar_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return float(tensor.item())


def setup_distributed() -> tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, local_rank, world_size
    return False, local_rank, world_size


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
                "original_image":   FloatTensor [3, H, W] normalized,
                "underwater_image": FloatTensor [3, H, W] normalized,
                "image_path":       str path to source image
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
        enable_augmentation: bool = True,
        depths_root: str | None = None,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.split = split
        self.enable_augmentation = enable_augmentation
        self.depths_root = Path(depths_root) if depths_root else None

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

        self.base_resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=self.IMG_MEAN, std=self.IMG_STD)

        logger.info(
            f"ColorImageDataset split={split}: {len(self.image_paths)} images from {self.data_root}"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_geometric_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply the same geometric augmentation before creating the underwater pair."""
        # Use a slightly larger crop scale range to improve robustness for scene depth.
        i, j, h, w = T.RandomResizedCrop.get_params(
            image,
            scale=(0.65, 1.0),
            ratio=(0.9, 1.1),
        )
        image = TF.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=self.img_size,
            interpolation=T.InterpolationMode.BILINEAR,
        )

        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.15:
            image = TF.vflip(image)

        if random.random() < 0.25:
            angle = random.uniform(-8.0, 8.0)
            image = TF.rotate(
                image,
                angle,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0,
            )
        return image

    def _apply_photometric_augmentation(self, img: torch.Tensor) -> torch.Tensor:
        """Standard photometric jitter for the original (clean) branch."""
        # img: [3, H, W] in [0, 1]
        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        saturation = random.uniform(0.85, 1.15)
        hue = random.uniform(-0.03, 0.03)

        out = TF.adjust_brightness(img, brightness)
        out = TF.adjust_contrast(out, contrast)
        out = TF.adjust_saturation(out, saturation)
        out = TF.adjust_hue(out, hue)
        return out.clamp(0.0, 1.0)

    def _simulate_underwater(
        self, img: torch.Tensor, depth: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Physically based underwater/turbidity simulation.

        Model:
            I(x) = J(x) * T(x) + A * (1 - T(x))

        where:
            T(x) = exp(-c * d(x))

        c is the beam attenuation coefficient caused by water + suspended particles.
        Higher turbidity means larger scattering/attenuation coefficients.

        Args:
            img: RGB image tensor in [0, 1], shape (3, H, W)
            depth: Optional depth map, shape (H, W)
        """
        c, h, w = img.shape
        if c != 3:
            raise ValueError("Expected 3-channel RGB tensor.")

        device = img.device
        dtype = img.dtype

        # --------------------------------------------------
        # 1. Build depth map d(x)
        # --------------------------------------------------
        if depth is not None:
            if isinstance(depth, torch.Tensor):
                d = depth.to(device=device, dtype=dtype)
            else:
                d = torch.from_numpy(depth).to(device=device, dtype=dtype)

            # Replace invalid values
            valid = torch.isfinite(d) & (d > 0)

            if valid.any():
                d_min = torch.quantile(d[valid], 0.01)
                d_max = torch.quantile(d[valid], 0.99)

                d = torch.clamp(d, d_min, d_max)
                d = d / d_max
            else:
                d = torch.ones((h, w), dtype=dtype, device=device) * 0.5

            # Convert normalized depth to an effective water path length.
            # Keep this moderate to avoid black images.
            max_range = random.uniform(1.0, 4.0)
            d = 0.2 + d * max_range

        else:
            # If no depth is available, use a smooth synthetic depth field.
            yy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1)
            xx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w)

            # Farther toward the top or center depending on random scene layout
            if random.random() < 0.5:
                d = 0.5 + 2.5 * yy
            else:
                center_x = random.uniform(0.3, 0.7)
                center_y = random.uniform(0.3, 0.7)
                d = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
                d = 0.5 + 3.0 * d / (d.max() + 1e-6)

        d = d.clamp(0.1, 5.0)

        # --------------------------------------------------
        # 2. Physical coefficients
        # --------------------------------------------------
        # These are deliberately moderate.
        # Larger values create very dark/over-hazed images.
        #
        # Red is usually attenuated more strongly than green/blue.
        absorption = torch.tensor(
            [
                random.uniform(0.10, 0.35),  # red absorption
                random.uniform(0.04, 0.16),  # green absorption
                random.uniform(0.02, 0.10),  # blue absorption
            ],
            dtype=dtype,
            device=device,
        ).view(3, 1, 1)

        # Suspended-particle scattering coefficient.
        # This is the main turbidity parameter.
        scattering_strength = random.uniform(0.03, 0.22)

        scattering = torch.tensor(
            [
                scattering_strength * random.uniform(0.8, 1.2),
                scattering_strength * random.uniform(0.9, 1.3),
                scattering_strength * random.uniform(1.0, 1.5),
            ],
            dtype=dtype,
            device=device,
        ).view(3, 1, 1)

        beam_attenuation = absorption + scattering

        # --------------------------------------------------
        # 3. Transmission map T(x)
        # --------------------------------------------------
        transmission = torch.exp(-beam_attenuation * d.view(1, h, w))

        # Prevent total collapse to black.
        transmission = transmission.clamp(0.25, 1.0)

        # --------------------------------------------------
        # 4. Water-light / in-scattered ambient light
        # --------------------------------------------------
        water_light = torch.tensor(
            [
                random.uniform(0.03, 0.12),
                random.uniform(0.12, 0.32),
                random.uniform(0.16, 0.42),
            ],
            dtype=dtype,
            device=device,
        ).view(3, 1, 1)

        direct = img * transmission
        inscatter = water_light * (1.0 - transmission)

        underwater = direct + inscatter

        # --------------------------------------------------
        # 5. Forward scattering: depth-dependent blur
        # --------------------------------------------------
        # Forward scattering reduces sharpness, but should be mild.
        if random.random() < 0.6:
            blurred = TF.gaussian_blur(
                underwater,
                kernel_size=3,
                sigma=random.uniform(0.3, 1.0),
            )

            # More blur where transmission is lower.
            blur_weight = (1.0 - transmission.mean(dim=0, keepdim=True)).clamp(0.0, 0.4)
            underwater = underwater * (1.0 - blur_weight) + blurred * blur_weight

        # --------------------------------------------------
        # 6. Mild contrast loss
        # --------------------------------------------------
        contrast = random.uniform(0.88, 1.00)
        underwater = TF.adjust_contrast(underwater.clamp(0.0, 1.0), contrast)

        # --------------------------------------------------
        # 7. Mild camera noise
        # --------------------------------------------------
        noise_std = random.uniform(0.0, 0.006)
        if noise_std > 0:
            underwater = underwater + torch.randn_like(underwater) * noise_std

        return underwater.clamp(0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.enable_augmentation and self.split == "train":
            image = self._apply_geometric_augmentation(image)
        else:
            image = self.base_resize(image)

        original = self.to_tensor(image).clamp(0.0, 1.0)

        if self.enable_augmentation and self.split == "train":
            original = self._apply_photometric_augmentation(original)

        # Try to load precomputed depth if available
        depth = None
        if self.depths_root is not None:
            rel_path = image_path.relative_to(self.data_root)
            depth_path = self.depths_root / rel_path.with_suffix(".npy")
            if depth_path.exists():
                try:
                    depth_np = np.load(str(depth_path)).astype(np.float32)
                    # Resize depth to match image resolution if needed
                    if depth_np.shape != (self.img_size[0], self.img_size[1]):
                        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
                        depth_resized = F.interpolate(
                            depth_t,
                            size=self.img_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        depth = depth_resized.squeeze(0).squeeze(0)
                    else:
                        depth = torch.from_numpy(depth_np)
                except Exception as e:
                    logger.warning(f"Failed to load depth from {depth_path}: {e}")
                    depth = None

        underwater = self._simulate_underwater(original, depth=depth)

        original = self.normalize(original)
        underwater = self.normalize(underwater)

        # --------------------------------------------------
        # DEBUG: save original and simulated underwater images
        # --------------------------------------------------
        if True:
            from torchvision.utils import save_image
            from pathlib import Path

            debug_dir = Path(getattr(self, "debug_save_dir", "debug_underwater_samples"))
            debug_dir.mkdir(parents=True, exist_ok=True)

            stem = image_path.stem

            save_image(
                original.clamp(0.0, 1.0),
                debug_dir / f"{idx:06d}_{stem}_original.png",
            )

            save_image(
                underwater.clamp(0.0, 1.0),
                debug_dir / f"{idx:06d}_{stem}_underwater.png",
            )

            # Optional side-by-side comparison
            comparison = torch.cat(
                [
                    original.clamp(0.0, 1.0),
                    underwater.clamp(0.0, 1.0),
                ],
                dim=2,  # concatenate along width
            )

            save_image(
                comparison,
                debug_dir / f"{idx:06d}_{stem}_comparison.png",
            )
        # --------------------------------------------------

        return {
            "original_image": original,
            "underwater_image": underwater,
            "image_path": str(image_path),
        }


def build_color_image_dataloader(
    data_root: str,
    split: str = "all",
    img_size: tuple[int, int] = (518, 518),
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool | None = None,
    sampler: torch.utils.data.Sampler | None = None,
    train_ratio: float = 0.98,
    seed: int = 42,
    enable_augmentation: bool = True,
    depths_root: str | None = None,
) -> DataLoader:
    """Convenience factory for paired (original, underwater) RGB dataloader."""
    dataset = ColorImageDataset(
        data_root=data_root,
        split=split,
        img_size=img_size,
        train_ratio=train_ratio,
        seed=seed,
        enable_augmentation=enable_augmentation,
        depths_root=depths_root,
    )

    if shuffle is None:
        shuffle = split == "train"
    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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


def depth_consistency_loss(
    pred_underwater: torch.Tensor,
    pred_original_teacher: torch.Tensor,
    conf_underwater: torch.Tensor | None = None,
    conf_original_teacher: torch.Tensor | None = None,
    grad_weight: float = 0.5,
    conf_weight: float = 0.1,
) -> torch.Tensor:
    """
    Self-distillation loss for underwater adaptation.

    The model prediction on clean/original image acts as teacher target,
    and the underwater branch is optimized to match it.
    """
    teacher = pred_original_teacher.detach().clamp(min=1e-3)
    student = pred_underwater.clamp(min=1e-3)

    valid = torch.isfinite(teacher) & torch.isfinite(student) & (teacher > 1e-3)
    if not valid.any():
        return torch.tensor(0.0, device=pred_underwater.device)

    log_diff = torch.abs(torch.log(student[valid]) - torch.log(teacher[valid])).mean()

    grad = gradient_loss(
        student,
        teacher,
        torch.ones_like(student, dtype=torch.bool),
    )
    loss = log_diff + grad_weight * grad

    if conf_underwater is not None:
        conf_s = conf_underwater.clamp(min=1e-6)
        if conf_original_teacher is not None:
            conf_t = conf_original_teacher.detach().clamp(min=1e-6)
            conf_target = conf_t
        else:
            conf_target = torch.ones_like(conf_s)
        conf_l1 = torch.abs(conf_s[valid] - conf_target[valid]).mean()
        loss = loss + conf_weight * conf_l1

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
        # Keep task_type unset for custom vision models so PEFT does not inject
        # NLP-specific kwargs (e.g., input_ids) into forward().
        task_type=None,
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
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    core_model = unwrap_model(model).model

    for step, batch in enumerate(loader):
        images     = batch["image"].to(device)        # (B, 3, H, W)
        depth_gt   = batch["depth_gt"].to(device)     # (B, H, W)
        valid_mask = batch["valid_mask"].to(device)   # (B, H, W)

        # DA3 expects (B, N, 3, H, W) – single-frame mode: N=1
        images_5d = images.unsqueeze(1)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = core_model(images_5d)

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

        raw_loss = loss.detach()
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()

        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch} | step {step}/{len(loader)} | loss {raw_loss.item():.4f}"
            )

    return total_loss / len(loader)


def train_one_epoch_underwater(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    consistency_grad_weight: float = 0.5,
    consistency_conf_weight: float = 0.1,
    grad_accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    core_model = unwrap_model(model).model

    for step, batch in enumerate(loader):
        original = batch["original_image"].to(device)
        underwater = batch["underwater_image"].to(device)

        original_5d = original.unsqueeze(1)
        underwater_5d = underwater.unsqueeze(1)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                teacher_out = core_model(original_5d)
                teacher_depth = teacher_out["depth"][:, 0]
                teacher_conf = teacher_out.get("depth_conf", None)
                if teacher_conf is not None:
                    teacher_conf = teacher_conf[:, 0]
                del teacher_out

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            student_out = core_model(underwater_5d)
            student_depth = student_out["depth"][:, 0]
            student_conf = student_out.get("depth_conf", None)
            if student_conf is not None:
                student_conf = student_conf[:, 0]

            if student_depth.shape[-2:] != teacher_depth.shape[-2:]:
                student_depth = F.interpolate(
                    student_depth.unsqueeze(1),
                    size=teacher_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                if student_conf is not None:
                    student_conf = F.interpolate(
                        student_conf.unsqueeze(1),
                        size=teacher_depth.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

            loss = depth_consistency_loss(
                pred_underwater=student_depth,
                pred_original_teacher=teacher_depth,
                conf_underwater=student_conf,
                conf_original_teacher=teacher_conf,
                grad_weight=consistency_grad_weight,
                conf_weight=consistency_conf_weight,
            )

        raw_loss = loss.detach()
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        if step % 50 == 0:
            logger.info(
                f"Epoch {epoch} | step {step}/{len(loader)} | consistency_loss {raw_loss.item():.4f}"
            )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    core_model = unwrap_model(model).model
    abs_rel_sum = 0.0
    delta1_sum  = 0.0
    count       = 0

    for batch in loader:
        images     = batch["image"].to(device).unsqueeze(1)
        depth_gt   = batch["depth_gt"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output = core_model(images)

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


@torch.no_grad()
def validate_underwater(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    consistency_grad_weight: float = 0.5,
    consistency_conf_weight: float = 0.1,
) -> dict[str, float]:
    model.eval()
    core_model = unwrap_model(model).model
    total_consistency = 0.0
    count = 0

    for batch in loader:
        original = batch["original_image"].to(device)
        underwater = batch["underwater_image"].to(device)

        original_5d = original.unsqueeze(1)
        underwater_5d = underwater.unsqueeze(1)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            teacher_out = core_model(original_5d)
            student_out = core_model(underwater_5d)

        teacher_depth = teacher_out["depth"][:, 0]
        teacher_conf = teacher_out.get("depth_conf", None)
        if teacher_conf is not None:
            teacher_conf = teacher_conf[:, 0]

        student_depth = student_out["depth"][:, 0]
        student_conf = student_out.get("depth_conf", None)
        if student_conf is not None:
            student_conf = student_conf[:, 0]

        if student_depth.shape[-2:] != teacher_depth.shape[-2:]:
            student_depth = F.interpolate(
                student_depth.unsqueeze(1),
                size=teacher_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if student_conf is not None:
                student_conf = F.interpolate(
                    student_conf.unsqueeze(1),
                    size=teacher_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        loss = depth_consistency_loss(
            pred_underwater=student_depth,
            pred_original_teacher=teacher_depth,
            conf_underwater=student_conf,
            conf_original_teacher=teacher_conf,
            grad_weight=consistency_grad_weight,
            conf_weight=consistency_conf_weight,
        )
        total_consistency += loss.item()
        count += 1

    return {"consistency": total_consistency / max(1, count)}


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
    p.add_argument(
        "--training_mode",
        default="supervised",
        choices=["supervised", "underwater_consistency"],
        help="supervised: use RGB+depth labels, underwater_consistency: image-only self-distillation.",
    )
    p.add_argument("--img_height", type=int, default=518)
    p.add_argument("--img_width",  type=int, default=518)
    p.add_argument("--max_depth",  type=float, default=80.0)
    p.add_argument("--train_ratio", type=float, default=0.98,
                   help="Train/val split ratio for image-only mode.")
    p.add_argument("--depths_root", default=None,
                   help="Optional directory with precomputed depth maps (.npy files). "
                        "Must mirror the directory structure of data_root.")
    p.add_argument("--disable_augmentation", action="store_true",
                   help="Disable data augmentation in image-only mode.")
    p.add_argument("--consistency_grad_weight", type=float, default=0.5)
    p.add_argument("--consistency_conf_weight", type=float, default=0.1)

    # Training
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Number of gradient accumulation steps.")
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
    distributed, local_rank, world_size = setup_distributed()
    set_seed(args.seed + get_rank())

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    if is_main_process():
        logger.info(f"Using device: {device} | distributed={distributed} world_size={world_size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base model
    # ------------------------------------------------------------------
    if args.pretrained_path:
        if is_main_process():
            logger.info(f"Loading pretrained weights from: {args.pretrained_path}")
        model = DepthAnything3.from_pretrained(args.pretrained_path)
    else:
        if is_main_process():
            logger.info(f"Initialising model from config: {args.model_name}")
        model = DepthAnything3(model_name=args.model_name)

    model = model.to(device)

    # ------------------------------------------------------------------
    # 2. Apply LoRA (or resume adapter)
    # ------------------------------------------------------------------
    if args.resume:
        if is_main_process():
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

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    if is_main_process():
        logger.info(f"Trainable params: {trainable:,} / {total:,} "
                    f"({100 * trainable / total:.2f}%)")

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    img_size = (args.img_height, args.img_width)
    train_sampler = None
    val_sampler = None
    if args.training_mode == "underwater_consistency":
        train_dataset = ColorImageDataset(
            data_root=args.data_root,
            split="train",
            img_size=img_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
            enable_augmentation=not args.disable_augmentation,
        )
        val_dataset = ColorImageDataset(
            data_root=args.data_root,
            split="val",
            img_size=img_size,
            train_ratio=args.train_ratio,
            seed=args.seed,
            enable_augmentation=False,
        )
        if distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = build_color_image_dataloader(
            data_root=args.data_root,
            split="train",
            img_size=img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
            train_ratio=args.train_ratio,
            seed=args.seed,
            enable_augmentation=not args.disable_augmentation,
            depths_root=args.depths_root,
        )
        val_loader = build_color_image_dataloader(
            data_root=args.data_root,
            split="val",
            img_size=img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_sampler,
            train_ratio=args.train_ratio,
            seed=args.seed,
            enable_augmentation=False,
            depths_root=args.depths_root,
        )
    else:
        train_dataset = DepthDataset(args.data_root, split="train",
                                      img_size=img_size, max_depth=args.max_depth)
        val_dataset   = DepthDataset(args.data_root, split="val",
                                      img_size=img_size, max_depth=args.max_depth)

        if distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
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
    best_score = float("inf")
    global_step  = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        if is_main_process():
            logger.info(f"=== Epoch {epoch}/{args.epochs} ===")

        if args.training_mode == "underwater_consistency":
            train_loss = train_one_epoch_underwater(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch,
                args.grad_clip,
                args.consistency_grad_weight,
                args.consistency_conf_weight,
                args.grad_accum_steps,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, epoch, args.grad_clip, args.grad_accum_steps
            )
        train_loss = reduce_scalar_mean(train_loss, device)
        scheduler.step(global_step + len(train_loader))
        global_step += len(train_loader)

        if args.training_mode == "underwater_consistency":
            metrics = validate_underwater(
                model,
                val_loader,
                device,
                args.consistency_grad_weight,
                args.consistency_conf_weight,
            )
            metrics["consistency"] = reduce_scalar_mean(metrics["consistency"], device)
            if is_main_process():
                logger.info(
                    f"Epoch {epoch} | train_loss={train_loss:.4f} | "
                    f"val_consistency={metrics['consistency']:.4f}"
                )
            current_score = metrics["consistency"]
        else:
            metrics = validate(model, val_loader, device)
            metrics["abs_rel"] = reduce_scalar_mean(metrics["abs_rel"], device)
            metrics["delta1"] = reduce_scalar_mean(metrics["delta1"], device)
            if is_main_process():
                logger.info(
                    f"Epoch {epoch} | train_loss={train_loss:.4f} | "
                    f"abs_rel={metrics['abs_rel']:.4f} | delta1={metrics['delta1']:.4f}"
                )
            current_score = metrics["abs_rel"]

        # Save best
        if is_main_process() and current_score < best_score:
            best_score = current_score
            save_path = output_dir / "best_lora"
            _save_checkpoint(model, optimizer, scaler, epoch, metrics, save_path, args)
            logger.info(f"  → new best saved to {save_path}")

        # Periodic checkpoint
        if is_main_process() and epoch % args.save_every == 0:
            save_path = output_dir / f"epoch_{epoch:04d}"
            _save_checkpoint(model, optimizer, scaler, epoch, metrics, save_path, args)

    if is_main_process():
        logger.info("Training complete.")

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


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

    model_unwrapped = unwrap_model(model)

    # Save only the LoRA adapter (tiny – no full model weights)
    if hasattr(model_unwrapped.model, "save_pretrained"):
        model_unwrapped.model.save_pretrained(str(save_path / "lora_adapter"))

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
