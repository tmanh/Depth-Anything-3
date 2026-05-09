"""
Quick test script to verify DA3 model inference API.
Run this to ensure the model can be called correctly.
"""
import sys
import torch
from PIL import Image
from pathlib import Path

# Try to import from our train.py imports
try:
    from torchvision import transforms as T
except ImportError:
    # Fallback: create simple transforms manually
    T = None

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

# Load a sample image
sample_images = list(Path("../dataset").rglob("*.jpg"))[:1]
if not sample_images:
    sample_images = list(Path("../dataset").rglob("*.png"))[:1]

if not sample_images:
    print("ERROR: No images found in ../dataset")
    exit(1)

print(f"Testing with image: {sample_images[0]}")

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create transforms - skip if torchvision not available
if T is not None:
    img_transform = T.Compose([
        T.Resize((518, 518), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    print("WARNING: torchvision not available, using simple tensor conversion")
    import numpy as np
    def img_transform(img):
        # Simple resize and normalize
        img = img.resize((518, 518), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_t - mean) / std

# Load model
print("Loading DA3 model...")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
model = model.to(device)
model.eval()

# Load image
img = Image.open(sample_images[0]).convert("RGB")
img_tensor = img_transform(img).unsqueeze(0)  # (1, 3, H, W)

print(f"Image tensor shape: {img_tensor.shape}")

# Try different calling conventions
with torch.no_grad():
    print("\n--- Testing model.model(batch_5d) ---")
    try:
        batch_5d = img_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
        print(f"Input shape: {batch_5d.shape}")
        outputs = model.model(batch_5d)
        print(f"Output type: {type(outputs)}")
        if outputs is not None:
            if isinstance(outputs, dict):
                print(f"Output keys: {list(outputs.keys())}")
                if "depth" in outputs:
                    depth = outputs["depth"]
                    print(f"Depth shape: {depth.shape}")
                    print("✓ model.model() works!")
            else:
                print(f"Output is {type(outputs)}: {outputs}")
        else:
            print("✗ model.model() returned None")
    except Exception as e:
        print(f"✗ model.model() failed: {e}")

    print("\n--- Testing model(batch_5d) (wrapper) ---")
    try:
        batch_5d = img_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
        print(f"Input shape: {batch_5d.shape}")
        outputs = model(batch_5d)
        print(f"Output type: {type(outputs)}")
        if outputs is not None:
            if isinstance(outputs, dict):
                print(f"Output keys: {list(outputs.keys())}")
            else:
                print(f"Output is {type(outputs)}: {outputs}")
        else:
            print("✗ model() returned None")
    except Exception as e:
        print(f"✗ model() failed: {e}")

print("\nDone!")
