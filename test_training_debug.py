#!/usr/bin/env python3
"""Quick test to reproduce the training error with full traceback."""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux2_lora.data.dataset import LoRADataset, create_dataloader

# Find the dataset that was used
dataset_paths = [
    Path("./examples/sample_dataset"),
    Path("./dataset"),
    Path("./data"),
    Path("../dataset"),
    Path("../data"),
]

dataset_path = None
for p in dataset_paths:
    if p.exists() and p.is_dir():
        dataset_path = p
        break

if dataset_path is None:
    print("ERROR: Could not find dataset directory")
    print("Tried:", dataset_paths)
    sys.exit(1)

print(f"Found dataset: {dataset_path}")

# Create dataset
try:
    dataset = LoRADataset(
        data_dir=dataset_path,
        resolution=1024,
        caption_sources=["txt", "caption", "json", "exif"],
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
except Exception as e:
    print(f"ERROR creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create dataloader
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=1,  # Use batch_size=1 for small dataset
    num_workers=0,
    pin_memory=True,
    shuffle=True,
    drop_last=False,  # Don't drop last incomplete batch
)
print(f"✓ DataLoader created")

# Try to get one batch
print("\nTrying to fetch first batch...")
try:
    data_iter = iter(dataloader)
    batch = next(data_iter)
    print(f"✓ Got batch!")
    print(f"  Batch keys: {batch.keys()}")
    print(f"  batch['images'] type: {type(batch['images'])}")
    print(f"  batch['images'] shape: {batch['images'].shape}")
    print(f"  batch['images'] dtype: {batch['images'].dtype}")
    print(f"  batch['captions'] length: {len(batch['captions'])}")
    print(f"  First caption: {batch['captions'][0][:100]}...")

    # Try moving to device
    print("\nTrying to move batch to CUDA...")
    images = batch["images"].to("cuda:0", non_blocking=True)
    print(f"✓ Successfully moved to CUDA")
    print(f"  Shape: {images.shape}")
    print(f"  Device: {images.device}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    print("\nFULL TRACEBACK:")
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
