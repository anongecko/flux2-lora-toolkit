#!/usr/bin/env python3
"""
Simple script to export a single LoRA safetensors file for distribution.

Usage:
    python export_lora.py                     # Export best checkpoint
    python export_lora.py step_00001800       # Export specific checkpoint
    python export_lora.py --checkpoint-dir ./output/checkpoints --output my_model.safetensors
"""

import argparse
import shutil
from pathlib import Path


def export_lora(checkpoint_dir: Path, checkpoint_name: str, output_path: Path):
    """Extract just the LoRA adapter file from a checkpoint."""

    # Find checkpoint directory
    if checkpoint_name == "best":
        best_link = checkpoint_dir / "best"
        if best_link.exists() and best_link.is_symlink():
            checkpoint_path = checkpoint_dir / best_link.readlink()
        else:
            raise FileNotFoundError("No 'best' checkpoint found")
    else:
        checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Find adapter file
    adapter_file = checkpoint_path / "adapter_model.safetensors"

    if not adapter_file.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors found in {checkpoint_path}")

    # Copy to output
    shutil.copy2(adapter_file, output_path)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"✓ Exported LoRA successfully!")
    print(f"  From: {checkpoint_path.name}")
    print(f"  To: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print()
    print("This file is ready to upload to HuggingFace or use for inference!")


def main():
    parser = argparse.ArgumentParser(
        description="Export a single LoRA safetensors file for distribution"
    )
    parser.add_argument(
        "checkpoint_name",
        nargs="?",
        default="best",
        help="Checkpoint name (e.g., 'step_00001500', 'best'). Default: best"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./output/checkpoints"),
        help="Checkpoint directory path. Default: ./output/checkpoints"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output/exported_lora.safetensors"),
        help="Output file path. Default: ./output/exported_lora.safetensors"
    )

    args = parser.parse_args()

    try:
        export_lora(args.checkpoint_dir, args.checkpoint_name, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
