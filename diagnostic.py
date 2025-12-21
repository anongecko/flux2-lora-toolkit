#!/usr/bin/env python3
"""
FLUX2 LoRA Training Diagnostic Script
Run this to diagnose issues with your setup before training.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 14):
        print("❌ ERROR: Python 3.14+ required")
        return False
    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        import diffusers

        print(f"✅ Diffusers version: {diffusers.__version__}")
        import transformers

        print(f"✅ Transformers version: {transformers.__version__}")
        from diffusers import FluxPipeline

        print("✅ FluxPipeline import OK")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(
                f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB"
            )
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        return False


def check_model_path(model_path):
    """Check if FLUX model path is valid and complete."""
    if not model_path:
        print("❌ No model path provided")
        return False

    path = Path(model_path)
    if not path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False

    if not path.is_dir():
        print(f"❌ Model path is not a directory: {model_path}")
        return False

    # Check for model_index.json
    if not (path / "model_index.json").exists():
        print("❌ Missing model_index.json")
        return False

    # Detect FLUX version and validate components
    import json

    try:
        with open(path / "model_index.json", "r") as f:
            model_index = json.load(f)

        class_name = model_index.get("_class_name", "")
        if "Flux2" in class_name:
            flux_version = "FLUX2"
            required_components = ["transformer", "text_encoder", "tokenizer", "vae", "scheduler"]
        else:
            flux_version = "FLUX1"
            required_components = [
                "transformer",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "vae",
                "scheduler",
            ]

        missing_components = []
        for component in required_components:
            if not (path / component).exists():
                missing_components.append(component)

        if missing_components:
            print(f"❌ Missing {flux_version} components: {missing_components}")
            return False

        print(f"✅ {flux_version} model path valid: {model_path}")
        print(f"   Found all required components: {required_components}")
        return True

    except json.JSONDecodeError:
        print("❌ Invalid model_index.json")
        return False
    except Exception as e:
        print(f"❌ Error validating model: {e}")
        return False


def main():
    print("🔍 FLUX2 LoRA Training Diagnostic")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        return 1

    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    print()

    # Check CUDA
    cuda_ok = check_cuda()
    print()

    # Check model path if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not check_model_path(model_path):
            return 1
    else:
        print("💡 Tip: Run with model path to check: python diagnostic.py /path/to/flux2-model")

    print()
    print("🎯 Diagnostic complete!")
    if cuda_ok:
        print("✅ Your system appears ready for FLUX2 LoRA training")
    else:
        print("⚠️  CUDA not available - training will be slow on CPU")

    return 0


if __name__ == "__main__":
    sys.exit(main())
