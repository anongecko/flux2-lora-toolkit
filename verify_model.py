#!/usr/bin/env python3
"""
Verify FLUX2-dev model completeness
"""

import sys
from pathlib import Path


def verify_flux2_model(model_path):
    """Verify FLUX2-dev model has all required components."""

    path = Path(model_path)

    if not path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False

    if not path.is_dir():
        print(f"❌ Model path is not a directory: {model_path}")
        return False

    # FLUX2-dev specific components (different from FLUX1)
    required_components = {
        "transformer": "Main diffusion model",
        "text_encoder": "CLIP text encoder",
        "text_encoder_2": "T5 text encoder (FLUX2-dev specific)",
        "tokenizer": "CLIP tokenizer",
        "tokenizer_2": "T5 tokenizer (FLUX2-dev specific)",
        "vae": "Variational Autoencoder",
        "scheduler": "Diffusion scheduler",
    }

    print(f"🔍 Checking FLUX2-dev model at: {model_path}")
    print()

    missing = []
    present = []

    for component, description in required_components.items():
        component_path = path / component
        if component_path.exists():
            present.append(f"✅ {component} - {description}")
        else:
            missing.append(f"❌ {component} - {description}")

    # Check model_index.json
    if (path / "model_index.json").exists():
        present.append("✅ model_index.json - Pipeline configuration")
    else:
        missing.append("❌ model_index.json - Pipeline configuration")

    print("Present components:")
    for item in present:
        print(f"  {item}")

    print()
    if missing:
        print("Missing components:")
        for item in missing:
            print(f"  {item}")

        print()
        print("🚨 This appears to be an incomplete FLUX2-dev model!")
        print("   FLUX2-dev requires text_encoder_2 and tokenizer_2, which FLUX1 doesn't have.")
        print("   Please download black-forest-labs/FLUX.2-dev from HuggingFace.")
        return False
    else:
        print("🎉 FLUX2-dev model appears complete!")
        return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_model.py /path/to/flux2/model")
        sys.exit(1)

    model_path = sys.argv[1]
    success = verify_flux2_model(model_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
