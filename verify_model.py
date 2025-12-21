#!/usr/bin/env python3
"""
Verify FLUX2-dev model completeness
"""

import sys
from pathlib import Path


def detect_flux_version(model_path):
    """Detect which FLUX version the model is."""
    import json

    model_index_path = Path(model_path) / "model_index.json"
    if not model_index_path.exists():
        return None

    try:
        with open(model_index_path, "r") as f:
            model_index = json.load(f)

        class_name = model_index.get("_class_name", "")
        if "Flux2" in class_name:
            return "FLUX2"
        else:
            return "FLUX1"
    except Exception:
        return None


def verify_flux_model(model_path):
    """Verify FLUX model has all required components."""

    path = Path(model_path)

    if not path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False

    if not path.is_dir():
        print(f"❌ Model path is not a directory: {model_path}")
        return False

    # Detect FLUX version
    flux_version = detect_flux_version(model_path)

    if flux_version == "FLUX2":
        print(f"🔍 Checking FLUX2 model at: {model_path}")
        # FLUX2 components (based on the model_index.json shown)
        required_components = {
            "transformer": "Main diffusion model",
            "text_encoder": "Text encoder (Mistral3ForConditionalGeneration)",
            "tokenizer": "Tokenizer (PixtralProcessor)",
            "vae": "Variational Autoencoder",
            "scheduler": "Diffusion scheduler",
        }
    elif flux_version == "FLUX1":
        print(f"🔍 Checking FLUX1 model at: {model_path}")
        # FLUX1 components
        required_components = {
            "transformer": "Main diffusion model",
            "text_encoder": "CLIP text encoder",
            "text_encoder_2": "T5 text encoder",
            "tokenizer": "CLIP tokenizer",
            "tokenizer_2": "T5 tokenizer",
            "vae": "Variational Autoencoder",
            "scheduler": "Diffusion scheduler",
        }
    else:
        print(f"🔍 Checking unknown FLUX model at: {model_path}")
        # Generic FLUX components
        required_components = {
            "transformer": "Main diffusion model",
            "text_encoder": "Text encoder",
            "vae": "Variational Autoencoder",
            "scheduler": "Diffusion scheduler",
        }

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
        if flux_version:
            present.append(f"✅ Detected {flux_version} pipeline")
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
        print(f"🚨 This appears to be an incomplete {flux_version or 'FLUX'} model!")
        return False
    else:
        print(f"🎉 {flux_version or 'FLUX'} model appears complete!")
        return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_model.py /path/to/flux/model")
        sys.exit(1)

    model_path = sys.argv[1]
    success = verify_flux_model(model_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
