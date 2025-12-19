"""
Gradio web interface for Flux2-dev LoRA Training Toolkit.

This module provides a user-friendly web interface for training, monitoring,
and evaluating LoRA models using Gradio.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from flux2_lora.ui.gradio_app import LoRATrainingApp


def main():
    """Launch the Gradio web interface."""
    app = LoRATrainingApp()

    # Launch the app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public access
        show_error=True,
    )


if __name__ == "__main__":
    main()
