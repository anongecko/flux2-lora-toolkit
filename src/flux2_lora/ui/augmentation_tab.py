"""
Augmentation tab for the Gradio interface.

Provides dataset augmentation capabilities for generating additional training samples.
"""

import os
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp

from .help_utils import help_system


def create_augmentation_tab(app: "LoRATrainingApp"):
    """
    Create the augmentation tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Dataset Augmentation Help & Tips", open=False):
        gr.Markdown(help_system.get_augmentation_help_text())

    with gr.Row():
        with gr.Column(scale=1):
            # Left column: Augmentation setup
            gr.Markdown("## 🎨 Dataset Augmentation")

            with gr.Group():
                gr.Markdown("### Input Dataset")

                # Dataset selection
                aug_dataset_source = gr.Radio(
                    choices=["Upload ZIP", "Local Directory"],
                    value="Upload ZIP",
                    label="Dataset Source",
                )

                aug_dataset_upload = gr.File(
                    label="Upload Dataset (ZIP)",
                    file_types=[".zip"],
                    visible=True,
                )

                aug_dataset_path = gr.Textbox(
                    label="Dataset Directory Path",
                    placeholder="/path/to/dataset",
                    visible=False,
                )

                aug_dataset_status = gr.Textbox(
                    label="Dataset Status",
                    value="No dataset selected",
                    interactive=False,
                )

                gr.Markdown("### Augmentation Settings")

                # Core settings
                aug_enabled = gr.Checkbox(
                    value=True,
                    label="Enable Augmentation",
                )

                aug_samples = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Number of Augmented Samples",
                )

                aug_preserve_originals = gr.Checkbox(
                    value=True,
                    label="Include Original Samples",
                )

                gr.Markdown("### Image Augmentations")

                # Image augmentation controls
                aug_image_flip = gr.Checkbox(
                    value=True,
                    label="Horizontal Flip",
                )

                aug_image_brightness = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.1,
                    step=0.05,
                    label="Brightness Variation",
                )

                aug_image_contrast = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.1,
                    step=0.05,
                    label="Contrast Variation",
                )

                gr.Markdown("### Text Augmentations")

                # Text augmentation controls
                aug_text_synonyms = gr.Checkbox(
                    value=True,
                    label="Synonym Replacement",
                )

                aug_text_deletion = gr.Checkbox(
                    value=False,
                    label="Random Word Deletion",
                )

                gr.Markdown("### Quality Controls")

                aug_preserve_quality = gr.Checkbox(
                    value=True,
                    label="Preserve Image Quality",
                )

                aug_max_per_sample = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Max Augmentations per Sample",
                )

                gr.Markdown("### Output Settings")

                aug_output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./augmented_dataset",
                )

                # Control buttons
                aug_generate_btn = gr.Button(
                    "🚀 Generate Augmented Dataset",
                    variant="primary",
                    size="lg",
                )

                aug_status = gr.Textbox(
                    label="Augmentation Status",
                    value="Ready to augment",
                    interactive=False,
                    lines=3,
                )

        with gr.Column(scale=2):
            # Right column: Results and preview
            gr.Markdown("## 📊 Augmentation Results")

            with gr.Group():
                # Statistics
                aug_stats = gr.JSON(
                    label="Augmentation Statistics",
                    value={},
                )

                # Preview gallery
                gr.Markdown("### Sample Preview")
                aug_preview_gallery = gr.Gallery(
                    label="Augmented Samples Preview",
                    columns=4,
                    height=300,
                    allow_preview=True,
                )

                # Comparison view
                gr.Markdown("### Before/After Comparison")
                with gr.Row():
                    aug_original_image = gr.Image(
                        label="Original",
                        height=200,
                    )
                    aug_augmented_image = gr.Image(
                        label="Augmented",
                        height=200,
                    )

                with gr.Row():
                    aug_original_caption = gr.Textbox(
                        label="Original Caption",
                        lines=2,
                        interactive=False,
                    )
                    aug_augmented_caption = gr.Textbox(
                        label="Augmented Caption",
                        lines=2,
                        interactive=False,
                    )

            # Download section
            gr.Markdown("### 📥 Download Results")
            aug_download_btn = gr.Button(
                "📦 Download Augmented Dataset (ZIP)",
                variant="secondary",
            )

            aug_download_status = gr.Textbox(
                label="Download Status",
                value="",
                interactive=False,
            )

    # State variables
    aug_active = gr.State(False)
    aug_results = gr.State({})
    aug_dataset_path = gr.State(None)
    aug_output_path = gr.State(None)

    # Event handlers
    def aug_update_dataset_visibility(source):
        """Update dataset input visibility."""
        if source == "Upload ZIP":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    aug_dataset_source.change(
        fn=aug_update_dataset_visibility,
        inputs=[aug_dataset_source],
        outputs=[aug_dataset_upload, aug_dataset_path],
    )

    # Dataset upload handler
    def aug_handle_dataset_upload(app, file_obj):
        """Handle dataset upload for augmentation."""
        if file_obj:
            from .training_tab import handle_dataset_upload

            status, path = handle_dataset_upload(app, file_obj)
            return status
        return "No file uploaded"

    aug_dataset_upload.change(
        fn=lambda file_obj: aug_handle_dataset_upload(app, file_obj),
        inputs=[aug_dataset_upload],
        outputs=[aug_dataset_status],
    )

    # Dataset path handler
    def aug_handle_dataset_path(app, path):
        """Handle dataset path input."""
        if path and path.strip():
            from .training_tab import handle_dataset_path

            status, dataset_path = handle_dataset_path(app, path.strip())
            return status
        return "No dataset path provided"

    aug_dataset_path.change(
        fn=lambda path: aug_handle_dataset_path(app, path),
        inputs=[aug_dataset_path],
        outputs=[aug_dataset_status],
    )

    # Generate augmentation handler
    def aug_generate_handler(
        app,
        dataset_path,
        enabled,
        samples,
        preserve_originals,
        image_flip,
        brightness,
        contrast,
        text_synonyms,
        text_deletion,
        preserve_quality,
        max_per_sample,
        output_dir,
        aug_active,
        aug_results,
    ):
        """Handle augmentation generation."""
        if aug_active:
            return "Augmentation is already running", aug_active, aug_results

        if not dataset_path:
            return (
                "❌ No dataset selected. Please upload or specify a dataset path.",
                aug_active,
                aug_results,
            )

        try:
            # Validate output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Start augmentation in background
            def augmentation_thread():
                """Run augmentation in background thread."""
                try:
                    from flux2_lora.data.augmentation import DatasetAugmenter, AugmentationConfig
                    from flux2_lora.data.dataset import LoRADataset
                    import shutil
                    from PIL import Image

                    app.update_training_state(
                        "aug_status", f"🚀 Starting augmentation of {samples} samples..."
                    )

                    # Load original dataset
                    original_dataset = LoRADataset(dataset_path)
                    app.update_training_state(
                        "aug_status", f"✅ Loaded {len(original_dataset)} original samples"
                    )

                    # Copy originals if requested
                    if preserve_originals:
                        app.update_training_state("aug_status", "📋 Copying original samples...")
                        for file_path in Path(dataset_path).iterdir():
                            if file_path.is_file():
                                shutil.copy2(file_path, output_path / file_path.name)

                    # Configure augmentations
                    aug_config = AugmentationConfig(
                        enabled=enabled,
                        probability=1.0,  # Always augment for generation
                        preserve_quality=preserve_quality,
                        max_augmentations_per_sample=max_per_sample,
                        image_augmentations={
                            "geometric": {"horizontal_flip": {"enabled": image_flip}},
                            "color": {
                                "brightness": {"enabled": brightness > 0, "limit": brightness},
                                "contrast": {"enabled": contrast > 0, "limit": contrast},
                            },
                        },
                        text_augmentations={
                            "synonym_replacement": {
                                "enabled": text_synonyms,
                                "probability": 0.2,
                                "max_replacements": 2,
                            },
                            "random_deletion": {"enabled": text_deletion, "probability": 0.1},
                        },
                    )

                    augmenter = DatasetAugmenter(aug_config)

                    # Generate samples
                    generated_samples = []
                    for i in range(samples):
                        # Sample random original
                        idx = i % len(original_dataset)  # Cycle through dataset
                        sample = original_dataset[idx]

                        # Load and augment
                        image = Image.open(sample["image_path"])
                        augmented_image, augmented_caption = augmenter.augment_sample(
                            image, sample["caption"]
                        )

                        # Save augmented sample
                        base_name = Path(sample["image_path"]).stem
                        aug_name = "08d"

                        # Save image
                        img_path = output_path / f"{aug_name}.jpg"
                        augmented_image.save(img_path, "JPEG", quality=95)

                        # Save caption
                        caption_path = output_path / f"{aug_name}.txt"
                        with open(caption_path, "w", encoding="utf-8") as f:
                            f.write(augmented_caption)

                        generated_samples.append(
                            {
                                "image_path": str(img_path),
                                "caption": augmented_caption,
                                "original_caption": sample["caption"],
                            }
                        )

                        # Update progress
                        if (i + 1) % 10 == 0:
                            app.update_training_state(
                                "aug_status", f"🎨 Generated {i + 1}/{samples} augmented samples..."
                            )

                    # Store results
                    results = {
                        "original_samples": len(original_dataset),
                        "generated_samples": len(generated_samples),
                        "total_samples": len(original_dataset) + len(generated_samples)
                        if preserve_originals
                        else len(generated_samples),
                        "output_directory": str(output_path),
                        "augmentation_config": aug_config.__dict__,
                        "samples": generated_samples[:5],  # Store first 5 for preview
                    }

                    app.update_training_state("aug_results", results)
                    app.update_training_state("aug_output_path", str(output_path))
                    app.update_training_state(
                        "aug_status",
                        f"✅ Augmentation complete! Generated {len(generated_samples)} samples.",
                    )

                except Exception as e:
                    app.update_training_state("aug_status", f"❌ Augmentation failed: {str(e)}")
                finally:
                    app.update_training_state("aug_active", False)

            # Start thread
            thread = threading.Thread(target=augmentation_thread, daemon=True)
            thread.start()

            return f"🚀 Starting augmentation of {samples} samples...", True, {}

        except Exception as e:
            return f"❌ Setup failed: {e}", aug_active, aug_results

    aug_generate_btn.click(
        fn=lambda *args: aug_generate_handler(app, *args),
        inputs=[
            aug_dataset_path,
            aug_enabled,
            aug_samples,
            aug_preserve_originals,
            aug_image_flip,
            aug_image_brightness,
            aug_image_contrast,
            aug_text_synonyms,
            aug_text_deletion,
            aug_preserve_quality,
            aug_max_per_sample,
            aug_output_dir,
            aug_active,
            aug_results,
        ],
        outputs=[aug_status, aug_active, aug_results],
    )

    # Download handler
    def aug_download_handler(aug_output_path):
        """Handle dataset download."""
        if aug_output_path and Path(aug_output_path).exists():
            # Create ZIP file
            import zipfile

            zip_path = Path(aug_output_path) / "augmented_dataset.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file_path in Path(aug_output_path).iterdir():
                    if file_path.is_file() and file_path.name != "augmented_dataset.zip":
                        zipf.write(file_path, file_path.name)

            return str(zip_path)
        return None

    aug_download_btn.click(
        fn=aug_download_handler,
        inputs=[aug_output_path],
        outputs=[aug_download_status],
    )

    # Update UI periodically
    def aug_update_ui(app):
        """Update augmentation UI components."""
        aug_active = app.get_training_state("aug_active", False)
        status = app.get_training_state("aug_status", "Ready to augment")
        results = app.get_training_state("aug_results", {})

        # Prepare preview images and captions
        preview_images = []
        original_image = None
        augmented_image = None
        original_caption = ""
        augmented_caption = ""

        if results and "samples" in results and results["samples"]:
            # Show preview gallery
            for sample in results["samples"][:8]:  # Show first 8
                preview_images.append(sample["image_path"])

            # Show before/after comparison for first sample
            if len(results["samples"]) > 0:
                sample = results["samples"][0]
                try:
                    augmented_image = sample["image_path"]
                    augmented_caption = sample["caption"]
                    original_caption = sample.get("original_caption", "")

                    # For original image, we'd need to find the original file
                    # This is simplified for the demo
                except Exception:
                    pass

        return (
            results,  # Stats
            preview_images,  # Preview gallery
            original_image,  # Original image
            augmented_image,  # Augmented image
            original_caption,  # Original caption
            augmented_caption,  # Augmented caption
            gr.update(interactive=not aug_active),  # Generate button
            gr.update(visible=bool(results)),  # Download button
        )

    # Set up periodic UI updates
    aug_timer = gr.Timer(2.0)  # Update every 2 seconds
    aug_timer.tick(
        fn=lambda: aug_update_ui(app),
        outputs=[
            aug_stats,
            aug_preview_gallery,
            aug_original_image,
            aug_augmented_image,
            aug_original_caption,
            aug_augmented_caption,
            aug_generate_btn,
            aug_download_btn,
        ],
    )
