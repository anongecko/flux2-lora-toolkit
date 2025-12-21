"""
Training tab for the Gradio interface.

Provides controls for configuring and running LoRA training jobs.
"""

import os
import tempfile
import zipfile
import shutil
from pathlib import Path
import threading
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp
from .help_utils import help_system

logger = logging.getLogger(__name__)


def handle_dataset_upload(app: "LoRATrainingApp", file_obj) -> tuple[str, Optional[Path]]:
    """
    Handle dataset ZIP file upload.

    Args:
        app: Main application instance
        file_obj: Gradio file object

    Returns:
        Tuple of (status_message, dataset_path)
    """
    if not file_obj:
        return "No file uploaded", None

    try:
        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="flux2_lora_dataset_"))
        zip_path = Path(file_obj.name)

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Register the uploaded file
        file_id = app.register_uploaded_file(temp_dir, "dataset")

        # Validate dataset structure
        validation_result = validate_dataset_structure(temp_dir)

        if validation_result["valid"]:
            status = f"✅ Dataset uploaded successfully: {validation_result['image_count']} images, {validation_result['caption_count']} captions"
            app.update_training_state("dataset_path", str(temp_dir))
            app.update_training_state("dataset_info", validation_result)
            return status, temp_dir
        else:
            # Clean up invalid dataset
            shutil.rmtree(temp_dir, ignore_errors=True)
            return f"❌ Invalid dataset: {validation_result['error']}", None

    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        return f"❌ Upload failed: {str(e)}", None


def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
    """
    Validate dataset structure and count files.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Validation results dictionary
    """
    try:
        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = []
        caption_files = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
                elif file_path.suffix.lower() in {".txt", ".caption"} or file_path.name.endswith(
                    ".caption"
                ):
                    caption_files.append(file_path)

        if not image_files:
            return {"valid": False, "error": "No image files found in dataset"}

        # Check for captions (should have at least some)
        caption_ratio = len(caption_files) / len(image_files) if image_files else 0

        result = {
            "valid": True,
            "image_count": len(image_files),
            "caption_count": len(caption_files),
            "caption_ratio": caption_ratio,
            "image_extensions": list(set(f.suffix for f in image_files)),
        }

        # Warning for low caption ratio
        if caption_ratio < 0.5:
            result["warning"] = ".1f"

        return result

    except Exception as e:
        return {"valid": False, "error": f"Validation failed: {str(e)}"}


def handle_model_path(model_path: str) -> str:
    """
    Validate FLUX model path and return concise status.

    Args:
        model_path: Path to model directory

    Returns:
        Concise status message string
    """
    if not model_path or not model_path.strip():
        return "Enter model path above"

    try:
        from pathlib import Path

        path = Path(model_path.strip())

        if not path.exists():
            return "❌ Model path does not exist"

        if not path.is_dir():
            return "❌ Path is not a directory"

        # Check for model_index.json
        if not (path / "model_index.json").exists():
            return "❌ Missing full model"

        # Detect FLUX version and validate components
        import json

        try:
            with open(path / "model_index.json", "r") as f:
                model_index = json.load(f)

            class_name = model_index.get("_class_name", "")
            if "Flux2" in class_name:
                flux_version = "FLUX2"
                required_components = [
                    "transformer",
                    "text_encoder",
                    "tokenizer",
                    "vae",
                    "scheduler",
                ]
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
                return "❌ Missing full model"
            else:
                return f"✅ {flux_version} Model Present"

        except json.JSONDecodeError:
            return "❌ Missing full model"
        except Exception as e:
            return "❌ Missing full model"

    except Exception as e:
        return "❌ Missing full model"


def handle_dataset_path(app: "LoRATrainingApp", dataset_path: str) -> tuple[str, Optional[Path]]:
    """
    Handle local dataset directory path.

    Args:
        app: Main application instance
        dataset_path: Path to dataset directory

    Returns:
        Tuple of (status_message, dataset_path)
    """
    if not dataset_path or not dataset_path.strip():
        return "No dataset path provided", None

    try:
        path = Path(dataset_path.strip())

        if not path.exists():
            return f"❌ Dataset path does not exist: {dataset_path}", None

        if not path.is_dir():
            return f"❌ Path is not a directory: {dataset_path}", None

        # Validate dataset structure
        validation_result = validate_dataset_structure(path)

        if validation_result["valid"]:
            status = f"✅ Dataset loaded: {validation_result['image_count']} images, {validation_result['caption_count']} captions"
            app.update_training_state("dataset_path", str(path))
            app.update_training_state("dataset_info", validation_result)
            return status, path
        else:
            return f"❌ Invalid dataset: {validation_result['error']}", None

    except Exception as e:
        logger.error(f"Dataset path validation failed: {e}")
        return f"❌ Path validation failed: {str(e)}", None


def start_training_background(
    app: "LoRATrainingApp", config: Dict[str, Any], progress_callback: callable = None
) -> Dict[str, Any]:
    """
    Start training in background thread.

    Args:
        app: Main application instance
        config: Training configuration
        progress_callback: Callback for progress updates

    Returns:
        Training results
    """
    try:
        # Import training components
        from ..core.model_loader import ModelLoader
        from ..core.trainer import LoRATrainer
        from ..data.dataset import LoRADataset, create_dataloader
        from ..utils.config_manager import TrainingConfig

        # Debug: Log config type and contents
        logger.info(f"Config type: {type(config)}")
        if isinstance(config, dict):
            logger.info(f"Config keys: {list(config.keys())}")
            logger.info(f"Config base_model: {config.get('base_model', 'NOT SET')}")
        else:
            logger.info(f"Config has model attr: {hasattr(config, 'model')}")
            if hasattr(config, "model"):
                logger.info(
                    f"Config.model.base_model: {getattr(config.model, 'base_model', 'NOT SET')}"
                )

        # Update app state
        app.reset_training_state()
        app.update_training_state("is_training", True)
        app.update_training_state("status_message", "Initializing training...")

        # Create training config
        from ..utils.config_manager import Config

        training_config = Config()

        # Handle both dict and Config object inputs
        if isinstance(config, dict):
            try:
                # Update config with UI parameters from dict
                training_config.data.dataset_path = config.get("dataset_path", "")
                training_config.model.base_model = config.get(
                    "base_model", "/path/to/black-forest-labs/FLUX.2-dev"
                )
                training_config.model.device = config.get("device", "auto")
                training_config.lora.rank = config.get("rank", 16)
                training_config.lora.alpha = config.get("alpha", 16)
                training_config.training.max_steps = config.get("max_steps", 1000)
                training_config.training.batch_size = config.get("batch_size", 4)
                training_config.training.learning_rate = config.get("learning_rate", 1e-4)
            except AttributeError as e:
                logger.error(f"Failed to set config attributes: {e}")
                raise ValueError(f"Config object missing expected attributes: {e}")
        elif hasattr(config, "model") and hasattr(config, "data"):
            # If config is already a Config object, use it directly
            logger.warning("Config object passed instead of dict, using directly")
            training_config = config
        else:
            raise ValueError(
                f"Invalid config type: {type(config)}. Expected dict or Config object."
            )

        # Set additional config values (handle both dict and Config object cases)
        if isinstance(config, dict):
            training_config.output.output_dir = config.get("output_dir", "./output")
            training_config.output.checkpoint_every_n_steps = config.get("checkpoint_every", 100)
        # If config is a Config object, these values should already be set
        training_config.logging.tensorboard = config.get("tensorboard", True)
        training_config.validation.every_n_steps = config.get("validation_every", 50)

        if progress_callback:
            progress_callback(0.1, "Loading model...")

        # Load model
        model_loader = ModelLoader()
        model, _ = model_loader.load_flux2_dev(
            model_name=training_config.model.base_model,
            device=training_config.model.device,
        )

        if progress_callback:
            progress_callback(0.3, "Loading dataset...")

        # Load dataset
        # Map caption_format to caption_sources list
        caption_sources = (
            [training_config.data.caption_format]
            if training_config.data.caption_format != "auto"
            else ["txt", "caption", "json", "exif"]
        )

        train_dataset = LoRADataset(
            data_dir=training_config.data.dataset_path,
            resolution=training_config.data.resolution,
            caption_sources=caption_sources,
            cache_images=training_config.data.cache_images,
            validate_captions=training_config.data.validate_captions,
        )

        # Create dataloader
        train_dataloader = create_dataloader(
            dataset=train_dataset,
            batch_size=training_config.training.batch_size,
            num_workers=training_config.data.num_workers,
            pin_memory=training_config.data.pin_memory,
            shuffle=True,
        )

        if progress_callback:
            progress_callback(0.5, "Initializing trainer...")

        # Initialize trainer
        trainer = LoRATrainer(
            model=model,
            config=training_config,
            output_dir=training_config.output.output_dir,
        )

        # Start training
        app.update_training_state("total_steps", training_config.training.max_steps)

        if progress_callback:
            progress_callback(0.7, "Starting training...")

        # Training loop with progress updates
        training_results = trainer.train_with_progress_callback(
            train_dataloader=train_dataloader,
            num_steps=training_config.training.max_steps,
            progress_callback=lambda step, loss, metrics: update_training_progress(
                app, step, loss, metrics, progress_callback
            ),
        )

        app.update_training_state("is_training", False)
        app.update_training_state("status_message", "Training completed successfully")

        return {"success": True, "results": training_results}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        app.update_training_state("is_training", False)
        app.update_training_state("status_message", f"Training failed: {str(e)}")
        return {"success": False, "error": str(e)}


def update_training_progress(
    app: "LoRATrainingApp",
    step: int,
    loss: float,
    metrics: Dict[str, Any],
    progress_callback: callable = None,
):
    """
    Update training progress in app state.

    Args:
        app: Main application instance
        step: Current training step
        loss: Current loss value
        metrics: Training metrics
        progress_callback: UI progress callback
    """
    # Update app state
    app.update_training_state("current_step", step)
    app.update_training_state("progress", step / app.get_training_state("total_steps", 1))

    # Update loss history
    loss_history = app.get_training_state("loss_history", [])
    loss_history.append(loss)
    if len(loss_history) > 100:  # Keep last 100 values
        loss_history = loss_history[-100:]
    app.update_training_state("loss_history", loss_history)

    # Update validation samples if available
    if "validation_samples" in metrics:
        app.update_training_state("validation_samples", metrics["validation_samples"])

    # Update status
    total_steps = app.get_training_state("total_steps", 1)
    progress_pct = int((step / total_steps) * 100)
    app.update_training_state(
        "status_message", f"Training step {step}/{total_steps} ({progress_pct}%) - Loss: {loss:.6f}"
    )

    # Call UI progress callback
    if progress_callback:
        progress_callback(step / total_steps, f"Step {step}/{total_steps} - Loss: {loss:.6f}")


def create_training_tab(app: "LoRATrainingApp"):
    """
    Create the training tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Training Help & Tips", open=False):
        gr.Markdown(help_system.get_training_help_text())

        # Additional feature explanations
        with gr.Accordion("🎯 Understanding LoRA Training", open=False):
            gr.Markdown(help_system.get_feature_overview()["lora_training"])

        with gr.Accordion("👁️ Validation Sampling Explained", open=False):
            gr.Markdown(help_system.get_feature_overview()["validation_sampling"])

        with gr.Accordion("⚙️ Training Presets Guide", open=False):
            gr.Markdown(help_system.get_feature_overview()["preset_system"])

        with gr.Accordion("📊 Monitoring Training Progress", open=False):
            gr.Markdown("""
            ### How to Monitor Your Training

            #### Loss Plot
            - **What it shows**: How well your LoRA is learning from the data
            - **What to look for**: Steady downward trend (good learning)
            - **Problems to watch**: Flat line (not learning), erratic jumps (unstable)
            - **Normal range**: Starts high (2.0-5.0), decreases to 0.1-0.5

            #### Validation Samples
            - **What they show**: Preview images of what your LoRA can generate
            - **Frequency**: Every 50-100 steps by default
            - **What to check**: Do images increasingly match your subject/style?
            - **Red flags**: No visible improvement, random/unrelated content

            #### Progress Indicators
            - **Step counter**: Current training iteration
            - **Time remaining**: Estimated completion time
            - **GPU memory**: Monitor for out-of-memory errors
            - **Status messages**: Current training phase and any issues

            #### When to Stop Training
            - **Good signs**: Loss stabilized, validation samples look correct
            - **Overfitting signs**: Validation samples become too similar to training images
            - **Early stopping**: Loss increases or validation quality decreases
            - **Recommended**: Test checkpoints every 100-200 steps
            """)

        with gr.Accordion("🚨 Troubleshooting Common Issues", open=False):
            gr.Markdown("""
            ### Training Problems & Solutions

            #### "CUDA out of memory" Error
            **Symptoms**: Training crashes with GPU memory error
            **Solutions**:
            - Reduce batch size (try 2 or 4)
            - Lower LoRA rank (try 16 instead of 32)
            - Close other GPU applications
            - Enable gradient checkpointing (automatic)
            - Reduce image resolution if needed

            #### Loss Not Decreasing
            **Symptoms**: Loss stays high or decreases very slowly
            **Solutions**:
            - Check dataset quality (blurry/low-res images?)
            - Verify captions match images accurately
            - Increase learning rate slightly (1e-4 → 2e-4)
            - Ensure adequate dataset size (10+ images minimum)
            - Check for corrupted or mismatched image/caption pairs

            #### Validation Samples Look Wrong
            **Symptoms**: Generated images don't match your subject
            **Solutions**:
            - Review captions (are they descriptive enough?)
            - Check image quality and consistency
            - Ensure trigger words are used in captions
            - Verify dataset represents your subject well
            - Consider changing LoRA type (character vs style vs concept)

            #### Training is Too Slow
            **Symptoms**: Training takes much longer than expected
            **Solutions**:
            - Check GPU utilization (should be near 100%)
            - Reduce batch size for faster iterations
            - Enable torch.compile if available
            - Use faster GPU if multiple are available
            - Consider smaller dataset for testing

            #### Poor Final Results
            **Symptoms**: LoRA works but results aren't great
            **Solutions**:
            - Add more diverse training images
            - Improve caption quality and detail
            - Train for more steps (2000+ instead of 1000)
            - Try different LoRA rank or learning rate
            - Review and refine your dataset
            """)

    with gr.Row():
        with gr.Column(scale=1):
            # Left column: Configuration
            with gr.Row():
                gr.Markdown("## ⚙️ Training Configuration")
                gr.Button("❓", size="sm", elem_classes=["help-button"]).click(
                    fn=lambda: None,  # Would open help modal
                    inputs=[],
                    outputs=[],
                )

            # Quick tips banner
            with gr.Group(elem_classes=["tips-banner"]):
                gr.Markdown("""
                **💡 Quick Tips:**
                • Upload ZIP or specify local dataset path
                • Choose preset based on your subject type
                • Monitor loss and validation samples
                • Stop when results look good (avoid overfitting)
                """)

            with gr.Group(elem_classes=["training-config"]):
                gr.Markdown("### 🤖 Model Configuration")

                # Base model selection
                base_model = gr.Textbox(
                    label="Base Model",
                    value="/path/to/black-forest-labs/FLUX.2-dev",
                    info="Local path to downloaded FLUX model directory (e.g., '/home/user/models/black-forest-labs/FLUX.2-dev')",
                )

                # Model status indicator
                model_status = gr.Textbox(
                    value="Enter model path above to check status",
                    label="Model Status",
                    interactive=False,
                    lines=1,
                )

                # Device selection
                device = gr.Dropdown(
                    label="Device",
                    choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
                    value="auto",
                    info="Device to run the model on (auto = GPU if available)",
                )

                gr.Markdown("""
                **📝 Model Notes:**
                - **Required**: Download FLUX2-dev locally from HuggingFace (black-forest-labs/FLUX.2-dev) and specify the local path
                - **Authentication**: Model requires HuggingFace authentication - download manually first
                - **GPU Required**: Training requires high-end GPU (A100 or H100 recommended for LoRA)
                """)

                gr.Markdown("### Dataset")

                # Dataset guidance
                with gr.Accordion("📋 Dataset Preparation Guide", open=False):
                    gr.Markdown("""
                    ### Creating Your Training Dataset

                    #### File Structure
                    ```
                    my_dataset/
                    ├── image_001.jpg
                    ├── image_001.txt    # "Detailed caption describing image_001.jpg"
                    ├── image_002.jpg
                    ├── image_002.txt    # "Detailed caption describing image_002.jpg"
                    └── ...
                    ```

                    #### Caption Writing Tips
                    - **Be Specific**: "A young woman with long brown hair, wearing a blue dress, standing in a garden"
                    - **Include Details**: Mention colors, poses, settings, lighting, expressions
                    - **Use Trigger Words**: Include words you'll use to activate your LoRA
                    - **Be Consistent**: Use similar vocabulary across all captions
                    - **Length**: Aim for 10-20 words per caption

                    #### Image Quality Guidelines
                    - **Resolution**: Minimum 1024x1024 pixels (higher is better)
                    - **Consistency**: Similar style, lighting, and quality across all images
                    - **Subject Focus**: Your main subject should be clearly visible
                    - **Format**: JPG, PNG, or WebP (RGB color space)

                    #### Quantity Recommendations
                    - **Minimum**: 10-15 images for basic results
                    - **Good**: 30-50 images for reliable training
                    - **Excellent**: 100+ images for complex subjects

                    #### Common Mistakes to Avoid
                    ❌ Images without captions
                    ❌ Blurry or low-quality photos
                    ❌ Inconsistent art styles
                    ❌ Subject not clearly visible
                    ❌ Too few images (< 10)
                    ❌ Captions that don't match images
                    """)

                # Dataset selection
                dataset_source = gr.Radio(
                    choices=["Upload ZIP", "Local Directory"],
                    value="Upload ZIP",
                    label="Dataset Source",
                    interactive=True,
                )

                dataset_upload = gr.File(
                    label="Upload Dataset (ZIP)",
                    file_types=[".zip"],
                    visible=True,
                )

                dataset_dir = gr.Textbox(
                    label="Dataset Directory Path",
                    placeholder="/path/to/dataset",
                    visible=False,
                )

                # Dataset validation status
                dataset_status = gr.Textbox(
                    label="Dataset Status",
                    value="No dataset selected",
                    interactive=False,
                )

                gr.Markdown("### LoRA Configuration")

                # Configuration controls
                with gr.Row():
                    undo_btn = gr.Button(
                        "↶ Undo",
                        size="sm",
                        interactive=False,
                    )
                    redo_btn = gr.Button("↷ Redo", size="sm", interactive=False)
                    reset_btn = gr.Button(
                        "🔄 Reset to Defaults",
                        size="sm",
                        variant="secondary",
                    )

                # Real-time validation status
                validation_status = gr.HTML(
                    value="<div style='padding: 10px; background: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; color: #2e7d32;'><strong>✓ Configuration Valid</strong></div>",
                    elem_id="config-validation-status",
                )

                # Preset selection with examples
                try:
                    from ..utils.config_manager import config_manager

                    available_presets = config_manager.list_presets()
                    preset_choices = (
                        [p.capitalize() for p in available_presets]
                        if available_presets
                        else ["Character", "Style", "Concept"]
                    )
                except Exception:
                    preset_choices = ["Character", "Style", "Concept"]

                # Get smart defaults
                smart_defaults = app.get_smart_defaults("training")
                default_preset = smart_defaults.get(
                    "preset", preset_choices[0] if preset_choices else "Character"
                )

                preset = gr.Dropdown(
                    choices=preset_choices,
                    label="Training Preset",
                    value=default_preset,
                )

                # Example configurations
                with gr.Accordion("📚 Example Setups", open=False):
                    gr.Markdown("""
                    ### Character LoRA Examples

                    **Anime Character:**
                    - Images: 20-50 character art pieces
                    - Captions: "anime style portrait of [character_name], detailed face, expressive eyes"
                    - Trigger: Character's name or unique identifier

                    **Real Person:**
                    - Images: 30-100 photos from different angles/poses
                    - Captions: "Photo of [person_name], [pose_description], [setting]"
                    - Trigger: Person's name

                    **Creature/Fantasy:**
                    - Images: 15-40 illustrations of the creature
                    - Captions: "Fantasy art of [creature_name], [pose], detailed scales/fur/feathers"
                    - Trigger: Creature type or specific name

                    ### Style LoRA Examples

                    **Art Style (Van Gogh):**
                    - Images: 25-50 paintings in target style
                    - Captions: "Oil painting in the style of Van Gogh, [subject], swirling brushstrokes"
                    - Trigger: "in the style of Van Gogh"

                    **Digital Art:**
                    - Images: 20-40 digital artworks
                    - Captions: "Digital art of [subject], vibrant colors, detailed shading"
                    - Trigger: "digital art style"

                    **Photography Style:**
                    - Images: 30-60 photos in consistent style
                    - Captions: "Professional photograph of [subject], [lighting], [composition]"
                    - Trigger: "professional photography"

                    ### Concept LoRA Examples

                    **Object (Vintage Cars):**
                    - Images: 20-40 photos of vintage cars
                    - Captions: "Vintage car, [model/year], [color], detailed chrome, classic design"
                    - Trigger: "vintage car"

                    **Scene (Mountain Landscapes):**
                    - Images: 25-50 mountain landscape photos
                    - Captions: "Mountain landscape, [time_of_day], dramatic peaks, [weather]"
                    - Trigger: "mountain landscape"

                    **Abstract Concept:**
                    - Images: 15-35 examples of the concept
                    - Captions: "Abstract representation of [concept], [visual_elements]"
                    - Trigger: Concept name
                    """)

                # Advanced settings (progressive disclosure based on experience level)
                exp_level = app.workflow_state.get("user_experience_level", "beginner")
                advanced_open = exp_level in ["intermediate", "advanced"]

                with gr.Accordion("🔧 Advanced Settings", open=advanced_open):
                    rank = gr.Slider(
                        minimum=4,
                        maximum=128,
                        value=smart_defaults.get("rank", 16),
                        step=4,
                        label="LoRA Rank",
                    )

                    alpha = gr.Slider(
                        minimum=1,
                        maximum=64,
                        value=smart_defaults.get("alpha", 16),
                        label="LoRA Alpha",
                    )

                    learning_rate = gr.Number(
                        value=smart_defaults.get("learning_rate", 1e-4),
                        label="Learning Rate",
                        minimum=1e-6,
                        maximum=1e-2,
                        step=1e-5,
                        info="Learning rate for training (default: 0.0001)",
                    )

                    max_steps = gr.Number(
                        value=smart_defaults.get("max_steps", 1000),
                        label="Max Training Steps",
                        minimum=100,
                        maximum=10000,
                    )

                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=smart_defaults.get("batch_size", 4),
                        label="Batch Size",
                    )

                gr.Markdown("### Training Controls")

                # Training buttons
                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg",
                    )
                    stop_btn = gr.Button(
                        "⏹️ Stop Training",
                        variant="stop",
                        interactive=False,
                    )
                    pause_btn = gr.Button(
                        "⏸️ Pause/Resume",
                        variant="secondary",
                        interactive=False,
                    )

                # Status display with guidance
                training_status = gr.Textbox(
                    label="Training Status",
                    value="Ready to train",
                    interactive=False,
                    lines=3,
                )

                # Training guidance based on status
                training_guidance = gr.Markdown(
                    value="""
                    **🎯 Ready to Train!**
                    Upload your dataset and click "Start Training" to begin.
                    Training typically takes 1-4 hours depending on your settings.
                    """,
                    visible=True,
                )

        with gr.Column(scale=2):
            # Right column: Progress and monitoring
            gr.Markdown("## 📊 Training Progress")

            with gr.Group(elem_classes=["progress-section"]):
                # Progress bar
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Training Progress (%)",
                    interactive=False,
                )

                # Current step info
                step_info = gr.Textbox(
                    label="Current Step",
                    value="0 / 0",
                    interactive=False,
                )

                # Loss plot
                loss_plot = gr.LinePlot(
                    label="Training Loss",
                    x="Step",
                    y="Loss",
                    title="Training Loss Over Time",
                )

                # Validation samples
                gr.Markdown("### Validation Samples")
                validation_gallery = gr.Gallery(
                    label="Generated Samples",
                    columns=3,
                    height="auto",
                    allow_preview=True,
                )

                # Log output
                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    autoscroll=True,
                )

    # State variables for UI updates
    training_active = gr.State(False)
    current_config = gr.State({})

    # Event handlers
    def update_dataset_visibility(source):
        """Update dataset input visibility based on source selection."""
        if source == "Upload ZIP":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    dataset_source.change(
        fn=update_dataset_visibility, inputs=[dataset_source], outputs=[dataset_upload, dataset_dir]
    )

    # Dataset upload handler
    def handle_upload_wrapper(file_obj):
        """Wrapper for dataset upload handling."""
        if file_obj:
            status, path = handle_dataset_upload(app, file_obj)
            return status, str(path) if path else ""
        return "No file uploaded", ""

    dataset_upload.change(
        fn=handle_upload_wrapper, inputs=[dataset_upload], outputs=[dataset_status, dataset_dir]
    )

    # Dataset path handler
    def handle_path_wrapper(path):
        """Wrapper for dataset path handling."""
        if path and path.strip():
            status, dataset_path = handle_dataset_path(app, path.strip())
            return status
        return "No dataset path provided"

    dataset_dir.change(fn=handle_path_wrapper, inputs=[dataset_dir], outputs=[dataset_status])

    # Model path handler
    def handle_model_path_wrapper(path):
        """Wrapper for model path handling."""
        if path and path.strip():
            status = handle_model_path(path.strip())
            return status
        return "No model path provided"

    base_model.change(fn=handle_model_path_wrapper, inputs=[base_model], outputs=[model_status])

    # Configuration validation functions
    def validate_training_config(
        preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_path
    ):
        """Validate training configuration in real-time."""
        config = {
            "preset": preset,
            "rank": rank,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "dataset_path": dataset_path,
        }

        validation = app.validate_configuration(config, "training")

        if validation["is_valid"]:
            if validation["warnings"]:
                status_html = f"""
                <div style='padding: 10px; background: #fff3e0; border: 1px solid #ff9800; border-radius: 5px; color: #e65100;'>
                    <strong>⚠️ Configuration Valid with Warnings</strong><br>
                    {"<br>".join(f"• {w}" for w in validation["warnings"])}
                </div>
                """
            else:
                status_html = """
                <div style='padding: 10px; background: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; color: #2e7d32;'>
                    <strong>✓ Configuration Valid</strong>
                </div>
                """
        else:
            status_html = f"""
            <div style='padding: 10px; background: #ffebee; border: 1px solid #f44336; border-radius: 5px; color: #c62828;'>
                <strong>❌ Configuration Invalid</strong><br>
                {"<br>".join(f"• {e}" for e in validation["errors"])}
            </div>
            """

        return status_html

    # Configuration management handlers
    def undo_config():
        """Undo the last configuration change."""
        prev_config = app.undo_config_change()
        if prev_config:
            return {
                preset: prev_config.get("preset", "Character"),
                rank: prev_config.get("rank", 16),
                alpha: prev_config.get("alpha", 16),
                learning_rate: prev_config.get("learning_rate", 1e-4),
                max_steps: prev_config.get("max_steps", 1000),
                batch_size: prev_config.get("batch_size", 4),
            }
        return {}

    def redo_config():
        """Redo a configuration change."""
        next_config = app.redo_config_change()
        if next_config:
            return {
                preset: next_config.get("preset", "Character"),
                rank: next_config.get("rank", 16),
                alpha: next_config.get("alpha", 16),
                learning_rate: next_config.get("learning_rate", 1e-4),
                max_steps: next_config.get("max_steps", 1000),
                batch_size: next_config.get("batch_size", 4),
            }
        return {}

    def reset_to_defaults():
        """Reset configuration to smart defaults."""
        defaults = app.get_smart_defaults("training")
        app.add_notification("Configuration reset to smart defaults", "info")
        return {
            preset: defaults.get("preset", "Character"),
            rank: defaults.get("rank", 16),
            alpha: defaults.get("alpha", 16),
            learning_rate: defaults.get("learning_rate", 1e-4),
            max_steps: defaults.get("max_steps", 1000),
            batch_size: defaults.get("batch_size", 4),
        }

    # Undo/Redo/Reset button handlers
    undo_btn.click(
        fn=undo_config, outputs=[preset, rank, alpha, learning_rate, max_steps, batch_size]
    )

    redo_btn.click(
        fn=redo_config, outputs=[preset, rank, alpha, learning_rate, max_steps, batch_size]
    )

    reset_btn.click(
        fn=reset_to_defaults, outputs=[preset, rank, alpha, learning_rate, max_steps, batch_size]
    )

    # Parameter validation triggers
    def validate_and_save_config(preset_val, rank_val, alpha_val, lr_val, steps_val, batch_val):
        """Validate config and save snapshot for undo/redo."""
        config = {
            "preset": preset_val,
            "rank": rank_val,
            "alpha": alpha_val,
            "learning_rate": lr_val,
            "max_steps": steps_val,
            "batch_size": batch_val,
            "dataset_path": app.get_training_state("dataset_path", ""),
        }

        # Save config snapshot
        app.save_config_snapshot(config, "training")

        # Enable undo/redo buttons
        can_undo = app.current_config_index > 0
        can_redo = app.current_config_index < len(app.config_history) - 1

        # Validate and return status
        validation_html = validate_training_config(
            preset_val, rank_val, alpha_val, lr_val, steps_val, batch_val, config["dataset_path"]
        )

        return validation_html, gr.update(interactive=can_undo), gr.update(interactive=can_redo)

    preset.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    rank.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    alpha.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    learning_rate.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    max_steps.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    batch_size.change(
        fn=validate_and_save_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size],
        outputs=[validation_status, undo_btn, redo_btn],
    )

    rank.change(
        fn=validate_training_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_dir],
        outputs=[validation_status],
    )

    alpha.change(
        fn=validate_training_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_dir],
        outputs=[validation_status],
    )

    learning_rate.change(
        fn=validate_training_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_dir],
        outputs=[validation_status],
    )

    max_steps.change(
        fn=validate_training_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_dir],
        outputs=[validation_status],
    )

    batch_size.change(
        fn=validate_training_config,
        inputs=[preset, rank, alpha, learning_rate, max_steps, batch_size, dataset_dir],
        outputs=[validation_status],
    )

    # Start training handler
    def start_training_handler(
        app,
        base_model,
        device,
        preset,
        rank,
        alpha,
        learning_rate,
        max_steps,
        batch_size,
        training_active,
        current_config,
    ):
        """Handle training start with validation."""
        try:
            if training_active:
                return "Training is already running", training_active, current_config

            # Get dataset path
            dataset_path = app.get_training_state("dataset_path")
            if not dataset_path:
                return (
                    "❌ No dataset selected. Please upload or specify a dataset path.",
                    training_active,
                    current_config,
                )

            # Validate model path
            if not base_model or not Path(base_model).exists():
                return (
                    f"❌ Invalid model path: '{base_model}'. Please specify a valid path to the downloaded FLUX2-dev model directory.",
                    training_active,
                    current_config,
                )

            # Validate that it's a directory with FLUX model files
            if not Path(base_model).is_dir():
                return (
                    f"❌ Model path must be a directory: '{base_model}'. Please specify the path to the FLUX2-dev model folder.",
                    training_active,
                    current_config,
                )

            # Check for basic FLUX model structure
            if not (Path(base_model) / "model_index.json").exists():
                return (
                    f"❌ Model directory missing model_index.json: '{base_model}'. "
                    "Please ensure you have downloaded the complete black-forest-labs/FLUX.2-dev model.",
                    training_active,
                    current_config,
                )

            # Validate based on detected FLUX version
            import json

            try:
                with open(Path(base_model) / "model_index.json", "r") as f:
                    model_index = json.load(f)

                class_name = model_index.get("_class_name", "")
                if "Flux2" in class_name:
                    # FLUX2-dev: single text encoder/tokenizer
                    required_components = [
                        "transformer",
                        "text_encoder",
                        "tokenizer",
                        "vae",
                        "scheduler",
                    ]
                else:
                    # FLUX1: dual text encoder/tokenizer
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
                    if not (Path(base_model) / component).exists():
                        missing_components.append(component)

                if missing_components:
                    flux_version = "FLUX2-dev" if "Flux2" in class_name else "FLUX1"
                    return (
                        f"❌ {flux_version} model incomplete: missing {missing_components}",
                        training_active,
                        current_config,
                    )
            except Exception:
                # If we can't read model_index.json, do basic validation
                pass

            # Validate device
            if device not in ["auto", "cpu"] and not device.startswith("cuda"):
                return (
                    f"❌ Invalid device: '{device}'. Use 'auto', 'cpu', or 'cuda:X' where X is GPU ID.",
                    training_active,
                    current_config,
                )

            # Validate and sanitize inputs
            try:
                learning_rate_val = float(learning_rate)
                if learning_rate_val <= 0:
                    learning_rate_val = 1e-4  # Default learning rate
            except (ValueError, TypeError):
                learning_rate_val = 1e-4  # Default learning rate

            # Ensure learning rate is within valid bounds
            learning_rate_val = max(1e-6, min(1e-2, learning_rate_val))

            # Build training config
            config = {
                "dataset_path": dataset_path,
                "base_model": base_model,
                "device": device,
                "preset": preset.lower(),
                "rank": int(rank),
                "alpha": int(alpha),
                "learning_rate": learning_rate_val,
                "max_steps": int(max_steps),
                "batch_size": int(batch_size),
                "output_dir": "./output",
                "checkpoint_every": 100,
                "validation_every": 50,
                "tensorboard": True,
            }

            # Auto-save configuration
            if app.user_prefs.get("auto_save_configs", True):
                app.update_user_preference("last_training_config", config)
                app.update_user_preference("last_dataset_path", dataset_path)
                app.update_user_preference("preferred_preset", preset)
                app.update_user_preference("default_batch_size", int(batch_size))
                app.update_user_preference("default_steps", int(max_steps))
                app.add_notification("Configuration auto-saved", "success")

            # Add to operation queue
            operation_id = len(app.operation_queue)
            app.add_operation(f"Training LoRA ({preset})", "running", 0.0)

            # Start training in background thread
            def progress_callback(progress, message):
                """Update progress in UI."""
                app.update_training_state("progress", progress)
                app.update_training_state("status_message", message)
                app.update_operation_progress(operation_id, progress * 100)

            def training_thread():
                """Run training in background thread."""
                try:
                    result = start_training_background(app, config, progress_callback)
                    if result["success"]:
                        app.update_training_state(
                            "status_message", "Training completed successfully!"
                        )
                        app.update_operation_progress(operation_id, 100.0, "completed")
                        app.add_notification("Training completed successfully!", "success")
                        app.update_workflow_step("train")
                        app.workflow_state["training_completed"] = True
                    else:
                        # Enhanced error handling with analysis and suggestions
                        error_analysis = app.analyze_error_and_suggest_fixes(
                            Exception(result["error"]), "training"
                        )

                        error_msg = f"Training failed: {result['error']}"
                        if error_analysis["recovery_actions"]:
                            error_msg += "\n\n💡 Suggested fixes:\n" + "\n".join(
                                f"• {action}" for action in error_analysis["recovery_actions"][:3]
                            )

                        app.update_training_state("status_message", error_msg)
                        app.update_operation_progress(operation_id, 0.0, "failed")

                        # Add detailed error notification
                        app.add_notification(
                            f"Training failed: {error_analysis['error_message'][:100]}...",
                            "error",
                            10000,  # Longer duration for errors
                        )

                        # Add troubleshooting notification
                        if error_analysis["troubleshooting_steps"]:
                            troubleshooting_msg = (
                                "🔧 Quick fix: " + error_analysis["troubleshooting_steps"][0]
                            )
                            app.add_notification(troubleshooting_msg, "warning", 8000)
                except Exception as e:
                    logger.error(f"Training thread error: {e}")
                    app.update_training_state("status_message", f"Training error: {str(e)}")
                    app.update_operation_progress(operation_id, 0.0, "failed")
                    app.add_notification(f"Training error: {str(e)}", "error")
                finally:
                    app.update_training_state("is_training", False)

                # Save config snapshot for undo/redo
                app.save_config_snapshot(config, "training")

            # Start training thread
            thread = threading.Thread(target=training_thread, daemon=True)
            thread.start()

            app.add_notification("Training started in background", "info")
            return "🚀 Training started! Check progress below.", True, config

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            error_msg = f"❌ Failed to start training: {str(e)}"
            return error_msg, training_active, current_config

    def start_training_wrapper(
        base_model_val,
        device_val,
        preset_val,
        rank_val,
        alpha_val,
        learning_rate_val,
        max_steps_val,
        batch_size_val,
        training_active_val,
        current_config_val,
    ):
        """Wrapper function for Gradio compatibility."""
        return start_training_handler(
            app,
            base_model_val,
            device_val,
            preset_val,
            rank_val,
            alpha_val,
            learning_rate_val,
            max_steps_val,
            batch_size_val,
            training_active_val,
            current_config_val,
        )

    start_btn.click(
        fn=start_training_wrapper,
        inputs=[
            base_model,
            device,
            preset,
            rank,
            alpha,
            learning_rate,
            max_steps,
            batch_size,
            training_active,
            current_config,
        ],
        outputs=[training_status, training_active, current_config],
    )

    def stop_training_handler(app, training_active):
        """Handle training stop."""
        if not training_active:
            return "No training is currently running", training_active

        # Set stop flag (trainer will check this)
        app.update_training_state("should_stop", True)
        app.update_training_state("status_message", "Stopping training...")

        return "⏹️ Stopping training...", training_active

    stop_btn.click(
        fn=lambda active: stop_training_handler(app, active),
        inputs=[training_active],
        outputs=[training_status, training_active],
    )

    # Update UI based on training state
    def update_ui_components():
        """Update UI components based on current training state."""
        is_training = app.get_training_state("is_training", False)
        progress = app.get_training_state("progress", 0.0)
        current_step = app.get_training_state("current_step", 0)
        total_steps = app.get_training_state("total_steps", 0)
        status_message = app.get_training_state("status_message", "Ready to train")
        loss_history = app.get_training_state("loss_history", [])
        validation_samples = app.get_training_state("validation_samples", [])

        # Prepare loss plot data
        import pandas as pd

        if loss_history:
            loss_data = pd.DataFrame(
                {"Step": list(range(1, len(loss_history) + 1)), "Loss": loss_history}
            )
        else:
            loss_data = pd.DataFrame(columns=["Step", "Loss"])

        # Generate contextual guidance based on training state
        if not is_training and current_step == 0:
            guidance_text = """
            **🎯 Ready to Train!**
            Upload your dataset and click "Start Training" to begin.
            Training typically takes 1-4 hours depending on your settings.
            """
        elif is_training and current_step < total_steps * 0.1:
            guidance_text = """
            **🚀 Training Started!**
            Monitor the loss plot for steady downward trend.
            First validation samples will appear soon.
            """
        elif is_training and current_step < total_steps * 0.5:
            guidance_text = """
            **📈 Training in Progress**
            Loss should be decreasing steadily. Check validation samples
            for early signs of learning. This is normal training behavior.
            """
        elif is_training and current_step >= total_steps * 0.5:
            guidance_text = """
            **🎯 Mid-to-Late Training**
            Watch for overfitting - validation samples should still look natural.
            Consider stopping if quality peaks or starts declining.
            """
        elif not is_training and current_step > 0:
            guidance_text = """
            **✅ Training Completed!**
            Check the evaluation tab to test your trained LoRA.
            Compare multiple checkpoints to find the best one.
            """
        else:
            guidance_text = """
            **⏸️ Training Paused**
            Resume when ready or stop to save current progress.
            """

        return (
            progress * 100,  # Progress bar (0-100)
            f"{current_step} / {total_steps}",
            loss_data,
            validation_samples,
            status_message,
            guidance_text,
            gr.update(interactive=not is_training),  # Start button
            gr.update(interactive=is_training),  # Stop button
            gr.update(interactive=is_training),  # Pause button
        )

    # Set up periodic UI updates
    timer = gr.Timer(1.0)  # Update every second
    timer.tick(
        fn=update_ui_components,
        outputs=[
            progress_bar,
            step_info,
            loss_plot,
            validation_gallery,
            training_status,
            training_guidance,
            start_btn,
            stop_btn,
            pause_btn,
        ],
    )
