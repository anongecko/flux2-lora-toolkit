"""
Unified logging interface for training monitoring and visualization.

This module provides a comprehensive logging system that integrates TensorBoard
for local visualization and optional Weights & Biases for cloud-based experiment tracking.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

# Optional dependencies
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .metrics import MetricsComputer

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Unified logging interface for TensorBoard and Weights & Biases.

    Features:
    - TensorBoard integration with organized scalar groups
    - Optional Weights & Biases cloud logging
    - System metrics monitoring (GPU memory, utilization)
    - Training metrics visualization
    - Image logging for validation samples
    - Custom scalar groups for better organization

    Scalar Groups:
    - training/loss: Loss curves and related metrics
    - training/learning: Learning rate and optimizer metrics
    - training/gradients: Gradient norms and statistics
    - system/resources: GPU memory, CPU usage, steps/sec
    - validation/metrics: Validation loss and quality metrics
    - validation/samples: Image grids and generation metrics
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enable_quality_metrics: bool = True,
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory for TensorBoard logs
            use_wandb: Whether to enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_entity: W&B entity (team/user)
            experiment_name: Name for this experiment run
            config: Configuration dict to log to W&B
            tags: Tags for W&B run categorization
            enable_quality_metrics: Whether to enable CLIP-based quality metrics
        """
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.experiment_name = experiment_name

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir), comment=experiment_name or "")

        # Initialize Weights & Biases
        self.wandb_run = None
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning(
                    "Weights & Biases requested but not available. Install with: pip install wandb"
                )
            else:
                try:
                    self.wandb_run = wandb.init(
                        project=wandb_project or "flux2-lora-training",
                        entity=wandb_entity,
                        name=experiment_name,
                        config=config,
                        tags=tags,
                        dir=str(self.log_dir),
                    )
                    logger.info(f"Initialized Weights & Biases logging: {wandb_project}")
                except Exception as e:
                    logger.error(f"Failed to initialize Weights & Biases: {e}")
                    self.use_wandb = False

        # Initialize metrics computer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics_computer = MetricsComputer(device=device) if enable_quality_metrics else None

        logger.info(f"TrainingLogger initialized. TensorBoard logs: {self.log_dir}")
        if self.use_wandb:
            logger.info(f"Weights & Biases enabled: {wandb_project}")
        if self.metrics_computer:
            logger.info("Quality metrics enabled with CLIP model")
        else:
            logger.info("Quality metrics disabled")

    def log_scalar(self, name: str, value: float, step: int, group: Optional[str] = None):
        """
        Log a single scalar value.

        Args:
            name: Metric name
            value: Scalar value
            step: Training step
            group: Optional group prefix (e.g., 'training', 'system')
        """
        tag = f"{group}/{name}" if group else name

        # TensorBoard logging
        self.tb_writer.add_scalar(tag, value, step)

        # Weights & Biases logging
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)

    def log_scalars(
        self, group: str, values: Dict[str, float], step: int, main_tag: Optional[str] = None
    ):
        """
        Log multiple related scalars under a group.

        Args:
            group: Group name (e.g., 'training', 'system')
            values: Dictionary of metric names to values
            step: Training step
            main_tag: Optional main tag for grouped scalars
        """
        # TensorBoard grouped scalars
        self.tb_writer.add_scalars(group, values, step)

        # Individual scalars for W&B
        if self.use_wandb and self.wandb_run:
            wandb_values = {f"{group}/{k}": v for k, v in values.items()}
            self.wandb_run.log(wandb_values, step=step)

        # Log main tag if provided
        if main_tag:
            main_value = sum(values.values()) / len(values) if values else 0.0
            self.log_scalar(main_tag, main_value, step, group)

    def log_training_metrics(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        step_time: float,
        memory_usage_mb: float,
        step: int,
    ):
        """
        Log comprehensive training metrics.

        Args:
            loss: Current training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
            step_time: Time per step in seconds
            memory_usage_mb: GPU memory usage in MB
            step: Training step
        """
        # Training metrics
        self.log_scalars("training/loss", {"loss": loss}, step)

        self.log_scalars(
            "training/learning", {"learning_rate": learning_rate, "grad_norm": grad_norm}, step
        )

        # System metrics
        steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
        self.log_scalars(
            "system/resources",
            {
                "step_time": step_time,
                "steps_per_sec": steps_per_sec,
                "memory_usage_mb": memory_usage_mb,
            },
            step,
        )

    def log_validation_metrics(
        self,
        validation_loss: Optional[float] = None,
        images: Optional[List["PIL.Image"]] = None,
        prompts: Optional[List[str]] = None,
        training_images: Optional[List["PIL.Image"]] = None,
        clip_score: Optional[float] = None,
        diversity_score: Optional[float] = None,
        step: int = 0,
    ):
        """
        Log validation metrics, computing quality metrics automatically if possible.

        Args:
            validation_loss: Validation loss (optional)
            images: Generated validation images for automatic metric computation
            prompts: Corresponding prompts for CLIP score computation
            training_images: Training images for overfitting detection
            clip_score: Pre-computed CLIP score (overrides automatic computation)
            diversity_score: Pre-computed diversity score (overrides automatic computation)
            step: Training step
        """
        metrics = {}

        if validation_loss is not None:
            metrics["validation_loss"] = validation_loss

        # Compute metrics automatically if we have the necessary components
        if self.metrics_computer and images and prompts:
            try:
                # CLIP score
                if clip_score is None:
                    clip_score = self.metrics_computer.compute_clip_score(images, prompts)
                metrics["clip_score"] = clip_score

                # Diversity metrics
                if diversity_score is None:
                    diversity_metrics = self.metrics_computer.compute_diversity_score(images)
                    metrics.update(diversity_metrics)
                else:
                    metrics["diversity_score"] = diversity_score

                # Overfitting detection (if training images provided)
                if training_images:
                    overfitting_metrics = self.metrics_computer.detect_overfitting(
                        images, training_images
                    )
                    metrics.update(overfitting_metrics)

                    # Comprehensive quality score
                    quality_metrics = self.metrics_computer.compute_comprehensive_quality_score(
                        "", images, prompts, training_images
                    )
                    metrics["quality_score"] = quality_metrics["quality_score"]

            except Exception as e:
                logger.warning(f"Failed to compute automatic validation metrics: {e}")
        else:
            # Use provided metrics
            if clip_score is not None:
                metrics["clip_score"] = clip_score
            if diversity_score is not None:
                metrics["diversity_score"] = diversity_score

        if metrics:
            self.log_scalars("validation/metrics", metrics, step)

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, "PIL.Image"],
        step: int,
        dataformats: str = "CHW",
    ):
        """
        Log a single image.

        Args:
            name: Image name/tag
            image: Image tensor or PIL Image
            step: Training step
            dataformats: Image format ('CHW', 'HWC', etc.)
        """
        self.tb_writer.add_image(name, image, step, dataformats=dataformats)

        # W&B image logging
        if self.use_wandb and self.wandb_run:
            try:
                # Convert tensor to PIL if needed
                if isinstance(image, torch.Tensor):
                    import torchvision.transforms as transforms

                    to_pil = transforms.ToPILImage()
                    if dataformats == "CHW":
                        pil_image = to_pil(image)
                    else:
                        # Assume HWC
                        pil_image = to_pil(image.permute(2, 0, 1))
                else:
                    pil_image = image

                self.wandb_run.log({name: wandb.Image(pil_image)}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log image to W&B: {e}")

    def log_images(
        self,
        name: str,
        images: List[Union[torch.Tensor, "PIL.Image"]],
        step: int,
        nrow: int = 8,
        dataformats: str = "CHW",
    ):
        """
        Log a grid of images.

        Args:
            name: Grid name/tag
            images: List of images
            step: Training step
            nrow: Number of images per row in grid
            dataformats: Image format
        """
        if not images:
            return

        # Create image grid for TensorBoard
        try:
            import torchvision

            if isinstance(images[0], torch.Tensor):
                # Convert to CHW format for make_grid
                if dataformats == "HWC":
                    tensor_images = [img.permute(2, 0, 1) for img in images]
                else:
                    tensor_images = images

                grid = torchvision.utils.make_grid(tensor_images, nrow=nrow, normalize=True)
                self.tb_writer.add_image(name, grid, step)
            else:
                # PIL Images - log individually with indexed names
                for i, img in enumerate(images):
                    self.log_image(f"{name}_{i}", img, step)
        except ImportError:
            logger.warning("torchvision not available for image grid creation")
        except Exception as e:
            logger.warning(f"Failed to create image grid: {e}")

        # W&B logging
        if self.use_wandb and self.wandb_run:
            try:
                wandb_images = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        import torchvision.transforms as transforms

                        to_pil = transforms.ToPILImage()
                        if dataformats == "CHW":
                            pil_img = to_pil(img)
                        else:
                            pil_img = to_pil(img.permute(2, 0, 1))
                        wandb_images.append(wandb.Image(pil_img))
                    else:
                        wandb_images.append(wandb.Image(img))

                self.wandb_run.log({name: wandb_images}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log images to W&B: {e}")

    def log_hyperparameters(self, config: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None):
        """
        Log hyperparameters and optionally initial metrics.

        Args:
            config: Hyperparameter configuration
            metrics: Initial metric values
        """
        self.tb_writer.add_hparams(config, metrics or {})

        if self.use_wandb and self.wandb_run:
            # W&B config is already logged during init
            pass

    def log_text(self, name: str, text: str, step: int):
        """
        Log text information.

        Args:
            name: Text tag
            text: Text content
            step: Training step
        """
        self.tb_writer.add_text(name, text, step)

        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({name: text}, step=step)

    def flush(self):
        """Flush all writers to ensure data is written."""
        self.tb_writer.flush()

    def close(self):
        """Close all loggers and clean up resources."""
        self.tb_writer.close()

        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()

        logger.info("TrainingLogger closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
        """
        Get current system metrics.

        Returns:
            Dictionary of system metrics
        """
        metrics = {}

        try:
            import psutil

            # CPU metrics
            metrics["cpu_percent"] = psutil.cpu_percent()
            metrics["cpu_memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_id = torch.cuda.current_device()
                metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(gpu_id) / (1024**2)
                metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(gpu_id) / (1024**2)

                # GPU utilization (if available)
                try:
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["gpu_utilization_percent"] = util.gpu
                    pynvml.nvmlShutdown()
                except ImportError:
                    pass
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")

        return metrics

    def log_system_metrics(self, step: int):
        """
        Log current system metrics.

        Args:
            step: Training step
        """
        metrics = self.get_system_metrics()
        if metrics:
            self.log_scalars("system/hardware", metrics, step)
