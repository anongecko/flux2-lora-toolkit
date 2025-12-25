"""
Main training loop for Flux2-dev LoRA training.

This module provides the core training orchestrator with proper error handling,
monitoring, and checkpoint management.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .optimizer import OptimizerManager
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.config_manager import Config
from ..utils.hardware_utils import hardware_manager
from ..monitoring import TrainingLogger
from ..monitoring.metrics import MetricsComputer
from ..monitoring.validator import ValidationSampler, create_validation_function
from ..monitoring.callbacks import (
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    ValidationCallback,
    LRSchedulerCallback,
)

logger = logging.getLogger(__name__)
console = Console()


class LoRATrainer:
    """
    Main LoRA training orchestrator for Flux2-dev.

    Features:
    - Mixed precision training with gradient scaling
    - Gradient clipping and accumulation
    - Automatic checkpointing and resume capability
    - Real-time training metrics and progress tracking
    - Memory optimization and error recovery
    - Configurable validation and monitoring
    """

    def __init__(
        self, model, config: Config, output_dir: Union[str, Path], device: Optional[str] = None
    ):
        """
        Initialize LoRA trainer.

        Args:
            model: Flux2-dev model with LoRA adapters
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
            device: Target device ('auto', 'cpu', 'cuda', 'cuda:X', or None for auto-detect)
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)

        # Setup device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                gpu_id = hardware_manager.select_best_gpu()
                self.device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # DON'T move model to device - the model loader already handles device placement
        # with CPU offloading if needed. Moving it here would break CPU offloading by
        # trying to move all components (including text_encoder) to GPU, causing OOM.
        # The model is already on the correct device(s) from model_loader.load_flux2_dev()
        console.print(f"[dim]Trainer using device: {self.device} (model already placed by loader)[/dim]")

        # Initialize components
        self.optimizer_manager = None
        self.checkpoint_manager = None
        self.scaler = None
        self.logger = None
        self.metrics_computer = None
        self.validation_sampler = None
        self.validation_fn = None
        self.callback_manager = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.training_start_time = None
        self.step_times = []
        self.loss_history = []
        self.should_stop = False
        self._progress_callback = None

        # Metrics tracking
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "step_time": [],
            "memory_usage": [],
        }

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        console.print(f"[bold green]LoRA Trainer Initialized[/bold green]")
        console.print(f"  Device: {self.device}")
        console.print(f"  Output: {self.output_dir}")
        console.print(f"  Mixed precision: {self.config.training.mixed_precision}")

    def _set_model_mode(self, training: bool = True):
        """
        Set model to training or eval mode.

        Args:
            training: True for training mode, False for eval mode
        """
        # Handle Flux pipeline vs regular model
        if hasattr(self.model, 'transformer'):
            # Flux pipeline - set mode on transformer
            if training:
                self.model.transformer.train()
            else:
                self.model.transformer.eval()
        elif hasattr(self.model, 'train'):
            # Regular nn.Module
            if training:
                self.model.train()
            else:
                self.model.eval()

    def setup_training(self, total_steps: int):
        """
        Setup training components.

        Args:
            total_steps: Total number of training steps
        """
        console.print("[bold blue]Setting up training components[/bold blue]")

        # Create optimizer manager
        self.optimizer_manager = OptimizerManager(
            self.model, self.config.training, total_steps, self.device
        )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.output_dir,
            checkpoints_limit=self.config.output.checkpoints_limit,
            save_optimizer_state=True,
            save_scheduler_state=True,
            verify_integrity=True,
        )

        # Create training logger
        self.logger = TrainingLogger(
            log_dir=self.logs_dir,
            use_wandb=self.config.logging.wandb,
            wandb_project=self.config.logging.wandb_project,
            experiment_name=f"flux2-lora-{self.config.lora.rank}r-{self.config.training.learning_rate:.0e}",
            config=self.config.to_dict(),
            enable_quality_metrics=self.config.logging.enable_quality_metrics,
        )

        # Create metrics computer
        self.metrics_computer = self.logger.metrics_computer

        # Setup validation if enabled
        if self.config.validation.enable:
            self.validation_sampler = ValidationSampler(
                model=self.model,
                config=self.config.validation,
                device=self.device,
                training_logger=self.logger,
                trigger_word=self.config.lora.trigger_word,
            )

            # Create validation output directory
            validation_output_dir = self.output_dir / "validation_samples"

            # Create validation function
            self.validation_fn = create_validation_function(
                sampler=self.validation_sampler,
                output_dir=validation_output_dir,
            )

            console.print(
                f"[green]✓ Validation enabled: sampling every {self.config.validation.every_n_steps} steps[/green]"
            )

        # Setup callbacks
        self._setup_callbacks()

        # Setup gradient scaler
        self.scaler = self.optimizer_manager.scaler

        # Enable gradient checkpointing if configured
        if self.config.training.gradient_checkpointing:
            if hasattr(self.model.transformer, "gradient_checkpointing_enable"):
                self.model.transformer.gradient_checkpointing_enable()
                console.print("[green]✓ Gradient checkpointing enabled[/green]")
            else:
                console.print("[yellow]Warning: Gradient checkpointing not supported[/yellow]")

        # Set model to training mode
        self._set_model_mode(training=True)

        console.print("[green]✓ Training setup complete[/green]")

    def _setup_callbacks(self):
        """Setup training callbacks based on configuration."""
        console.print("[bold blue]Setting up callbacks[/bold blue]")

        callbacks = []

        # Checkpoint callback
        if self.config.callbacks.enable_checkpoint:
            checkpoint_callback = CheckpointCallback(
                save_every_n_steps=self.config.callbacks.checkpoint_every_n_steps,
                output_dir=self.checkpoints_dir,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                monitor_metric=self.config.callbacks.checkpoint_monitor_metric,
                save_top_k=self.config.callbacks.checkpoint_save_top_k,
            )
            callbacks.append(checkpoint_callback)

        # Early stopping callback
        if self.config.callbacks.enable_early_stopping:
            early_stopping_callback = EarlyStoppingCallback(
                monitor=self.config.callbacks.early_stopping_monitor,
                patience=self.config.callbacks.early_stopping_patience,
                min_delta=self.config.callbacks.early_stopping_min_delta,
                restore_best_weights=self.config.callbacks.early_stopping_restore_best,
            )
            callbacks.append(early_stopping_callback)

        # Validation callback
        if self.config.callbacks.enable_validation_callback and self.validation_sampler:
            validation_callback = ValidationCallback(
                validator=self.validation_sampler,
                every_n_steps=self.config.callbacks.validation_callback_every_n_steps,
                log_images=self.config.callbacks.validation_callback_log_images,
            )
            callbacks.append(validation_callback)

        # LR scheduler callback
        if self.config.callbacks.enable_lr_scheduler_callback:
            lr_callback = LRSchedulerCallback(
                scheduler_step_interval=self.config.callbacks.lr_scheduler_step_interval,
            )
            callbacks.append(lr_callback)

        # Create callback manager
        self.callback_manager = CallbackManager(callbacks)

        console.print(f"[green]✓ Setup {len(callbacks)} callbacks[/green]")

    def train_with_progress_callback(
        self,
        train_dataloader: DataLoader,
        num_steps: int,
        progress_callback: Optional[callable] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop with progress callback support.

        Args:
            train_dataloader: Training data loader
            num_steps: Number of training steps
            progress_callback: Optional callback function(step, loss, metrics)
            resume_from: Path to checkpoint to resume from

        Returns:
            Training results dictionary
        """
        # Store progress callback
        self._progress_callback = progress_callback

        # Call regular train method
        return self.train(train_dataloader, num_steps, resume_from)

    def stop_training(self):
        """Stop training gracefully."""
        self.should_stop = True
        logger.info("Training stop requested")

    def train(
        self,
        train_dataloader: DataLoader,
        num_steps: int,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            num_steps: Number of training steps
            resume_from: Path to checkpoint to resume from
            validation_fn: Optional validation function

        Returns:
            Training results dictionary
        """
        console.print(f"[bold blue]Starting LoRA training[/bold blue]")
        console.print(f"  Total steps: {num_steps}")
        console.print(f"  Batch size: {self.config.training.batch_size}")
        console.print(f"  Learning rate: {self.config.training.learning_rate}")

        # Setup training
        self.setup_training(num_steps)

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)

        # Initialize training state
        self.training_start_time = time.time()
        self._current_dataloader = train_dataloader  # Store for validation access
        data_iter = iter(train_dataloader)

        # Call on_train_begin callbacks
        if self.callback_manager:
            self.callback_manager.on_train_begin(self)

        # Training loop
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Training (step {self.global_step}/{num_steps})", total=num_steps
                )

                while self.global_step < num_steps and not self.should_stop:
                    # Get next batch
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        # Start new epoch
                        data_iter = iter(train_dataloader)
                        batch = next(data_iter)
                        self.current_epoch += 1

                    # Training step
                    step_result = self._training_step(batch)

                    if step_result["success"]:
                        # Update metrics
                        self._update_metrics(step_result)

                        # Update progress
                        progress.update(
                            task,
                            advance=1,
                            description=f"Training (step {self.global_step}/{num_steps}, loss: {step_result['loss']:.6f})",
                        )

                        # Call progress callback if provided
                        if self._progress_callback:
                            try:
                                metrics = {
                                    "loss": step_result["loss"],
                                    "learning_rate": self.optimizer_manager.get_lr(),
                                    "step": self.global_step,
                                }
                                if hasattr(self, "metrics_computer") and self.metrics_computer:
                                    training_metrics = (
                                        self.metrics_computer.compute_training_metrics(
                                            self.model, step_result["loss"]
                                        )
                                    )
                                    metrics.update(training_metrics)

                                self._progress_callback(
                                    self.global_step, step_result["loss"], metrics
                                )
                            except Exception as e:
                                logger.warning(f"Progress callback failed: {e}")

                        # Checkpoint saving
                        if (
                            self.global_step % self.config.output.checkpoint_every_n_steps == 0
                            or self.global_step == num_steps
                        ):
                            self._save_checkpoint(step_result["loss"])

                        # Validation
                        if self.validation_fn and (
                            self.global_step % self.config.validation.every_n_steps == 0
                        ):
                            self._run_validation(self.validation_fn, step_result["loss"])

                            # Call on_epoch_end callbacks (validation cycle = epoch)
                            if self.callback_manager:
                                epoch_metrics = {
                                    "validation_loss": step_result.get("validation_loss", 0.0),
                                    "loss": step_result["loss"],
                                    "step": self.global_step,
                                }
                                validation_cycle = (
                                    self.global_step // self.config.validation.every_n_steps
                                )
                                self.callback_manager.on_epoch_end(
                                    self, validation_cycle, epoch_metrics
                                )

                        # Log metrics
                        if self.global_step % self.config.logging.log_every_n_steps == 0:
                            self._log_metrics()

                        # Call on_step_end callbacks
                        if self.callback_manager:
                            step_metrics = {
                                "loss": step_result["loss"],
                                "learning_rate": step_result["learning_rate"],
                                "grad_norm": step_result["grad_norm"],
                                "memory_usage": step_result["memory_usage"],
                                "step_time": step_result["step_time"],
                            }
                            self.callback_manager.on_step_end(self, self.global_step, step_metrics)

                    else:
                        console.print(f"[red]Training step failed: {step_result['error']}[/red]")
                        logger.error(f"Step {self.global_step} failed: {step_result['error']}")

                    self.global_step += 1

        except KeyboardInterrupt:
            console.print("[yellow]Training interrupted by user[/yellow]")
            self._save_checkpoint(
                self.metrics["loss"][-1] if self.metrics["loss"] else 0.0, force=True
            )

        except Exception as e:
            console.print(f"[red]Training failed: {e}[/red]")
            logger.error(f"Training failed: {e}")
            self._save_checkpoint(
                self.metrics["loss"][-1] if self.metrics["loss"] else 0.0, force=True
            )
            raise

        finally:
            # Always close logger
            if self.logger is not None:
                self.logger.close()

        # Call on_train_end callbacks
        if self.callback_manager:
            self.callback_manager.on_train_end(self)

        # Training completed
        training_time = time.time() - self.training_start_time
        if self.should_stop:
            console.print(f"[bold yellow]Training stopped early![/bold yellow]")
        else:
            console.print(f"[bold green]Training completed![/bold green]")
        console.print(f"  Total time: {training_time:.2f}s")
        console.print(
            f"  Final loss: {self.metrics['loss'][-1] if self.metrics['loss'] else 'N/A'}"
        )
        console.print(f"  Steps/sec: {self.global_step / training_time:.2f}")
        console.print(f"  Final step: {self.global_step}/{num_steps}")

        # Close logger
        if self.logger is not None:
            self.logger.close()

        return self._get_training_results()

    def _training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single training step.

        Args:
            batch: Training batch

        Returns:
            Step result dictionary
        """
        step_start_time = time.time()

        try:
            # Debug: Check batch structure
            if batch is None:
                raise ValueError("Batch is None!")
            if "images" not in batch:
                raise ValueError(f"Batch missing 'images' key. Keys: {batch.keys()}")
            if batch["images"] is None:
                raise ValueError("batch['images'] is None!")

            # Move batch to device
            images = batch["images"].to(self.device, non_blocking=True)
            captions = batch["captions"]

            # Zero gradients
            self.optimizer_manager.zero_grad()

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16
                    if self.config.training.mixed_precision == "bf16"
                    else torch.float16
                ):
                    loss = self._compute_loss(images, captions)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Clip gradients
                if self.config.training.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer_manager.optimizer)
                    self.optimizer_manager.clip_gradients()

                # Optimizer step
                self.scaler.step(self.optimizer_manager.optimizer)
                self.scaler.update()
            else:
                # Full precision training
                loss = self._compute_loss(images, captions)
                loss.backward()

                # Clip gradients
                if self.config.training.max_grad_norm > 0:
                    self.optimizer_manager.clip_gradients()

                # Optimizer step
                self.optimizer_manager.optimizer.step()

            # Update scheduler
            if hasattr(self.optimizer_manager.scheduler, "step"):
                self.optimizer_manager.scheduler.step()

            # Calculate step time
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)

            # Keep only recent step times for moving average
            if len(self.step_times) > 100:
                self.step_times.pop(0)

            # Get gradient norm
            grad_norm = self._compute_gradient_norm()

            # Get memory usage
            memory_usage = self._get_memory_usage()

            # Compute training metrics if available
            training_metrics = {}
            if self.metrics_computer:
                try:
                    training_metrics = self.metrics_computer.compute_training_metrics(
                        model=self.model,
                        loss=loss,
                        optimizer=self.optimizer_manager.optimizer,
                    )
                except Exception as e:
                    logger.debug(f"Failed to compute training metrics: {e}")

            return {
                "success": True,
                "loss": loss.item(),
                "step_time": step_time,
                "grad_norm": grad_norm,
                "memory_usage": memory_usage,
                "learning_rate": self.optimizer_manager.current_lr,
                "training_metrics": training_metrics,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "loss": 0.0,
                "step_time": 0.0,
                "grad_norm": 0.0,
                "memory_usage": 0.0,
                "learning_rate": 0.0,
                "error": str(e),
            }

    def _compute_loss(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """
        Compute training loss for Flux2-dev.

        Args:
            images: Input images
            captions: List of captions

        Returns:
            Computed loss tensor
        """
        batch_size = images.shape[0]

        # Generate random noise for diffusion training
        # For Flux2-dev, we need to match the expected input format
        noise = torch.randn_like(images)

        # Sample random timesteps
        timesteps = torch.randint(0, 1000, (batch_size,), device=images.device)
        timesteps = timesteps.long()

        # Add noise to images (forward diffusion)
        noisy_images = self._add_noise(images, noise, timesteps)

        # Get text embeddings from captions
        # This is a simplified version - in practice, you'd use the model's text encoder
        text_embeddings = self._encode_text(captions)

        # Predict noise using the model
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Call transformer with keyword arguments to avoid signature issues
            # Flux2Transformer expects: hidden_states, encoder_hidden_states, timestep (among others)
            model_output = self.model.transformer(
                hidden_states=noisy_images,
                encoder_hidden_states=text_embeddings,
                timestep=timesteps,
                return_dict=False,
            )

            # Extract predicted noise from model output
            # Model output is typically (sample,) or just the tensor
            if isinstance(model_output, tuple):
                predicted_noise = model_output[0]
            else:
                predicted_noise = model_output

        # Compute MSE loss between predicted and actual noise
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return loss

    def _add_noise(
        self, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to images for diffusion training.

        Args:
            images: Clean images
            noise: Random noise
            timesteps: Diffusion timesteps

        Returns:
            Noisy images
        """
        # Simplified noise addition - in practice, this would use the proper noise schedule
        # For now, we'll use a simple linear interpolation
        alpha = 1.0 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1)
        noisy_images = alpha.view(-1, 1, 1, 1) * images + (1.0 - alpha).view(-1, 1, 1, 1) * noise

        return noisy_images

    def _encode_text(self, captions: List[str]) -> torch.Tensor:
        """
        Encode text captions to embeddings.

        Args:
            captions: List of text captions

        Returns:
            Text embeddings tensor
        """
        # Simplified text encoding - in practice, this would use the model's text encoder
        # For now, we'll create dummy embeddings
        batch_size = len(captions)
        embedding_dim = 4096  # Typical for large language models

        # Create random embeddings (placeholder)
        embeddings = torch.randn(batch_size, 77, embedding_dim, device=self.device)

        return embeddings

    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm for monitoring."""
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is not None and "lora" in name.lower():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** (1.0 / 2)

    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**2)  # MB
            return memory_used
        return 0.0

    def _update_metrics(self, step_result: Dict[str, Any]):
        """Update training metrics."""
        self.metrics["loss"].append(step_result["loss"])
        self.metrics["learning_rate"].append(step_result["learning_rate"])
        self.metrics["grad_norm"].append(step_result["grad_norm"])
        self.metrics["step_time"].append(step_result["step_time"])
        self.metrics["memory_usage"].append(step_result["memory_usage"])

        # Store training metrics for logging
        if "training_metrics" in step_result:
            self._last_training_metrics = step_result["training_metrics"]

    def _save_checkpoint(self, loss: float, force: bool = False):
        """Save training checkpoint."""
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss

        # Save checkpoint
        result = self.checkpoint_manager.save_checkpoint(
            self.model,
            self.optimizer_manager,
            self.global_step,
            loss,
            self.config,
            metadata={
                "epoch": self.current_epoch,
                "avg_step_time": sum(self.step_times[-10:]) / len(self.step_times[-10:])
                if self.step_times
                else 0.0,
                "memory_usage_mb": self._get_memory_usage(),
                "metrics": {
                    "avg_loss": sum(self.metrics["loss"][-100:]) / len(self.metrics["loss"][-100:])
                    if self.metrics["loss"]
                    else 0.0,
                    "current_lr": self.optimizer_manager.current_lr,
                },
            },
            is_best=is_best,
        )

        if result["success"]:
            logger.info(f"Checkpoint saved at step {self.global_step}")
        else:
            logger.error(f"Failed to save checkpoint: {result['error']}")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        console.print(f"[bold blue]Resuming from checkpoint: {checkpoint_path}[/bold blue]")

        result = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer_manager, self.device
        )

        if result["success"]:
            self.global_step = result["step"]
            self.best_loss = result["loss"]
            console.print(f"[green]✓ Resumed from step {self.global_step}[/green]")
        else:
            console.print(f"[red]Failed to resume: {result['error']}[/red]")
            raise RuntimeError(f"Failed to resume from checkpoint: {result['error']}")

    def _run_validation(self, validation_fn: callable, current_loss: float):
        """Run validation during training."""
        try:
            console.print("[blue]Running validation...[/blue]")

            # Switch to eval mode
            self._set_model_mode(training=False)

            # Run validation
            validation_results = validation_fn(self.model, self.global_step)

            # Get validation images and prompts for metrics computation
            validation_images = None
            validation_prompts = None
            if self.validation_sampler:
                # Get the most recent validation samples
                validation_images = self.validation_sampler.get_samples_for_step(self.global_step)
                validation_prompts = self.validation_sampler.prompts

            # Log validation metrics with quality assessment
            if self.logger and validation_images and validation_prompts:
                try:
                    # Get training images for overfitting detection (first few from dataset)
                    training_images = None
                    if hasattr(self, "_training_images_cache") and self._training_images_cache:
                        training_images = self._training_images_cache
                    elif hasattr(self, "_current_dataloader"):
                        # Cache a few training images for overfitting detection
                        try:
                            batch = next(iter(self._current_dataloader))
                            if "images" in batch:
                                # Convert tensor images to PIL for comparison
                                import torchvision.transforms as transforms

                                to_pil = transforms.ToPILImage()
                                training_images = []
                                for i in range(
                                    min(10, len(batch["images"]))
                                ):  # Use first 10 images
                                    img_tensor = batch["images"][i]
                                    if img_tensor.dim() == 3:  # CHW format
                                        pil_img = to_pil(img_tensor)
                                        training_images.append(pil_img)
                                self._training_images_cache = training_images
                        except Exception as e:
                            logger.debug(f"Could not cache training images: {e}")

                    # Log validation metrics with automatic quality assessment
                    self.logger.log_validation_metrics(
                        validation_loss=current_loss,
                        images=validation_images,
                        prompts=validation_prompts,
                        training_images=training_images,
                        step=self.global_step,
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute validation quality metrics: {e}")
                    # Fallback to basic logging
                    self.logger.log_validation_metrics(
                        validation_loss=current_loss, step=self.global_step
                    )

            # Switch back to train mode
            self._set_model_mode(training=True)

            # Log validation results
            if validation_results:
                console.print(f"[green]Validation completed[/green]")
                for key, value in validation_results.items():
                    console.print(f"  {key}: {value}")

        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
            logger.error(f"Validation failed: {e}")

        finally:
            # Ensure model is back in train mode
            self._set_model_mode(training=True)

    def _log_metrics(self):
        """Log training metrics."""
        if not self.metrics["loss"]:
            return

        # Get recent metrics
        recent_window = min(100, len(self.metrics["loss"]))
        recent_loss = sum(self.metrics["loss"][-recent_window:]) / recent_window
        recent_lr = self.metrics["learning_rate"][-1]
        recent_grad_norm = self.metrics["grad_norm"][-1]
        recent_step_time = sum(self.metrics["step_time"][-recent_window:]) / recent_window
        recent_memory = self.metrics["memory_usage"][-1]

        # Calculate steps per second
        steps_per_sec = 1.0 / recent_step_time if recent_step_time > 0 else 0.0

        # Log to TensorBoard/W&B if logger is available
        if self.logger is not None:
            self.logger.log_training_metrics(
                loss=recent_loss,
                learning_rate=recent_lr,
                grad_norm=recent_grad_norm,
                step_time=recent_step_time,
                memory_usage_mb=recent_memory,
                step=self.global_step,
            )

            # Log additional training metrics if available
            if (
                self.metrics_computer
                and hasattr(self, "_last_training_metrics")
                and self._last_training_metrics
            ):
                # Log training metrics as scalars in the training group
                training_metric_scalars = {}
                for key, value in self._last_training_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        training_metric_scalars[key] = value

                if training_metric_scalars:
                    # Log under training/parameters group
                    self.logger.log_scalars(
                        "training/parameters", training_metric_scalars, self.global_step
                    )

            # Log system metrics occasionally
            if self.global_step % (self.config.logging.log_every_n_steps * 10) == 0:
                self.logger.log_system_metrics(self.global_step)

        # Log to console/Python logger
        logger.info(
            f"Step {self.global_step}: "
            f"loss={recent_loss:.6f}, "
            f"lr={recent_lr:.2e}, "
            f"grad_norm={recent_grad_norm:.4f}, "
            f"step_time={recent_step_time:.3f}s, "
            f"steps/sec={steps_per_sec:.2f}, "
            f"memory={recent_memory:.1f}MB"
        )

    def _get_training_results(self) -> Dict[str, Any]:
        """Get comprehensive training results."""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0.0

        return {
            "success": True,
            "global_step": self.global_step,
            "total_time": total_time,
            "steps_per_second": self.global_step / total_time if total_time > 0 else 0.0,
            "best_loss": self.best_loss,
            "final_loss": self.metrics["loss"][-1] if self.metrics["loss"] else 0.0,
            "avg_loss": sum(self.metrics["loss"]) / len(self.metrics["loss"])
            if self.metrics["loss"]
            else 0.0,
            "metrics": self.metrics,
            "output_dir": str(self.output_dir),
            "checkpoints_saved": len(self.checkpoint_manager.list_checkpoints()),
            "best_checkpoint": self.checkpoint_manager.get_best_checkpoint(),
        }

    def evaluate(
        self, eval_dataloader: DataLoader, max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on evaluation dataset.

        Args:
            eval_dataloader: Evaluation data loader
            max_steps: Maximum number of evaluation steps

        Returns:
            Evaluation results
        """
        console.print("[bold blue]Running evaluation[/bold blue]")

        self._set_model_mode(training=False)
        eval_losses = []
        step_count = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                if max_steps and step_count >= max_steps:
                    break

                # Move batch to device
                images = batch["images"].to(self.device, non_blocking=True)
                captions = batch["captions"]

                # Compute loss
                loss = self._compute_loss(images, captions)
                eval_losses.append(loss.item())

                step_count += 1

        # Calculate metrics
        avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0.0

        results = {
            "avg_loss": avg_loss,
            "num_steps": step_count,
            "eval_losses": eval_losses,
        }

        console.print(f"[green]Evaluation completed[/green]")
        console.print(f"  Average loss: {avg_loss:.6f}")
        console.print(f"  Steps evaluated: {step_count}")

        return results

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for monitoring."""
        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "device": self.device,
            "model_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "memory_usage_mb": self._get_memory_usage(),
            "recent_avg_loss": sum(self.metrics["loss"][-10:]) / len(self.metrics["loss"][-10:])
            if self.metrics["loss"]
            else 0.0,
            "current_lr": self.optimizer_manager.current_lr if self.optimizer_manager else 0.0,
            "steps_per_second": 1.0 / (sum(self.step_times[-10:]) / len(self.step_times[-10:]))
            if self.step_times
            else 0.0,
        }
