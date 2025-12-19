"""
Training callbacks system for Flux2 LoRA training.

This module provides a flexible callback system that allows custom logic to be executed
at different points during training (start, step end, epoch end, training end).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from rich.console import Console

from ..monitoring.validator import ValidationSampler

logger = logging.getLogger(__name__)
console = Console()


class TrainingCallback(ABC):
    """
    Base class for training callbacks.

    Callbacks allow custom logic to be executed at specific points during training.
    Override the methods you need in your custom callback.
    """

    def __init__(self):
        """Initialize callback."""
        pass

    @abstractmethod
    def on_train_begin(self, trainer):
        """
        Called at the beginning of training.

        Args:
            trainer: The LoRATrainer instance
        """
        pass

    @abstractmethod
    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """
        Called after each training step.

        Args:
            trainer: The LoRATrainer instance
            step: Current training step
            metrics: Dictionary of metrics from the training step
        """
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """
        Called after each epoch/validation cycle.

        For step-based training, this is called after validation runs.

        Args:
            trainer: The LoRATrainer instance
            epoch: Current epoch number (or validation cycle number)
            metrics: Dictionary of metrics from the epoch
        """
        pass

    @abstractmethod
    def on_train_end(self, trainer):
        """
        Called at the end of training.

        Args:
            trainer: The LoRATrainer instance
        """
        pass


class CheckpointCallback(TrainingCallback):
    """
    Callback that saves checkpoints at regular intervals.

    This callback handles automatic checkpoint saving during training.
    """

    def __init__(
        self,
        save_every_n_steps: int = 500,
        output_dir: Union[str, Path] = "./checkpoints",
        save_best_only: bool = False,
        monitor_metric: str = "loss",
        save_top_k: int = 3,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_every_n_steps: Save checkpoint every N steps
            output_dir: Directory to save checkpoints
            save_best_only: If True, only save when metric improves
            monitor_metric: Metric to monitor for "best" checkpoints
            save_top_k: Number of best checkpoints to keep
        """
        super().__init__()
        self.save_every_n_steps = save_every_n_steps
        self.output_dir = Path(output_dir)
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.save_top_k = save_top_k

        self.best_metric_value = float("inf") if monitor_metric == "loss" else float("-inf")
        self.saved_checkpoints = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer):
        """Called at training start."""
        console.print(
            f"[green]✓ Checkpoint callback initialized: saving every {self.save_every_n_steps} steps[/green]"
        )

    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """Save checkpoint if needed."""
        if step % self.save_every_n_steps == 0:
            self._save_checkpoint(trainer, step, metrics)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Save checkpoint at epoch end if configured."""
        pass  # Handled in on_step_end

    def on_train_end(self, trainer):
        """Save final checkpoint."""
        self._save_checkpoint(trainer, trainer.global_step, {}, is_final=True)

    def _save_checkpoint(self, trainer, step: int, metrics: Dict[str, Any], is_final: bool = False):
        """Save a checkpoint."""
        try:
            checkpoint_path = self.output_dir / f"checkpoint_step_{step}.safetensors"

            # Save using trainer's checkpoint manager
            result = trainer.checkpoint_manager.save_checkpoint(
                trainer.model,
                trainer.optimizer_manager,
                step,
                metrics.get("loss", 0.0),
                trainer.config,
                metadata={
                    "step": step,
                    "metrics": metrics,
                    "is_final": is_final,
                },
                is_best=self._is_best_checkpoint(metrics),
            )

            if result["success"]:
                checkpoint_path = result["checkpoint_path"]
                self.saved_checkpoints.append(str(checkpoint_path))

                # Clean up old checkpoints if needed
                self._cleanup_old_checkpoints()

                console.print(f"[green]✓ Checkpoint saved: {checkpoint_path}[/green]")
                logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
            else:
                console.print(f"[red]✗ Failed to save checkpoint: {result['error']}[/red]")
                logger.error(f"Failed to save checkpoint at step {step}: {result['error']}")

        except Exception as e:
            console.print(f"[red]✗ Checkpoint callback error: {e}[/red]")
            logger.error(f"Checkpoint callback error at step {step}: {e}")

    def _is_best_checkpoint(self, metrics: Dict[str, Any]) -> bool:
        """Check if this is the best checkpoint so far."""
        if not self.save_best_only:
            return False

        current_value = metrics.get(self.monitor_metric, 0.0)

        if self.monitor_metric == "loss":
            is_best = current_value < self.best_metric_value
        else:
            is_best = current_value > self.best_metric_value

        if is_best:
            self.best_metric_value = current_value

        return is_best

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to maintain save_top_k limit."""
        if len(self.saved_checkpoints) <= self.save_top_k:
            return

        # Sort by modification time, keep newest
        checkpoints_with_time = []
        for cp in self.saved_checkpoints:
            path = Path(cp)
            if path.exists():
                checkpoints_with_time.append((path.stat().st_mtime, cp))

        checkpoints_with_time.sort(reverse=True)  # Newest first
        to_keep = [cp for _, cp in checkpoints_with_time[: self.save_top_k]]
        to_remove = [cp for _, cp in checkpoints_with_time[self.save_top_k :]]

        for cp in to_remove:
            try:
                Path(cp).unlink()
                console.print(f"[blue]Removed old checkpoint: {cp}[/blue]")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {cp}: {e}")

        self.saved_checkpoints = to_keep


class EarlyStoppingCallback(TrainingCallback):
    """
    Callback that stops training early if validation loss doesn't improve.

    This prevents overfitting and saves compute resources.
    """

    def __init__(
        self,
        monitor: str = "validation_loss",
        patience: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor ("validation_loss", "loss", etc.)
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_value = float("inf") if "loss" in monitor else float("-inf")
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_train_begin(self, trainer):
        """Called at training start."""
        console.print(
            f"[green]✓ Early stopping initialized: monitoring {self.monitor}, patience={self.patience}[/green]"
        )

    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """Check for early stopping at step end."""
        pass  # Early stopping typically happens at epoch/validation end

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            return  # Metric not available

        # Check if improvement occurred
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait_count = 0
            self.best_epoch = epoch

            # Save best weights if requested
            if self.restore_best_weights:
                self.best_weights = self._get_model_weights(trainer.model)

            console.print(
                f"[blue]Early stopping metric improved: {self.monitor} = {current_value:.6f}[/blue]"
            )
        else:
            self.wait_count += 1
            console.print(
                f"[blue]Early stopping: {self.monitor} = {current_value:.6f}, wait_count = {self.wait_count}/{self.patience}[/blue]"
            )

            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                console.print(f"[yellow]Early stopping triggered at epoch {epoch}[/yellow]")
                logger.info(f"Early stopping triggered at epoch {epoch}")

                # Restore best weights if requested
                if self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(trainer.model, self.best_weights)
                    console.print(
                        f"[blue]Restored best weights from epoch {self.best_epoch}[/blue]"
                    )

                # Signal trainer to stop
                trainer.should_stop = True

    def on_train_end(self, trainer):
        """Called at training end."""
        if self.stopped_epoch > 0:
            console.print(f"[yellow]Training stopped early at epoch {self.stopped_epoch}[/yellow]")

    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value represents an improvement."""
        if "loss" in self.monitor:
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)

    def _get_model_weights(self, model) -> Dict[str, torch.Tensor]:
        """Get model weights for restoration."""
        return {name: param.clone() for name, param in model.named_parameters()}

    def _set_model_weights(self, model, weights: Dict[str, torch.Tensor]):
        """Set model weights from saved state."""
        for name, param in model.named_parameters():
            if name in weights:
                param.data.copy_(weights[name].data)


class ValidationCallback(TrainingCallback):
    """
    Callback that runs validation during training.

    This callback integrates validation sampling into the training loop.
    """

    def __init__(
        self,
        validator: ValidationSampler,
        every_n_steps: int = 100,
        log_images: bool = True,
    ):
        """
        Initialize validation callback.

        Args:
            validator: ValidationSampler instance
            every_n_steps: Run validation every N steps
            log_images: Whether to log generated images
        """
        super().__init__()
        self.validator = validator
        self.every_n_steps = every_n_steps
        self.log_images = log_images

    def on_train_begin(self, trainer):
        """Called at training start."""
        console.print(
            f"[green]✓ Validation callback initialized: running every {self.every_n_steps} steps[/green]"
        )

    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """Run validation if needed."""
        if step % self.every_n_steps == 0:
            self._run_validation(trainer, step)

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Validation is handled in on_step_end."""
        pass

    def on_train_end(self, trainer):
        """Run final validation."""
        self._run_validation(trainer, trainer.global_step, is_final=True)

    def _run_validation(self, trainer, step: int, is_final: bool = False):
        """Run validation sampling."""
        try:
            console.print("[blue]Running validation sampling...[/blue]")

            # Generate validation samples
            images = self.validator.generate_samples(step)

            # Log images if requested
            if self.log_images and trainer.logger:
                # Create a grid of images
                if images:
                    # Log to TensorBoard
                    trainer.logger.log_validation_samples(images, step)

                    # Save to disk
                    output_dir = trainer.output_dir / "validation_samples"
                    output_dir.mkdir(exist_ok=True)

                    for i, img in enumerate(images):
                        img_path = output_dir / f"step_{step}_sample_{i}.png"
                        img.save(img_path)

            console.print(f"[green]✓ Validation completed at step {step}[/green]")
            logger.info(f"Validation completed at step {step}")

        except Exception as e:
            console.print(f"[red]✗ Validation failed: {e}[/red]")
            logger.error(f"Validation failed at step {step}: {e}")


class LRSchedulerCallback(TrainingCallback):
    """
    Callback that handles learning rate scheduling.

    This callback steps the learning rate scheduler at appropriate intervals.
    """

    def __init__(self, scheduler_step_interval: str = "step"):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler_step_interval: When to step scheduler ("step" or "epoch")
        """
        super().__init__()
        self.scheduler_step_interval = scheduler_step_interval

    def on_train_begin(self, trainer):
        """Called at training start."""
        console.print("[green]✓ LR scheduler callback initialized[/green]")

    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """Step scheduler if configured for step interval."""
        if self.scheduler_step_interval == "step" and trainer.optimizer_manager.scheduler:
            trainer.optimizer_manager.scheduler.step()

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Step scheduler if configured for epoch interval."""
        if self.scheduler_step_interval == "epoch" and trainer.optimizer_manager.scheduler:
            trainer.optimizer_manager.scheduler.step()

    def on_train_end(self, trainer):
        """Called at training end."""
        pass


class CallbackManager:
    """
    Manager for training callbacks.

    This class handles registration and execution of multiple callbacks.
    """

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        """
        Initialize callback manager.

        Args:
            callbacks: List of callbacks to register
        """
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainingCallback):
        """Add a callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: TrainingCallback):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_train_begin(self, trainer):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_begin(trainer)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}.on_train_begin: {e}")

    def on_step_end(self, trainer, step: int, metrics: Dict[str, Any]):
        """Call on_step_end for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_step_end(trainer, step, metrics)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}.on_step_end: {e}")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_epoch_end(trainer, epoch, metrics)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}.on_epoch_end: {e}")

    def on_train_end(self, trainer):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            try:
                callback.on_train_end(trainer)
            except Exception as e:
                logger.error(f"Error in callback {callback.__class__.__name__}.on_train_end: {e}")
