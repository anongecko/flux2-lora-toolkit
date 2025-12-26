"""
Optimizer and scheduler factories for Flux2-dev LoRA training.

This module provides factory classes for creating optimizers and learning rate
schedulers with proper configuration and validation for LoRA training.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ConstantLR,
    LinearLR,
    PolynomialLR,
    StepLR,
    ExponentialLR,
    OneCycleLR,
)
from torch.optim.swa_utils import AveragedModel

from ..utils.config_manager import TrainingConfig

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory class for creating optimizers with proper configuration."""

    @staticmethod
    def create_optimizer(
        model_parameters,
        config: TrainingConfig,
        param_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.

        Args:
            model_parameters: Model parameters to optimize (generator or list)
            config: Training configuration
            param_groups: Optional parameter groups for differential learning rates

        Returns:
            Configured optimizer instance

        Raises:
            ValueError: If optimizer type is not supported
        """
        logger.info(f"Creating optimizer: {config.optimizer}")
        logger.info(f"Learning rate: {config.learning_rate}")

        # Convert generator to list if needed
        if hasattr(model_parameters, "__iter__") and not isinstance(model_parameters, list):
            model_parameters = list(model_parameters)

        # Default parameter groups
        if param_groups is None:
            param_groups = [{"params": model_parameters}]
        else:
            # Convert generators to lists in parameter groups
            for group in param_groups:
                if (
                    "params" in group
                    and hasattr(group["params"], "__iter__")
                    and not isinstance(group["params"], list)
                ):
                    group["params"] = list(group["params"])
                # Add learning rate to parameter groups if not specified
                if "lr" not in group:
                    group["lr"] = config.learning_rate

        try:
            if config.optimizer.lower() == "adamw":
                optimizer = AdamW(
                    param_groups,
                    lr=config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.01,  # Standard for LoRA training
                    amsgrad=False,
                )
                logger.info("Created AdamW optimizer with weight_decay=0.01")

            elif config.optimizer.lower() == "adam":
                optimizer = Adam(
                    param_groups,
                    lr=config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                    amsgrad=False,
                )
                logger.info("Created Adam optimizer with weight_decay=0.0")

            elif config.optimizer.lower() == "sgd":
                optimizer = SGD(
                    param_groups,
                    lr=config.learning_rate,
                    momentum=0.9,
                    weight_decay=0.0,
                    nesterov=True,
                )
                logger.info("Created SGD optimizer with momentum=0.9")

            else:
                raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        except Exception as e:
            logger.error(f"Failed to create optimizer {config.optimizer}: {e}")
            raise

        return optimizer

    @staticmethod
    def get_parameter_groups(
        model, lora_lr_multiplier: float = 1.0, bias_lr_multiplier: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups for differential learning rates.

        Args:
            model: Model to create parameter groups for
            lora_lr_multiplier: Learning rate multiplier for LoRA parameters
            bias_lr_multiplier: Learning rate multiplier for bias parameters

        Returns:
            List of parameter groups
        """
        lora_params = []
        bias_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "lora" in name.lower():
                lora_params.append(param)
            elif "bias" in name.lower():
                bias_params.append(param)
            else:
                other_params.append(param)

        param_groups = []

        if lora_params:
            param_groups.append(
                {"params": lora_params, "lr": lora_lr_multiplier, "name": "lora_parameters"}
            )
            logger.info(f"LoRA parameters: {len(lora_params)} tensors")

        if bias_params:
            param_groups.append(
                {"params": bias_params, "lr": bias_lr_multiplier, "name": "bias_parameters"}
            )
            logger.info(f"Bias parameters: {len(bias_params)} tensors")

        if other_params:
            param_groups.append({"params": other_params, "lr": 1.0, "name": "other_parameters"})
            logger.info(f"Other trainable parameters: {len(other_params)} tensors")

        return param_groups

    @staticmethod
    def validate_optimizer_config(config: TrainingConfig) -> list[str]:
        """
        Validate optimizer configuration and return warnings.

        Args:
            config: Training configuration

        Returns:
            List of warning messages
        """
        warnings = []

        if config.learning_rate > 1e-3:
            warnings.append("High learning rate may cause training instability")

        if config.learning_rate < 1e-6:
            warnings.append("Very low learning rate may result in slow training")

        if config.optimizer.lower() == "sgd" and config.learning_rate > 1e-2:
            warnings.append("SGD with high learning rate may diverge")

        return warnings


class SchedulerFactory:
    """Factory class for creating learning rate schedulers."""

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        total_steps: int,
        warmup_steps: Optional[int] = None,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler based on configuration.

        Args:
            optimizer: Optimizer to schedule
            config: Training configuration
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps (overrides config)

        Returns:
            Configured scheduler instance

        Raises:
            ValueError: If scheduler type is not supported
        """
        warmup_steps = warmup_steps or config.warmup_steps

        logger.info(f"Creating scheduler: {config.scheduler}")
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Total steps: {total_steps}")

        try:
            if config.scheduler.lower() == "constant":
                scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
                logger.info("Created constant learning rate scheduler")

            elif config.scheduler.lower() == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=total_steps - warmup_steps, eta_min=0.0, last_epoch=-1
                )
                logger.info("Created cosine annealing scheduler")

            elif config.scheduler.lower() == "linear":
                scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=total_steps - warmup_steps,
                    last_epoch=-1,
                )
                logger.info("Created linear decay scheduler")

            elif config.scheduler.lower() == "polynomial":
                scheduler = PolynomialLR(
                    optimizer, total_iters=total_steps - warmup_steps, power=2.0, last_epoch=-1
                )
                logger.info("Created polynomial decay scheduler")

            elif config.scheduler.lower() == "step":
                # Step LR with configurable milestones
                milestones = [total_steps // 2, total_steps // 4, total_steps // 8]
                gamma = 0.5
                scheduler = StepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
                logger.info(f"Created step scheduler with milestones: {milestones}")

            elif config.scheduler.lower() == "exponential":
                scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
                logger.info("Created exponential decay scheduler")

            elif config.scheduler.lower() == "onecycle":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=config.learning_rate,
                    total_steps=total_steps,
                    pct_start=warmup_steps / total_steps if warmup_steps > 0 else 0.03,
                    anneal_strategy="cos",
                    div_factor=25.0,
                    final_div_factor=1e4,
                )
                logger.info("Created OneCycle learning rate scheduler")

            else:
                raise ValueError(f"Unsupported scheduler: {config.scheduler}")

        except Exception as e:
            logger.error(f"Failed to create scheduler {config.scheduler}: {e}")
            raise

        return scheduler

    @staticmethod
    def create_warmup_scheduler(
        optimizer: torch.optim.Optimizer, config: TrainingConfig, total_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create a warmup-only scheduler that transitions to main scheduler.

        Args:
            optimizer: Optimizer to schedule
            config: Training configuration
            total_steps: Total number of training steps

        Returns:
            Combined scheduler with warmup
        """
        from torch.optim.lr_scheduler import SequentialLR, LinearLR

        warmup = LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=config.warmup_steps,
            last_epoch=-1,
        )

        # Create main scheduler for remaining steps
        main_config = config
        main_config.warmup_steps = 0  # Disable warmup in main scheduler

        main_scheduler = SchedulerFactory.create_scheduler(optimizer, main_config, total_steps)

        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, main_scheduler], milestones=[config.warmup_steps]
        )

        logger.info(f"Created combined scheduler with {config.warmup_steps} warmup steps")
        return scheduler

    @staticmethod
    def validate_scheduler_config(config: TrainingConfig, total_steps: int) -> list[str]:
        """
        Validate scheduler configuration and return warnings.

        Args:
            config: Training configuration
            total_steps: Total number of training steps

        Returns:
            List of warning messages
        """
        warnings = []

        if config.warmup_steps >= total_steps:
            warnings.append("Warmup steps should be less than total training steps")

        if config.warmup_steps > total_steps // 2:
            warnings.append("Warmup period is very long (>50% of training)")

        if config.scheduler.lower() == "cosine" and config.warmup_steps == 0:
            warnings.append("Cosine scheduler typically benefits from warmup")

        if config.scheduler.lower() == "onecycle" and config.warmup_steps > 0:
            warnings.append(
                "OneCycle scheduler has built-in warmup, consider setting warmup_steps=0"
            )

        return warnings


class GradientScalerFactory:
    """Factory for creating gradient scalers for mixed precision training."""

    @staticmethod
    def create_scaler(
        config: TrainingConfig, device: str = "cuda"
    ) -> Optional[torch.cuda.amp.GradScaler]:
        """
        Create gradient scaler based on precision configuration.

        Args:
            config: Training configuration
            device: Target device

        Returns:
            Gradient scaler instance or None for full precision
        """
        if config.mixed_precision.lower() == "no":
            logger.info("Using full precision training (no gradient scaler)")
            return None

        elif config.mixed_precision.lower() in ["fp16", "bf16"]:
            if device.startswith("cuda"):
                scaler = torch.amp.GradScaler(
                    'cuda',
                    init_scale=2**16,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                    enabled=True,
                )
                logger.info(f"Created gradient scaler for {config.mixed_precision}")
                return scaler
            else:
                logger.warning("Gradient scaler only supported on CUDA, using full precision")
                return None

        else:
            logger.warning(f"Unknown precision: {config.mixed_precision}, using full precision")
            return None


class OptimizerManager:
    """High-level manager for optimizer and scheduler configuration."""

    def __init__(self, model, config: TrainingConfig, total_steps: int, device: str = "cuda"):
        """
        Initialize optimizer manager.

        Args:
            model: Model to optimize (can be a Flux pipeline or nn.Module)
            config: Training configuration
            total_steps: Total number of training steps
            device: Target device
        """
        self.model = model
        self.config = config
        self.total_steps = total_steps
        self.device = device

        # Handle Flux pipeline vs regular model
        # For Flux pipelines, the trainable parameters are in the transformer
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'parameters'):
            # This is a Flux pipeline - get parameters from transformer
            trainable_model = model.transformer
            logger.info("Detected Flux pipeline - using transformer for optimization")
        elif hasattr(model, 'parameters'):
            # This is a regular nn.Module
            trainable_model = model
        else:
            raise ValueError("Model must have either 'parameters()' method or 'transformer.parameters()' method")

        # Create parameter groups
        base_params = [p for p in trainable_model.parameters() if p.requires_grad]

        if len(base_params) > 0:
            # Create parameter groups with proper formatting
            self.param_groups = OptimizerFactory.get_parameter_groups(trainable_model)
        else:
            # No trainable parameters
            self.param_groups = [{"params": []}]

        # Create optimizer
        if len(base_params) > 0:
            self.optimizer = OptimizerFactory.create_optimizer(base_params, config)
        else:
            # Create a dummy optimizer for testing
            self.optimizer = AdamW([torch.randn(1)], lr=config.learning_rate)

        # Create scheduler
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, config, total_steps)

        # Create gradient scaler
        self.scaler = GradientScalerFactory.create_scaler(config, device)

        # Track learning rate
        self.current_lr = config.learning_rate

        logger.info("Optimizer manager initialized")
        self._log_configuration()

    def _log_configuration(self):
        """Log the optimizer and scheduler configuration."""
        logger.info("=== Optimizer Configuration ===")
        logger.info(f"Optimizer: {self.config.optimizer}")
        logger.info(f"Scheduler: {self.config.scheduler}")
        logger.info(f"Base LR: {self.config.learning_rate}")
        logger.info(f"Mixed precision: {self.config.mixed_precision}")
        logger.info(f"Warmup steps: {self.config.warmup_steps}")
        logger.info(f"Max grad norm: {self.config.max_grad_norm}")
        logger.info(f"Parameter groups: {len(self.param_groups)}")
        logger.info("================================")

    def step(self) -> Dict[str, Any]:
        """
        Perform one optimizer and scheduler step.

        Returns:
            Dictionary with step information
        """
        step_info = {"lr": self.current_lr, "grad_norm": None, "skipped": False}

        # Get current learning rate from scheduler
        if hasattr(self.scheduler, "get_last_lr"):
            last_lr = self.scheduler.get_last_lr()
            if last_lr:
                self.current_lr = last_lr[0]
                step_info["lr"] = self.current_lr

        return step_info

    def zero_grad(self):
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad(set_to_none=True)

    def clip_gradients(self):
        """Clip gradients to prevent explosion."""
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.max_grad_norm
            )

    def get_state_dict(self) -> Dict[str, Any]:
        """Get optimizer and scheduler state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if hasattr(self.scheduler, "state_dict")
            else None,
            "current_lr": self.current_lr,
            "config": {
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
                "learning_rate": self.config.learning_rate,
                "mixed_precision": self.config.mixed_precision,
                "max_grad_norm": self.config.max_grad_norm,
                "warmup_steps": self.config.warmup_steps,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer and scheduler state from checkpoint."""
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            logger.info("Loaded optimizer state")

        if "scheduler" in state_dict and state_dict["scheduler"] is not None:
            if hasattr(self.scheduler, "load_state_dict"):
                self.scheduler.load_state_dict(state_dict["scheduler"])
                logger.info("Loaded scheduler state")

        if "current_lr" in state_dict:
            self.current_lr = state_dict["current_lr"]
            logger.info(f"Restored learning rate: {self.current_lr}")

    def save_hyperparameters(self, path: str):
        """Save optimizer hyperparameters to file."""
        import json
        from pathlib import Path

        hyperparams = {
            "optimizer": self.config.optimizer,
            "scheduler": self.config.scheduler,
            "learning_rate": self.config.learning_rate,
            "mixed_precision": self.config.mixed_precision,
            "max_grad_norm": self.config.max_grad_norm,
            "warmup_steps": self.config.warmup_steps,
            "total_steps": self.total_steps,
            "parameter_groups": [
                {
                    "name": group.get("name", "unknown"),
                    "param_count": len(group["params"]),
                    "lr_multiplier": group.get("lr", 1.0),
                }
                for group in self.param_groups
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(hyperparams, f, indent=2)

        logger.info(f"Saved optimizer hyperparameters to {path}")
