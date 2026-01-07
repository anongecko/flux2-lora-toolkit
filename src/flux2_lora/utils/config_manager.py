"""
Configuration management utilities for Flux2 LoRA training.

This module provides configuration loading, validation, and management
functionality for training LoRA models on Flux2-dev.
"""

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jsonschema import Draft7Validator, ValidationError
from rich.console import Console

console = Console()


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    base_model: str = "/path/to/black-forest-labs/FLUX.2-dev"
    dtype: str = "bfloat16"
    device: str = "cuda"
    cache_dir: Optional[str] = None
    torch_compile: bool = True
    attention_implementation: str = "default"

    def __post_init__(self):
        """Validate model configuration."""
        valid_dtypes = ["float32", "float16", "bfloat16"]
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")

        valid_attention = ["default", "flash_attention_2", "xformers"]
        if self.attention_implementation not in valid_attention:
            raise ValueError(
                f"attention_implementation must be one of {valid_attention}, "
                f"got {self.attention_implementation}"
            )


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"])
    use_dora: bool = False
    trigger_word: Optional[str] = None

    def __post_init__(self):
        """Validate LoRA configuration."""
        if not (1 <= self.rank <= 256):
            raise ValueError(f"rank must be between 1 and 256, got {self.rank}")

        if not (0.0 <= self.alpha <= 128.0):
            raise ValueError(f"alpha must be between 0.0 and 128.0, got {self.alpha}")

        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    learning_rate: float = 1e-4
    batch_size: int = 4
    max_steps: int = 1000
    gradient_accumulation_steps: int = 4
    optimizer: str = "adamw"
    scheduler: str = "constant"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    seed: int = 42

    def __post_init__(self):
        """Validate training configuration."""
        if not (1e-7 <= self.learning_rate <= 1e-1):
            raise ValueError(
                f"learning_rate must be between 1e-7 and 1e-1, got {self.learning_rate}"
            )

        if not (1 <= self.batch_size <= 32):
            raise ValueError(f"batch_size must be between 1 and 32, got {self.batch_size}")

        if not (1 <= self.max_steps <= 100000):
            raise ValueError(f"max_steps must be between 1 and 100000, got {self.max_steps}")


@dataclass
class DataConfig:
    """Data configuration parameters."""

    dataset_path: str = "./dataset"
    resolution: int = 1024
    caption_format: str = "auto"
    center_crop: bool = True
    random_flip: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    cache_images: bool = False
    validate_captions: bool = True
    min_caption_length: int = 3
    max_caption_length: int = 1000

    def __post_init__(self):
        """Validate data configuration."""
        valid_resolutions = [256, 512, 768, 1024, 1536, 2048]
        if self.resolution not in valid_resolutions:
            raise ValueError(
                f"resolution must be one of {valid_resolutions}, got {self.resolution}"
            )

        valid_formats = ["txt", "caption", "json", "auto"]
        if self.caption_format not in valid_formats:
            raise ValueError(
                f"caption_format must be one of {valid_formats}, got {self.caption_format}"
            )

        if not (0 <= self.num_workers <= 16):
            raise ValueError(f"num_workers must be between 0 and 16, got {self.num_workers}")


@dataclass
class ValidationConfig:
    """Validation configuration parameters."""

    enable: bool = True
    prompts: List[str] = field(
        default_factory=lambda: [
            "A photo of [TRIGGER_WORD] smiling",
            "A portrait in natural lighting",
            "A close-up photo",
            "[TRIGGER_WORD] wearing a black t-shirt",
            "A candid shot outdoors",
            "A professional headshot",
            "[TRIGGER_WORD] with different expressions",
        ]
    )
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    every_n_steps: int = 100
    num_samples: int = 1

    def __post_init__(self):
        """Validate validation configuration."""
        if not (1 <= self.num_inference_steps <= 100):
            raise ValueError(
                f"num_inference_steps must be between 1 and 100, got {self.num_inference_steps}"
            )

        if not (0.0 <= self.guidance_scale <= 20.0):
            raise ValueError(
                f"guidance_scale must be between 0.0 and 20.0, got {self.guidance_scale}"
            )


@dataclass
class OutputConfig:
    """Output configuration parameters."""

    output_dir: str = "./output"
    checkpoint_every_n_steps: int = 500
    checkpoints_limit: int = 5
    save_model_config: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration parameters.

    Uses nested dict structure to match the data.augmentation module.
    """

    enabled: bool = False
    probability: float = 0.5
    preserve_quality: bool = True
    max_augmentations_per_sample: int = 3

    # Nested augmentation configs (compatible with data.augmentation module)
    image_augmentations: Dict[str, Any] = field(default_factory=dict)
    text_augmentations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate augmentation configuration."""
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be between 0.0 and 1.0, got {self.probability}")

        if not (1 <= self.max_augmentations_per_sample <= 10):
            raise ValueError(
                f"max_augmentations_per_sample must be between 1 and 10, got {self.max_augmentations_per_sample}"
            )


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""

    log_level: str = "INFO"
    log_dir: str = "./logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "flux2-lora-training"
    log_every_n_steps: int = 10
    enable_quality_metrics: bool = True

    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")


@dataclass
class SecurityConfig:
    """Security and resource limits configuration."""

    max_file_size_mb: int = 50
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".txt", ".json"]
    )
    max_training_time_hours: float = 24.0
    memory_limit_gb: int = 80

    def __post_init__(self):
        """Validate security configuration."""
        if not (1 <= self.max_file_size_mb <= 1000):
            raise ValueError(
                f"max_file_size_mb must be between 1 and 1000, got {self.max_file_size_mb}"
            )

        if not (0.1 <= self.max_training_time_hours <= 168.0):
            raise ValueError(
                f"max_training_time_hours must be between 0.1 and 168.0, got {self.max_training_time_hours}"
            )


@dataclass
class QuantizationConfig:
    """Quantization (QLoRA) configuration for memory optimization."""

    enabled: bool = False
    bits: int = 8  # 8 or 4
    compute_dtype: str = "bfloat16"

    def __post_init__(self):
        """Validate quantization configuration."""
        if self.bits not in [4, 8]:
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")

        valid_dtypes = ["float16", "bfloat16", "float32"]
        if self.compute_dtype not in valid_dtypes:
            raise ValueError(
                f"compute_dtype must be one of {valid_dtypes}, got {self.compute_dtype}"
            )


@dataclass
class MemoryOptimizationConfig:
    """Memory optimization configuration for large models like Flux2-dev."""

    # Quantization settings
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Attention optimization
    enable_attention_slicing: bool = True
    attention_slice_size: str = "auto"  # "auto", "max", or specific number

    # VAE optimization
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True

    # CPU offloading (fallback, very slow)
    sequential_cpu_offload: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryOptimizationConfig":
        """Create MemoryOptimizationConfig from dictionary."""
        quantization_dict = config_dict.get("quantization", {})
        quantization_config = QuantizationConfig(**quantization_dict)

        return cls(
            quantization=quantization_config,
            enable_attention_slicing=config_dict.get("enable_attention_slicing", True),
            attention_slice_size=config_dict.get("attention_slice_size", "auto"),
            enable_vae_slicing=config_dict.get("enable_vae_slicing", True),
            enable_vae_tiling=config_dict.get("enable_vae_tiling", True),
            sequential_cpu_offload=config_dict.get("sequential_cpu_offload", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quantization": self.quantization.__dict__,
            "enable_attention_slicing": self.enable_attention_slicing,
            "attention_slice_size": self.attention_slice_size,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_vae_tiling": self.enable_vae_tiling,
            "sequential_cpu_offload": self.sequential_cpu_offload,
        }


@dataclass
class CallbacksConfig:
    """Callback configuration parameters."""

    enable_checkpoint: bool = True
    checkpoint_every_n_steps: int = 500
    checkpoint_save_best_only: bool = False
    checkpoint_monitor_metric: str = "loss"
    checkpoint_save_top_k: int = 3

    enable_early_stopping: bool = False
    early_stopping_monitor: str = "validation_loss"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_restore_best: bool = True

    enable_validation_callback: bool = True
    validation_callback_every_n_steps: int = 100
    validation_callback_log_images: bool = True

    enable_lr_scheduler_callback: bool = True
    lr_scheduler_step_interval: str = "step"

    def __post_init__(self):
        """Validate callbacks configuration."""
        if self.checkpoint_every_n_steps <= 0:
            raise ValueError("checkpoint_every_n_steps must be > 0")

        if self.checkpoint_save_top_k <= 0:
            raise ValueError("checkpoint_save_top_k must be > 0")

        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0")

        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")

        valid_monitors = ["loss", "validation_loss"]
        if self.early_stopping_monitor not in valid_monitors:
            raise ValueError(f"early_stopping_monitor must be one of {valid_monitors}")

        if self.validation_callback_every_n_steps <= 0:
            raise ValueError("validation_callback_every_n_steps must be > 0")

        valid_intervals = ["step", "epoch"]
        if self.lr_scheduler_step_interval not in valid_intervals:
            raise ValueError(f"lr_scheduler_step_interval must be one of {valid_intervals}")


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    memory_optimization: MemoryOptimizationConfig = field(
        default_factory=MemoryOptimizationConfig
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Extract sub-configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        lora_config = LoRAConfig(**config_dict.get("lora", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        validation_config = ValidationConfig(**config_dict.get("validation", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        security_config = SecurityConfig(**config_dict.get("security", {}))
        callbacks_config = CallbacksConfig(**config_dict.get("callbacks", {}))
        augmentation_config = AugmentationConfig(**config_dict.get("augmentation", {}))

        # Parse memory_optimization with nested quantization
        memory_opt_dict = config_dict.get("memory_optimization", {})
        memory_optimization_config = MemoryOptimizationConfig.from_dict(memory_opt_dict)

        return cls(
            model=model_config,
            lora=lora_config,
            training=training_config,
            data=data_config,
            validation=validation_config,
            output=output_config,
            logging=logging_config,
            security=security_config,
            callbacks=callbacks_config,
            augmentation=augmentation_config,
            memory_optimization=memory_optimization_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "validation": self.validation.__dict__,
            "output": self.output.__dict__,
            "logging": self.logging.__dict__,
            "security": self.security.__dict__,
            "callbacks": self.callbacks.__dict__,
            "augmentation": self.augmentation.__dict__,
            "memory_optimization": self.memory_optimization.to_dict(),
        }


class ConfigManager:
    """Configuration manager for loading and validating configs."""

    def __init__(self, schema_path: Optional[str] = None):
        """Initialize config manager.

        Args:
            schema_path: Path to JSON schema file. If None, uses default schema.
        """
        if schema_path is None:
            # Use schema file in configs directory
            current_dir = Path(__file__).parent.parent.parent.parent
            self.schema_path = current_dir / "configs" / "schema.json"
        else:
            self.schema_path = Path(schema_path)

        self._load_schema()

    def _load_schema(self):
        """Load JSON schema for validation."""
        try:
            with open(self.schema_path, "r") as f:
                self._schema = json.load(f)
                self._validator = Draft7Validator(self._schema)
        except Exception as e:
            console.print(f"[red]Error loading schema: {e}[/red]")
            self._schema = None
            self._validator = None

    def load_config(self, config_path: str) -> Config:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
            ValueError: If config values are invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

                # Convert scientific notation strings to floats
                def convert_scientific_notation(obj):
                    if isinstance(obj, dict):
                        return {k: convert_scientific_notation(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_scientific_notation(item) for item in obj]
                    elif isinstance(obj, str) and "e" in obj.lower():
                        try:
                            return float(obj)
                        except ValueError:
                            return obj
                    return obj

                config_dict = convert_scientific_notation(config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")

        # Validate against schema if available
        if self._validator is not None:
            self._validate_schema(config_dict, str(config_path))

        # Create Config object (this will also validate individual fields)
        return Config.from_dict(config_dict)

    def _validate_schema(self, config_dict: Dict[str, Any], config_path: str):
        """Validate configuration against JSON schema."""
        try:
            self._validator.validate(config_dict)
        except ValidationError as e:
            console.print(f"[red]Configuration validation error in {config_path}:[/red]")
            if hasattr(e, "errors"):
                for error in e.errors:
                    console.print(f"  • {error}")
            else:
                console.print(f"  • {e.message}")
            raise ValueError(f"Configuration validation failed: {e.message}")

    def validate_config_values(self, config: Config) -> List[str]:
        """Validate configuration values and return list of warnings."""
        warnings = []

        # Check for common configuration issues
        if config.training.learning_rate > 1e-3:
            warnings.append("High learning rate may cause training instability")

        if config.training.batch_size > 8 and not config.training.gradient_checkpointing:
            warnings.append("Large batch size may cause GPU memory issues")

        if config.lora.rank > 64:
            warnings.append("High LoRA rank may increase overfitting risk")

        if config.data.resolution > 1024:
            warnings.append("High resolution requires more GPU memory")

        if config.validation.every_n_steps > config.training.max_steps // 2:
            warnings.append("Validation frequency too low for effective monitoring")

        return warnings

    def get_preset_config(self, preset_name: str) -> Config:
        """Load preset configuration.

        Args:
            preset_name: Name of preset (character, style, concept)

        Returns:
            Preset Config object

        Raises:
            FileNotFoundError: If preset file doesn't exist
        """
        current_dir = Path(__file__).parent.parent.parent.parent
        preset_path = current_dir / "configs" / "presets" / f"{preset_name}.yaml"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")

        return self.load_config(str(preset_path))

    def list_presets(self) -> List[str]:
        """List available preset configurations.

        Returns:
            List of preset names
        """
        current_dir = Path(__file__).parent.parent.parent.parent
        presets_dir = current_dir / "configs" / "presets"

        if not presets_dir.exists():
            return []

        presets = []
        for preset_file in presets_dir.glob("*.yaml"):
            presets.append(preset_file.stem)

        return presets


# Global config manager instance
config_manager = ConfigManager()
