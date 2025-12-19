"""
LoRA configuration for Flux2-dev training.

This module provides LoRA-specific configuration classes and utilities
for configuring LoRA adapters on Flux2-dev model.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from peft import LoraConfig, TaskType


@dataclass
class FluxLoRAConfig:
    """Flux2-dev specific LoRA configuration.
    
    This class extends the standard PEFT LoRA configuration with
    Flux2-dev specific settings and validation.
    """
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "to_k",
            "to_q", 
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "add_out_proj"
        ]
    )
    use_dora: bool = False
    bias: str = "none"
    init_lora_weights: bool = True
    
    def __post_init__(self):
        """Validate LoRA configuration parameters."""
        if not (1 <= self.rank <= 256):
            raise ValueError(f"rank must be between 1 and 256, got {self.rank}")
        
        if not (0.1 <= self.alpha <= 128.0):
            raise ValueError(f"alpha must be between 0.1 and 128.0, got {self.alpha}")
        
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got {self.dropout}")
        
        valid_bias = ["none", "all", "lora_only"]
        if self.bias not in valid_bias:
            raise ValueError(f"bias must be one of {valid_bias}, got {self.bias}")
        
        # Validate target modules
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
    
    def to_peft_config(self, task_type: str = "FEATURE_EXTRACTION") -> LoraConfig:
        """Convert to PEFT LoraConfig.
        
        Args:
            task_type: Task type for PEFT configuration
            
        Returns:
            PEFT LoraConfig object
        """
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=task_type,
            init_lora_weights=self.init_lora_weights,
            use_dora=self.use_dora,
        )
    
    def get_trainable_parameters_count(self, base_model_params: int) -> int:
        """Estimate number of trainable LoRA parameters.
        
        Args:
            base_model_params: Number of parameters in base model
            
        Returns:
            Estimated number of LoRA parameters
        """
        # LoRA adds two matrices (A and B) for each target module
        # A: rank x in_features, B: out_features x rank
        # Simplified estimation based on rank and number of target modules
        params_per_adapter = 2 * self.rank * 768  # Assuming 768 hidden size
        return params_per_adapter * len(self.target_modules)
    
    def get_memory_overhead_mb(self) -> float:
        """Estimate memory overhead for LoRA adapters.
        
        Returns:
            Estimated memory overhead in MB
        """
        # Rough estimation: 4 bytes per parameter (float32)
        param_count = self.get_trainable_parameters_count(0)
        return (param_count * 4) / (1024 * 1024)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FluxLoRAConfig":
        """Create FluxLoRAConfig from dictionary.
        
        Args:
            config_dict: Dictionary with LoRA configuration
            
        Returns:
            FluxLoRAConfig object
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "use_dora": self.use_dora,
            "bias": self.bias,
            "init_lora_weights": self.init_lora_weights,
        }
    
    def copy(self, **overrides) -> "FluxLoRAConfig":
        """Create a copy with optional overrides.
        
        Args:
            **overrides: Parameters to override
            
        Returns:
            New FluxLoRAConfig object
        """
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)


class LoRAConfigPresets:
    """Predefined LoRA configuration presets."""
    
    @staticmethod
    def character_lora() -> FluxLoRAConfig:
        """Preset for character LoRA training.
        
        Character LoRAs typically need higher rank to capture fine details.
        """
        return FluxLoRAConfig(
            rank=32,
            alpha=32.0,
            dropout=0.1,
            target_modules=[
                "to_k",
                "to_q",
                "to_v", 
                "to_out.0",
            ],
            use_dora=False,
        )
    
    @staticmethod
    def style_lora() -> FluxLoRAConfig:
        """Preset for style LoRA training.
        
        Style LoRAs can use lower rank as they learn broader patterns.
        """
        return FluxLoRAConfig(
            rank=16,
            alpha=16.0,
            dropout=0.05,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_q_proj", 
                "add_v_proj",
                "add_out_proj"
            ],
            use_dora=True,  # DoRA helps with style consistency
        )
    
    @staticmethod
    def concept_lora() -> FluxLoRAConfig:
        """Preset for concept LoRA training.
        
        Concept LoRAs balance between character and style requirements.
        """
        return FluxLoRAConfig(
            rank=24,
            alpha=24.0,
            dropout=0.0,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ],
            use_dora=False,
        )
    
    @staticmethod
    def lightweight_lora() -> FluxLoRAConfig:
        """Preset for lightweight LoRA with minimal memory usage.
        
        Useful for quick experiments or limited GPU memory.
        """
        return FluxLoRAConfig(
            rank=8,
            alpha=8.0,
            dropout=0.0,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
            ],
            use_dora=False,
        )
    
    @staticmethod
    def high_quality_lora() -> FluxLoRAConfig:
        """Preset for high-quality LoRA with maximum expressiveness.
        
        Requires significant GPU memory but provides best results.
        """
        return FluxLoRAConfig(
            rank=64,
            alpha=64.0,
            dropout=0.1,
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_q_proj",
                "add_v_proj", 
                "add_out_proj"
            ],
            use_dora=True,
        )
    
    @classmethod
    def get_preset(cls, preset_name: str) -> FluxLoRAConfig:
        """Get preset configuration by name.
        
        Args:
            preset_name: Name of preset (character, style, concept, lightweight, high_quality)
            
        Returns:
            FluxLoRAConfig object
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        presets = {
            "character": cls.character_lora,
            "style": cls.style_lora,
            "concept": cls.concept_lora,
            "lightweight": cls.lightweight_lora,
            "high_quality": cls.high_quality_lora,
        }
        
        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        
        return presets[preset_name]()
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available preset names.
        
        Returns:
            List of preset names
        """
        return ["character", "style", "concept", "lightweight", "high_quality"]


def validate_lora_config(config: FluxLoRAConfig) -> List[str]:
    """Validate LoRA configuration and return warnings.
    
    Args:
        config: LoRA configuration to validate
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check for potential issues
    if config.rank > 32:
        warnings.append("High rank may increase overfitting risk and memory usage")
    
    if config.alpha / config.rank > 2.0:
        warnings.append("High alpha/rank ratio may cause training instability")
    
    if config.alpha / config.rank < 0.5:
        warnings.append("Low alpha/rank ratio may result in weak LoRA effect")
    
    if config.dropout > 0.2:
        warnings.append("High dropout may impair LoRA learning")
    
    if len(config.target_modules) < 4:
        warnings.append("Few target modules may limit LoRA expressiveness")
    
    if config.use_dora and config.rank < 16:
        warnings.append("DoRA with low rank may not provide significant benefits")
    
    return warnings


def estimate_lora_memory_usage(config: FluxLoRAConfig, batch_size: int = 4) -> Dict[str, float]:
    """Estimate memory usage for LoRA training.
    
    Args:
        config: LoRA configuration
        batch_size: Training batch size
        
    Returns:
        Dictionary with memory estimates in MB
    """
    # Base model memory (rough estimate for Flux2-dev)
    base_model_memory = 24000  # MB for bfloat16
    
    # LoRA parameters memory
    lora_params_memory = config.get_memory_overhead_mb()
    
    # Gradient memory (same as parameters for Adam)
    gradient_memory = lora_params_memory
    
    # Optimizer state memory (2x parameters for AdamW)
    optimizer_memory = lora_params_memory * 2
    
    # Activation memory (depends on batch size and resolution)
    activation_memory = batch_size * 1024 * 1024 * 4 / (1024 * 1024)  # Rough estimate
    
    # Total memory
    total_memory = (
        base_model_memory +
        lora_params_memory +
        gradient_memory +
        optimizer_memory +
        activation_memory
    )
    
    return {
        "base_model_mb": base_model_memory,
        "lora_parameters_mb": lora_params_memory,
        "gradients_mb": gradient_memory,
        "optimizer_mb": optimizer_memory,
        "activations_mb": activation_memory,
        "total_mb": total_memory,
    }