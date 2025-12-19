"""
Flux2-dev LoRA Training Toolkit

A comprehensive toolkit for training high-quality LoRA models for Flux2-dev
with real-time monitoring, automatic quality assessment, and an intuitive interface.
"""

__version__ = "0.1.0"
__author__ = "Flux2 LoRA Toolkit Team"
__email__ = "contact@example.com"

# Import key classes and functions for easy access
from flux2_lora.utils.config_manager import (
    Config,
    ConfigManager,
    config_manager,
)
from flux2_lora.utils.hardware_utils import (
    HardwareManager,
    hardware_manager,
    GPUInfo,
    SystemInfo,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Config",
    "ConfigManager", 
    "config_manager",
    "HardwareManager",
    "hardware_manager",
    "GPUInfo",
    "SystemInfo",
]