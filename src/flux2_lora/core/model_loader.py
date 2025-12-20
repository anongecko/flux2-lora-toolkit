"""
Model loading utilities for Flux2-dev LoRA training.

This module provides utilities for loading Flux2-dev model and
injecting LoRA adapters using PEFT.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import FluxPipeline
from diffusers.utils import is_accelerate_available
from peft import get_peft_model, set_peft_model_state_dict
from rich.console import Console

from .lora_config import FluxLoRAConfig
from ..utils.hardware_utils import hardware_manager

console = Console()
logger = logging.getLogger(__name__)


class ModelLoader:
    """Utilities for loading Flux2-dev model with LoRA support."""

    def __init__(self):
        """Initialize model loader."""
        self._model_cache = {}
        self._device_cache = None

    @staticmethod
    def load_flux2_dev(
        model_name: str = "/path/to/black-forest-labs/FLUX.2-dev",
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        torch_compile: bool = True,
        attention_implementation: str = "default",
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
    ) -> Tuple[FluxPipeline, Dict[str, any]]:
        """Load Flux2-dev model and prepare for LoRA training.

        Args:
            model_name: HuggingFace model name or path
            dtype: Data type for model weights
            device: Target device (auto-detect if None)
            cache_dir: Model cache directory
            torch_compile: Whether to compile model with torch.compile
            attention_implementation: Attention implementation to use
            low_cpu_mem_usage: Whether to use low CPU memory loading
            use_safetensors: Whether to prefer safetensors format

        Returns:
            Tuple of (pipeline, metadata)

        Raises:
            RuntimeError: If model loading fails
            ValueError: If invalid parameters provided
        """
        console.print(f"[bold blue]Loading Flux2-dev model: {model_name}[/bold blue]")

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                gpu_id = hardware_manager.select_best_gpu()
                device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
            else:
                device = "cpu"

        console.print(f"Using device: {device}")
        console.print(f"Using dtype: {dtype}")

        # Validate dtype compatibility
        if dtype == torch.bfloat16 and device != "cpu":
            if not torch.cuda.is_bf16_supported():
                console.print(
                    "[yellow]Warning: bfloat16 not supported, falling back to float16[/yellow]"
                )
                dtype = torch.float16

        # Prepare loading kwargs
        loading_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": use_safetensors,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }

        if cache_dir:
            loading_kwargs["cache_dir"] = cache_dir

        # Set attention implementation
        if attention_implementation != "default":
            loading_kwargs["variant"] = attention_implementation

        try:
            # Load the pipeline
            pipeline = FluxPipeline.from_pretrained(model_name, **loading_kwargs)

            # Move to device
            pipeline = pipeline.to(device)

            # Enable memory efficient attention if available
            if attention_implementation == "flash_attention_2":
                try:
                    pipeline.transformer.enable_xformers_memory_efficient_attention()
                    console.print("[green]Enabled Flash Attention 2[/green]")
                except Exception as e:
                    console.print(f"[yellow]Flash Attention 2 not available: {e}[/yellow]")

            # Compile model for better performance
            if torch_compile and device != "cpu":
                try:
                    pipeline.transformer = torch.compile(
                        pipeline.transformer, mode="reduce-overhead"
                    )
                    console.print("[green]Model compiled with torch.compile[/green]")
                except Exception as e:
                    console.print(f"[yellow]torch.compile failed: {e}[/yellow]")

            # Get model metadata
            metadata = ModelLoader._get_model_metadata(pipeline, device, dtype)

            console.print(f"[green]✓ Model loaded successfully[/green]")
            console.print(f"  Parameters: {metadata['total_parameters']:,}")
            console.print(f"  Memory: {metadata['memory_gb']:.1f}GB")

            return pipeline, metadata

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise RuntimeError(f"Failed to load Flux2-dev model: {e}")

    @staticmethod
    def _get_model_metadata(
        pipeline: FluxPipeline, device: str, dtype: torch.dtype
    ) -> Dict[str, any]:
        """Get metadata about loaded model.

        Args:
            pipeline: Loaded Flux pipeline
            device: Device model is on
            dtype: Data type of model

        Returns:
            Dictionary with model metadata
        """
        # Count parameters
        total_params = sum(p.numel() for p in pipeline.transformer.parameters())
        trainable_params = sum(
            p.numel() for p in pipeline.transformer.parameters() if p.requires_grad
        )

        # Estimate memory usage
        if device.startswith("cuda"):
            if torch.cuda.is_available():
                gpu_id = int(device.split(":")[-1]) if ":" in device else 0
                memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
            else:
                memory_used = 0
        else:
            memory_used = 0

        return {
            "model_name": "/path/to/black-forest-labs/FLUX.2-dev",
            "device": device,
            "dtype": str(dtype),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_gb": memory_used,
            "has_lora": False,  # Will be updated after LoRA injection
        }

    @staticmethod
    def inject_lora(
        pipeline: FluxPipeline,
        lora_config: FluxLoRAConfig,
        adapter_name: str = "default",
    ) -> Tuple[FluxPipeline, Dict[str, any]]:
        """Inject LoRA adapters into Flux2-dev model.

        Args:
            pipeline: Loaded Flux pipeline
            lora_config: LoRA configuration
            adapter_name: Name for the LoRA adapter

        Returns:
            Tuple of (pipeline_with_lora, injection_metadata)

        Raises:
            RuntimeError: If LoRA injection fails
        """
        console.print(f"[bold blue]Injecting LoRA adapters[/bold blue]")
        console.print(f"  Rank: {lora_config.rank}")
        console.print(f"  Alpha: {lora_config.alpha}")
        console.print(f"  Target modules: {len(lora_config.target_modules)}")

        try:
            # Convert to PEFT config
            peft_config = lora_config.to_peft_config()

            # Apply LoRA to the transformer
            pipeline.transformer = get_peft_model(pipeline.transformer, peft_config)

            # Set adapter name
            if hasattr(pipeline.transformer, "set_adapter"):
                pipeline.transformer.set_adapter(adapter_name)

            # Freeze base model parameters
            ModelLoader._freeze_base_model(pipeline.transformer)

            # Enable training mode for LoRA parameters
            pipeline.transformer.train()

            # Get injection metadata
            injection_metadata = ModelLoader._get_injection_metadata(
                pipeline.transformer, lora_config
            )

            console.print(f"[green]✓ LoRA injected successfully[/green]")
            console.print(f"  LoRA parameters: {injection_metadata['lora_parameters']:,}")
            console.print(f"  Total trainable: {injection_metadata['trainable_parameters']:,}")
            console.print(f"  Memory overhead: {injection_metadata['memory_overhead_mb']:.1f}MB")

            return pipeline, injection_metadata

        except Exception as e:
            console.print(f"[red]Error injecting LoRA: {e}[/red]")
            raise RuntimeError(f"Failed to inject LoRA adapters: {e}")

    @staticmethod
    def _freeze_base_model(model):
        """Freeze all non-LoRA parameters in the model.

        Args:
            model: Model with LoRA adapters
        """
        for name, param in model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

    @staticmethod
    def _get_injection_metadata(model, lora_config: FluxLoRAConfig) -> Dict[str, any]:
        """Get metadata about LoRA injection.

        Args:
            model: Model with LoRA adapters
            lora_config: LoRA configuration used

        Returns:
            Dictionary with injection metadata
        """
        # Count LoRA parameters
        lora_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
                if "lora" in name.lower():
                    lora_params += param.numel()

        # Get list of LoRA modules
        lora_modules = []
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                lora_modules.append(name)

        return {
            "lora_config": lora_config.to_dict(),
            "lora_parameters": lora_params,
            "trainable_parameters": trainable_params,
            "lora_modules": lora_modules,
            "memory_overhead_mb": lora_config.get_memory_overhead_mb(),
            "injection_successful": True,
        }

    @staticmethod
    def verify_lora_injection(model) -> Dict[str, any]:
        """Verify LoRA parameters and print statistics.

        Args:
            model: Model to verify

        Returns:
            Dictionary with verification results
        """
        console.print("[bold blue]Verifying LoRA injection[/bold blue]")

        total_params = 0
        trainable_params = 0
        lora_params = 0
        frozen_params = 0

        lora_module_names = []

        for name, param in model.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                if "lora" in name.lower():
                    lora_params += param.numel()
            else:
                frozen_params += param.numel()

        # Check for LoRA modules
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                lora_module_names.append(name)

        # Verification results
        verification = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "lora_parameters": lora_params,
            "frozen_parameters": frozen_params,
            "lora_modules": lora_module_names,
            "lora_module_count": len(lora_module_names),
            "trainable_percentage": (trainable_params / total_params) * 100,
            "lora_percentage": (lora_params / total_params) * 100,
            "base_model_frozen": frozen_params > 0,
            "has_lora_modules": len(lora_module_names) > 0,
        }

        # Print verification results
        console.print(f"Total parameters: {total_params:,}")
        console.print(
            f"Trainable parameters: {trainable_params:,} ({verification['trainable_percentage']:.2f}%)"
        )
        console.print(f"LoRA parameters: {lora_params:,} ({verification['lora_percentage']:.2f}%)")
        console.print(f"Frozen parameters: {frozen_params:,}")
        console.print(f"LoRA modules: {verification['lora_module_count']}")

        # Verification checks
        issues = []

        if not verification["has_lora_modules"]:
            issues.append("No LoRA modules found")

        if verification["lora_parameters"] == 0:
            issues.append("No LoRA parameters are trainable")

        if verification["trainable_parameters"] != verification["lora_parameters"]:
            issues.append("Non-LoRA parameters are trainable")

        if verification["lora_percentage"] > 5.0:
            issues.append(
                f"LoRA parameter percentage too high: {verification['lora_percentage']:.2f}%"
            )

        if issues:
            console.print("[red]Verification issues found:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
            verification["verification_passed"] = False
        else:
            console.print("[green]✓ LoRA injection verification passed[/green]")
            verification["verification_passed"] = True

        return verification

    @staticmethod
    def test_forward_pass(
        pipeline: FluxPipeline,
        prompt: str = "A photo of a cat",
        num_inference_steps: int = 4,  # Minimal steps for testing
        guidance_scale: float = 7.0,
        height: int = 512,
        width: int = 512,
    ) -> Dict[str, any]:
        """Test forward pass with LoRA active.

        Args:
            pipeline: Pipeline with LoRA adapters
            prompt: Test prompt
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            height: Image height
            width: Image width

        Returns:
            Dictionary with test results

        Raises:
            RuntimeError: If forward pass fails
        """
        console.print(f"[bold blue]Testing forward pass[/bold blue]")
        console.print(f"Prompt: '{prompt}'")

        try:
            # Set model to eval mode for inference
            pipeline.transformer.eval()

            # Run inference
            with torch.no_grad():
                result = pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    return_dict=True,
                )

            # Get generated image
            images = result.images
            if not images:
                raise RuntimeError("No images generated")

            # Test results
            test_results = {
                "success": True,
                "prompt": prompt,
                "num_images": len(images),
                "image_size": f"{width}x{height}",
                "inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "error": None,
            }

            console.print(f"[green]✓ Forward pass successful[/green]")
            console.print(f"  Generated {len(images)} image(s)")
            console.print(f"  Image size: {width}x{height}")

            return test_results

        except Exception as e:
            console.print(f"[red]Forward pass failed: {e}[/red]")
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e),
                "num_images": 0,
            }
        finally:
            # Set back to train mode
            pipeline.transformer.train()

    @staticmethod
    def save_lora_weights(
        pipeline: FluxPipeline,
        output_path: str,
        adapter_name: str = "default",
    ) -> Dict[str, any]:
        """Save LoRA weights to disk.

        Args:
            pipeline: Pipeline with LoRA adapters
            output_path: Path to save LoRA weights
            adapter_name: Name of adapter to save

        Returns:
            Dictionary with save results
        """
        console.print(f"[bold blue]Saving LoRA weights[/bold blue]")
        console.print(f"Output path: {output_path}")

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save LoRA weights
            if hasattr(pipeline.transformer, "save_pretrained"):
                pipeline.transformer.save_pretrained(
                    output_path,
                    safe_serialization=True,
                    selected_adapters=[adapter_name],
                )
            else:
                # Fallback for older PEFT versions
                lora_state_dict = pipeline.transformer.state_dict()
                lora_only = {k: v for k, v in lora_state_dict.items() if "lora" in k}
                torch.save(lora_only, output_path / "lora_weights.safetensors")

            console.print(f"[green]✓ LoRA weights saved[/green]")

            return {
                "success": True,
                "output_path": str(output_path),
                "adapter_name": adapter_name,
                "error": None,
            }

        except Exception as e:
            console.print(f"[red]Error saving LoRA weights: {e}[/red]")
            return {
                "success": False,
                "output_path": output_path,
                "adapter_name": adapter_name,
                "error": str(e),
            }

    @staticmethod
    def load_lora_weights(
        pipeline: FluxPipeline,
        lora_path: str,
        adapter_name: str = "default",
    ) -> Dict[str, any]:
        """Load LoRA weights from disk.

        Args:
            pipeline: Pipeline to load LoRA into
            lora_path: Path to LoRA weights
            adapter_name: Name for loaded adapter

        Returns:
            Dictionary with load results
        """
        console.print(f"[bold blue]Loading LoRA weights[/bold blue]")
        console.print(f"LoRA path: {lora_path}")

        try:
            lora_path = Path(lora_path)

            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA path not found: {lora_path}")

            # Load LoRA weights
            if hasattr(pipeline.transformer, "load_adapter"):
                pipeline.transformer.load_adapter(lora_path, adapter_name)
            else:
                # Fallback for older PEFT versions
                lora_weights = torch.load(lora_path / "lora_weights.safetensors")
                set_peft_model_state_dict(pipeline.transformer, lora_weights, adapter_name)

            console.print(f"[green]✓ LoRA weights loaded[/green]")

            return {
                "success": True,
                "lora_path": str(lora_path),
                "adapter_name": adapter_name,
                "error": None,
            }

        except Exception as e:
            console.print(f"[red]Error loading LoRA weights: {e}[/red]")
            return {
                "success": False,
                "lora_path": lora_path,
                "adapter_name": adapter_name,
                "error": str(e),
            }

    @staticmethod
    def clear_model_cache():
        """Clear model cache and free memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        console.print("[green]Model cache cleared[/green]")


# Global model loader instance
model_loader = ModelLoader()
