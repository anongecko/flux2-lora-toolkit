"""
Model loading utilities for Flux2-dev LoRA training.

This module provides utilities for loading Flux2-dev model and
injecting LoRA adapters using PEFT.
"""

import gc
import logging
import os
from pathlib import Path

import torch
from diffusers import Flux2Pipeline, FluxPipeline
from peft import get_peft_model, set_peft_model_state_dict
from rich.console import Console

from ..utils.hardware_utils import hardware_manager
from .lora_config import FluxLoRAConfig

console = Console()
logger = logging.getLogger(__name__)


def _print_memory_diagnostic(device: str = "cuda:0"):
    """Print detailed memory diagnostic information."""
    if not torch.cuda.is_available():
        return

    try:
        gpu_id = int(device.split(":")[-1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(gpu_id)
        total_gb = props.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        free_gb = total_gb - reserved_gb

        console.print("[blue]" + "="*60 + "[/blue]")
        console.print("[bold blue]GPU Memory Diagnostic[/bold blue]")
        console.print(f"GPU: {props.name}")
        console.print(f"Total: {total_gb:.2f}GB | Allocated: {allocated_gb:.2f}GB | Reserved: {reserved_gb:.2f}GB | Free: {free_gb:.2f}GB")
        console.print("[blue]" + "="*60 + "[/blue]")
    except Exception as e:
        console.print(f"[yellow]Could not get memory diagnostic: {e}[/yellow]")


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
        device: str | None = None,
        cache_dir: str | None = None,
        torch_compile: bool = True,
        attention_implementation: str = "default",
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        force_cpu_loading: bool = False,
        _retry_attempt: int = 0,
    ) -> tuple[FluxPipeline, dict[str, any]]:
        """Load Flux2-dev model and prepare for LoRA training.

        Args:
            model_name: HuggingFace model name or path
            dtype: Data type for model weights
             device: Target device ('auto', 'cpu', 'cuda', 'cuda:X', or None for auto-detect)
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
        console.print(f"[bold blue]Loading FLUX model: {model_name}[/bold blue]")

        # Check for existing memory allocations that might indicate unclean state
        if torch.cuda.is_available():
            try:
                existing_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                existing_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                if existing_allocated > 1.0 or existing_reserved > 1.0:  # More than 1GB
                    console.print(
                        f"[yellow]⚠️  Warning: Found existing GPU allocations: {existing_allocated:.2f}GB allocated, {existing_reserved:.2f}GB reserved[/yellow]"
                    )
                    console.print(
                        "[yellow]This may indicate unclean state from previous runs. Consider restarting the process/VM.[/yellow]"
                    )
            except Exception:
                pass

        # Aggressive memory cleanup before loading
        if torch.cuda.is_available():
            console.print("[yellow]Performing aggressive GPU memory cleanup...[/yellow]")

            # Show current memory usage before cleanup
            try:
                current_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                current_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                props = torch.cuda.get_device_properties(0)
                total_memory = props.total_memory / (1024**3)  # GB
                console.print(
                    f"[blue]Memory before cleanup: {current_allocated:.2f}GB allocated, {current_reserved:.2f}GB reserved, {total_memory:.2f}GB total[/blue]"
                )
            except Exception:
                pass

            # Force complete GPU memory reset
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Multiple garbage collection passes
            for _ in range(3):
                gc.collect()

            # Reset CUDA allocator state if possible
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                console.print("[green]✓ Reset CUDA memory statistics[/green]")
            except Exception:
                pass

            # Show memory usage after cleanup
            try:
                after_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                after_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                console.print(
                    f"[blue]Memory after cleanup: {after_allocated:.2f}GB allocated, {after_reserved:.2f}GB reserved[/blue]"
                )
            except Exception:
                pass

            console.print("[green]✓ Aggressive GPU memory cleanup completed[/green]")

        # Clear any existing cached models to prevent memory accumulation
        try:
            from diffusers import utils

            # Clear model cache if it exists
            if hasattr(utils, "MODEL_CACHE"):
                utils.MODEL_CACHE.clear()
                console.print("[green]✓ Cleared diffusers model cache[/green]")
        except Exception:
            pass  # Ignore if cache clearing fails

        # Set memory optimization environment variables for GPU memory management
        if torch.cuda.is_available():
            # Force expandable segments to prevent memory fragmentation on large GPUs
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            console.print(
                "[green]✓ Enabled expandable memory segments for GPU (max_split_size_mb:512)[/green]"
            )

            # Detect H100 GPU and apply specific optimizations
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ""
            if "H100" in gpu_name:
                console.print(f"[green]✓ Detected H100 GPU: {gpu_name}[/green]")
                # H100-specific optimizations are already handled in dtype selection and attention implementation
                console.print("[green]✓ H100 optimizations enabled[/green]")

        # Determine which pipeline to use
        pipeline_class = ModelLoader._detect_flux_pipeline_class(model_name)

        # Validate model directory if it's a local path
        if Path(model_name).exists() and Path(model_name).is_dir():
            if not ModelLoader._validate_flux_model_directory(model_name, pipeline_class):
                raise ValueError(
                    f"Invalid FLUX model directory: {model_name}. "
                    "Please ensure you have downloaded the complete FLUX model files."
                )

        # Auto-detect device if not specified or set to "auto"
        if device is None or device == "auto":
            if torch.cuda.is_available():
                gpu_id = hardware_manager.select_best_gpu()
                device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda"
            else:
                device = "cpu"

        console.print(f"Using device: {device}")
        console.print(f"Using dtype: {dtype}")
        console.print(f"DEBUG: dtype type = {type(dtype)}, dtype value = {dtype}")

        # Estimate memory requirements for Flux2-dev model
        if device.startswith("cuda"):
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                console.print(f"[blue]DEBUG: Starting memory estimation for dtype={dtype}[/blue]")
                # Accurate estimate based on the TARGET dtype (what model will be after conversion)
                target_dtype = dtype  # This is what the model will be after potential conversion
                if target_dtype == torch.bfloat16:
                    bytes_per_param = 2
                elif target_dtype == torch.float32:
                    bytes_per_param = 4
                elif target_dtype == torch.float16:
                    bytes_per_param = 2
                else:
                    bytes_per_param = 2  # default to 2

                console.print(
                    f"[blue]DEBUG: target_dtype={target_dtype}, bytes_per_param={bytes_per_param}[/blue]"
                )
                model_weights_gb = (32e9 * bytes_per_param) / (1024**3)  # Full model weights

                # For LoRA training, we only need gradients for LoRA parameters (~100M params), not full model
                # But during loading, the full model must be loaded
                lora_params_estimate = 100e6  # Rough estimate for LoRA parameters
                lora_memory_gb = (lora_params_estimate * bytes_per_param * 3) / (
                    1024**3
                )  # params + gradients + optimizer

                # Loading overhead: we load in bfloat16 first, then convert
                # So loading overhead is based on bfloat16 size, but final memory is target dtype
                loading_dtype = torch.bfloat16  # We always load in bfloat16 first
                loading_bytes_per_param = (
                    2
                    if loading_dtype == torch.bfloat16
                    else 4
                    if loading_dtype == torch.float32
                    else 2
                )
                loading_model_gb = (32e9 * loading_bytes_per_param) / (1024**3)

                # Accurate overhead based on loading strategy
                if force_cpu_loading:
                    # CPU→GPU transfer: mainly temporary allocations
                    loading_overhead_gb = loading_model_gb * 0.15  # 15%
                    strategy = "CPU-first (low GPU overhead)"
                elif gpu_memory_gb >= 80:
                    if dtype == torch.bfloat16:
                        loading_overhead_gb = loading_model_gb * 0.20  # 20%
                        strategy = "Direct GPU (native dtype)"
                    else:
                        # Need temp space for conversion, but PyTorch frees as it converts
                        loading_overhead_gb = loading_model_gb * 0.25  # 25%
                        strategy = f"Direct GPU (bf16→{dtype} conversion)"
                else:
                    loading_overhead_gb = loading_model_gb * 0.30  # 30%
                    strategy = "Default loading"

                total_estimated_gb = loading_model_gb + loading_overhead_gb  # Loading memory
                console.print(f"[blue]Strategy: {strategy}[/blue]")
                final_memory_gb = model_weights_gb + (
                    model_weights_gb * 0.1
                )  # Final memory after conversion

                console.print(f"[blue]Estimated peak: {total_estimated_gb:.1f}GB ({loading_model_gb:.1f}GB + {loading_overhead_gb:.1f}GB overhead)[/blue]")
                console.print(
                    f"[blue]Training: Model={model_weights_gb:.1f}GB + LoRA={lora_memory_gb:.1f}GB = {model_weights_gb + lora_memory_gb:.1f}GB total[/blue]"
                )

                loading_percent = (total_estimated_gb / gpu_memory_gb) * 100
                training_percent = ((model_weights_gb + lora_memory_gb) / gpu_memory_gb) * 100

                console.print(
                    f"[blue]Loading memory: {loading_percent:.1f}% of GPU ({total_estimated_gb:.1f}GB peak)[/blue]"
                )
                console.print(
                    f"[blue]Training memory: {training_percent:.1f}% of GPU ({model_weights_gb + lora_memory_gb:.1f}GB sustained)[/blue]"
                )

                available_memory_gb = gpu_memory_gb - (torch.cuda.memory_allocated(0) / (1024**3))

                if total_estimated_gb > available_memory_gb:
                    console.print(f"[red]❌ INSUFFICIENT MEMORY[/red]")
                    console.print(f"  Need: {total_estimated_gb:.1f}GB | Available: {available_memory_gb:.1f}GB | Shortfall: {total_estimated_gb - available_memory_gb:.1f}GB")
                    console.print("[cyan]💡 Solutions:[/cyan]")
                    if dtype == torch.float32:
                        console.print("  1. Use bfloat16/float16 (50% less memory)")
                    console.print("  2. Set force_cpu_loading=True")
                    console.print("  3. Close other GPU programs")
                elif total_estimated_gb > available_memory_gb * 0.85:
                    console.print(f"[yellow]⚠️  TIGHT: {total_estimated_gb:.1f}GB needed, {available_memory_gb:.1f}GB available ({(total_estimated_gb/available_memory_gb)*100:.1f}%)[/yellow]")
                    console.print("[yellow]Loading will proceed but monitor for OOM[/yellow]")
                else:
                    console.print(f"[green]✅ SUFFICIENT: {total_estimated_gb:.1f}GB needed, {available_memory_gb:.1f}GB available[/green]")
            except Exception as e:
                console.print(f"[dim]Could not estimate memory requirements: {e}[/dim]")

        # Validate dtype compatibility and optimize for memory
        if dtype == torch.bfloat16 and device != "cpu":
            if not torch.cuda.is_bf16_supported():
                console.print(
                    "[yellow]Warning: bfloat16 not supported, falling back to float16[/yellow]"
                )
                dtype = torch.float16
        elif dtype == torch.bfloat16 and device.startswith("cuda"):
            # For H100 GPUs, bfloat16 is optimal, but let's be more aggressive with memory
            console.print(f"[green]✓ Using bfloat16 for H100 GPU (optimal precision)[/green]")

        # For memory-constrained loading, consider float16 for even lower memory usage
        if device.startswith("cuda"):
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < 80:  # Less than 80GB total
                if dtype == torch.float32:
                    dtype = torch.float16
                    console.print(
                        f"[yellow]⚠️  Low GPU memory detected ({gpu_memory_gb:.1f}GB), switching to float16[/yellow]"
                    )
            elif gpu_memory_gb >= 80 and dtype == torch.float32:
                dtype = torch.bfloat16  # Use bfloat16 for high-memory GPUs
                console.print(
                    f"[green]✓ Using bfloat16 for high-memory GPU ({gpu_memory_gb:.1f}GB)[/green]"
                )

        # Initialize loading strategy FIRST
        load_on_cpu_first = False

        # Prepare loading kwargs with memory optimizations
        # NOTE: We intentionally DO NOT use device_map="cuda" because:
        # 1. It can cause OOM during loading with partial model stuck in GPU memory
        # 2. We have more control loading to CPU first, then moving to GPU
        # 3. The slight overhead of CPU->GPU transfer is worth the reliability
        loading_kwargs = {
            "use_safetensors": use_safetensors,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            # No device_map - load to CPU first, then move to GPU after
        }

        print(f"DEBUG: Flux2Pipeline doesn't accept dtype parameter, will convert after loading")
        print(f"DEBUG: Target dtype = {dtype}, load_on_cpu_first = {load_on_cpu_first}")

        # For GPU devices with sufficient memory, load directly to GPU
        if device.startswith("cuda"):
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            console.print(f"[blue]GPU detected: {gpu_memory_gb:.1f}GB total memory[/blue]")

            # Determine loading strategy
            # For large models like FLUX, we ALWAYS use CPU-first loading for reliability:
            # 1. Load model to CPU (with low_cpu_mem_usage=True for efficiency)
            # 2. Convert dtype on CPU if needed
            # 3. Move to GPU component by component
            # This prevents OOM during loading leaving partial models stuck in GPU memory
            if force_cpu_loading:
                console.print("[yellow]⚠️  CPU-first loading explicitly requested[/yellow]")
            else:
                console.print(f"[green]✓ Using CPU-first loading (safer for large models)[/green]")
                console.print(f"[blue]Will load to CPU, then move to GPU ({gpu_memory_gb:.1f}GB available)[/blue]")
            load_on_cpu_first = True  # ALWAYS use CPU-first for reliability
        else:
            loading_kwargs["device_map"] = device

        if cache_dir:
            loading_kwargs["cache_dir"] = cache_dir

        console.print(
            f"[green]✓ Optimized loading kwargs: low_cpu_mem_usage={low_cpu_mem_usage}[/green]"
        )

        # Set attention implementation
        if attention_implementation != "default":
            loading_kwargs["variant"] = attention_implementation

        # Try loading with current dtype, fallback to float16 if needed
        loading_dtype = dtype
        try:
            # Load the pipeline using the detected class
            if load_on_cpu_first:
                # Load on CPU first (no device_map), then convert and move to GPU
                cpu_loading_kwargs = {k: v for k, v in loading_kwargs.items() if k != "device_map"}
                pipeline = pipeline_class.from_pretrained(model_name, **cpu_loading_kwargs)
                console.print(f"[green]✓ Loaded model on CPU[/green]")

                # Check ACTUAL loaded dtype - Flux2Pipeline often loads as float32!
                try:
                    actual_dtype = next(pipeline.transformer.parameters()).dtype
                    console.print(f"[blue]DEBUG: Model loaded with actual dtype = {actual_dtype}[/blue]")
                except Exception:
                    actual_dtype = torch.float32  # Assume float32 if can't detect

                # Convert to target dtype if needed (compare ACTUAL vs TARGET)
                if actual_dtype != dtype:
                    console.print(
                        f"[yellow]⚠️  Converting model from {actual_dtype} to {dtype} on CPU...[/yellow]"
                    )
                    pipeline = pipeline.to(dtype)
                    console.print(f"[green]✓ Model converted to {dtype} on CPU[/green]")
                else:
                    console.print(
                        f"[blue]DEBUG: No conversion needed, already {dtype}[/blue]"
                    )
            else:
                # Standard loading (non-CUDA devices)
                pipeline = pipeline_class.from_pretrained(model_name, **loading_kwargs)

                # Check ACTUAL loaded dtype - Flux2Pipeline often loads as float32!
                try:
                    actual_dtype = next(pipeline.transformer.parameters()).dtype
                    console.print(f"[blue]DEBUG: Model loaded with actual dtype = {actual_dtype}[/blue]")
                except Exception:
                    actual_dtype = torch.float32  # Assume float32 if can't detect

                # Convert to target dtype if needed (compare ACTUAL vs TARGET)
                if actual_dtype != dtype:
                    console.print(
                        f"[yellow]⚠️  Converting model from {actual_dtype} to {dtype}...[/yellow]"
                    )
                    pipeline = pipeline.to(dtype)
                    console.print(f"[green]✓ Model converted to {dtype}[/green]")
                else:
                    console.print(
                        f"[blue]DEBUG: No conversion needed, already {dtype}[/blue]"
                    )

            # Check actual model dtype after loading
            try:
                # Get dtype of first transformer parameter as representative
                sample_param = next(pipeline.transformer.parameters())
                actual_dtype = sample_param.dtype
                console.print(
                    f"[blue]DEBUG: Model loaded with actual dtype = {actual_dtype}[/blue]"
                )

                # Calculate expected memory based on actual dtype
                bytes_per_param = (
                    2
                    if actual_dtype == torch.bfloat16
                    else 4
                    if actual_dtype == torch.float32
                    else 2
                )
                expected_model_gb = (32e9 * bytes_per_param) / (1024**3)
                console.print(
                    f"[blue]DEBUG: Expected model size = {expected_model_gb:.1f}GB[/blue]"
                )
            except Exception as e:
                console.print(f"[yellow]Could not check model dtype: {e}[/yellow]")

            # Move pipeline to target device if needed
            gpu_memory_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if torch.cuda.is_available()
                else 0
            )

            if load_on_cpu_first and device.startswith("cuda"):
                # CPU-first loading: move from CPU to GPU component by component
                console.print(f"[green]✓ Moving model to {device} (component by component)[/green]")

                # Check GPU memory before moving
                gpu_memory_before = (
                    torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
                )
                console.print(
                    f"[blue]DEBUG: GPU memory before move: {gpu_memory_before:.2f}GB[/blue]"
                )

                # Move each major component separately to reduce peak memory
                # Calculate total model size first to determine if CPU offloading is needed
                try:
                    print("=" * 60)
                    print("CALCULATING MODEL SIZE FOR CPU OFFLOADING CHECK")
                    print("=" * 60)

                    components = [
                        ('transformer', pipeline.transformer),
                        ('vae', pipeline.vae),
                        ('text_encoder', pipeline.text_encoder),
                    ]

                    # Calculate total size
                    total_params = 0
                    component_sizes = {}
                    for name, component in components:
                        if component is not None:
                            param_count = sum(p.numel() for p in component.parameters())
                            param_size_gb = (param_count * 2) / (1024**3)  # 2 bytes for bfloat16
                            component_sizes[name] = {'params': param_count, 'size_gb': param_size_gb}
                            total_params += param_count
                            print(f"  {name}: {param_count/1e9:.2f}B params = {param_size_gb:.1f}GB")

                    total_size_gb = (total_params * 2) / (1024**3)
                    print(f"TOTAL: {total_params/1e9:.2f}B params = {total_size_gb:.1f}GB")
                    print(f"GPU available: {gpu_memory_gb:.1f}GB")

                    # Check if model fits in GPU
                    gpu_free = gpu_memory_gb - (torch.cuda.memory_allocated(0) / (1024**3))
                    print(f"GPU free: {gpu_free:.1f}GB")
                    print(f"Model fits in GPU? {total_size_gb:.1f}GB <= {gpu_free * 0.95:.1f}GB (95% of free): {total_size_gb <= gpu_free * 0.95}")

                    if total_size_gb > gpu_free * 0.95:  # Leave 5% margin
                        print(">>> ENABLING CPU OFFLOADING FOR TEXT_ENCODER <<<")
                        console.print(f"[yellow]⚠️  Model ({total_size_gb:.1f}GB) exceeds GPU memory ({gpu_free:.1f}GB free)[/yellow]")
                        console.print(f"[yellow]⚠️  Enabling CPU offloading for text_encoder[/yellow]")
                        cpu_offload_text_encoder = True
                    else:
                        print(">>> ALL COMPONENTS WILL GO TO GPU <<<")
                        cpu_offload_text_encoder = False

                    print("=" * 60)

                    for name, component in components:
                        if component is not None:
                            info = component_sizes[name]
                            gpu_before = torch.cuda.memory_allocated(0) / (1024**3)

                            # CPU offload text_encoder if needed
                            if name == 'text_encoder' and cpu_offload_text_encoder:
                                console.print(f"  Keeping {name} on CPU ({info['params']/1e9:.2f}B params, ~{info['size_gb']:.1f}GB) - CPU offloading")
                                # Ensure it stays on CPU
                                component.to('cpu')
                                continue

                            console.print(f"  Moving {name} to GPU ({info['params']/1e9:.2f}B params, ~{info['size_gb']:.1f}GB)...")
                            component.to(device)

                            # Force cleanup and check memory
                            gc.collect()
                            torch.cuda.empty_cache()
                            gpu_after = torch.cuda.memory_allocated(0) / (1024**3)
                            console.print(f"    GPU memory: {gpu_before:.1f}GB → {gpu_after:.1f}GB (+{gpu_after-gpu_before:.1f}GB)")

                    if cpu_offload_text_encoder:
                        console.print(f"[green]✓ Model loaded with CPU offloading (text_encoder on CPU, rest on GPU)[/green]")
                        console.print(f"[yellow]Note: Text encoding will run on CPU, which is slower but enables training[/yellow]")
                    else:
                        console.print(f"[green]✓ All components on GPU ({device})[/green]")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Get current memory state
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                        free_gb = gpu_memory_gb - reserved_gb

                        console.print(f"[red]❌ Out of Memory while moving {name} to GPU[/red]")
                        console.print(f"[yellow]Component: {name} ({param_count/1e9:.2f}B params, ~{param_size_gb:.1f}GB)[/yellow]")
                        console.print(f"[yellow]GPU Memory: {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved, {free_gb:.1f}GB free[/yellow]")
                        console.print(f"[yellow]GPU Total: {gpu_memory_gb:.1f}GB[/yellow]")
                        console.print("")
                        console.print("[cyan]💡 This shouldn't happen with a 93GB H100![/cyan]")
                        console.print("[cyan]Possible causes:[/cyan]")
                        console.print("  1. Another process is using GPU memory")
                        console.print("  2. Memory fragmentation from previous runs")
                        console.print("  3. The model is larger than expected")
                        console.print("")
                        console.print("[cyan]Solutions:[/cyan]")
                        console.print("  1. Run: nvidia-smi to check GPU usage")
                        console.print("  2. Restart Python process")
                        console.print("  3. Reboot the server if issue persists")

                        raise RuntimeError(
                            f"OOM moving {name} to GPU. Allocated: {allocated_gb:.1f}GB, "
                            f"Reserved: {reserved_gb:.1f}GB, Free: {free_gb:.1f}GB"
                        ) from e
                    else:
                        raise
            elif not device.startswith("cuda"):
                # CPU target - model already on CPU
                console.print(f"[green]✓ Model loaded on {device}[/green]")

            # SUCCESS PATH: Return pipeline and metadata
            # Get model metadata for successful loading
            metadata = ModelLoader._get_model_metadata(pipeline, device, dtype)
            console.print("[green]✓ Model loaded successfully[/green]")
            console.print(f"  Parameters: {metadata['total_parameters']:,}")
            console.print(f"  Memory: {metadata['memory_gb']:.1f}GB")

            return pipeline, metadata

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and loading_dtype == torch.bfloat16:
                console.print(f"[yellow]⚠️  OOM with bfloat16, trying float16 instead[/yellow]")

                # Emergency fallback: reload and convert to float16
                console.print("[red]🔄 Reloading and converting to float16[/red]")

                # CRITICAL: Delete the failed pipeline to release GPU memory
                console.print("[yellow]Releasing failed pipeline from GPU memory...[/yellow]")
                try:
                    if 'pipeline' in dir() and pipeline is not None:
                        # Delete all component references first
                        if hasattr(pipeline, 'transformer'):
                            del pipeline.transformer
                        if hasattr(pipeline, 'vae'):
                            del pipeline.vae
                        if hasattr(pipeline, 'text_encoder'):
                            del pipeline.text_encoder
                        del pipeline
                        console.print("[green]✓ Deleted failed pipeline references[/green]")
                except Exception as del_err:
                    console.print(f"[yellow]Note: Could not delete pipeline: {del_err}[/yellow]")

                # Aggressive memory cleanup - must happen AFTER deleting pipeline
                if torch.cuda.is_available():
                    console.print("[yellow]Performing emergency GPU memory cleanup...[/yellow]")
                    # Multiple rounds of cleanup
                    for i in range(3):
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Check memory after cleanup
                    freed_gb = torch.cuda.memory_allocated(0) / (1024**3)
                    console.print(f"[blue]GPU memory after cleanup: {freed_gb:.2f}GB allocated[/blue]")

                # ALWAYS use CPU loading for fallback - GPU loading already failed once
                fallback_kwargs = {
                    "use_safetensors": use_safetensors,
                    "low_cpu_mem_usage": low_cpu_mem_usage,
                    # No device_map - load to CPU by default
                }

                console.print("[yellow]⚠️  Reloading model to CPU (safer fallback)[/yellow]")

                # Reload to CPU and convert to float16
                pipeline = pipeline_class.from_pretrained(model_name, **fallback_kwargs)
                console.print("[green]✓ Model loaded to CPU[/green]")

                # Convert to float16 on CPU
                console.print("[yellow]Converting to float16 on CPU...[/yellow]")
                pipeline = pipeline.to(torch.float16)
                loading_dtype = torch.float16
                console.print(f"[green]✓ Model converted to float16[/green]")

                # Move to GPU component by component to reduce peak memory
                if device.startswith("cuda"):
                    console.print(f"[green]✓ Moving float16 model to {device} (component by component)[/green]")

                    components = [
                        ('transformer', pipeline.transformer),
                        ('vae', pipeline.vae),
                        ('text_encoder', pipeline.text_encoder),
                    ]

                    for name, component in components:
                        if component is not None:
                            console.print(f"  Moving {name} to GPU...")
                            component.to(device)
                            torch.cuda.empty_cache()

                    console.print(f"[green]✓ All components on GPU ({device})[/green]")
                else:
                    console.print(f"[green]✓ Float16 model loaded on {device}[/green]")

                console.print(
                    f"[yellow]⚠️  Successfully loaded with float16 instead of bfloat16[/yellow]"
                )
            else:
                raise

            # Enable memory efficient attention if available
            if attention_implementation == "flash_attention_2":
                try:
                    # Try Flash Attention 2 first (optimal for H100)
                    pipeline.transformer.enable_xformers_memory_efficient_attention()
                    console.print("[green]✓ Enabled Flash Attention 2[/green]")
                except Exception as e:
                    console.print(f"[yellow]Flash Attention 2 not available: {e}[/yellow]")
                    # Fallback to xformers if available
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                        console.print(
                            "[green]✓ Enabled xFormers memory efficient attention[/green]"
                        )
                    except Exception as e2:
                        console.print(f"[yellow]xFormers not available: {e2}[/yellow]")

            # Skip torch.compile during initial loading to reduce memory pressure
            # Will compile after training setup if requested
            if torch_compile and device != "cpu":
                console.print(
                    f"[yellow]⚠️  Skipping torch.compile during loading to reduce memory pressure[/yellow]"
                )
                console.print(f"[dim]Will compile after training setup if needed[/dim]")

            # Get model metadata
            metadata = ModelLoader._get_model_metadata(pipeline, device, dtype)

            console.print("[green]✓ Model loaded successfully[/green]")
            console.print(f"  Parameters: {metadata['total_parameters']:,}")
            console.print(f"  Memory: {metadata['memory_gb']:.1f}GB")

            return pipeline, metadata

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            raise RuntimeError(f"Failed to load Flux2-dev model: {e}")

    @staticmethod
    def _validate_flux_model_directory(model_path: str, detected_pipeline_class=None):
        """
        Validate that a local directory contains the expected FLUX2-dev model files.

        Args:
            model_path: Path to the model directory

        Returns:
            True if directory appears to contain complete FLUX2-dev model files
        """
        path = Path(model_path)

        # Check for model_index.json (required)
        if not (path / "model_index.json").exists():
            console.print(f"[red]Missing model_index.json in {model_path}[/red]")
            return False

        # Use provided pipeline class or detect it
        if detected_pipeline_class is not None:
            pipeline_class = detected_pipeline_class
        else:
            # Check model_index.json to determine which FLUX version this is
            pipeline_class = ModelLoader._detect_flux_pipeline_class(str(path))

        if pipeline_class == Flux2Pipeline:
            # FLUX2-dev requires these components (single text encoder/tokenizer architecture)
            required_components = ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"]
        else:
            # FLUX1 requires these components (dual text encoder/tokenizer architecture)
            required_components = [
                "transformer",
                "text_encoder",
                "text_encoder_2",
                "vae",
                "tokenizer",
                "tokenizer_2",
                "scheduler",
            ]

        missing_components = []
        for component in required_components:
            if not (path / component).exists():
                missing_components.append(component)

        if missing_components:
            flux_version = "FLUX2-dev" if pipeline_class == Flux2Pipeline else "FLUX1"
            console.print(f"[red]Missing {flux_version} components: {missing_components}[/red]")
            return False

        # Check for key config files (pipeline-specific)
        if pipeline_class == Flux2Pipeline:
            # FLUX2-dev config files (single text encoder/tokenizer)
            key_files = [
                "transformer/config.json",
                "text_encoder/config.json",
                "vae/config.json",
                "tokenizer/config.json",
            ]
        else:
            # FLUX1 config files (dual text encoder/tokenizer)
            key_files = [
                "transformer/config.json",
                "text_encoder/config.json",
                "text_encoder_2/config.json",
                "vae/config.json",
                "tokenizer/config.json",
                "tokenizer_2/config.json",
            ]

        missing_files = []
        for file_path in key_files:
            if not (path / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            flux_version = "FLUX2-dev" if pipeline_class == Flux2Pipeline else "FLUX1"
            console.print(
                f"[yellow]Warning: Missing {flux_version} config files: {missing_files}[/yellow]"
            )

        console.print(f"[green]FLUX model validation passed for {model_path}[/green]")
        return True

    @staticmethod
    def _detect_flux_pipeline_class(model_path: str):
        """
        Detect which FLUX pipeline class to use based on model_index.json.

        Args:
            model_path: Path to the model directory

        Returns:
            Pipeline class (FluxPipeline or Flux2Pipeline)
        """
        model_index_path = Path(model_path) / "model_index.json"

        if not model_index_path.exists():
            # Default to FluxPipeline if no model_index.json
            console.print("[yellow]No model_index.json found, defaulting to FluxPipeline[/yellow]")
            return FluxPipeline

        try:
            import json

            with open(model_index_path) as f:
                model_index = json.load(f)

            class_name = model_index.get("_class_name", "")

            if "Flux2" in class_name:
                console.print("[green]Detected Flux2Pipeline model[/green]")
                return Flux2Pipeline
            else:
                console.print("[green]Detected FluxPipeline model[/green]")
                return FluxPipeline

        except Exception as e:
            console.print(
                f"[yellow]Could not read model_index.json: {e}, defaulting to FluxPipeline[/yellow]"
            )
            return FluxPipeline

    @staticmethod
    def _get_model_metadata(
        pipeline: FluxPipeline, device: str, dtype: torch.dtype
    ) -> dict[str, any]:
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
    ) -> tuple[FluxPipeline, dict[str, any]]:
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
        console.print("[bold blue]Injecting LoRA adapters[/bold blue]")
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

            console.print("[green]✓ LoRA injected successfully[/green]")
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
    def _get_injection_metadata(model, lora_config: FluxLoRAConfig) -> dict[str, any]:
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
    def verify_lora_injection(model) -> dict[str, any]:
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
    ) -> dict[str, any]:
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
        console.print("[bold blue]Testing forward pass[/bold blue]")
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

            console.print("[green]✓ Forward pass successful[/green]")
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
    ) -> dict[str, any]:
        """Save LoRA weights to disk.

        Args:
            pipeline: Pipeline with LoRA adapters
            output_path: Path to save LoRA weights
            adapter_name: Name of adapter to save

        Returns:
            Dictionary with save results
        """
        console.print("[bold blue]Saving LoRA weights[/bold blue]")
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

            console.print("[green]✓ LoRA weights saved[/green]")

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
    ) -> dict[str, any]:
        """Load LoRA weights from disk.

        Args:
            pipeline: Pipeline to load LoRA into
            lora_path: Path to LoRA weights
            adapter_name: Name for loaded adapter

        Returns:
            Dictionary with load results
        """
        console.print("[bold blue]Loading LoRA weights[/bold blue]")
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

            console.print("[green]✓ LoRA weights loaded[/green]")

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
