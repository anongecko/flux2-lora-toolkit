"""
Hardware utilities for GPU detection and memory management.

This module provides utilities for detecting available hardware,
monitoring GPU memory usage, and optimizing training performance.
"""

import gc
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil
import torch
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class GPUInfo:
    """Information about a GPU."""
    id: int
    name: str
    memory_total: int  # MB
    memory_free: int   # MB
    memory_used: int    # MB
    utilization: float  # Percentage
    temperature: Optional[float] = None  # Celsius
    power_usage: Optional[float] = None  # Watts
    is_available: bool = True


@dataclass
class SystemInfo:
    """Information about the system hardware."""
    platform: str
    cpu_count: int
    memory_total: int  # GB
    memory_available: int  # GB
    gpus: List[GPUInfo]
    cuda_available: bool
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None


class HardwareManager:
    """Hardware detection and management utilities."""

    def __init__(self):
        """Initialize hardware manager."""
        self._system_info = None
        self._gpu_info_cache = None
        self._cache_timeout = 5.0  # Cache GPU info for 5 seconds

    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information.
        
        Returns:
            SystemInfo object with hardware details
        """
        if self._system_info is None:
            self._system_info = self._collect_system_info()
        return self._system_info

    def _collect_system_info(self) -> SystemInfo:
        """Collect system information."""
        # Basic system info
        system_platform = platform.system()
        cpu_count = os.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total = memory.total // (1024**3)  # GB
        memory_available = memory.available // (1024**3)  # GB
        
        # GPU info
        gpus = self.get_gpu_info()
        
        # CUDA info
        cuda_available = torch.cuda.is_available()
        cuda_version = None
        torch_version = None
        
        if cuda_available:
            cuda_version = torch.version.cuda
            torch_version = torch.__version__
        
        return SystemInfo(
            platform=system_platform,
            cpu_count=cpu_count,
            memory_total=memory_total,
            memory_available=memory_available,
            gpus=gpus,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            torch_version=torch_version,
        )

    def get_gpu_info(self, force_refresh: bool = False) -> List[GPUInfo]:
        """Get information about available GPUs.
        
        Args:
            force_refresh: Force refresh of GPU information
            
        Returns:
            List of GPUInfo objects
        """
        if not torch.cuda.is_available():
            return []

        # Check cache
        if not force_refresh and self._gpu_info_cache is not None:
            # TODO: Add cache timeout logic
            pass

        gpu_info_list = []
        
        for gpu_id in range(torch.cuda.device_count()):
            gpu_info = self._get_single_gpu_info(gpu_id)
            gpu_info_list.append(gpu_info)
        
        self._gpu_info_cache = gpu_info_list
        return gpu_info_list

    def _get_single_gpu_info(self, gpu_id: int) -> GPUInfo:
        """Get information for a single GPU."""
        # Basic torch info
        props = torch.cuda.get_device_properties(gpu_id)
        name = props.name
        memory_total = props.total_memory // (1024**2)  # MB
        
        # Memory usage
        memory_reserved = torch.cuda.memory_reserved(gpu_id) // (1024**2)  # MB
        memory_allocated = torch.cuda.memory_allocated(gpu_id) // (1024**2)  # MB
        memory_free = memory_total - memory_reserved
        memory_used = memory_allocated
        
        # Try to get more detailed info from nvidia-smi
        try:
            smi_info = self._get_nvidia_smi_info(gpu_id)
            utilization = smi_info.get("utilization", 0.0)
            temperature = smi_info.get("temperature")
            power_usage = smi_info.get("power_usage")
        except Exception:
            # Fallback if nvidia-smi is not available
            utilization = 0.0
            temperature = None
            power_usage = None
        
        return GPUInfo(
            id=gpu_id,
            name=name,
            memory_total=memory_total,
            memory_free=memory_free,
            memory_used=memory_used,
            utilization=utilization,
            temperature=temperature,
            power_usage=power_usage,
            is_available=True,
        )

    def _get_nvidia_smi_info(self, gpu_id: int) -> Dict[str, float]:
        """Get GPU information from nvidia-smi.
        
        Returns:
            Dictionary with GPU metrics
        """
        try:
            # Query nvidia-smi for specific GPU
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                if len(values) >= 3:
                    return {
                        "utilization": float(values[0]),
                        "temperature": float(values[1]),
                        "power_usage": float(values[2]),
                    }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return {}

    def print_system_info(self):
        """Print formatted system information."""
        info = self.get_system_info()
        
        console.print("\n[bold blue]System Information[/bold blue]")
        console.print(f"Platform: {info.platform}")
        console.print(f"CPU Cores: {info.cpu_count}")
        console.print(f"Memory: {info.memory_available:.1f}GB / {info.memory_total:.1f}GB")
        
        if info.cuda_available:
            console.print(f"CUDA: Available (v{info.cuda_version})")
            console.print(f"PyTorch: v{info.torch_version}")
        else:
            console.print("[red]CUDA: Not Available[/red]")
        
        if info.gpus:
            self.print_gpu_info(info.gpus)
        else:
            console.print("[yellow]No GPUs detected[/yellow]")

    def print_gpu_info(self, gpu_list: Optional[List[GPUInfo]] = None):
        """Print formatted GPU information.
        
        Args:
            gpu_list: List of GPUInfo objects. If None, gets current GPU info.
        """
        if gpu_list is None:
            gpu_list = self.get_gpu_info()
        
        if not gpu_list:
            console.print("[yellow]No GPUs detected[/yellow]")
            return
        
        table = Table(title="GPU Information")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Name", style="white", width=30)
        table.add_column("Memory", style="green", width=15)
        table.add_column("Utilization", style="yellow", width=12)
        table.add_column("Temperature", style="red", width=12)
        table.add_column("Power", style="blue", width=10)
        
        for gpu in gpu_list:
            memory_str = f"{gpu.memory_used}/{gpu.memory_total}MB"
            util_str = f"{gpu.utilization:.1f}%"
            temp_str = f"{gpu.temperature:.0f}°C" if gpu.temperature else "N/A"
            power_str = f"{gpu.power_usage:.0f}W" if gpu.power_usage else "N/A"
            
            table.add_row(
                str(gpu.id),
                gpu.name,
                memory_str,
                util_str,
                temp_str,
                power_str
            )
        
        console.print(table)

    def select_best_gpu(self, min_memory_mb: int = 0) -> Optional[int]:
        """Select the best GPU for training.
        
        Args:
            min_memory_mb: Minimum required memory in MB
            
        Returns:
            GPU ID of best GPU, or None if no suitable GPU found
        """
        gpu_list = self.get_gpu_info()
        
        if not gpu_list:
            return None
        
        # Filter GPUs with enough memory
        suitable_gpus = [
            gpu for gpu in gpu_list 
            if gpu.memory_free >= min_memory_mb
        ]
        
        if not suitable_gpus:
            console.print(
                f"[red]No GPU with at least {min_memory_mb}MB free memory found[/red]"
            )
            return None
        
        # Select GPU with most free memory
        best_gpu = max(suitable_gpus, key=lambda gpu: gpu.memory_free)
        
        console.print(
            f"[green]Selected GPU {best_gpu.id}: {best_gpu.name} "
            f"({best_gpu.memory_free}MB free)[/green]"
        )
        
        return best_gpu.id

    def optimize_memory_settings(self, config) -> Dict[str, any]:
        """Optimize memory settings based on available hardware.
        
        Args:
            config: Training configuration object
            
        Returns:
            Dictionary with optimized settings
        """
        gpu_list = self.get_gpu_info()
        
        if not gpu_list:
            return {}
        
        # Use the GPU with most free memory
        best_gpu = max(gpu_list, key=lambda gpu: gpu.memory_free)
        free_memory_gb = best_gpu.memory_free / 1024
        
        optimizations = {}
        
        # Batch size optimization
        current_batch_size = config.training.batch_size
        if free_memory_gb < 16:
            # Low memory GPU
            if current_batch_size > 2:
                optimizations["batch_size"] = max(1, current_batch_size // 2)
                optimizations["gradient_accumulation_steps"] = (
                    config.training.gradient_accumulation_steps * 2
                )
                console.print(
                    f"[yellow]Reduced batch size to {optimizations['batch_size']} "
                    f"for low memory GPU[/yellow]"
                )
        
        # Gradient checkpointing for low memory
        if free_memory_gb < 24:
            optimizations["gradient_checkpointing"] = True
            console.print("[yellow]Enabled gradient checkpointing for memory optimization[/yellow]")
        
        # Mixed precision recommendation
        if config.training.mixed_precision == "no" and free_memory_gb < 32:
            optimizations["mixed_precision"] = "bf16"
            console.print("[yellow]Recommended mixed precision (bf16) for memory savings[/yellow]")
        
        # DataLoader workers
        cpu_count = os.cpu_count()
        if config.data.num_workers > cpu_count:
            optimizations["num_workers"] = cpu_count
            console.print(
                f"[yellow]Reduced data workers to {cpu_count} (CPU core limit)[/yellow]"
            )
        
        return optimizations

    def clear_gpu_cache(self, gpu_id: Optional[int] = None):
        """Clear GPU cache.
        
        Args:
            gpu_id: Specific GPU to clear, or None for all GPUs
        """
        if not torch.cuda.is_available():
            return
        
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        console.print("[green]GPU cache cleared[/green]")

    def monitor_memory_usage(self, gpu_id: int = 0) -> Dict[str, float]:
        """Monitor current memory usage.
        
        Args:
            gpu_id: GPU ID to monitor
            
        Returns:
            Dictionary with memory usage statistics
        """
        if not torch.cuda.is_available():
            return {}
        
        with torch.cuda.device(gpu_id):
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)   # GB
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)  # GB
            
            props = torch.cuda.get_device_properties(gpu_id)
            total_memory = props.total_memory / (1024**3)  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "total_gb": total_memory,
                "free_gb": total_memory - reserved,
                "utilization_percent": (allocated / total_memory) * 100,
            }

    def check_h100_optimization(self) -> List[str]:
        """Check if system is optimized for H100 GPU.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        gpu_list = self.get_gpu_info()
        
        if not gpu_list:
            return ["No GPU detected"]
        
        # Check for H100
        has_h100 = any("H100" in gpu.name for gpu in gpu_list)
        
        if has_h100:
            # Check CUDA version
            cuda_version = torch.version.cuda
            if cuda_version and tuple(map(int, cuda_version.split('.'))) < (12, 1):
                recommendations.append("Upgrade to CUDA 12.1+ for optimal H100 performance")
            
            # Check PyTorch version
            torch_version = torch.__version__
            # This is a simplified check - in practice you'd want more sophisticated version checking
            if "2.0" not in torch_version and "2.1" not in torch_version:
                recommendations.append("Upgrade to PyTorch 2.0+ for H100 optimizations")
            
            # Check for bfloat16 support
            if not torch.cuda.is_bf16_supported():
                recommendations.append("bfloat16 not supported - consider using fp16")
            
            # Check memory
            h100_gpus = [gpu for gpu in gpu_list if "H100" in gpu.name]
            for gpu in h100_gpus:
                if gpu.memory_total < 80000:  # Less than 80GB
                    recommendations.append(f"GPU {gpu.id} may not be full 96GB H100")
        
        return recommendations

    def estimate_training_time(
        self, 
        dataset_size: int, 
        batch_size: int, 
        steps_per_second: float = 0.5
    ) -> Dict[str, str]:
        """Estimate training time based on dataset and hardware.
        
        Args:
            dataset_size: Number of training samples
            batch_size: Training batch size
            steps_per_second: Estimated training speed (steps/second)
            
        Returns:
            Dictionary with time estimates
        """
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch  # Assume 1 epoch for estimation
        
        seconds = total_steps / steps_per_second
        minutes = seconds / 60
        hours = minutes / 60
        
        return {
            "total_steps": str(total_steps),
            "estimated_time_seconds": f"{seconds:.1f}",
            "estimated_time_minutes": f"{minutes:.1f}",
            "estimated_time_hours": f"{hours:.2f}",
        }


# Global hardware manager instance
hardware_manager = HardwareManager()