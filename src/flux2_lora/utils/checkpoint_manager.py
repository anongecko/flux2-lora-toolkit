"""
Checkpoint management utilities for Flux2-dev LoRA training.

This module provides safe checkpoint saving and loading with proper validation,
metadata tracking, and recovery capabilities.
"""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from rich.console import Console

from ..utils.config_manager import Config

logger = logging.getLogger(__name__)
console = Console()


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup for LoRA training.
    
    Features:
    - Safe checkpoint saving with atomic writes
    - Metadata tracking and validation
    - Automatic cleanup of old checkpoints
    - Checkpoint integrity verification
    - Resume capability with state restoration
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        checkpoints_limit: int = 5,
        save_optimizer_state: bool = True,
        save_scheduler_state: bool = True,
        verify_integrity: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            checkpoints_limit: Maximum number of checkpoints to keep
            save_optimizer_state: Whether to save optimizer state
            save_scheduler_state: Whether to save scheduler state
            verify_integrity: Whether to verify checkpoint integrity after saving
        """
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_limit = checkpoints_limit
        self.save_optimizer_state = save_optimizer_state
        self.save_scheduler_state = save_scheduler_state
        self.verify_integrity = verify_integrity
        
        # Create directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_registry = {'checkpoints': {}, 'version': '1.0'}
    
    def _load_checkpoint_registry(self) -> Dict[str, Any]:
        """Load checkpoint registry from disk."""
        registry_path = self.checkpoints_dir / "registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint registry: {e}")
        
        return {'checkpoints': {}, 'version': '1.0'}
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk."""
        registry_path = self.checkpoints_dir / "registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.checkpoint_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _update_registry(self, checkpoint_name: str, checkpoint_data: Dict[str, Any]):
        """Update checkpoint registry with new checkpoint."""
        self.checkpoint_registry['checkpoints'][checkpoint_name] = {
            'step': checkpoint_data['step'],
            'loss': checkpoint_data['loss'],
            'timestamp': checkpoint_data['timestamp'],
            'is_best': checkpoint_data.get('is_best', False),
        }
        self._save_checkpoint_registry()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond the limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.checkpoints_limit:
            return
        
        # Keep the best checkpoint regardless of age
        best_checkpoint = self.get_best_checkpoint()
        checkpoints_to_keep = set()
        
        if best_checkpoint:
            checkpoints_to_keep.add(best_checkpoint['name'])
        
        # Keep the most recent checkpoints up to the limit
        for checkpoint in checkpoints:
            if len(checkpoints_to_keep) >= self.checkpoints_limit:
                break
            checkpoints_to_keep.add(checkpoint['name'])
        
        # Remove old checkpoints
        for checkpoint in checkpoints:
            if checkpoint['name'] not in checkpoints_to_keep:
                checkpoint_path = self.checkpoints_dir / checkpoint['name']
                try:
                    shutil.rmtree(checkpoint_path)
                    console.print(f"[yellow]Removed old checkpoint: {checkpoint['name']}[/yellow]")
                    
                    # Remove from registry
                    if checkpoint['name'] in self.checkpoint_registry['checkpoints']:
                        del self.checkpoint_registry['checkpoints'][checkpoint['name']]
                    
                except Exception as e:
                    logger.error(f"Failed to remove checkpoint {checkpoint['name']}: {e}")
        
        self._save_checkpoint_registry()
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_dir in self.checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir() or checkpoint_dir.name == "best":
                continue
            
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoints.append({
                        'name': checkpoint_dir.name,
                        'path': str(checkpoint_dir),
                        'step': metadata.get('step', 0),
                        'loss': metadata.get('loss', 0.0),
                        'timestamp': metadata.get('timestamp', 0),
                        'is_best': metadata.get('is_best', False),
                    })
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint metadata for {checkpoint_dir.name}: {e}")
        
        # Sort by step (newest first)
        checkpoints.sort(key=lambda x: x['step'], reverse=True)
        return checkpoints
    
    def save_checkpoint(
        self,
        model,
        optimizer_manager,
        step: int,
        loss: float,
        config: Config,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Dict[str, Any]:
        """
        Save training checkpoint with all necessary state.
        
        Args:
            model: Model to save
            optimizer_manager: Optimizer manager with state
            step: Current training step
            loss: Current loss value
            config: Training configuration
            metadata: Additional metadata to save
            is_best: Whether this is the best checkpoint
            
        Returns:
            Dictionary with save results
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        checkpoint_name = f"step_{step:08d}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        temp_path = checkpoint_path.with_suffix(".tmp")
        
        console.print(f"[bold blue]Saving checkpoint: {checkpoint_name}[/bold blue]")
        
        try:
            # Create temp directory first
            temp_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data
            # Handle config conversion to dict safely
            try:
                if isinstance(config, dict):
                    config_dict = config
                elif hasattr(config, 'to_dict'):
                    # Check if to_dict is actually callable
                    to_dict_attr = getattr(config, 'to_dict', None)
                    if callable(to_dict_attr):
                        config_dict = to_dict_attr()
                    else:
                        # to_dict is not callable (might be a dict itself)
                        if isinstance(to_dict_attr, dict):
                            config_dict = to_dict_attr
                        else:
                            config_dict = config.__dict__ if hasattr(config, '__dict__') else {}
                else:
                    # Fallback: convert to dict using __dict__
                    config_dict = config.__dict__ if hasattr(config, '__dict__') else {}
            except Exception as e:
                logger.warning(f"Failed to convert config to dict: {e}, using empty dict")
                config_dict = {}

            checkpoint_data = {
                'step': step,
                'loss': loss,
                'timestamp': time.time(),
                'config': config_dict,
                'metadata': metadata or {},
                'is_best': is_best,
            }
            
            # Save LoRA weights
            try:
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'save_pretrained'):
                    model.transformer.save_pretrained(
                        str(temp_path),
                        safe_serialization=True,
                        selected_adapters=["default"]
                    )
                    checkpoint_data['lora_saved'] = True
                else:
                    # Fallback: save state dict
                    # Handle Flux pipeline vs regular model
                    if hasattr(model, 'transformer'):
                        state_dict = model.transformer.state_dict()
                    else:
                        state_dict = model.state_dict()

                    lora_state_dict = {
                        k: v for k, v in state_dict.items()
                        if "lora" in k.lower()
                    }
                    torch.save(lora_state_dict, temp_path / "lora_weights.safetensors")
                    checkpoint_data['lora_saved'] = True
            except Exception as e:
                logger.warning(f"Failed to save LoRA weights via save_pretrained, using fallback: {e}")
                # Fallback: save state dict
                # Handle Flux pipeline vs regular model
                if hasattr(model, 'transformer'):
                    state_dict = model.transformer.state_dict()
                else:
                    state_dict = model.state_dict()

                lora_state_dict = {
                    k: v for k, v in state_dict.items()
                    if "lora" in k.lower()
                }
                torch.save(lora_state_dict, temp_path / "lora_weights.safetensors")
                checkpoint_data['lora_saved'] = True
            
            # Save optimizer state
            if self.save_optimizer_state and optimizer_manager:
                optimizer_state = optimizer_manager.get_state_dict()
                torch.save(optimizer_state, temp_path / "optimizer_state.pt")
                checkpoint_data['optimizer_saved'] = True
            
            # Save scheduler state
            if self.save_scheduler_state and optimizer_manager:
                if hasattr(optimizer_manager.scheduler, 'state_dict'):
                    scheduler_state = optimizer_manager.scheduler.state_dict()
                    torch.save(scheduler_state, temp_path / "scheduler_state.pt")
                    checkpoint_data['scheduler_saved'] = True
            
            # Save checkpoint metadata
            # Handle Flux pipeline vs regular model for getting device/dtype
            if hasattr(model, 'transformer'):
                try:
                    first_param = next(model.transformer.parameters())
                    device_str = str(first_param.device)
                    dtype_str = str(first_param.dtype)
                except StopIteration:
                    device_str = 'unknown'
                    dtype_str = 'unknown'
            else:
                try:
                    first_param = next(model.parameters())
                    device_str = str(first_param.device)
                    dtype_str = str(first_param.dtype)
                except StopIteration:
                    device_str = 'unknown'
                    dtype_str = 'unknown'

            checkpoint_data['checkpoint_metadata'] = {
                'version': '1.0',
                'model_type': 'flux2_lora',
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device': device_str,
                'dtype': dtype_str,
            }
            
            # Save metadata file
            with open(temp_path / "metadata.json", 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Verify integrity if enabled
            if self.verify_integrity:
                self._verify_checkpoint_integrity(temp_path)
            
            # Atomic move to final location
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            temp_path.rename(checkpoint_path)
            
            # Update registry
            self._update_registry(checkpoint_name, checkpoint_data)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Create "best" symlink if this is the best checkpoint
            if is_best:
                best_path = self.checkpoints_dir / "best"
                if best_path.exists():
                    best_path.unlink()
                best_path.symlink_to(checkpoint_name)
                console.print("[green]✓ Updated best checkpoint symlink[/green]")
            
            console.print(f"[green]✓ Checkpoint saved successfully[/green]")
            console.print(f"  Path: {checkpoint_path}")
            console.print(f"  Step: {step}")
            console.print(f"  Loss: {loss:.6f}")
            
            return {
                'success': True,
                'checkpoint_path': str(checkpoint_path),
                'checkpoint_name': checkpoint_name,
                'step': step,
                'loss': loss,
                'error': None
            }
            
        except Exception as e:
            # Cleanup temp directory on failure
            if temp_path.exists():
                shutil.rmtree(temp_path)

            console.print(f"[red]Error saving checkpoint: {e}[/red]")
            logger.error(f"Checkpoint save failed: {e}")
            # Add traceback for debugging
            import traceback
            logger.error(f"Checkpoint save traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'checkpoint_path': str(checkpoint_path),
                'checkpoint_name': checkpoint_name,
                'step': step,
                'loss': loss,
                'error': str(e)
            }
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path):
        """Verify checkpoint file integrity."""
        required_files = ["metadata.json"]
        
        # Check for LoRA weights
        has_peft_files = any(
            checkpoint_path.glob("adapter_*.safetensors") or
            checkpoint_path.glob("adapter_*.bin")
        )
        has_lora_file = (checkpoint_path / "lora_weights.safetensors").exists()
        
        if not (has_peft_files or has_lora_file):
            raise RuntimeError("No LoRA weights found in checkpoint")
        
        # Verify metadata
        metadata_path = checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            raise RuntimeError("Checkpoint metadata file missing")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required metadata fields
            required_fields = ['step', 'loss', 'timestamp', 'config']
            for field in required_fields:
                if field not in metadata:
                    raise RuntimeError(f"Missing required metadata field: {field}")
        
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid checkpoint metadata JSON: {e}")
        
        # Verify LoRA weights file integrity
        if has_lora_file:
            lora_path = checkpoint_path / "lora_weights.safetensors"
            try:
                # Try to load the file to verify integrity
                torch.load(lora_path, map_location='cpu')
            except Exception as e:
                raise RuntimeError(f"Corrupted LoRA weights file: {e}")
        
        logger.debug(f"Checkpoint integrity verified: {checkpoint_path.name}")
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the best checkpoint based on loss.
        
        Returns:
            Best checkpoint info or None if no checkpoints
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Find checkpoint marked as best, or fallback to lowest loss
        best_checkpoint = None
        for checkpoint in checkpoints:
            if checkpoint['is_best']:
                best_checkpoint = checkpoint
                break
        
        if best_checkpoint is None:
            # Fallback to checkpoint with lowest loss
            best_checkpoint = min(checkpoints, key=lambda x: x['loss'])
        
        return best_checkpoint
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint.
        
        Returns:
            Latest checkpoint info or None if no checkpoints
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint
            
        Returns:
            Checkpoint information dictionary or None if not found
        """
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            return None
        
        metadata_path = checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get file sizes
            file_sizes = {}
            total_size = 0
            for file_path in checkpoint_path.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_sizes[file_path.name] = size
                    total_size += size
            
            return {
                'name': checkpoint_name,
                'path': str(checkpoint_path),
                'step': metadata.get('step', 0),
                'loss': metadata.get('loss', 0.0),
                'timestamp': metadata.get('timestamp', 0),
                'is_best': metadata.get('is_best', False),
                'config': metadata.get('config'),
                'metadata': metadata.get('metadata', {}),
                'file_sizes': file_sizes,
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': len(file_sizes),
            }
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint info for {checkpoint_name}: {e}")
            return None
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model,
        optimizer_manager = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load training checkpoint and restore state.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load weights into
            optimizer_manager: Optimizer manager to restore state
            device: Target device for loading
            
        Returns:
            Dictionary with load results
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        console.print(f"[bold blue]Loading checkpoint: {checkpoint_path.name}[/bold blue]")
        
        try:
            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Verify checkpoint integrity
            if self.verify_integrity:
                self._verify_checkpoint_integrity(checkpoint_path)
            
            # Load LoRA weights
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'load_adapter'):
                # Use PEFT's load_adapter method
                model.transformer.load_adapter(str(checkpoint_path), "default")
                console.print("[green]✓ LoRA weights loaded via PEFT[/green]")
            else:
                # Fallback: load state dict
                lora_weights_path = checkpoint_path / "lora_weights.safetensors"
                if lora_weights_path.exists():
                    lora_state_dict = torch.load(lora_weights_path, map_location=device)
                    model.load_state_dict(lora_state_dict, strict=False)
                    console.print("[green]✓ LoRA weights loaded from safetensors[/green]")
                else:
                    console.print("[yellow]Warning: No LoRA weights found[/yellow]")
            
            # Load optimizer state
            if self.save_optimizer_state and optimizer_manager:
                optimizer_state_path = checkpoint_path / "optimizer_state.pt"
                if optimizer_state_path.exists():
                    optimizer_state = torch.load(optimizer_state_path, map_location=device)
                    optimizer_manager.load_state_dict(optimizer_state)
                    console.print("[green]✓ Optimizer state restored[/green]")
                else:
                    console.print("[yellow]Warning: No optimizer state found[/yellow]")
            
            # Load scheduler state
            if self.save_scheduler_state and optimizer_manager:
                scheduler_state_path = checkpoint_path / "scheduler_state.pt"
                if scheduler_state_path.exists():
                    scheduler_state = torch.load(scheduler_state_path, map_location=device)
                    if hasattr(optimizer_manager.scheduler, 'load_state_dict'):
                        optimizer_manager.scheduler.load_state_dict(scheduler_state)
                        console.print("[green]✓ Scheduler state restored[/green]")
                else:
                    console.print("[yellow]Warning: No scheduler state found[/yellow]")
            
            console.print(f"[green]✓ Checkpoint loaded successfully[/green]")
            console.print(f"  Step: {checkpoint_data['step']}")
            console.print(f"  Loss: {checkpoint_data['loss']:.6f}")
            
            return {
                'success': True,
                'checkpoint_path': str(checkpoint_path),
                'step': checkpoint_data['step'],
                'loss': checkpoint_data['loss'],
                'config': checkpoint_data.get('config'),
                'metadata': checkpoint_data.get('metadata', {}),
                'error': None
            }
            
        except Exception as e:
            console.print(f"[red]Error loading checkpoint: {e}[/red]")
            logger.error(f"Checkpoint load failed: {e}")
            
            return {
                'success': False,
                'checkpoint_path': str(checkpoint_path),
                'step': 0,
                'loss': 0.0,
                'config': None,
                'metadata': {},
                'error': str(e)
            }
    
    def export_checkpoint(
        self,
        checkpoint_name: str,
        output_path: Union[str, Path],
        include_optimizer: bool = False
    ) -> Dict[str, Any]:
        """
        Export checkpoint for sharing or deployment.
        
        Args:
            checkpoint_name: Name of checkpoint to export
            output_path: Output path for exported checkpoint
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Export results dictionary
        """
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        output_path = Path(output_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        
        console.print(f"[bold blue]Exporting checkpoint: {checkpoint_name}[/bold blue]")
        
        try:
            # Create export directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy essential files
            files_to_copy = ["metadata.json"]
            
            # Copy LoRA weights
            for file in checkpoint_path.glob("*.safetensors"):
                files_to_copy.append(file.name)
            
            for file in checkpoint_path.glob("*.bin"):
                files_to_copy.append(file.name)
            
            # Optionally copy optimizer state
            if include_optimizer:
                files_to_copy.append("optimizer_state.pt")
                files_to_copy.append("scheduler_state.pt")
            
            # Copy files
            for file_name in files_to_copy:
                src_file = checkpoint_path / file_name
                if src_file.exists():
                    dst_file = output_path / file_name
                    shutil.copy2(src_file, dst_file)
            
            # Create export manifest
            manifest = {
                'exported_from': str(checkpoint_path),
                'export_timestamp': time.time(),
                'files_included': files_to_copy,
                'includes_optimizer': include_optimizer,
                'checkpoint_name': checkpoint_name,
            }
            
            with open(output_path / "export_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            console.print(f"[green]✓ Checkpoint exported successfully[/green]")
            console.print(f"  Output: {output_path}")
            console.print(f"  Files: {len(files_to_copy)}")
            
            return {
                'success': True,
                'output_path': str(output_path),
                'files_count': len(files_to_copy),
                'error': None
            }
            
        except Exception as e:
            console.print(f"[red]Error exporting checkpoint: {e}[/red]")
            return {
                'success': False,
                'output_path': str(output_path),
                'files_count': 0,
                'error': str(e)
            }


# Global checkpoint manager instance
checkpoint_manager = None


def get_checkpoint_manager(output_dir: str, **kwargs) -> CheckpointManager:
    """Get or create checkpoint manager instance."""
    global checkpoint_manager
    if checkpoint_manager is None or checkpoint_manager.output_dir != Path(output_dir):
        checkpoint_manager = CheckpointManager(output_dir, **kwargs)
    return checkpoint_manager