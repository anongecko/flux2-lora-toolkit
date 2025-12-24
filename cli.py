"""
Command-line interface for Flux2 LoRA Training Toolkit.

This module provides a comprehensive CLI for training, evaluating,
and managing LoRA models for Flux2-dev.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from flux2_lora.utils.config_manager import ConfigManager, config_manager
from flux2_lora.utils.hardware_utils import hardware_manager

# Initialize Rich console for beautiful output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="flux2-lora",
    help="Flux2-dev LoRA Training Toolkit - Train high-quality LoRA models",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add subcommands
train_app = typer.Typer(help="Training commands")
eval_app = typer.Typer(help="Evaluation commands")
data_app = typer.Typer(help="Dataset management commands")
system_app = typer.Typer(help="System information and diagnostics")

app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(system_app, name="system")


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        from flux2_lora import __version__

        console.print(f"Flux2 LoRA Training Toolkit v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, help="Show version and exit"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
):
    """🚀 Flux2-dev LoRA Training Toolkit

    A comprehensive toolkit for training high-quality LoRA models for Flux2-dev
    with real-time monitoring, automatic quality assessment, and an intuitive interface.

    \b
    GETTING STARTED:
      • Web Interface: Run 'flux2-lora app' for the graphical interface (recommended for beginners)
      • Quick Training: 'flux2-lora train --preset character --dataset ./data --output ./output'
      • Optimization: 'flux2-lora train optimize --dataset ./data' (advanced)
      • Help: Use --help with any command for detailed options

    \b
    WORKFLOWS:
      1. Prepare Dataset: Organize images with descriptive captions
      2. Analyze Dataset: Use 'flux2-lora data analyze' to check data quality
      3. Optimize (Optional): Run 'flux2-lora train optimize' for best settings
      4. Train LoRA: Use presets or optimized config for training
      5. Evaluate: Test checkpoints and compare performance with 'flux2-lora eval'
      6. Iterate: Refine based on results and train again if needed

    \b
    COMMON COMMANDS:
      flux2-lora app                                           # Launch web interface
      flux2-lora data analyze --dataset ./data                 # Check dataset quality
      flux2-lora train optimize --dataset ./data               # Find best settings
      flux2-lora train --preset character --dataset ./photos   # Train character LoRA
      flux2-lora eval assess-quality --checkpoint model.safetensors  # Test quality
      flux2-lora eval compare model1.safetensors model2.safetensors   # Compare models
      flux2-lora system info                                   # Check system compatibility

    \b
    WEB INTERFACE FEATURES:
      • Visual dataset upload and validation
      • Real-time training progress with graphs
      • Interactive quality assessment and comparison
      • Hyperparameter optimization with presets
      • Comprehensive help and guidance throughout

    For detailed documentation, visit: https://github.com/your-repo/docs
    """
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)


@train_app.command()
def train(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file (alternative to --preset)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to training dataset directory containing images and captions",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: str = typer.Option(
        "./output",
        "--output",
        "-o",
        help="Output directory for checkpoints, logs, and validation samples",
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume training from checkpoint file path",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use preset configuration: 'character' (people/creatures), 'style' (artistic styles), 'concept' (objects/scenes)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configuration and dataset without starting training",
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-s",
        help="Override number of training steps (default: preset-specific)",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr",
        help="Override learning rate (default: 1e-4 for character/style, 5e-5 for concept)",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override batch size (reduce if GPU memory issues occur)",
    ),
    dtype: Optional[str] = typer.Option(
        None,
        "--dtype",
        help="Override data type: 'float16', 'bfloat16', 'float32' (default: config-specific)",
    ),
    use_wandb: Optional[bool] = typer.Option(
        None,
        "--wandb/--no-wandb",
        help="Enable/disable Weights & Biases experiment tracking",
    ),
    force_cpu_loading: bool = typer.Option(
        False,
        "--force-cpu-loading",
        help="Force CPU-first loading strategy (slower but more reliable for memory issues)",
    ),
):
    """🎨 Train a LoRA model on Flux2-dev

    This command trains a LoRA adapter for the Flux2-dev model using your dataset.

    \b
    BASIC USAGE:
      # Quick training with character preset
      flux2-lora train --preset character --dataset ./my_photos

      # Training with custom config
      flux2-lora train --config my_config.yaml --dataset ./data

    \b
    PRESETS:
      character  → Optimized for training characters, people, or creatures
      style      → Optimized for artistic styles and painting techniques
      concept    → Optimized for objects, scenes, or abstract concepts

    \b
    DATASET FORMAT:
      Your dataset directory should contain:
      • Images: .jpg, .png, .webp files (1024x1024+ recommended)
      • Captions: .txt files with same name as images
      • Example: photo_001.jpg + photo_001.txt

    \b
    TRAINING TIPS:
      • Start with 10-50 high-quality images
      • Use descriptive captions including trigger words
      • Monitor loss - should steadily decrease
      • Save checkpoints every 100-200 steps
      • Stop when validation samples look good (avoid overfitting)

    \b
    GPU MEMORY ISSUES:
      • Use --dtype float16 for 50% less memory usage (recommended for memory issues)
      • Use --force-cpu-loading for reliable loading (slower but works with corrupted GPU memory)
      • Reduce --batch-size (try 2 or 4)
      • Lower LoRA rank (try 16 instead of 32)
      • Close other GPU applications
    """

    console.print("[bold blue]🚀 Starting LoRA Training[/bold blue]")

    # Validate that either config or preset is provided
    if not config and not preset:
        console.print("[red]❌ Either --config or --preset must be specified[/red]")
        raise typer.Exit(1)

    try:
        # Load configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading configuration...", total=None)

            if preset:
                # Load preset configuration
                base_config = config_manager.get_preset_config(preset)
                console.print(f"✅ Loaded preset configuration: [green]{preset}[/green]")
            else:
                # Load custom configuration
                base_config = config_manager.load_config(config)
                console.print(f"✅ Loaded configuration: [green]{config}[/green]")

            # Apply command-line overrides
            if steps:
                base_config.training.max_steps = steps
                console.print(f"✅ Override steps: [green]{steps}[/green]")

            if learning_rate:
                base_config.training.learning_rate = learning_rate
                console.print(f"✅ Override learning rate: [green]{learning_rate}[/green]")

            if batch_size:
                base_config.training.batch_size = batch_size
                console.print(f"✅ Override batch size: [green]{batch_size}[/green]")

            if dtype:
                base_config.model.dtype = dtype
                console.print(f"✅ Override dtype: [green]{dtype}[/green]")

            if use_wandb is not None:
                base_config.logging.wandb = use_wandb
                console.print(f"✅ Override WandB: [green]{use_wandb}[/green]")

            # Set dataset and output paths
            base_config.data.dataset_path = dataset
            base_config.output.output_dir = output_dir

            progress.update(task, description="Validating configuration...")

            # Validate configuration
            warnings = config_manager.validate_config_values(base_config)
            if warnings:
                console.print("[yellow]⚠️  Configuration warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")

            progress.update(task, description="Checking system requirements...")

            # Check system requirements
            system_info = hardware_manager.get_system_info()
            if not system_info.cuda_available:
                console.print("[red]❌ CUDA not available - training requires GPU[/red]")
                raise typer.Exit(1)

            if not system_info.gpus:
                console.print("[red]❌ No GPUs detected[/red]")
                raise typer.Exit(1)

            # Select best GPU
            best_gpu_id = hardware_manager.select_best_gpu(
                min_memory_mb=24576  # Require at least 24GB for FLUX2 LoRA training
            )
            if best_gpu_id is None:
                console.print("[red]❌ No suitable GPU found[/red]")
                raise typer.Exit(1)

            base_config.model.device = f"cuda:{best_gpu_id}"

            # Get optimization recommendations
            optimizations = hardware_manager.optimize_memory_settings(base_config)
            if optimizations:
                console.print("[yellow]💡 Optimization recommendations:[/yellow]")
                for key, value in optimizations.items():
                    console.print(f"  • {key}: {value}")
                    # Apply optimizations
                    if hasattr(base_config.training, key):
                        setattr(base_config.training, key, value)
                    elif hasattr(base_config.data, key):
                        setattr(base_config.data, key, value)

            progress.update(task, description="Validating dataset...")

            # Validate dataset
            if not Path(dataset).exists():
                console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
                raise typer.Exit(1)

            # TODO: Add dataset validation logic
            console.print(f"✅ Dataset found: [green]{dataset}[/green]")

            if dry_run:
                console.print("[green]✅ Dry run completed - configuration is valid[/green]")
                console.print("\n[bold]Configuration Summary:[/bold]")
                print_config_summary(base_config)
                raise typer.Exit(0)

            progress.update(task, description="Initializing trainer...")

            # Initialize and run trainer
            progress.update(task, description="Loading model and dataset...")

            # Import training components
            from flux2_lora.core.model_loader import ModelLoader
            from flux2_lora.core.trainer import LoRATrainer
            from flux2_lora.data.dataset import LoRADataset, create_dataloader

            # Load model
            model_loader = ModelLoader()
            model, model_metadata = model_loader.load_flux2_dev(
                model_name=base_config.model.base_model,
                dtype=getattr(torch, base_config.model.dtype),
                device=base_config.model.device,
                cache_dir=base_config.model.cache_dir,
                torch_compile=base_config.model.torch_compile,
                attention_implementation=base_config.model.attention_implementation,
                force_cpu_loading=force_cpu_loading,
            )

            # Load dataset
            # Map caption_format to caption_sources list
            caption_sources = (
                [base_config.data.caption_format]
                if base_config.data.caption_format != "auto"
                else ["txt", "caption", "json", "exif"]
            )

            train_dataset = LoRADataset(
                data_dir=dataset,
                resolution=base_config.data.resolution,
                caption_sources=caption_sources,
                cache_images=base_config.data.cache_images,
                validate_captions=base_config.data.validate_captions,
                augmentation_config=base_config.augmentation.__dict__,
            )

            # Create dataloader
            train_dataloader = create_dataloader(
                dataset=train_dataset,
                batch_size=base_config.training.batch_size,
                num_workers=base_config.data.num_workers,
                pin_memory=base_config.data.pin_memory,
                shuffle=True,
            )

            progress.update(task, description="Initializing trainer...")

            # Initialize trainer
            trainer = LoRATrainer(model=model, config=base_config, output_dir=output_dir)

            # Determine number of steps
            num_steps = base_config.training.max_steps

            progress.update(task, description="Starting training...")

            console.print("\n[bold]Configuration Summary:[/bold]")
            print_config_summary(base_config)

            console.print(f"\n[green]✅ Ready to train with {len(system_info.gpus)} GPU(s)[/green]")
            console.print(f"   • Selected GPU: {system_info.gpus[best_gpu_id].name}")
            console.print(f"   • Dataset: {dataset} ({len(train_dataset)} images)")
            console.print(f"   • Output: {output_dir}")
            console.print(f"   • Steps: {num_steps}")
            console.print(f"   • Batch size: {base_config.training.batch_size}")
            console.print(f"   • Learning rate: {base_config.training.learning_rate}")

            # Start training
            console.print(f"\n[bold blue]🚀 Starting LoRA Training[/bold blue]")

            # Run training
            training_results = trainer.train(
                train_dataloader=train_dataloader, num_steps=num_steps, resume_from=resume_from
            )

            # Print results
            if training_results["success"]:
                console.print(f"\n[bold green]🎉 Training completed successfully![/bold green]")
                console.print(f"   • Total steps: {training_results['global_step']}")
                console.print(f"   • Training time: {training_results['total_time']:.2f}s")
                console.print(f"   • Steps/sec: {training_results['steps_per_second']:.2f}")
                console.print(f"   • Best loss: {training_results['best_loss']:.6f}")
                console.print(f"   • Checkpoints saved: {training_results['checkpoints_saved']}")
                console.print(f"   • Output directory: {training_results['output_dir']}")
            else:
                console.print(f"\n[red]❌ Training failed[/red]")
                console.print(f"   Error: {training_results.get('error', 'Unknown error')}")
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@train_app.command()
def resume(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to checkpoint file to resume from",
        exists=True,
    ),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        "-s",
        help="Additional training steps",
    ),
):
    """Resume training from a checkpoint."""
    console.print(f"[bold blue]🔄 Resuming training from {checkpoint}[/bold blue]")
    console.print("[yellow]🚧 Resume functionality not yet implemented[/yellow]")


@train_app.command()
def optimize(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to training dataset directory for optimization",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: str = typer.Option(
        "./optimization_results",
        "--output",
        "-o",
        help="Output directory for optimization results and trials",
    ),
    trials: int = typer.Option(
        50,
        "--trials",
        "-t",
        help="Number of optimization trials (default: 50)",
        min=5,
        max=200,
    ),
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout in hours for optimization (optional)",
        min=0.1,
    ),
    study_name: str = typer.Option(
        "flux2_lora_optimization",
        "--study-name",
        help="Name for the Optuna study",
    ),
    base_model: str = typer.Option(
        "/path/to/black-forest-labs/FLUX.2-dev",
        "--base-model",
        help="Local path to FLUX2-dev model directory",
    ),
    max_steps: int = typer.Option(
        500,
        "--max-steps",
        help="Training steps per trial (default: 500 for speed)",
        min=100,
        max=2000,
    ),
):
    """🎯 Optimize hyperparameters for LoRA training using Optuna.

    Automatically finds the best LoRA rank, alpha, learning rate, batch size,
    and gradient accumulation settings for your dataset.

    \b
    OPTIMIZATION PROCESS:
      1. Runs multiple training trials with different hyperparameters
      2. Evaluates each trial's performance using quality metrics
      3. Uses Bayesian optimization to find optimal settings
      4. Provides final recommendations for production training

    \b
    WHAT GETS OPTIMIZED:
      • LoRA rank (4-128) - Model capacity vs. training speed
      • LoRA alpha (4-128) - LoRA strength scaling
      • Learning rate (1e-6 to 1e-2) - Training convergence speed
      • Batch size (1, 2, 4, 8, 16) - GPU memory utilization
      • Gradient accumulation (1, 2, 4, 8) - Effective batch size

    \b
    EXPECTED DURATION:
      • 50 trials: 10-20 hours (depending on GPU and dataset size)
      • Each trial: ~10-30 minutes for 500 steps
      • Use --trials 20 for quicker results during development

    \b
    OUTPUT FILES:
      • best_config.yaml - Optimal configuration for production training
      • optimization_results.json - Complete optimization summary
      • trials_data.json - Detailed results from all trials
      • trial_N/ directories - Individual trial checkpoints and logs

    \b
    EXAMPLE USAGE:
      # Basic optimization (50 trials)
      flux2-lora train optimize --dataset ./my_dataset

      # Quick optimization for testing
      flux2-lora train optimize --dataset ./data --trials 20 --max-steps 300

      # Custom output location
      flux2-lora train optimize --dataset ./data --output ./my_optimization
    """
    try:
        from flux2_lora.optimization import create_optimizer

        console.print("[bold blue]🎯 Starting Hyperparameter Optimization[/bold blue]")
        console.print(f"[blue]Dataset:[/blue] {dataset}")
        console.print(f"[blue]Trials:[/blue] {trials}")
        console.print(f"[blue]Output:[/blue] {output_dir}")

        if timeout:
            console.print(f"[blue]Timeout:[/blue] {timeout} hours")

        # Check if Optuna is available
        try:
            import optuna
        except ImportError:
            console.print("[red]❌ Optuna is required for hyperparameter optimization[/red]")
            console.print("Install with: pip install optuna")
            console.print("Or: pip install flux2-lora-training-toolkit[optimization]")
            raise typer.Exit(1)

        # Create optimizer
        optimizer = create_optimizer(
            n_trials=trials,
            dataset_path=dataset,
            output_dir=output_dir,
            timeout_hours=timeout,
            max_steps=max_steps,
        )

        # Run optimization with progress
        with console.status("[bold green]Running hyperparameter optimization...[/bold green]"):
            results = optimizer.optimize(
                dataset_path=dataset, base_model=base_model, study_name=study_name
            )

        # Display results
        console.print("\n[bold green]✅ Optimization Complete![/bold green]")

        # Best parameters
        console.print("\n[bold]🏆 Best Hyperparameters Found:[/bold]")
        best_params = results["best_params"]
        console.print(f"  • LoRA Rank: [cyan]{best_params['rank']}[/cyan]")
        console.print(f"  • LoRA Alpha: [cyan]{best_params['alpha']}[/cyan]")
        console.print(f"  • Learning Rate: [cyan]{best_params['learning_rate']:.2e}[/cyan]")
        console.print(f"  • Batch Size: [cyan]{best_params['batch_size']}[/cyan]")
        console.print(
            f"  • Gradient Accumulation: [cyan]{best_params['gradient_accumulation']}[/cyan]"
        )
        console.print(f"  • Quality Score: [green]{results['best_score']:.4f}[/green]")

        # Trial statistics
        console.print("\n[bold]📊 Optimization Statistics:[/bold]")
        console.print(f"  • Total Trials: [cyan]{results['n_trials']}[/cyan]")
        console.print(f"  • Completed: [green]{results['completed_trials']}[/green]")
        console.print(f"  • Pruned: [yellow]{results['pruned_trials']}[/yellow]")
        console.print(f"  • Failed: [red]{results['failed_trials']}[/red]")

        # Output files
        console.print("\n[bold]📄 Output Files:[/bold]")
        console.print(f"  • Best Config: [cyan]{output_dir}/best_config.yaml[/cyan]")
        console.print(f"  • Results Summary: [cyan]{output_dir}/optimization_results.json[/cyan]")
        console.print(f"  • Trial Data: [cyan]{output_dir}/trials_data.json[/cyan]")

        # Next steps
        console.print("\n[bold]🚀 Next Steps:[/bold]")
        console.print("  1. Review the best_config.yaml file")
        console.print("  2. Run production training with optimized settings:")
        console.print(f"     [dim]flux2-lora train --config {output_dir}/best_config.yaml[/dim]")
        console.print("  3. Use the optimized settings as a starting point for fine-tuning")

        console.print(
            "\n[green]🎉 Optimization complete! Your LoRA training is now optimized.[/green]"
        )

    except Exception as e:
        console.print(f"[red]❌ Optimization failed: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command()
def compare(
    checkpoints: list[str] = typer.Argument(..., help="List of checkpoint files to compare"),
    prompts: Optional[list[str]] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Test prompts (can be specified multiple times)",
    ),
    output: str = typer.Option(
        "./comparison",
        "--output",
        "-o",
        help="Output directory for comparison results",
    ),
    steps: int = typer.Option(
        25,
        "--steps",
        "-s",
        help="Number of inference steps",
    ),
    guidance_scale: float = typer.Option(
        7.5,
        "--guidance-scale",
        "-g",
        help="Guidance scale for generation",
    ),
    seed: Optional[int] = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducible results",
    ),
):
    """Compare multiple checkpoints side-by-side."""
    from flux2_lora.evaluation import CheckpointComparator

    console.print("[bold blue]🔍 Comparing checkpoints[/bold blue]")

    if len(checkpoints) < 2:
        console.print("[red]Error: At least 2 checkpoints are required for comparison[/red]")
        raise typer.Exit(1)

    # Validate checkpoint files exist
    for checkpoint in checkpoints:
        if not os.path.exists(checkpoint):
            console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
            raise typer.Exit(1)

    try:
        # Create comparator
        comparator = CheckpointComparator(
            checkpoint_paths=checkpoints,
            test_prompts=prompts,
            output_dir=output,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Run comparison
        with console.status("[bold green]Running comparison...[/bold green]"):
            results = comparator.run_full_comparison()

        # Display results
        console.print("[green]✅ Comparison complete![/green]")
        console.print(f"[blue]📊 Checkpoints compared:[/blue] {len(checkpoints)}")
        console.print(f"[blue]🖼️  Test prompts:[/blue] {len(comparator.test_prompts)}")
        console.print(f"[blue]📄 Report saved to:[/blue] {results['report_path']}")

        # Show metrics summary
        console.print("\n[bold]📈 Metrics Summary:[/bold]")
        for checkpoint_name, metrics in results["metrics"].items():
            clip_score = metrics.get("clip_score", 0)
            diversity = metrics.get("diversity_score", 0)
            console.print(".3f")

    except Exception as e:
        console.print(f"[red]Error during comparison: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command()
def test(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Checkpoint file to test",
        exists=True,
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Test prompt",
    ),
    output: str = typer.Option(
        "./test_output",
        "--output",
        "-o",
        help="Output directory for test results",
    ),
):
    """Test a single checkpoint with a prompt."""
    console.print("[bold blue]🧪 Testing checkpoint[/bold blue]")
    console.print("[yellow]🚧 Testing functionality not yet implemented[/yellow]")


@eval_app.command()
def assess_quality(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Checkpoint file to assess",
        exists=True,
    ),
    prompts: Optional[list[str]] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Test prompts (can be specified multiple times)",
    ),
    training_data: Optional[str] = typer.Option(
        None,
        "--training-data",
        "-t",
        help="Path to training dataset for overfitting detection",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: str = typer.Option(
        "./quality_assessment.json",
        "--output",
        "-o",
        help="Output file for assessment results",
    ),
    samples_per_prompt: int = typer.Option(
        3,
        "--samples-per-prompt",
        "-s",
        help="Number of samples to generate per prompt",
    ),
    steps: int = typer.Option(
        25,
        "--steps",
        help="Number of inference steps",
    ),
    guidance_scale: float = typer.Option(
        7.5,
        "--guidance-scale",
        "-g",
        help="Guidance scale for generation",
    ),
):
    """Assess quality of a single checkpoint."""
    from flux2_lora.evaluation import QualityAssessor
    import json

    console.print("[bold blue]📊 Assessing checkpoint quality[/bold blue]")

    # Load training images if provided
    training_images = None
    if training_data:
        try:
            from flux2_lora.data.dataset import LoRADataset

            dataset = LoRADataset(training_data)
            training_images = dataset.get_sample_images(max_images=50)  # Sample training images
            console.print(
                f"[green]Loaded {len(training_images)} training images for overfitting detection[/green]"
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load training data: {e}[/yellow]")

    # Default prompts if none provided
    if not prompts:
        prompts = [
            "A portrait of a person with distinctive features",
            "A landscape scene with mountains and water",
            "A detailed close-up of an object",
            "An artistic composition with dramatic lighting",
        ]

    try:
        assessor = QualityAssessor()

        with console.status("[bold green]Assessing quality...[/bold green]"):
            results = assessor.assess_checkpoint_quality(
                checkpoint_path=checkpoint,
                test_prompts=prompts,
                training_images=training_images,
                num_samples_per_prompt=samples_per_prompt,
                generation_kwargs={
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                },
            )

        # Save results
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Display summary
        console.print("[green]✅ Quality assessment complete![/green]")
        console.print(f"[blue]📄 Results saved to:[/blue] {output}")
        console.print(f"[blue]📊 Quality Score:[/blue] {results.get('quality_score', 0):.3f}")
        console.print(f"[blue]🎯 CLIP Score:[/blue] {results.get('clip_score', 0):.3f}")
        console.print(f"[blue]🎨 Diversity Score:[/blue] {results.get('diversity_score', 0):.3f}")
        if "is_overfitting" in results:
            status = "⚠️ Yes" if results["is_overfitting"] else "✅ No"
            console.print(f"[blue]🔍 Overfitting:[/blue] {status}")

    except Exception as e:
        console.print(f"[red]Error during quality assessment: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command()
def select_best(
    checkpoints: list[str] = typer.Argument(..., help="List of checkpoint files to evaluate"),
    prompts: Optional[list[str]] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Test prompts (can be specified multiple times)",
    ),
    training_data: Optional[str] = typer.Option(
        None,
        "--training-data",
        "-t",
        help="Path to training dataset for overfitting detection",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    quality_weight: float = typer.Option(
        0.5,
        "--quality-weight",
        help="Weight for quality score in selection (0.0-1.0)",
    ),
    diversity_weight: float = typer.Option(
        0.5,
        "--diversity-weight",
        help="Weight for diversity score in selection (0.0-1.0)",
    ),
    overfitting_penalty: float = typer.Option(
        0.2,
        "--overfitting-penalty",
        help="Penalty weight for overfitting risk (0.0-1.0)",
    ),
    output: str = typer.Option(
        "./best_checkpoint_selection.json",
        "--output",
        "-o",
        help="Output file for selection results",
    ),
    samples_per_prompt: int = typer.Option(
        2,
        "--samples-per-prompt",
        help="Number of samples per prompt (lower for faster comparison)",
    ),
):
    """Select the best checkpoint based on quality metrics."""
    from flux2_lora.evaluation import QualityAssessor, BestCheckpointSelector
    import json

    console.print("[bold blue]🏆 Selecting best checkpoint[/bold blue]")

    # Validate inputs
    if not checkpoints:
        console.print("[red]Error: At least one checkpoint is required[/red]")
        raise typer.Exit(1)

    # Check checkpoint files exist
    for checkpoint in checkpoints:
        if not os.path.exists(checkpoint):
            console.print(f"[red]Error: Checkpoint not found: {checkpoint}[/red]")
            raise typer.Exit(1)

    # Load training images if provided
    training_images = None
    if training_data:
        try:
            from flux2_lora.data.dataset import LoRADataset

            dataset = LoRADataset(training_data)
            training_images = dataset.get_sample_images(max_images=30)  # Smaller sample for speed
            console.print(f"[green]Loaded {len(training_images)} training images[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load training data: {e}[/yellow]")

    # Default prompts if none provided
    if not prompts:
        prompts = [
            "A portrait of a person",
            "A landscape scene",
            "An object in detail",
        ]

    try:
        assessor = QualityAssessor()
        selector = BestCheckpointSelector(
            quality_weight=quality_weight,
            diversity_weight=diversity_weight,
            overfitting_penalty=overfitting_penalty,
        )

        with console.status("[bold green]Evaluating checkpoints...[/bold green]"):
            comparison_results = assessor.compare_checkpoints_quality(
                checkpoint_paths=checkpoints,
                test_prompts=prompts,
                training_images=training_images,
                num_samples_per_prompt=samples_per_prompt,
            )

        with console.status("[bold green]Selecting best checkpoint...[/bold green]"):
            selection_results = selector.select_best_checkpoint(comparison_results, explain=True)

        # Save results
        with open(output, "w") as f:
            json.dump(
                {
                    "selection": selection_results,
                    "comparison": comparison_results,
                },
                f,
                indent=2,
                default=str,
            )

        # Display results
        console.print("[green]✅ Best checkpoint selection complete![/green]")
        console.print(f"[blue]🏆 Selected:[/blue] {selection_results['selected_checkpoint']}")
        console.print(
            f"[blue]📊 Composite Score:[/blue] {selection_results['composite_score']:.3f}"
        )
        console.print(f"[blue]📄 Full results saved to:[/blue] {output}")

        # Show explanation
        if "explanation" in selection_results:
            console.print("\n[bold]📋 Selection Explanation:[/bold]")
            console.print(selection_results["explanation"])

    except Exception as e:
        console.print(f"[red]Error during checkpoint selection: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command()
def test_prompts(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Checkpoint file to test",
        exists=True,
    ),
    concept: Optional[str] = typer.Option(
        None,
        "--concept",
        help="Concept name (auto-detected from checkpoint if not provided)",
    ),
    trigger_word: Optional[str] = typer.Option(
        None,
        "--trigger-word",
        help="Trigger word to test (auto-detected if not provided)",
    ),
    output: str = typer.Option(
        "./prompt_test_report.md",
        "--output",
        "-o",
        help="Output file for test report",
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        help="Report format (markdown or html)",
    ),
    samples_per_prompt: int = typer.Option(
        3,
        "--samples-per-prompt",
        help="Number of samples to generate per prompt",
    ),
    categories: Optional[list[str]] = typer.Option(
        None,
        "--category",
        help="Test categories to include (basic, positioning, composition, complexity, negative)",
    ),
):
    """Run comprehensive prompt testing suite."""
    from flux2_lora.evaluation import PromptTester

    console.print("[bold blue]🧪 Running prompt testing suite[/bold blue]")

    try:
        # Initialize tester
        tester = PromptTester(
            checkpoint_path=checkpoint,
            trigger_word=trigger_word,
        )

        # Create test suite
        with console.status("[bold green]Creating test suite...[/bold green]"):
            test_suite = tester.create_test_suite(concept=concept)

        # Filter by categories if specified
        if categories:
            test_suite = [test for test in test_suite if test.category in categories]

        console.print(f"[green]Created test suite with {len(test_suite)} prompts[/green]")

        # Show category breakdown
        category_counts = {}
        for test in test_suite:
            category_counts[test.category] = category_counts.get(test.category, 0) + 1

        console.print("[blue]Category breakdown:[/blue]")
        for category, count in category_counts.items():
            console.print(f"  - {category}: {count} tests")

        # Run tests with progress
        results = {"results": [], "analysis": {}, "summary": {}, "test_suite_info": {}}

        def progress_callback(current, total, result):
            console.print(
                f"[dim]Completed test {current}/{total}: {result.test.name} ({result.success_rating})[/dim]"
            )

        with console.status("[bold green]Running prompt tests...[/bold green]"):
            results = tester.run_test_suite(test_suite, progress_callback=progress_callback)

        # Generate report
        with console.status("[bold green]Generating report...[/bold green]"):
            report_path = tester.generate_report(results, output_path=output, format=format)

        # Display summary
        summary = results["summary"]
        analysis = results["analysis"]

        console.print("[green]✅ Prompt testing complete![/green]")
        console.print(f"[blue]📊 Tests run:[/blue] {summary.get('total_tests', 0)}")
        console.print(f"[blue]🎯 Success rate:[/blue] {summary.get('success_rate', 0):.1%}")
        console.print(f"[blue]📈 Average score:[/blue] {summary.get('average_score', 0):.3f}")
        console.print(f"[blue]📄 Report saved to:[/blue] {report_path}")

        # Show top categories
        if analysis.get("category_performance"):
            console.print("\n[bold]🏆 Best performing categories:[/bold]")
            sorted_cats = sorted(
                analysis["category_performance"].items(),
                key=lambda x: x[1]["average_score"],
                reverse=True,
            )
            for cat, perf in sorted_cats[:3]:
                console.print(".3f")

        # Show insights
        if analysis.get("recommendations"):
            console.print("\n[bold]💡 Key recommendations:[/bold]")
            for rec in analysis["recommendations"][:2]:
                console.print(f"  • {rec}")

    except Exception as e:
        console.print(f"[red]Error during prompt testing: {e}[/red]")
        raise typer.Exit(1)


@eval_app.command()
def test_prompt_interactive(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Checkpoint file to test",
        exists=True,
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Custom prompt to test",
    ),
    samples: int = typer.Option(
        3,
        "--samples",
        "-s",
        help="Number of samples to generate",
    ),
    output: str = typer.Option(
        "./interactive_test.png",
        "--output",
        "-o",
        help="Output file for generated image",
    ),
):
    """Test a single custom prompt interactively."""
    from flux2_lora.evaluation import PromptTester

    console.print("[bold blue]🎨 Testing custom prompt[/bold blue]")

    try:
        # Initialize tester
        tester = PromptTester(checkpoint_path=checkpoint)
        tester.load_checkpoint()

        # Generate image
        with console.status("[bold green]Generating image...[/bold green]"):
            with torch.no_grad():
                result = tester.model(
                    prompt=prompt,
                    num_inference_steps=tester.num_inference_steps,
                    guidance_scale=tester.guidance_scale,
                    num_images_per_prompt=samples,
                    height=1024,
                    width=1024,
                )

        # Save first image
        result.images[0].save(output)
        console.print(f"[green]✅ Image generated and saved to: {output}[/green]")
        console.print(f"[blue]📝 Prompt:[/blue] {prompt}")
        console.print(f"[blue]🖼️  Samples generated:[/blue] {len(result.images)}")

        if len(result.images) > 1:
            # Save additional samples with numbered names
            base_path = Path(output)
            stem = base_path.stem
            suffix = base_path.suffix

            for i, img in enumerate(result.images[1:], 1):
                sample_path = base_path.parent / f"{stem}_sample_{i}{suffix}"
                img.save(sample_path)
                console.print(f"[dim]Additional sample saved: {sample_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error during prompt testing: {e}[/red]")
        raise typer.Exit(1)


@data_app.command()
def analyze(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for analysis report",
    ),
):
    """Analyze dataset and generate statistics."""
    console.print("[bold blue]📊 Analyzing dataset[/bold blue]")
    console.print("[yellow]🚧 Dataset analysis not yet implemented[/yellow]")


@data_app.command()
def validate(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to dataset directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix common issues",
    ),
):
    """Validate dataset for common issues."""
    console.print("[bold blue]✅ Validating dataset[/bold blue]")
    console.print("[yellow]🚧 Dataset validation not yet implemented[/yellow]")


@data_app.command()
def augment(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Path to dataset directory to augment",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: str = typer.Option(
        "./augmented_dataset",
        "--output",
        "-o",
        help="Output directory for augmented dataset",
    ),
    samples: int = typer.Option(
        100,
        "--samples",
        "-s",
        help="Number of augmented samples to generate",
        min=10,
        max=1000,
    ),
    image_augmentations: bool = typer.Option(
        True,
        "--image-augs/--no-image-augs",
        help="Enable/disable image augmentations",
    ),
    text_augmentations: bool = typer.Option(
        True,
        "--text-augs/--no-text-augs",
        help="Enable/disable text augmentations",
    ),
    preserve_originals: bool = typer.Option(
        True,
        "--preserve-originals/--no-preserve-originals",
        help="Include original images in augmented dataset",
    ),
):
    """🎨 Generate augmented dataset samples for training.

    Creates additional training samples by applying various augmentations
    to your existing dataset. Helps improve model generalization and robustness.

    \b
    AUGMENTATION TYPES:
      • Image: Horizontal flips, brightness/contrast adjustments
      • Text: Synonym replacement, random word operations

    \b
    WHEN TO USE:
      • Small datasets (< 50 images) - expand effective dataset size
      • Limited variety - add different angles, lighting, compositions
      • Overfitting prevention - create diverse training samples
      • Generalization improvement - make models more robust

    \b
    BEST PRACTICES:
      • Keep augmentation intensity reasonable (not too extreme)
      • Include original samples alongside augmented ones
      • Review augmented samples before training
      • Balance augmentation types for diversity

    \b
    EXAMPLE USAGE:
      # Basic augmentation (100 samples)
      flux2-lora data augment --dataset ./my_dataset

      # Advanced augmentation with custom settings
      flux2-lora data augment --dataset ./data --samples 200 --output ./big_dataset --no-text-augs
    """
    try:
        from flux2_lora.data.augmentation import DatasetAugmenter, AugmentationConfig
        from flux2_lora.data.dataset import LoRADataset
        import shutil
        from PIL import Image

        console.print("[bold blue]🎨 Generating Augmented Dataset[/bold blue]")
        console.print(f"[blue]Input dataset:[/blue] {dataset}")
        console.print(f"[blue]Output directory:[/blue] {output_dir}")
        console.print(f"[blue]Target samples:[/blue] {samples}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy original dataset if preserving originals
        if preserve_originals:
            console.print("📋 Copying original dataset...")
            for file_path in Path(dataset).iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, output_path / file_path.name)

        # Load original dataset
        original_dataset = LoRADataset(dataset)
        console.print(f"✅ Loaded {len(original_dataset)} original samples")

        # Configure augmentations
        aug_config = AugmentationConfig(
            enabled=True,
            probability=1.0,  # Always augment for generation
            image_augmentations={
                "geometric": {"horizontal_flip": {"enabled": image_augmentations}},
                "color": {
                    "brightness": {"enabled": image_augmentations, "limit": 0.1},
                    "contrast": {"enabled": image_augmentations, "limit": 0.1},
                },
            }
            if image_augmentations
            else {},
            text_augmentations={
                "synonym_replacement": {
                    "enabled": text_augmentations,
                    "probability": 0.2,
                    "max_replacements": 2,
                }
            }
            if text_augmentations
            else {},
        )

        augmenter = DatasetAugmenter(aug_config)

        # Generate augmented samples
        generated = 0
        original_count = len(original_dataset) if preserve_originals else 0

        console.print("🎨 Generating augmented samples...")
        with console.status("[bold green]Generating augmentations...[/bold green]"):
            for i in range(samples):
                # Sample random original
                idx = random.randint(0, len(original_dataset) - 1)
                sample = original_dataset[idx]

                # Apply augmentation
                image, caption = augmenter.augment_sample(
                    Image.open(sample["image_path"]), sample["caption"]
                )

                # Save augmented sample
                base_name = Path(sample["image_path"]).stem
                aug_name = "08d"

                # Save image
                image.save(output_path / f"{aug_name}.jpg", "JPEG", quality=95)

                # Save caption
                with open(output_path / f"{aug_name}.txt", "w", encoding="utf-8") as f:
                    f.write(caption)

                generated += 1

                if generated % 10 == 0:
                    console.print(f"  Generated {generated}/{samples} samples...")

        total_samples = original_count + generated
        console.print("\n[bold green]✅ Augmentation Complete![/bold green]")
        console.print(f"[blue]📊 Original samples:[/blue] {len(original_dataset)}")
        console.print(f"[blue]🎨 Generated samples:[/blue] {generated}")
        console.print(f"[blue]📂 Total samples:[/blue] {total_samples}")
        console.print(f"[blue]📁 Output directory:[/blue] {output_dir}")

        # Show augmentation statistics
        stats = augmenter.get_augmentation_stats()
        console.print("\n[bold]Augmentation Summary:[/bold]")
        if stats.get("image_augmentations"):
            console.print(
                f"[blue]🖼️  Image augmentations enabled:[/blue] {sum(stats['image_augmentations'].values())}"
            )
        if stats.get("text_augmentations"):
            console.print(
                f"[blue]📝 Text augmentations enabled:[/blue] {stats['text_augmentations']['enabled']}"
            )

        console.print(
            "\n[green]🎉 Dataset augmentation complete! Use the augmented dataset for training.[/green]"
        )

    except Exception as e:
        console.print(f"[red]❌ Augmentation failed: {e}[/red]")
        raise typer.Exit(1)


@system_app.command()
def info():
    """Show system information and hardware details."""
    console.print("[bold blue]💻 System Information[/bold blue]")
    hardware_manager.print_system_info()


@system_app.command()
def gpu():
    """Show detailed GPU information."""
    console.print("[bold blue]🎮 GPU Information[/bold blue]")
    hardware_manager.print_gpu_info()


@system_app.command()
def optimize(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file to optimize",
        exists=True,
    ),
):
    """Get optimization recommendations for your hardware."""
    console.print("[bold blue]⚡ Hardware Optimization[/bold blue]")

    # Check H100 optimizations
    recommendations = hardware_manager.check_h100_optimization()
    if recommendations:
        console.print("[yellow]💡 Recommendations:[/yellow]")
        for rec in recommendations:
            console.print(f"  • {rec}")
    else:
        console.print("[green]✅ System appears optimized for H100[/green]")

    if config:
        try:
            cfg = config_manager.load_config(config)
            optimizations = hardware_manager.optimize_memory_settings(cfg)
            if optimizations:
                console.print("\n[yellow]🔧 Configuration optimizations:[/yellow]")
                for key, value in optimizations.items():
                    console.print(f"  • {key}: {value}")
        except Exception as e:
            console.print(f"[red]❌ Error loading config: {e}[/red]")


@system_app.command()
def presets():
    """List available configuration presets."""
    console.print("[bold blue]📋 Available Presets[/bold blue]")

    presets = config_manager.list_presets()
    if presets:
        table = Table()
        table.add_column("Preset Name", style="cyan")
        table.add_column("Description", style="white")

        preset_descriptions = {
            "character": "Optimized for training character-specific LoRAs",
            "style": "Optimized for training artistic style LoRAs",
            "concept": "Optimized for training concept/object LoRAs",
        }

        for preset in presets:
            description = preset_descriptions.get(preset, "Custom preset")
            table.add_row(preset, description)

        console.print(table)
    else:
        console.print("[yellow]No presets found[/yellow]")


def print_config_summary(config):
    """Print a summary of the configuration."""
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", config.model.base_model)
    table.add_row("Data Type", config.model.dtype)
    table.add_row("Device", config.model.device)
    table.add_row("", "")  # Separator
    table.add_row("LoRA Rank", str(config.lora.rank))
    table.add_row("LoRA Alpha", str(config.lora.alpha))
    table.add_row("LoRA Dropout", str(config.lora.dropout))
    table.add_row("", "")  # Separator
    table.add_row("Learning Rate", f"{config.training.learning_rate:.2e}")
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Max Steps", str(config.training.max_steps))
    table.add_row("Gradient Accumulation", str(config.training.gradient_accumulation_steps))
    table.add_row("Optimizer", config.training.optimizer)
    table.add_row("Scheduler", config.training.scheduler)
    table.add_row("", "")  # Separator
    table.add_row("Dataset", config.data.dataset_path)
    table.add_row("Resolution", f"{config.data.resolution}x{config.data.resolution}")
    table.add_row("Caption Format", config.data.caption_format)
    table.add_row("", "")  # Separator
    table.add_row("Output Directory", config.output.output_dir)
    table.add_row("Checkpoint Interval", f"Every {config.output.checkpoint_every_n_steps} steps")

    console.print(table)


if __name__ == "__main__":
    app()
