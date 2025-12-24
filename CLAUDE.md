# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **development toolkit for training LoRA (Low-Rank Adaptation) models on FLUX2-dev**, featuring real-time monitoring, automatic quality assessment, and both CLI and web interfaces. The project is optimized for H100/A100 GPUs and uses Python 3.14+.

**Important**: This targets FLUX2-dev (NOT FLUX1) which requires specific model components and configurations.

## Core Commands

### Setup and Installation
```bash
# Create virtual environment (Python 3.14 required)
python -m venv venv314
source venv314/bin/activate  # On Windows: venv314\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install with optimization features
pip install -e ".[optimization]"

# Verify installation
python cli.py system info
```

### Running the Application

**Web Interface (Recommended for most users):**
```bash
python app.py
# Access at http://localhost:7860
```

**CLI Training:**
```bash
# Train with preset
python cli.py train --preset character --dataset ./my_dataset --output ./output

# Train with custom config
python cli.py train --config my_config.yaml --dataset ./data --output ./results

# Override specific settings
python cli.py train --preset style --steps 2000 --lr 5e-5 --batch-size 4

# Force CPU-first loading (for GPU memory issues)
python cli.py train --preset character --dataset ./data --force-cpu-loading
```

**Hyperparameter Optimization:**
```bash
# Basic optimization (50 trials, ~10-20 hours)
python cli.py train optimize --dataset ./data

# Quick optimization for testing (20 trials)
python cli.py train optimize --dataset ./data --trials 20 --max-steps 300
```

**Evaluation Commands:**
```bash
# Compare multiple checkpoints
python cli.py eval compare checkpoint1.safetensors checkpoint2.safetensors --prompt "test"

# Assess checkpoint quality
python cli.py eval assess-quality --checkpoint model.safetensors --training-data ./dataset

# Run comprehensive prompt testing
python cli.py eval test-prompts --checkpoint model.safetensors --concept "my_character"

# Select best checkpoint automatically
python cli.py eval select-best checkpoint1.safetensors checkpoint2.safetensors
```

**Dataset Commands:**
```bash
# Analyze dataset quality
python cli.py data analyze --dataset ./my_dataset

# Validate dataset structure
python cli.py data validate --dataset ./data --fix

# Augment dataset (helpful for small datasets)
python cli.py data augment --dataset ./data --samples 100 --output ./augmented
```

**System Commands:**
```bash
# Show system/GPU information
python cli.py system info
python cli.py system gpu

# Get optimization recommendations
python cli.py system optimize --config my_config.yaml

# List available presets
python cli.py system presets
```

### Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest --cov=src/flux2_lora --cov-report=html

# Run specific test file
pytest tests/unit/test_config_manager.py -v

# Run integration tests (slower)
pytest tests/integration/ -v

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Run linter (ruff)
ruff check src/

# Auto-fix linting issues
ruff check --fix src/

# Format code (black)
black src/ tests/

# Type checking (mypy)
mypy src/flux2_lora/
```

## Architecture Overview

### Directory Structure
```
src/flux2_lora/
├── core/              # Core training components
│   ├── model_loader.py    # FLUX2-dev model loading with LoRA injection
│   ├── trainer.py         # Main training orchestrator
│   ├── lora_config.py     # LoRA configuration classes
│   └── optimizer.py       # Optimizer and scheduler management
├── data/              # Dataset handling
│   ├── dataset.py         # Dataset class and DataLoader
│   ├── caption_utils.py   # Caption loading and processing
│   └── augmentation.py    # Data augmentation pipeline
├── monitoring/        # Training monitoring and callbacks
│   ├── logger.py          # TensorBoard/W&B logging
│   ├── metrics.py         # Quality metrics (CLIP, etc.)
│   ├── validator.py       # Validation sampling during training
│   └── callbacks.py       # Training callbacks system
├── evaluation/        # Post-training evaluation
│   ├── checkpoint_compare.py   # Compare multiple checkpoints
│   ├── quality_metrics.py      # Quality assessment tools
│   └── prompt_testing.py       # Comprehensive prompt testing
├── optimization/      # Hyperparameter optimization
│   └── hyperparameter_optimizer.py  # Optuna-based optimization
├── ui/                # Gradio web interface
│   ├── gradio_app.py      # Main Gradio application
│   ├── training_tab.py    # Training interface
│   ├── evaluation_tab.py  # Evaluation interface
│   ├── dataset_tab.py     # Dataset tools
│   └── optimization_tab.py # Hyperparameter optimization UI
└── utils/             # Shared utilities
    ├── config_manager.py      # Configuration loading/validation
    ├── hardware_utils.py      # GPU detection and optimization
    └── checkpoint_manager.py  # Checkpoint saving/loading

configs/
├── base_config.yaml       # Base configuration template
└── presets/              # Optimized preset configurations
    ├── character.yaml    # For characters/people (rank=128, higher LR)
    ├── style.yaml        # For artistic styles (rank=64, balanced)
    └── concept.yaml      # For objects/scenes (rank=32, conservative)
```

### Key Design Patterns

**Configuration Management:**
- YAML-based configuration with validation via `ConfigManager`
- Three optimized presets: `character`, `style`, `concept`
- Command-line overrides supported for all config values
- Config schema validation with clear error messages

**Training Pipeline:**
1. `ModelLoader` loads FLUX2-dev and injects LoRA adapters via PEFT
2. `LoRADataset` handles image-caption pairs with preprocessing
3. `LoRATrainer` orchestrates training with callbacks and monitoring
4. `CheckpointManager` handles safetensors checkpoint saving/loading
5. `TrainingLogger` logs to TensorBoard and optionally W&B
6. `ValidationSampler` generates validation samples during training

**Evaluation Pipeline:**
1. `CheckpointComparator` compares multiple checkpoints side-by-side
2. `QualityAssessor` computes CLIP scores, diversity, overfitting detection
3. `PromptTester` runs comprehensive prompt testing suites
4. `BestCheckpointSelector` ranks checkpoints by composite quality score

**Memory Management:**
- Aggressive GPU memory cleanup before model loading
- CPU-first loading strategy available via `--force-cpu-loading`
- Expandable memory segments for large GPUs (H100/A100)
- Gradient checkpointing support for reduced memory usage
- Mixed precision training (bfloat16 default for H100)

## Important Implementation Details

### FLUX2-dev Specific Requirements

**Model Loading:**
- Uses `FluxPipeline` or `Flux2Pipeline` from diffusers
- FLUX2-dev requires different components than FLUX1
- Path format: `/path/to/black-forest-labs/FLUX.2-dev`
- Always use safetensors format for checkpoints (security)

**LoRA Configuration:**
- Target modules: `["to_k", "to_q", "to_v", "to_out.0"]` (attention layers)
- Rank range: 4-128 (higher = more capacity, slower training)
- Alpha typically equals rank for balanced scaling
- Dropout: 0.0-0.1 (0.0 default)

**Data Type Optimization:**
- H100: Use `bfloat16` (native support, best performance)
- A100: Use `bfloat16` or `float16`
- For memory issues: Use `float16` (50% less memory than float32)

### GPU Memory Issues

**Common Problem:** CUDA out of memory during model loading

**Solutions (in order of preference):**
1. Use `--force-cpu-loading` flag (slower but reliable)
2. Reduce batch size: `--batch-size 2` or `--batch-size 1`
3. Use `--dtype float16` for 50% less memory
4. Enable gradient accumulation instead of large batches
5. Close other GPU applications
6. Restart the process/VM to clear fragmented memory

**Memory Debugging:**
- Model loading prints memory stats before/after cleanup
- Watch for warnings about existing GPU allocations
- H100 detection enables specific optimizations automatically

### Training Best Practices

**Dataset Preparation:**
- Directory structure: `image_001.jpg` + `image_001.txt` (caption)
- Image resolution: 1024x1024 minimum recommended
- Caption quality matters: descriptive, specific captions work best
- Quantity: Start with 10-50 high-quality images
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

**Preset Selection:**
- `character`: For people, characters, creatures (rank=128, LR=1e-4)
- `style`: For artistic styles, painting techniques (rank=64, LR=5e-5)
- `concept`: For objects, scenes, abstract concepts (rank=32, LR=5e-5)

**Monitoring Training:**
- Loss should decrease steadily (check TensorBoard)
- Validation samples show training progress visually
- Stop when validation looks good (avoid overfitting)
- Typical training: 500-2000 steps depending on dataset size

**Checkpoint Selection:**
- Don't always use the last checkpoint (may be overfit)
- Compare multiple checkpoints with `eval compare`
- Use `eval select-best` for automatic selection
- Check overfitting with `eval assess-quality --training-data`

### Hyperparameter Optimization

**When to Use:**
- Have a representative dataset ready
- Want 10-30% quality improvement over defaults
- Dataset size differs significantly from typical (10-50 images)
- Custom use case not covered by presets

**What Gets Optimized:**
- LoRA rank (4-128): Model capacity
- LoRA alpha (4-128): Scaling factor
- Learning rate (1e-6 to 1e-2): Convergence speed
- Batch size (1, 2, 4, 8, 16): GPU memory usage
- Gradient accumulation (1, 2, 4, 8): Effective batch size

**Time Expectations:**
- 50 trials: 10-20 hours on H100
- 20 trials: 4-8 hours (good for testing)
- Each trial: ~10-30 minutes for 500 steps

**Output Files:**
- `best_config.yaml`: Use for production training
- `optimization_results.json`: Summary with best parameters
- `trials_data.json`: All trial results for analysis

## Configuration Files

### Base Config (`configs/base_config.yaml`)
Complete template with all available options. Key sections:
- `model`: Base model path, dtype, device, torch_compile
- `lora`: rank, alpha, dropout, target_modules
- `training`: learning_rate, batch_size, max_steps, optimizer, scheduler
- `data`: dataset_path, resolution, caption_format, augmentation
- `validation`: enable, prompts, frequency
- `output`: output_dir, checkpoint_every_n_steps, checkpoints_limit
- `logging`: tensorboard, wandb, log_level
- `callbacks`: checkpoint, early_stopping, validation, lr_scheduler

### Preset Configs (`configs/presets/`)
Optimized for specific use cases:
- **character.yaml**: rank=128, lr=1e-4, higher capacity for complex characters
- **style.yaml**: rank=64, lr=5e-5, balanced for artistic styles
- **concept.yaml**: rank=32, lr=5e-5, conservative for objects/scenes

## Common Debugging Scenarios

### Model Loading Fails with CUDA OOM
```bash
# Use CPU-first loading strategy
python cli.py train --preset character --dataset ./data --force-cpu-loading

# Or use float16 instead of bfloat16
python cli.py train --preset character --dataset ./data --dtype float16
```

### Training Loss Not Decreasing
Check:
1. LoRA actually injected? (trainer prints parameter stats at startup)
2. Learning rate too low? Try `--lr 1e-4` or `--lr 5e-5`
3. Dataset quality issues? Run `data validate --dataset ./data`
4. Batch size too small? Use gradient accumulation: config has `gradient_accumulation_steps`

### Validation Samples Look Identical to Training Images
Overfitting detected. Solutions:
1. Reduce training steps: `--steps 500` instead of 1000
2. Lower LoRA rank: edit config or use concept preset
3. Increase dataset size or use augmentation: `data augment`
4. Add regularization: set `lora.dropout: 0.1` in config

### Web Interface Not Accessible
```bash
# Check if port 7860 is in use
lsof -i :7860  # macOS/Linux
netstat -ano | findstr :7860  # Windows

# Use different port if needed (edit app.py)
# Or access via: http://localhost:7860
```

## Development Workflow

### Adding a New Feature

1. **Update relevant modules** in `src/flux2_lora/`
2. **Add tests** in `tests/unit/` or `tests/integration/`
3. **Update CLI** in `cli.py` if adding new command
4. **Update Gradio UI** in `src/flux2_lora/ui/` if user-facing
5. **Update configs** in `configs/` if new config options
6. **Run tests**: `pytest`
7. **Check code quality**: `ruff check src/` and `black src/`

### Modifying Training Logic

Key files to understand:
- `src/flux2_lora/core/trainer.py`: Main training loop (lines 1-150 show initialization)
- `src/flux2_lora/core/model_loader.py`: Model loading with memory management
- `src/flux2_lora/monitoring/callbacks.py`: Callback system for training events
- `src/flux2_lora/data/dataset.py`: Dataset loading and preprocessing

### Adding New Evaluation Metrics

1. Add metric computation to `src/flux2_lora/evaluation/quality_metrics.py`
2. Update `QualityAssessor.assess_checkpoint_quality()` method
3. Add CLI command in `cli.py` under `@eval_app.command()`
4. Add UI component in `src/flux2_lora/ui/evaluation_tab.py`
5. Add tests in `tests/unit/test_quality_metrics.py`

## Security Considerations

**Always Use Safetensors:**
- Never use `torch.load()` on untrusted checkpoints (pickle vulnerability)
- Use `safetensors` format exclusively
- `CheckpointManager` enforces this by default

**Input Validation:**
- File extensions whitelisted in config: `.jpg`, `.jpeg`, `.png`, `.webp`, `.txt`, `.json`
- Max file size enforced: 50MB default
- Path traversal prevention on all file operations
- Dataset validation before training starts

**Resource Limits:**
- Max training time: 24 hours default (configurable)
- Memory limits enforced via config
- GPU memory monitoring during training
- Timeout handling for long operations

## Additional Resources

- **README.md**: User-facing documentation with installation and usage
- **DEVELOPMENT.md**: Detailed development roadmap with implementation phases
- **pyproject.toml**: Package configuration and dependencies
- **configs/base_config.yaml**: Complete configuration reference with comments
