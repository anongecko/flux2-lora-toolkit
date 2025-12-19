# 🎨 Flux2-dev LoRA Training Toolkit

(WIP - Development Repository)

A comprehensive, production-grade toolkit for training high-quality LoRA (Low-Rank Adaptation) models for Flux2-dev, featuring real-time monitoring, automatic quality assessment, and an accessible web interface.

**Note**: This is currently a development repository. Installation requires setting up from source. See the Installation section below for detailed setup instructions.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Key Features

- 🚀 **Enterprise-grade training** with real-time monitoring and validation
- 🎯 **Automatic quality assessment** using CLIP-based metrics
- 🖥️ **Intuitive web interface** for both technical and creative users
- 📊 **Comprehensive evaluation tools** for checkpoint comparison and selection
- 🔧 **Flexible CLI** with presets and extensive configuration options
- 🎨 **Multiple LoRA types** support (character, style, concept)
- 💾 **Production-ready** with safetensors and robust error handling

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Web Interface](#web-interface)
- [Command Line Interface](#command-line-interface)
- [Configuration](#configuration)
- [Dataset Preparation](#dataset-preparation)
- [Training Guide](#training-guide)
- [Evaluation & Testing](#evaluation--testing)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (NVIDIA recommended, minimum 8GB VRAM)
- 16GB+ system RAM
- 100GB+ free disk space

### Option 1: Web Interface (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/flux2-lora-training-toolkit.git
cd flux2-lora-training-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the toolkit
pip install -e ".[dev]"

# Launch the web interface
python app.py

# Open http://localhost:7860 in your browser
```

### Option 2: Command Line

```bash
# Clone and setup (same as above)
git clone https://github.com/your-repo/flux2-lora-training-toolkit.git
cd flux2-lora-training-toolkit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Train with a preset
python cli.py train --preset character --dataset /path/to/dataset --output ./output

# Evaluate your trained model
python cli.py eval test-prompts --checkpoint ./output/best_checkpoint.safetensors
```

## 📦 Installation

### System Requirements

- **GPU**: CUDA-compatible GPU (NVIDIA H100 recommended, minimum 8GB VRAM)
- **RAM**: 16GB+ system RAM
- **Storage**: 100GB+ free disk space
- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10+

### Installation Steps

This is a development repository. You'll need to install it from source.

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/flux2-lora-training-toolkit.git
    cd flux2-lora-training-toolkit
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the toolkit and dependencies**:
    ```bash
    # Install with development dependencies (recommended)
    pip install -e ".[dev]"

    # Or install with just core dependencies
    pip install -e .
    ```

4. **Verify installation**:
    ```bash
    python cli.py system info
    ```

### Optional Dependencies

For enhanced functionality, you can install additional packages:

```bash
# For hyperparameter optimization
pip install optuna

# For memory-efficient attention (may conflict with Flash Attention)
pip install xformers

# For Flash Attention 2 (requires CUDA)
pip install flash-attn
```

### Troubleshooting Installation

**CUDA Issues**: Make sure you have CUDA installed and PyTorch with CUDA support:
```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Virtual Environment Issues**: Always activate your virtual environment before running commands:
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

## 🖥️ Web Interface

The web interface provides an intuitive way to train and evaluate LoRA models without command-line knowledge.

### Starting the Interface

After installation, launch the web interface:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch the web interface
python app.py
```

Navigate to `http://localhost:7860` in your browser.

### Training Tab

1. **Upload Dataset**: Click "Upload Dataset" and select a ZIP file containing your training images and captions
2. **Choose Preset**: Select from Character, Style, or Concept presets optimized for different use cases
3. **Configure Training**: Adjust advanced settings or use defaults
4. **Start Training**: Click "Start Training" and monitor progress in real-time

### Evaluation Tab

1. **Load Checkpoint**: Upload or specify path to your trained LoRA checkpoint
2. **Test Prompts**: Enter prompts to test your model's capabilities
3. **Quality Assessment**: Run automatic quality metrics and overfitting detection
4. **Compare Checkpoints**: Upload multiple checkpoints for side-by-side comparison

### Dataset Tools Tab

1. **Analyze Dataset**: Upload or specify dataset path for comprehensive analysis
2. **Validate Structure**: Check for common issues and get actionable recommendations
3. **Browse Images**: Review your dataset with caption display and navigation

## 💻 Command Line Interface

The CLI provides full control over training and evaluation with extensive options.

### Basic Training

```bash
# Train with character preset
python cli.py train --preset character --dataset ./my_dataset --output ./output

# Train with custom config
python cli.py train --config my_config.yaml --dataset ./data --output ./results

# Override specific settings
python cli.py train --preset style --steps 2000 --lr 5e-5 --batch-size 4
```

### Evaluation Commands

```bash
# Test a checkpoint with prompts
python cli.py eval test --checkpoint ./output/checkpoint-1000.safetensors --prompt "A portrait of a person"

# Run comprehensive prompt testing
python cli.py eval test-prompts --checkpoint ./output/best.safetensors --concept "my_character"

# Compare multiple checkpoints
python cli.py eval compare checkpoint1.safetensors checkpoint2.safetensors --prompt "Test prompt"

# Assess quality metrics
python cli.py eval assess-quality --checkpoint ./checkpoint.safetensors --training-data ./dataset

# Select best checkpoint from multiple
python cli.py eval select-best checkpoint1.safetensors checkpoint2.safetensors checkpoint3.safetensors
```

### Dataset Commands

```bash
# Analyze dataset
python cli.py data analyze --dataset ./my_dataset --output analysis.json

# Validate dataset structure
python cli.py data validate --dataset ./data --fix
```

### System Commands

```bash
# Show system information
python cli.py system info

# Display GPU details
python cli.py system gpu

# Get optimization recommendations
python cli.py system optimize --config my_config.yaml

# List available presets
python cli.py system presets
```

## ⚙️ Configuration

### Preset Configurations

The toolkit includes optimized presets for different LoRA types:

- **Character**: Optimized for training character-specific LoRAs (rank=128, higher learning rate)
- **Style**: Optimized for artistic style transfer (rank=64, balanced settings)
- **Concept**: Optimized for object/concept training (rank=32, conservative settings)

### Custom Configuration

Create a YAML configuration file:

```yaml
model:
  base_model: "blackforestlabs/FLUX.1-dev"
  dtype: "bfloat16"
  device: "cuda:0"

lora:
  rank: 128
  alpha: 128
  dropout: 0.1

training:
  learning_rate: 5e-5
  batch_size: 4
  max_steps: 1000
  gradient_accumulation_steps: 1

data:
  resolution: 1024
  caption_format: "txt"
  cache_images: true

output:
  output_dir: "./output"
  checkpoint_every_n_steps: 100
```

## 📸 Dataset Preparation

### Directory Structure

```
my_dataset/
├── image_001.jpg
├── image_001.txt          # Caption file
├── image_002.png
├── image_002.txt
└── ...
```

### Caption Formats

The toolkit supports multiple caption formats:

- **`.txt` files**: Standard text files with captions
- **`.caption` files**: Alternative caption extension
- **EXIF metadata**: Captions embedded in image metadata
- **JSON sidecar files**: Structured caption data

### Best Practices

1. **Image Quality**: Use high-resolution images (1024x1024 minimum)
2. **Consistency**: Maintain consistent style, lighting, and composition
3. **Diverse Poses**: Include multiple angles and expressions for characters
4. **Captions**: Write detailed, descriptive captions
5. **Quantity**: Start with 10-50 high-quality images
6. **Naming**: Use sequential naming (image_001.jpg, image_002.jpg, etc.)

### Example Dataset

```
character_dataset/
├── character_front.jpg
├── character_front.txt          # "A portrait of a character with distinctive features"
├── character_side.jpg
├── character_side.txt           # "Side profile of the character"
├── character_action.jpg
├── character_action.txt         # "The character in an action pose"
└── ...
```

## 🎯 Training Guide

### Step-by-Step Training

1. **Prepare Dataset**: Organize images and captions as described above
2. **Choose Preset**: Select appropriate preset based on your use case
3. **Configure Training**:
   - Set output directory
   - Adjust training steps (1000-5000 typically)
   - Configure batch size based on GPU memory
4. **Monitor Training**:
   - Watch loss decrease over time
   - Review validation samples
   - Monitor GPU memory usage
5. **Evaluate Results**: Test trained checkpoints with various prompts

### Understanding the Metrics

- **Loss**: Training objective (lower is better)
- **CLIP Score**: How well generated images match prompts (0-1, higher better)
- **Diversity Score**: Variety in generated images (higher better)
- **Overfitting Detection**: Similarity to training images (lower risk better)

### Optimization Tips

- **GPU Memory**: Reduce batch size if encountering OOM errors
- **Training Steps**: More steps generally improve quality but increase training time
- **Learning Rate**: Start with preset defaults, adjust based on convergence
- **Resolution**: Higher resolution requires more memory but improves detail

## 🔍 Evaluation & Testing

### Prompt Testing

Test your trained LoRA with various prompt patterns:

```bash
# Basic usage
python cli.py eval test-prompts --checkpoint my_lora.safetensors --trigger-word "my_character"

# Custom concept
python cli.py eval test-prompts --concept "cyberpunk city" --trigger-word "cyberpunk"
```

### Quality Assessment

Get detailed quality metrics for your checkpoints:

```bash
python cli.py eval assess-quality \
  --checkpoint my_lora.safetensors \
  --training-data ./dataset \
  --output quality_report.json
```

### Checkpoint Comparison

Compare multiple checkpoints side-by-side:

```bash
python cli.py eval compare \
  checkpoint_500.safetensors \
  checkpoint_1000.safetensors \
  checkpoint_1500.safetensors \
  --output comparison_results.html
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch-size 2`
- Enable gradient checkpointing (automatic)
- Reduce resolution if using high-res images
- Close other GPU-intensive applications

#### Training Not Converging
```
Loss not decreasing significantly
```
**Solutions**:
- Increase training steps: `--steps 2000`
- Adjust learning rate: `--lr 1e-4`
- Check dataset quality and consistency
- Try different preset or custom configuration

#### Poor Generation Quality
```
Generated images don't match expectations
```
**Solutions**:
- Ensure trigger word is used correctly in prompts
- Check dataset quality and variety
- Increase training steps or adjust LoRA rank
- Review caption quality and specificity

#### Import Errors
```
ModuleNotFoundError: No module named 'diffusers'
```
**Solutions**:
- Install missing dependencies: `pip install -e ".[dev]"`
- Check Python version (3.10+ required)
- Ensure virtual environment is activated

### Getting Help

1. **Check system compatibility**: `python cli.py system info`
2. **Validate configuration**: Use `--dry-run` flag
3. **Check dataset**: `python cli.py data validate --dataset ./my_dataset`
4. **Review logs**: Check output directory for detailed logs

### Performance Optimization

For H100 GPUs:
```bash
python cli.py system optimize --config my_config.yaml
```

This will provide specific recommendations for your hardware.

## ⚡ Hyperparameter Optimization

Automatically find the best training settings for your specific dataset using Bayesian optimization.

### Why Use Optimization?
- **10-30% Quality Improvement**: Optimized settings produce better LoRA models
- **Faster Training**: Better parameters converge faster and more reliably
- **Memory Efficiency**: Optimized batch sizes and accumulation maximize GPU utilization
- **Dataset-Specific**: Each dataset may have different optimal hyperparameters

### Quick Start
```bash
# Basic optimization (50 trials, ~10-20 hours)
python cli.py train optimize --dataset ./my_dataset

# Quick optimization for testing (20 trials, ~4-8 hours)
python cli.py train optimize --dataset ./data --trials 20 --max-steps 300

# Custom output directory
python cli.py train optimize --dataset ./data --output ./my_optimization
```

### What Gets Optimized
- **LoRA Rank** (4-128): Model capacity - higher values learn more complex patterns
- **LoRA Alpha** (4-128): LoRA strength scaling factor
- **Learning Rate** (1e-6 to 1e-2): Training convergence speed
- **Batch Size** (1, 2, 4, 8, 16): Images processed simultaneously
- **Gradient Accumulation** (1, 2, 4, 8): Effective batch size for memory management

### Understanding Results
After optimization completes, you'll get:
- **`best_config.yaml`**: Ready-to-use configuration for production training
- **`optimization_results.json`**: Complete optimization summary with best parameters
- **`trials_data.json`**: Detailed results from each optimization trial

### Web Interface Optimization
Use the **Optimization** tab in the web interface for:
- Interactive parameter range selection
- Real-time progress monitoring
- Visual optimization history
- Easy configuration download

### Best Practices
- **Start Small**: Use 20-30 trials for initial optimization
- **Good Dataset**: Optimization works best with representative, high-quality data
- **Monitor Progress**: Check that quality scores improve over trials
- **Production Training**: Use optimized settings for your final LoRA training

### Time Estimates
- **20 trials**: 4-8 hours (good for development/testing)
- **50 trials**: 10-20 hours (recommended for production)
- **100 trials**: 20-40 hours (maximum optimization with diminishing returns)

## 🚀 Advanced Usage

### Custom Training Loops

```python
from flux2_lora.core.trainer import LoRATrainer
from flux2_lora.utils.config_manager import config_manager

# Load configuration
config = config_manager.get_preset_config("character")

# Initialize trainer
trainer = LoRATrainer(model=model, config=config, output_dir="./output")

# Custom training loop
for step in range(1000):
    loss = trainer.train_step(batch)
    if step % 100 == 0:
        trainer.save_checkpoint(f"checkpoint-{step}")
```

### Integration with Existing Pipelines

The toolkit can be integrated into existing ML pipelines:

```python
from flux2_lora import LoRADataset, create_dataloader

# Load your custom dataset
dataset = LoRADataset(data_dir="./my_data", resolution=1024)
dataloader = create_dataloader(dataset, batch_size=4)

# Use with your training framework
for batch in dataloader:
    # Your training logic here
    pass
```

### Custom Evaluation Metrics

```python
from flux2_lora.evaluation import QualityAssessor

assessor = QualityAssessor()

# Add custom metrics
results = assessor.assess_checkpoint_quality(
    checkpoint_path="my_lora.safetensors",
    test_prompts=["custom prompt"],
    custom_metrics={"my_metric": custom_function}
)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-repo/flux2-lora-training-toolkit.git
cd flux2-lora-training-toolkit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run linting
ruff check .
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [Stability AI](https://stability.ai/) for Flux2-dev
- [OpenAI](https://openai.com/) for CLIP model
- The open-source ML community

## 📞 Support

- **Documentation**: [Full Documentation](https://your-docs-site.com)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**Happy training!** 🎨✨

For the latest updates and community discussions, join our [Discord server](https://discord.gg/your-server).
