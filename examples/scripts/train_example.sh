#!/usr/bin/env python3
"""
Example training script for Flux2-dev LoRA Training Toolkit.

This script demonstrates how to use the CLI to train a LoRA model
with various configuration options.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure you have activated the virtual environment and installed the package")
        return False

def main():
    """Run example training scenarios."""
    
    # Check if we're in the right directory
    if not Path("cli.py").exists():
        print("❌ Error: cli.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    print("🚀 Flux2-dev LoRA Training Toolkit - Example Training Script")
    print("This script demonstrates various training scenarios")
    
    # Example 1: Dry run with character preset
    print("\n" + "="*80)
    print("EXAMPLE 1: Dry run with character preset")
    print("="*80)
    
    cmd = [
        "python", "cli.py", "train",
        "--preset", "character",
        "--dataset", "./examples/sample_dataset",
        "--output", "./example_output/character",
        "--dry-run"
    ]
    
    if not run_command(cmd, "Dry run with character preset"):
        print("⚠️  Dry run failed, but continuing with other examples")
    
    # Example 2: Show system information
    print("\n" + "="*80)
    print("EXAMPLE 2: System information")
    print("="*80)
    
    cmd = ["python", "cli.py", "system", "info"]
    run_command(cmd, "System information check")
    
    # Example 3: List available presets
    print("\n" + "="*80)
    print("EXAMPLE 3: List available presets")
    print("="*80)
    
    cmd = ["python", "cli.py", "system", "presets"]
    run_command(cmd, "List presets")
    
    # Example 4: Hardware optimization recommendations
    print("\n" + "="*80)
    print("EXAMPLE 4: Hardware optimization recommendations")
    print("="*80)
    
    cmd = ["python", "cli.py", "system", "optimize"]
    run_command(cmd, "Hardware optimization")
    
    # Example 5: Help information
    print("\n" + "="*80)
    print("EXAMPLE 5: Help information")
    print("="*80)
    
    cmd = ["python", "cli.py", "train", "--help"]
    run_command(cmd, "Training command help")
    
    # Example 6: Custom configuration (commented out - requires actual config)
    print("\n" + "="*80)
    print("EXAMPLE 6: Custom configuration (commented out)")
    print("="*80)
    
    custom_config_example = [
        "# Example with custom configuration:",
        "python cli.py train \\",
        "    --config configs/base_config.yaml \\",
        "    --dataset ./examples/sample_dataset \\",
        "    --output ./example_output/custom \\",
        "    --steps 100 \\",
        "    --learning-rate 1e-4 \\",
        "    --batch-size 4",
        "",
        "# Example with parameter overrides:",
        "python cli.py train \\",
        "    --preset style \\",
        "    --dataset ./examples/sample_dataset \\",
        "    --output ./example_output/style \\",
        "    --steps 200 \\",
        "    --lr 5e-5 \\",
        "    --batch-size 2"
    ]
    
    for line in custom_config_example:
        print(line)
    
    print("\n" + "="*80)
    print("🎉 Examples completed!")
    print("="*80)
    
    print("\n📚 Next steps:")
    print("1. Prepare your dataset with images and captions")
    print("2. Choose a preset or create custom config")
    print("3. Run training with: python cli.py train --preset <preset> --dataset <path>")
    print("4. Monitor training progress and check output directory")
    print("5. Use evaluation commands to test your trained LoRA")
    
    print("\n📖 For more information:")
    print("- python cli.py --help  # General help")
    print("- python cli.py train --help  # Training command help")
    print("- python cli.py system info  # Check your system")
    print("- python cli.py system presets  # List available presets")

if __name__ == "__main__":
    main()