#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_optimization_imports():
    """Test that optimization module can be imported."""
    try:
        from flux2_lora.optimization import LoRAOptimizer, OptimizationConfig, create_optimizer

        print("✅ Optimization module imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config_creation():
    """Test that optimization config can be created."""
    try:
        from flux2_lora.optimization import OptimizationConfig

        config = OptimizationConfig(n_trials=10, max_steps=200)
        print(f"✅ Config created: {config.n_trials} trials, {config.max_steps} steps")
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False


def test_optimizer_creation():
    """Test that optimizer can be created."""
    try:
        from flux2_lora.optimization import create_optimizer

        optimizer = create_optimizer(n_trials=5, output_dir="./test_output")
        print("✅ Optimizer created successfully")
        return True
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return False


def test_optuna_availability():
    """Test if Optuna is available."""
    try:
        import optuna

        print(f"✅ Optuna available (version {optuna.__version__})")
        return True
    except ImportError:
        print("⚠️  Optuna not available - install with: pip install optuna")
        return False


if __name__ == "__main__":
    print("🧪 Testing Hyperparameter Optimization Module")
    print("=" * 50)

    tests = [
        test_optimization_imports,
        test_config_creation,
        test_optimizer_creation,
        test_optuna_availability,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Optimization module is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)
