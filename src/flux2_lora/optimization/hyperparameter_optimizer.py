"""
Hyperparameter optimization module for Flux2 LoRA Training Toolkit.

Uses Optuna to automatically find optimal hyperparameters for LoRA training.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from ..core.trainer import LoRATrainer
from ..core.model_loader import ModelLoader
from ..data.dataset import LoRADataset, create_dataloader
from ..evaluation.quality_metrics import QualityAssessor
from ..utils.config_manager import config_manager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""

    # Optimization settings
    n_trials: int = 50
    timeout_hours: Optional[float] = None
    n_jobs: int = 1

    # Search spaces
    rank_range: tuple[int, int] = (4, 128)
    alpha_range: tuple[int, int] = (4, 128)
    learning_rate_range: tuple[float, float] = (1e-6, 1e-2)
    batch_size_choices: List[int] = (1, 2, 4, 8, 16)
    gradient_accumulation_choices: List[int] = (1, 2, 4, 8)

    # Fixed settings for optimization
    max_steps: int = 500  # Shorter for optimization trials
    validation_every: int = 50
    enable_quality_metrics: bool = True

    # Early stopping
    patience: int = 3
    min_delta: float = 0.001

    # Output
    output_dir: str = "./optimization_results"
    save_trials: bool = True
    save_best_config: bool = True


class LoRAOptimizer:
    """
    Hyperparameter optimizer for LoRA training using Optuna.

    Automatically searches for optimal LoRA hyperparameters to maximize training quality.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the optimizer.

        Args:
            config: Optimization configuration
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install with: pip install optuna"
            )

        self.config = config
        self.study = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store training data paths (set later)
        self.dataset_path = None
        self.base_model = "blackforestlabs/FLUX.1-dev"

        logger.info(f"Initialized LoRA optimizer with {config.n_trials} trials")

    def setup_study(
        self,
        study_name: str = "flux2_lora_optimization",
        storage_url: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """
        Set up the Optuna study.

        Args:
            study_name: Name of the optimization study
            storage_url: URL for persistent storage (SQLite, etc.)
            load_if_exists: Whether to load existing study

        Returns:
            Optuna study object
        """
        # Set up storage if provided
        storage = None
        if storage_url:
            storage = optuna.storages.RDBStorage(storage_url)

        # Create or load study
        direction = "maximize"  # Maximize quality score
        sampler = TPESampler(seed=42)  # Reproducible results
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        logger.info(f"Created Optuna study: {study_name}")
        return self.study

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Quality score (higher is better)
        """
        try:
            # Sample hyperparameters
            rank = trial.suggest_int("rank", *self.config.rank_range)
            alpha = trial.suggest_int("alpha", *self.config.alpha_range)
            learning_rate = trial.suggest_float(
                "learning_rate", *self.config.learning_rate_range, log=True
            )
            batch_size = trial.suggest_categorical("batch_size", self.config.batch_size_choices)
            gradient_accumulation = trial.suggest_categorical(
                "gradient_accumulation", self.config.gradient_accumulation_choices
            )

            # Create configuration
            trial_config = {
                "model": {"base_model": self.base_model, "dtype": "bfloat16", "device": "auto"},
                "lora": {"rank": rank, "alpha": alpha, "dropout": 0.1},
                "training": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_steps": self.config.max_steps,
                    "gradient_accumulation_steps": gradient_accumulation,
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "validation_every": self.config.validation_every,
                },
                "data": {
                    "dataset_path": self.dataset_path,
                    "resolution": 1024,
                    "caption_format": "txt",
                    "cache_images": True,
                    "num_workers": 2,
                    "pin_memory": True,
                },
                "output": {
                    "output_dir": f"{self.output_dir}/trial_{trial.number}",
                    "checkpoint_every_n_steps": self.config.max_steps,  # Only save final
                    "tensorboard": False,  # Disable for optimization speed
                    "wandb": False,
                },
                "logging": {"tensorboard": False, "wandb": False},
                "quality_metrics": {"enable_quality_metrics": self.config.enable_quality_metrics},
            }

            # Convert to proper config object
            from ..utils.config_manager import TrainingConfig

            config = TrainingConfig.from_dict(trial_config)

            # Validate configuration
            warnings = config_manager.validate_config_values(config)
            if warnings:
                logger.warning(f"Trial {trial.number} config warnings: {warnings}")

            # Run training trial
            quality_score = self._run_training_trial(config, trial)

            # Report intermediate results
            trial.report(quality_score, step=self.config.max_steps)

            # Early stopping check
            if trial.should_prune():
                raise optuna.TrialPruned()

            logger.info(f"Trial {trial.number} completed with score: {quality_score:.4f}")
            return quality_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise

    def _run_training_trial(self, config, trial: optuna.Trial) -> float:
        """
        Run a single training trial and return quality score.

        Args:
            config: Training configuration
            trial: Optuna trial object

        Returns:
            Quality score (0-1, higher is better)
        """
        try:
            # Load model
            model_loader = ModelLoader()
            model, _ = model_loader.load_flux2_dev(
                model_name=config.model.base_model,
                dtype=getattr(__import__("torch"), config.model.dtype),
                device=config.model.device,
            )

            # Load dataset
            train_dataset = LoRADataset(
                data_dir=config.data.dataset_path,
                resolution=config.data.resolution,
                caption_format=config.data.caption_format,
                cache_images=config.data.cache_images,
                validate_captions=config.data.validate_captions,
            )

            # Create dataloader
            train_dataloader = create_dataloader(
                dataset=train_dataset,
                batch_size=config.training.batch_size,
                num_workers=config.data.num_workers,
                pin_memory=config.data.pin_memory,
                shuffle=True,
            )

            # Initialize trainer
            trainer = LoRATrainer(model=model, config=config, output_dir=config.output.output_dir)

            # Track progress for early stopping
            best_loss = float("inf")
            patience_counter = 0

            # Custom progress callback for optimization
            def progress_callback(step: int, loss: float, metrics: Dict[str, Any]):
                nonlocal best_loss, patience_counter

                # Report to Optuna for pruning
                trial.report(loss, step=step)

                # Early stopping logic
                if loss < best_loss - self.config.min_delta:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    # Signal early stopping
                    trainer.should_stop = True

                # Pruning check
                if trial.should_prune():
                    trainer.should_stop = True

            # Run training
            training_results = trainer.train_with_progress_callback(
                train_dataloader=train_dataloader,
                num_steps=config.training.max_steps,
                progress_callback=progress_callback,
            )

            # Evaluate final quality
            quality_score = self._evaluate_trial_quality(config, training_results)

            return quality_score

        except Exception as e:
            logger.error(f"Training trial failed: {e}")
            return 0.0  # Return worst possible score

    def _evaluate_trial_quality(self, config, training_results: Dict[str, Any]) -> float:
        """
        Evaluate the quality of a training trial.

        Args:
            config: Training configuration
            training_results: Results from training

        Returns:
            Quality score (0-1)
        """
        try:
            # Check if training completed successfully
            if not training_results.get("success", False):
                return 0.1  # Low but not zero score for failed training

            # Look for final checkpoint
            output_dir = Path(config.output.output_dir)
            checkpoints = list(output_dir.glob("*.safetensors"))

            if not checkpoints:
                logger.warning(f"No checkpoint found in {output_dir}")
                return 0.2

            checkpoint_path = checkpoints[0]  # Use the first (likely only) checkpoint

            # Assess quality
            assessor = QualityAssessor()
            quality_results = assessor.assess_checkpoint_quality(
                checkpoint_path=str(checkpoint_path),
                test_prompts=[
                    "A portrait of a person with distinctive features",
                    "A landscape scene with mountains",
                    "An object with detailed textures",
                ],
                num_samples_per_prompt=2,  # Keep light for optimization
                generation_kwargs={"num_inference_steps": 20, "guidance_scale": 7.5},
            )

            # Combine metrics into final score
            quality_score = quality_results.get("quality_score", 0.5)

            # Factor in training stability (lower final loss is better)
            final_loss = training_results.get("best_loss", 1.0)
            training_stability = max(0, 1.0 - final_loss)  # Convert loss to 0-1 scale

            # Weighted combination
            final_score = 0.7 * quality_score + 0.3 * training_stability

            return final_score

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return 0.3  # Return moderate score for evaluation failures

    def optimize(
        self,
        dataset_path: str,
        base_model: str = "blackforestlabs/FLUX.1-dev",
        study_name: str = "flux2_lora_optimization",
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            dataset_path: Path to training dataset
            base_model: Base model to use
            study_name: Name for the Optuna study

        Returns:
            Dictionary with optimization results
        """
        self.dataset_path = dataset_path
        self.base_model = base_model

        # Setup study
        self.setup_study(study_name=study_name)

        # Set timeout if specified
        timeout = None
        if self.config.timeout_hours:
            timeout = self.config.timeout_hours * 3600  # Convert to seconds

        logger.info(f"Starting optimization with {self.config.n_trials} trials")

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
        )

        # Get results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        results = {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(self.study.trials),
            "study_name": study_name,
            "completed_trials": len(
                [t for t in self.study.trials if t.state == optuna.TrialState.COMPLETE]
            ),
            "pruned_trials": len(
                [t for t in self.study.trials if t.state == optuna.TrialState.PRUNED]
            ),
            "failed_trials": len(
                [t for t in self.study.trials if t.state == optuna.TrialState.FAIL]
            ),
        }

        # Save results
        self._save_results(results)

        logger.info(f"Optimization completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to disk."""
        # Save best configuration
        if self.config.save_best_config:
            best_config_path = self.output_dir / "best_config.yaml"
            best_config = self._create_config_from_params(results["best_params"])
            config_manager.save_config(best_config, str(best_config_path))
            logger.info(f"Saved best configuration to {best_config_path}")

        # Save results summary
        results_path = self.output_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved optimization results to {results_path}")

        # Save trials data if requested
        if self.config.save_trials:
            trials_data = []
            for trial in self.study.trials:
                trial_info = {
                    "number": trial.number,
                    "state": str(trial.state),
                    "value": trial.value,
                    "params": trial.params,
                    "datetime_start": trial.datetime_start,
                    "datetime_complete": trial.datetime_complete,
                }
                trials_data.append(trial_info)

            trials_path = self.output_dir / "trials_data.json"
            with open(trials_path, "w") as f:
                json.dump(trials_data, f, indent=2, default=str)
            logger.info(f"Saved trials data to {trials_path}")

    def _create_config_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a full configuration from optimized parameters."""
        config = {
            "model": {"base_model": self.base_model, "dtype": "bfloat16", "device": "auto"},
            "lora": {"rank": params["rank"], "alpha": params["alpha"], "dropout": 0.1},
            "training": {
                "learning_rate": params["learning_rate"],
                "batch_size": params["batch_size"],
                "max_steps": 1000,  # Default for production use
                "gradient_accumulation_steps": params["gradient_accumulation"],
                "optimizer": "adamw",
                "scheduler": "cosine",
            },
            "data": {
                "dataset_path": self.dataset_path,
                "resolution": 1024,
                "caption_format": "txt",
                "cache_images": True,
                "num_workers": 4,
                "pin_memory": True,
            },
            "output": {
                "output_dir": "./output",
                "checkpoint_every_n_steps": 100,
                "tensorboard": True,
                "wandb": False,
            },
            "logging": {"tensorboard": True, "wandb": False},
        }
        return config

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of optimization trials.

        Returns:
            List of trial results
        """
        if not self.study:
            return []

        history = []
        for trial in self.study.trials:
            if trial.state == optuna.TrialState.COMPLETE:
                history.append(
                    {
                        "trial": trial.number,
                        "score": trial.value,
                        "params": trial.params,
                        "datetime": trial.datetime_complete,
                    }
                )

        return sorted(history, key=lambda x: x["score"], reverse=True)


def create_optimizer(
    n_trials: int = 50,
    dataset_path: str = None,
    output_dir: str = "./optimization_results",
    **kwargs,
) -> LoRAOptimizer:
    """
    Create a LoRA optimizer with sensible defaults.

    Args:
        n_trials: Number of optimization trials
        dataset_path: Path to training dataset
        output_dir: Output directory for results
        **kwargs: Additional configuration options

    Returns:
        Configured LoRA optimizer
    """
    config = OptimizationConfig(n_trials=n_trials, output_dir=output_dir, **kwargs)

    optimizer = LoRAOptimizer(config)

    if dataset_path:
        optimizer.dataset_path = dataset_path

    return optimizer
