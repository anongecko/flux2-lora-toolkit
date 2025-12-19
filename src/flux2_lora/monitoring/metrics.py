"""
Quality assessment and metrics computation for Flux2-dev LoRA training.

This module provides comprehensive metrics for monitoring training progress and
assessing LoRA quality, including CLIP-based image-text alignment, overfitting
detection, and diversity analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

# Optional dependencies
try:
    from transformers import CLIPProcessor, CLIPModel

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPProcessor = None
    CLIPModel = None

logger = logging.getLogger(__name__)


class MetricsComputer:
    """
    Compute various metrics for training monitoring and quality assessment.

    Features:
    - Training metrics: gradient norms, parameter statistics
    - Quality metrics: CLIP scores for image-text alignment
    - Overfitting detection: similarity between generated and training images
    - Diversity metrics: embedding variance across generated samples
    - Memory-efficient computation with optional GPU acceleration
    """

    def __init__(self, device: str = "cpu", clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize metrics computer.

        Args:
            device: Device to run computations on ('cpu', 'cuda', or 'auto')
            clip_model_name: CLIP model to use for quality assessment
        """
        # Handle auto device selection
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clip_model_name = clip_model_name

        # Initialize CLIP model for quality assessment
        self.clip_model = None
        self.clip_processor = None

        if CLIP_AVAILABLE:
            try:
                logger.info(f"Loading CLIP model: {clip_model_name}")
                self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                self.clip_model.eval()
                logger.info("CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load CLIP model: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            logger.warning("CLIP dependencies not available. Quality metrics will be disabled.")

    def compute_training_metrics(
        self,
        model: torch.nn.Module,
        loss: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Compute training-related metrics.

        Args:
            model: Current model
            loss: Current training loss
            optimizer: Optional optimizer for gradient analysis

        Returns:
            Dictionary of training metrics
        """
        metrics = {
            "loss": loss.item() if torch.is_tensor(loss) else float(loss),
        }

        try:
            # Gradient statistics (only for LoRA parameters)
            if optimizer is not None:
                total_norm = 0.0
                param_count = 0
                max_grad_norm = 0.0
                grad_norms = []

                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            max_grad_norm = max(max_grad_norm, param_norm.item())
                            grad_norms.append(param_norm.item())
                            param_count += 1

                if param_count > 0:
                    metrics["grad_norm_l2"] = total_norm**0.5
                    metrics["grad_norm_max"] = max_grad_norm
                    metrics["grad_norm_avg"] = sum(grad_norms) / len(grad_norms)
                    metrics["grad_norm_std"] = torch.std(torch.tensor(grad_norms)).item()

            # Parameter statistics for LoRA layers
            lora_params = []
            total_params = 0

            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    lora_params.append(param)
                total_params += param.numel()

            if lora_params:
                # LoRA parameter statistics
                lora_total = sum(p.numel() for p in lora_params)
                metrics["lora_param_count"] = lora_total
                metrics["lora_param_percentage"] = (lora_total / total_params) * 100

                # Parameter magnitude statistics
                param_magnitudes = []
                for param in lora_params:
                    param_magnitudes.extend(param.data.abs().flatten().tolist())

                if param_magnitudes:
                    param_tensor = torch.tensor(param_magnitudes)
                    metrics["lora_param_mean"] = param_tensor.mean().item()
                    metrics["lora_param_std"] = param_tensor.std().item()
                    metrics["lora_param_max"] = param_tensor.max().item()
                    metrics["lora_param_min"] = param_tensor.min().item()

            # Total parameter count
            metrics["total_param_count"] = total_params

        except Exception as e:
            logger.warning(f"Failed to compute training metrics: {e}")

        return metrics

    def compute_clip_score(
        self,
        images: List[Image.Image],
        prompts: List[str],
        batch_size: int = 8,
    ) -> float:
        """
        Compute CLIP score for image-text alignment.

        Args:
            images: List of PIL images
            prompts: Corresponding text prompts
            batch_size: Batch size for processing

        Returns:
            Average CLIP similarity score (0-1, higher is better)
        """
        if not self._clip_available():
            logger.warning("CLIP model not available for quality assessment")
            return 0.0

        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})"
            )

        if not images:
            return 0.0

        try:
            scores = []

            # Process in batches to avoid memory issues
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_prompts = prompts[i : i + batch_size]

                # Prepare inputs
                inputs = self.clip_processor(
                    text=batch_prompts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    # Compute similarity scores
                    logits_per_image = outputs.logits_per_image  # [batch_size, batch_size]
                    # For diagonal (image-text pairs), we want the diagonal elements
                    batch_scores = torch.diagonal(logits_per_image).cpu().numpy()
                    scores.extend(batch_scores)

            # Convert logits to probabilities and average
            scores_tensor = torch.tensor(scores)
            probabilities = torch.sigmoid(scores_tensor)
            avg_score = probabilities.mean().item()

            return avg_score

        except Exception as e:
            logger.error(f"Failed to compute CLIP score: {e}")
            return 0.0

    def compute_diversity_score(
        self,
        images: List[Image.Image],
        batch_size: int = 16,
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for a set of images.

        Args:
            images: List of PIL images
            batch_size: Batch size for processing

        Returns:
            Dictionary with diversity metrics
        """
        if not self._clip_available() or len(images) < 2:
            return {
                "diversity_score": 0.0,
                "embedding_variance": 0.0,
                "embedding_std": 0.0,
            }

        try:
            # Get CLIP image embeddings
            embeddings = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]

                inputs = self.clip_processor(
                    images=batch_images,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings.append(image_features.cpu())

            if embeddings:
                all_embeddings = torch.cat(embeddings, dim=0)

                # Compute pairwise similarities
                similarity_matrix = torch.mm(all_embeddings, all_embeddings.t())

                # Remove self-similarities (diagonal)
                mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool)
                similarities = similarity_matrix[~mask]

                # Diversity score: 1 - average similarity (higher is more diverse)
                avg_similarity = similarities.mean().item()
                diversity_score = 1.0 - avg_similarity

                # Embedding statistics
                embedding_variance = all_embeddings.var(dim=0).mean().item()
                embedding_std = all_embeddings.std(dim=0).mean().item()

                return {
                    "diversity_score": diversity_score,
                    "embedding_variance": embedding_variance,
                    "embedding_std": embedding_std,
                    "avg_similarity": avg_similarity,
                }
            else:
                return {
                    "diversity_score": 0.0,
                    "embedding_variance": 0.0,
                    "embedding_std": 0.0,
                    "avg_similarity": 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to compute diversity score: {e}")
            return {
                "diversity_score": 0.0,
                "embedding_variance": 0.0,
                "embedding_std": 0.0,
                "avg_similarity": 0.0,
            }

    def detect_overfitting(
        self,
        generated_images: List[Image.Image],
        training_images: List[Image.Image],
        threshold: float = 0.85,
    ) -> Dict[str, Union[float, bool]]:
        """
        Detect overfitting by comparing generated images to training images.

        Args:
            generated_images: Images generated by the model
            training_images: Original training images
            threshold: Similarity threshold above which overfitting is detected

        Returns:
            Dictionary with overfitting metrics
        """
        if not self._clip_available() or not generated_images or not training_images:
            return {
                "overfitting_detected": False,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "similarity_std": 0.0,
            }

        try:
            # Get embeddings for both sets
            def get_embeddings(images, batch_size=16):
                embeddings = []
                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    inputs = self.clip_processor(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        features = self.clip_model.get_image_features(**inputs)
                        features = features / features.norm(dim=-1, keepdim=True)
                        embeddings.append(features.cpu())

                return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 512)

            generated_embeddings = get_embeddings(generated_images)
            training_embeddings = get_embeddings(training_images)

            if generated_embeddings.numel() == 0 or training_embeddings.numel() == 0:
                return {
                    "overfitting_detected": False,
                    "max_similarity": 0.0,
                    "avg_similarity": 0.0,
                    "similarity_std": 0.0,
                }

            # Compute similarities between generated and training images
            similarities = torch.mm(generated_embeddings, training_embeddings.t())

            # For each generated image, find maximum similarity to any training image
            max_similarities, _ = similarities.max(dim=1)

            # Overall statistics
            avg_similarity = max_similarities.mean().item()
            max_similarity = max_similarities.max().item()
            similarity_std = max_similarities.std().item()

            # Detect overfitting
            overfitting_detected = avg_similarity > threshold

            return {
                "overfitting_detected": overfitting_detected,
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "similarity_std": similarity_std,
                "threshold": threshold,
            }

        except Exception as e:
            logger.error(f"Failed to detect overfitting: {e}")
            return {
                "overfitting_detected": False,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "similarity_std": 0.0,
            }

    def compute_comprehensive_quality_score(
        self,
        checkpoint_path: str,
        validation_images: List[Image.Image],
        validation_prompts: List[str],
        training_images: List[Image.Image],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute a comprehensive quality score combining multiple metrics.

        Args:
            checkpoint_path: Path to the checkpoint (for logging)
            validation_images: Images generated from validation prompts
            validation_prompts: Validation prompts used
            training_images: Original training images
            weights: Optional weights for different metrics

        Returns:
            Dictionary with comprehensive quality metrics
        """
        if weights is None:
            weights = {
                "clip_score": 0.4,
                "diversity_score": 0.3,
                "overfitting_penalty": 0.3,
            }

        # Compute individual metrics
        clip_score = self.compute_clip_score(validation_images, validation_prompts)
        diversity_metrics = self.compute_diversity_score(validation_images)
        overfitting_metrics = self.detect_overfitting(validation_images, training_images)

        # Extract diversity score
        diversity_score = diversity_metrics.get("diversity_score", 0.0)

        # Compute overfitting penalty (0 if no overfitting, 1 if severe overfitting)
        overfitting_penalty = 0.0
        if overfitting_metrics.get("overfitting_detected", False):
            # Scale penalty based on how much it exceeds threshold
            avg_similarity = overfitting_metrics.get("avg_similarity", 0.0)
            threshold = overfitting_metrics.get("threshold", 0.85)
            if avg_similarity > threshold:
                overfitting_penalty = min(1.0, (avg_similarity - threshold) / (1.0 - threshold))

        # Compute weighted score
        quality_score = (
            weights["clip_score"] * clip_score
            + weights["diversity_score"] * diversity_score
            - weights["overfitting_penalty"] * overfitting_penalty
        )

        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "quality_score": quality_score,
            "clip_score": clip_score,
            "diversity_score": diversity_score,
            "overfitting_penalty": overfitting_penalty,
            "overfitting_detected": overfitting_metrics.get("overfitting_detected", False),
            "max_similarity": overfitting_metrics.get("max_similarity", 0.0),
            "avg_similarity": overfitting_metrics.get("avg_similarity", 0.0),
            **diversity_metrics,  # Include all diversity metrics
        }

    def _clip_available(self) -> bool:
        """Check if CLIP model is available."""
        return self.clip_model is not None and self.clip_processor is not None


# Utility functions for easy integration


def create_metrics_computer(device: str = "auto") -> MetricsComputer:
    """
    Create a metrics computer with automatic device selection.

    Args:
        device: Device to use ('auto', 'cpu', or 'cuda')

    Returns:
        Configured MetricsComputer instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return MetricsComputer(device=device)


def compute_validation_metrics(
    metrics_computer: MetricsComputer,
    validation_images: List[Image.Image],
    validation_prompts: List[str],
    training_images: List[Image.Image],
    step: int,
) -> Dict[str, float]:
    """
    Compute all validation metrics for a training step.

    Args:
        metrics_computer: MetricsComputer instance
        validation_images: Generated validation images
        validation_prompts: Validation prompts used
        training_images: Training images for overfitting detection
        step: Training step

    Returns:
        Dictionary of validation metrics
    """
    metrics = {}

    try:
        # CLIP score
        clip_score = metrics_computer.compute_clip_score(validation_images, validation_prompts)
        metrics["clip_score"] = clip_score

        # Diversity metrics
        diversity_metrics = metrics_computer.compute_diversity_score(validation_images)
        metrics.update(diversity_metrics)

        # Overfitting detection
        overfitting_metrics = metrics_computer.detect_overfitting(
            validation_images, training_images
        )
        metrics.update(overfitting_metrics)

        # Comprehensive quality score
        quality_metrics = metrics_computer.compute_comprehensive_quality_score(
            "",  # checkpoint_path not needed here
            validation_images,
            validation_prompts,
            training_images,
        )
        metrics["quality_score"] = quality_metrics["quality_score"]

    except Exception as e:
        logger.error(f"Failed to compute validation metrics: {e}")
        # Return default values
        metrics.update(
            {
                "clip_score": 0.0,
                "diversity_score": 0.0,
                "overfitting_detected": False,
                "quality_score": 0.0,
            }
        )

    return metrics
