"""
Quality assessment and overfitting detection for Flux2-dev LoRA evaluation.

This module provides comprehensive quality assessment tools including CLIP-based
metrics, overfitting detection, diversity analysis, and automatic quality scoring
for LoRA checkpoint evaluation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from ..monitoring.metrics import MetricsComputer, create_metrics_computer
from ..core.model_loader import ModelLoader
from ..core.lora_config import FluxLoRAConfig
from .checkpoint_compare import CheckpointComparator

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Comprehensive LoRA quality assessment system.

    Provides multiple quality metrics including CLIP scores, aesthetic quality,
    diversity analysis, overfitting detection, and automatic quality scoring
    for checkpoint evaluation and selection.
    """

    def __init__(
        self,
        device: str = "auto",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        aesthetic_model_name: Optional[str] = None,  # Future: aesthetic predictor
    ):
        """
        Initialize quality assessor.

        Args:
            device: Device for computations ('auto', 'cpu', 'cuda')
            clip_model_name: CLIP model for quality assessment
            aesthetic_model_name: Optional aesthetic quality model
        """
        self.device = device
        self.clip_model_name = clip_model_name
        self.aesthetic_model_name = aesthetic_model_name

        # Initialize metrics computer
        self.metrics_computer = create_metrics_computer(device=device)
        # Override CLIP model name if different from default
        if clip_model_name != "openai/clip-vit-base-patch32":
            # Reinitialize with custom model (this would require modifying MetricsComputer)
            # For now, we'll use the default and log a warning
            logger.warning(f"Custom CLIP model {clip_model_name} not supported yet, using default")

        # Initialize model loader for checkpoint assessment
        self.model_loader = ModelLoader()

        logger.info(f"QualityAssessor initialized with device: {self.device}")

    def assess_checkpoint_quality(
        self,
        checkpoint_path: Union[str, Path],
        test_prompts: List[str],
        training_images: Optional[List[Image.Image]] = None,
        num_samples_per_prompt: int = 3,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of a single checkpoint.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            test_prompts: Prompts to test the checkpoint with
            training_images: Optional training images for overfitting detection
            num_samples_per_prompt: Number of images to generate per prompt
            generation_kwargs: Additional kwargs for image generation

        Returns:
            Dictionary with comprehensive quality assessment results
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Assessing checkpoint quality: {checkpoint_path.name}")

        # Load checkpoint
        model, metadata = self.model_loader.load_flux2_dev(device=self.device)
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(checkpoint_path))
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

        # Set generation defaults
        gen_kwargs = {
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
        }
        if generation_kwargs:
            gen_kwargs.update(generation_kwargs)

        # Generate test images
        generated_images = []
        prompt_image_map = {}

        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    images = model(
                        prompt=prompt, num_images_per_prompt=num_samples_per_prompt, **gen_kwargs
                    ).images

                    generated_images.extend(images)
                    prompt_image_map[prompt] = images
                    logger.debug(f"Generated {len(images)} images for prompt: {prompt[:50]}...")

                except Exception as e:
                    logger.error(f"Failed to generate for prompt '{prompt[:50]}...': {e}")
                    # Create placeholder images
                    placeholder = Image.new("RGB", (512, 512), color="gray")
                    generated_images.extend([placeholder] * num_samples_per_prompt)
                    prompt_image_map[prompt] = [placeholder] * num_samples_per_prompt

        # Create repeated prompts list for metrics
        repeated_prompts = []
        for prompt in test_prompts:
            repeated_prompts.extend([prompt] * num_samples_per_prompt)

        # Compute quality metrics
        results = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.stem,
            "test_prompts": test_prompts,
            "num_prompts": len(test_prompts),
            "num_samples_per_prompt": num_samples_per_prompt,
            "total_images": len(generated_images),
            "generation_kwargs": gen_kwargs,
        }

        # CLIP score for image-text alignment
        if self.metrics_computer._clip_available():
            try:
                clip_score = self.metrics_computer.compute_clip_score(
                    generated_images, repeated_prompts
                )
                results["clip_score"] = clip_score
                results["clip_score_per_prompt"] = (
                    clip_score / len(test_prompts) if test_prompts else 0
                )
                logger.debug(".3f")
            except Exception as e:
                logger.warning(f"Failed to compute CLIP score: {e}")
                results["clip_score"] = 0.0
                results["clip_score_per_prompt"] = 0.0
        else:
            results["clip_score"] = 0.0
            results["clip_score_per_prompt"] = 0.0
            logger.warning("CLIP not available for quality assessment")

        # Diversity score
        try:
            diversity_score = self.metrics_computer.compute_diversity_score(generated_images)
            results["diversity_score"] = diversity_score
            logger.debug(".3f")
        except Exception as e:
            logger.warning(f"Failed to compute diversity score: {e}")
            results["diversity_score"] = 0.0

        # Overfitting detection (if training images provided)
        if training_images:
            try:
                overfitting_results = self.metrics_computer.detect_overfitting(
                    generated_images, training_images, threshold=0.85
                )
                results["overfitting_analysis"] = overfitting_results
                results["is_overfitting"] = overfitting_results.get("is_overfitting", False)
                results["overfitting_similarity"] = overfitting_results.get("similarity_score", 0.0)
                logger.debug(".3f")
            except Exception as e:
                logger.warning(f"Failed to detect overfitting: {e}")
                results["overfitting_analysis"] = {"error": str(e)}
                results["is_overfitting"] = False
                results["overfitting_similarity"] = 0.0

        # Comprehensive quality score
        try:
            quality_score_results = self.metrics_computer.compute_comprehensive_quality_score(
                generated_images, repeated_prompts, training_images
            )
            results["quality_score"] = quality_score_results.get("quality_score", 0.0)
            results["quality_breakdown"] = quality_score_results
            logger.debug(".3f")
        except Exception as e:
            logger.warning(f"Failed to compute quality score: {e}")
            results["quality_score"] = 0.0
            results["quality_breakdown"] = {"error": str(e)}

        # Prompt adherence analysis
        try:
            prompt_adherence = self._analyze_prompt_adherence(prompt_image_map)
            results["prompt_adherence"] = prompt_adherence
        except Exception as e:
            logger.warning(f"Failed to analyze prompt adherence: {e}")
            results["prompt_adherence"] = {}

        return results

    def compare_checkpoints_quality(
        self,
        checkpoint_paths: List[Union[str, Path]],
        test_prompts: List[str],
        training_images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compare quality across multiple checkpoints.

        Args:
            checkpoint_paths: List of checkpoint paths
            test_prompts: Test prompts for evaluation
            training_images: Optional training images for overfitting detection
            **kwargs: Additional arguments passed to assess_checkpoint_quality

        Returns:
            Dictionary with comparison results and rankings
        """
        logger.info(f"Comparing quality across {len(checkpoint_paths)} checkpoints")

        checkpoint_results = {}
        for checkpoint_path in checkpoint_paths:
            try:
                result = self.assess_checkpoint_quality(
                    checkpoint_path, test_prompts, training_images, **kwargs
                )
                checkpoint_results[result["checkpoint_name"]] = result
                logger.info(
                    f"Assessed {result['checkpoint_name']}: quality_score={result.get('quality_score', 0):.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to assess checkpoint {checkpoint_path}: {e}")
                checkpoint_name = Path(checkpoint_path).stem
                checkpoint_results[checkpoint_name] = {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_name": checkpoint_name,
                    "error": str(e),
                    "quality_score": 0.0,
                }

        # Rank checkpoints by quality score
        rankings = sorted(
            checkpoint_results.items(), key=lambda x: x[1].get("quality_score", 0), reverse=True
        )

        comparison_results = {
            "checkpoint_results": checkpoint_results,
            "rankings": rankings,
            "best_checkpoint": rankings[0][0] if rankings else None,
            "num_checkpoints": len(checkpoint_paths),
            "num_prompts": len(test_prompts),
            "training_images_provided": training_images is not None,
        }

        logger.info(
            f"Quality comparison complete. Best checkpoint: {comparison_results['best_checkpoint']}"
        )
        return comparison_results

    def generate_overfitting_report(
        self,
        checkpoint_results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        training_images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Generate detailed overfitting analysis report.

        Args:
            checkpoint_results: Results from quality assessment
            output_path: Path to save report (auto-generated if None)
            training_images: Optional training images for comparison

        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = Path("./overfitting_report.html")

        output_path = Path(output_path)

        # Create HTML report
        html_content = self._create_overfitting_html_report(checkpoint_results, training_images)

        # Write report
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Overfitting report saved to: {output_path}")
        return str(output_path)

    def _analyze_prompt_adherence(
        self, prompt_image_map: Dict[str, List[Image.Image]]
    ) -> Dict[str, Any]:
        """Analyze how well generated images adhere to their prompts."""
        adherence_scores = {}

        for prompt, images in prompt_image_map.items():
            try:
                # Simple adherence analysis based on CLIP score consistency
                if len(images) > 1 and self.metrics_computer._clip_available():
                    repeated_prompts = [prompt] * len(images)
                    clip_score = self.metrics_computer.compute_clip_score(images, repeated_prompts)

                    # Calculate variance in CLIP scores (lower variance = better adherence)
                    # For now, just return the average CLIP score as adherence measure
                    adherence_scores[prompt] = {
                        "clip_score": clip_score,
                        "num_images": len(images),
                        "consistency_score": clip_score,  # Placeholder for consistency analysis
                    }
                else:
                    adherence_scores[prompt] = {
                        "clip_score": 0.0,
                        "num_images": len(images),
                        "consistency_score": 0.0,
                    }
            except Exception as e:
                adherence_scores[prompt] = {
                    "error": str(e),
                    "num_images": len(images),
                }

        return adherence_scores

    def _create_overfitting_html_report(
        self,
        checkpoint_results: Dict[str, Any],
        training_images: Optional[List[Image.Image]] = None,
    ) -> str:
        """Create HTML overfitting analysis report."""
        checkpoint_results_data = checkpoint_results.get("checkpoint_results", {})

        # Create overfitting table
        overfitting_table = """
        <table border="1" style="border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color: #f0f0f0;">
                <th style="padding: 8px;">Checkpoint</th>
                <th style="padding: 8px;">Overfitting Detected</th>
                <th style="padding: 8px;">Similarity Score</th>
                <th style="padding: 8px;">Quality Score</th>
                <th style="padding: 8px;">Diversity Score</th>
            </tr>
        """

        for checkpoint_name, results in checkpoint_results_data.items():
            is_overfitting = results.get("is_overfitting", False)
            similarity = results.get("overfitting_similarity", 0.0)
            quality = results.get("quality_score", 0.0)
            diversity = results.get("diversity_score", 0.0)

            color = "#ffcccc" if is_overfitting else "#ccffcc"
            status = "⚠️ Yes" if is_overfitting else "✅ No"

            overfitting_table += f"""
            <tr style="background-color: {color};">
                <td style="padding: 8px;">{checkpoint_name}</td>
                <td style="padding: 8px;">{status}</td>
                <td style="padding: 8px;">{similarity:.3f}</td>
                <td style="padding: 8px;">{quality:.3f}</td>
                <td style="padding: 8px;">{diversity:.3f}</td>
            </tr>
            """

        overfitting_table += "</table>"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LoRA Overfitting Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                table {{ width: 100%; }}
                th, td {{ text-align: left; }}
                .warning {{ background-color: #ffcccc; }}
                .good {{ background-color: #ccffcc; }}
            </style>
        </head>
        <body>
            <h1>LoRA Overfitting Analysis Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Checkpoints analyzed:</strong> {len(checkpoint_results_data)}</p>
                <p><strong>Training images provided:</strong> {"Yes" if training_images else "No"}</p>
                <p><strong>Overfitting threshold:</strong> 0.85 similarity score</p>
                <p><strong>Generated on:</strong> {__import__("time").strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <h2>Overfitting Analysis</h2>
            {overfitting_table}

            <h2>Analysis Notes</h2>
            <ul>
                <li><strong>Overfitting Detection:</strong> Identifies checkpoints that generate images too similar to training data</li>
                <li><strong>Similarity Score:</strong> Higher values indicate potential overfitting (threshold: 0.85)</li>
                <li><strong>Quality Score:</strong> Overall assessment combining multiple metrics</li>
                <li><strong>Diversity Score:</strong> Measures output variety (lower values may indicate mode collapse)</li>
            </ul>

            <h2>Recommendations</h2>
            <ul>
                <li>Consider early stopping if overfitting is detected</li>
                <li>Review training data if multiple checkpoints show overfitting</li>
                <li>Balance quality and diversity when selecting final checkpoint</li>
                <li>Use checkpoints with low overfitting risk for broader applicability</li>
            </ul>
        </body>
        </html>
        """

        return html


class BestCheckpointSelector:
    """
    Automatic best checkpoint selection based on comprehensive quality metrics.

    Analyzes multiple checkpoints and recommends the best one based on
    configurable criteria including quality, overfitting risk, and diversity.
    """

    def __init__(
        self,
        quality_weight: float = 0.7,
        diversity_weight: float = 0.3,
        overfitting_penalty: float = 0.2,
        min_quality_threshold: float = 0.0,
    ):
        """
        Initialize checkpoint selector.

        Args:
            quality_weight: Weight for overall quality score
            diversity_weight: Weight for diversity score
            overfitting_penalty: Penalty weight for overfitting risk
            min_quality_threshold: Minimum quality score required
        """
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.overfitting_penalty = overfitting_penalty
        self.min_quality_threshold = min_quality_threshold

        # Validate weights
        total_weight = quality_weight + diversity_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    def select_best_checkpoint(
        self,
        comparison_results: Dict[str, Any],
        explain: bool = True,
    ) -> Dict[str, Any]:
        """
        Select the best checkpoint from comparison results.

        Args:
            comparison_results: Results from QualityAssessor.compare_checkpoints_quality
            explain: Whether to include detailed explanation

        Returns:
            Dictionary with selection results and explanation
        """
        checkpoint_results = comparison_results.get("checkpoint_results", {})

        if not checkpoint_results:
            raise ValueError("No checkpoint results provided")

        # Calculate composite scores
        scored_checkpoints = []

        for checkpoint_name, results in checkpoint_results.items():
            # Skip checkpoints with errors
            if "error" in results:
                logger.warning(
                    f"Skipping checkpoint {checkpoint_name} due to error: {results['error']}"
                )
                continue

            quality_score = results.get("quality_score", 0.0)
            diversity_score = results.get("diversity_score", 0.0)
            is_overfitting = results.get("is_overfitting", False)

            # Apply minimum quality threshold
            if quality_score < self.min_quality_threshold:
                continue

            # Calculate composite score
            composite_score = (
                self.quality_weight * quality_score + self.diversity_weight * diversity_score
            )

            # Apply overfitting penalty
            if is_overfitting:
                composite_score *= 1.0 - self.overfitting_penalty

            scored_checkpoints.append(
                {
                    "name": checkpoint_name,
                    "composite_score": composite_score,
                    "quality_score": quality_score,
                    "diversity_score": diversity_score,
                    "is_overfitting": is_overfitting,
                    "results": results,
                }
            )

        if not scored_checkpoints:
            raise ValueError("No valid checkpoints found meeting criteria")

        # Sort by composite score
        scored_checkpoints.sort(key=lambda x: x["composite_score"], reverse=True)
        best_checkpoint = scored_checkpoints[0]

        result = {
            "selected_checkpoint": best_checkpoint["name"],
            "composite_score": best_checkpoint["composite_score"],
            "ranking": scored_checkpoints,
            "num_candidates": len(scored_checkpoints),
            "selection_criteria": {
                "quality_weight": self.quality_weight,
                "diversity_weight": self.diversity_weight,
                "overfitting_penalty": self.overfitting_penalty,
                "min_quality_threshold": self.min_quality_threshold,
            },
        }

        if explain:
            result["explanation"] = self._explain_selection(best_checkpoint, scored_checkpoints[1:])

        return result

    def _explain_selection(
        self, best_checkpoint: Dict[str, Any], other_checkpoints: List[Dict[str, Any]]
    ) -> str:
        """Generate explanation for why the best checkpoint was selected."""
        explanation = f"Selected '{best_checkpoint['name']}' as the best checkpoint with composite score {best_checkpoint['composite_score']:.3f}.\n\n"

        explanation += "Selection factors:\n"
        explanation += f"- Quality score: {best_checkpoint['quality_score']:.3f}\n"
        explanation += f"- Diversity score: {best_checkpoint['diversity_score']:.3f}\n"
        if best_checkpoint["is_overfitting"]:
            explanation += f"- Overfitting detected (penalty applied)\n"
        else:
            explanation += "- No overfitting detected\n"

        if other_checkpoints:
            explanation += f"\nCompared to {len(other_checkpoints)} other checkpoint(s):\n"
            for i, cp in enumerate(other_checkpoints[:3], 1):  # Show top 3 alternatives
                explanation += f"{i}. {cp['name']}: score {cp['composite_score']:.3f} "
                explanation += f"(quality: {cp['quality_score']:.3f}, diversity: {cp['diversity_score']:.3f})\n"

        explanation += f"\nWeights used: Quality {self.quality_weight:.1f}, Diversity {self.diversity_weight:.1f}, Overfitting penalty {self.overfitting_penalty:.1f}"

        return explanation
