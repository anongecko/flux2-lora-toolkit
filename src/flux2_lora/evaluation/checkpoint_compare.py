"""
Checkpoint comparison utilities for Flux2-dev LoRA evaluation.

This module provides tools for comparing multiple LoRA checkpoints side-by-side,
generating comparison grids, computing metrics, and creating HTML reports.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline

from ..core.model_loader import ModelLoader
from ..core.lora_config import FluxLoRAConfig
from ..monitoring.metrics import MetricsComputer
from ..utils.hardware_utils import hardware_manager

logger = logging.getLogger(__name__)


class CheckpointComparator:
    """
    Compare multiple LoRA checkpoints side-by-side.

    Features:
    - Load multiple checkpoints simultaneously
    - Generate images with identical prompts across all checkpoints
    - Create comparison grids for visual analysis
    - Compute quantitative metrics (CLIP scores, diversity, etc.)
    - Export HTML reports with results and analysis
    """

    def __init__(
        self,
        checkpoint_paths: List[str],
        test_prompts: Optional[List[str]] = None,
        device: str = "auto",
        output_dir: str = "./comparison_output",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = 42,
    ):
        """
        Initialize checkpoint comparator.

        Args:
            checkpoint_paths: List of paths to LoRA checkpoint files (.safetensors)
            test_prompts: List of test prompts to use for comparison
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            output_dir: Directory to save comparison results
            num_inference_steps: Number of inference steps for image generation
            guidance_scale: Guidance scale for image generation
            seed: Random seed for reproducible results
        """
        self.checkpoint_paths = [Path(cp) for cp in checkpoint_paths]
        self.test_prompts = test_prompts or self._get_default_prompts()
        # Handle device selection
        if device == "auto":
            system_info = hardware_manager.get_system_info()
            if system_info.cuda_available and system_info.gpus:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.output_dir = Path(output_dir)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

        # Initialize components
        self.metrics_computer = MetricsComputer(device=self.device)
        self.model_loader = ModelLoader()

        # Storage for results
        self.models: Dict[str, FluxPipeline] = {}
        self.results: Dict[str, Dict[str, List[Image.Image]]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized CheckpointComparator with {len(checkpoint_paths)} checkpoints")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Test prompts: {len(self.test_prompts)}")

    def _get_default_prompts(self) -> List[str]:
        """Get default test prompts for comparison."""
        return [
            "A portrait of a person with distinctive features",
            "A landscape scene with mountains and water",
            "A detailed close-up of an object",
            "An artistic composition with dramatic lighting",
            "A scene showing motion and energy",
        ]

    def load_checkpoints(self) -> None:
        """Load all checkpoints into memory."""
        logger.info("Loading checkpoints...")

        for checkpoint_path in self.checkpoint_paths:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint_name = checkpoint_path.stem
            logger.info(f"Loading checkpoint: {checkpoint_name}")

            try:
                # Load base model
                model, metadata = self.model_loader.load_flux2_dev(
                    device=self.device, dtype=torch.bfloat16
                )

                # Load LoRA weights
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(checkpoint_path))

                # Store model
                self.models[checkpoint_name] = model
                logger.info(f"Successfully loaded checkpoint: {checkpoint_name}")

            except Exception as e:
                logger.error(f"Failed to load checkpoint {checkpoint_name}: {e}")
                raise

    def generate_comparisons(self) -> Dict[str, Dict[str, List[Image.Image]]]:
        """
        Generate images for all checkpoints and prompts.

        Returns:
            Dictionary mapping checkpoint names to prompt results
            Format: {checkpoint_name: {prompt: [generated_images]}}
        """
        logger.info("Generating comparison images...")

        if not self.models:
            self.load_checkpoints()

        # Set seed for reproducible results
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        for checkpoint_name, model in self.models.items():
            logger.info(f"Generating images for checkpoint: {checkpoint_name}")
            self.results[checkpoint_name] = {}

            for prompt in self.test_prompts:
                logger.debug(f"Generating for prompt: {prompt[:50]}...")

                try:
                    # Generate image
                    with torch.no_grad():
                        images = model(
                            prompt=prompt,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=self.guidance_scale,
                            num_images_per_prompt=1,
                            height=1024,
                            width=1024,
                        ).images

                    self.results[checkpoint_name][prompt] = images

                except Exception as e:
                    logger.error(
                        f"Failed to generate for {checkpoint_name}, prompt '{prompt[:50]}...': {e}"
                    )
                    # Create placeholder image
                    placeholder = Image.new("RGB", (512, 512), color="gray")
                    draw = ImageDraw.Draw(placeholder)
                    draw.text((10, 250), "Generation Failed", fill="red")
                    self.results[checkpoint_name][prompt] = [placeholder]

        logger.info("Completed image generation for all checkpoints")
        return self.results

    def create_comparison_grid(
        self,
        results: Optional[Dict[str, Dict[str, List[Image.Image]]]] = None,
        max_images_per_prompt: int = 3,
    ) -> Image.Image:
        """
        Create side-by-side comparison grid.

        Args:
            results: Results dictionary (uses self.results if None)
            max_images_per_prompt: Maximum images to show per prompt

        Returns:
            PIL Image containing the comparison grid
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results available. Run generate_comparisons() first.")

        checkpoint_names = list(results.keys())
        num_checkpoints = len(checkpoint_names)
        num_prompts = len(self.test_prompts)

        # Image dimensions (assuming square images)
        img_width, img_height = 512, 512
        margin = 20
        header_height = 60
        label_height = 40

        # Calculate grid dimensions
        grid_width = num_checkpoints * (img_width + margin) + margin
        grid_height = num_prompts * (img_height + label_height + margin) + header_height + margin

        # Create grid image
        grid = Image.new("RGB", (grid_width, grid_height), "white")
        draw = ImageDraw.Draw(grid)

        # Try to load a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw headers
        y_offset = margin
        draw.text((margin, y_offset), "Checkpoint Comparison", fill="black", font=font)
        y_offset += header_height

        # Draw column headers (checkpoint names)
        for i, checkpoint_name in enumerate(checkpoint_names):
            x_pos = margin + i * (img_width + margin)
            # Truncate long names
            display_name = (
                checkpoint_name[:15] + "..." if len(checkpoint_name) > 18 else checkpoint_name
            )
            draw.text((x_pos, y_offset), display_name, fill="black", font=small_font)

        y_offset += label_height

        # Draw images
        for prompt_idx, prompt in enumerate(self.test_prompts):
            # Draw prompt label
            prompt_text = prompt[:40] + "..." if len(prompt) > 43 else prompt
            draw.text((margin, y_offset), prompt_text, fill="black", font=small_font)

            # Draw images for this prompt
            for checkpoint_idx, checkpoint_name in enumerate(checkpoint_names):
                x_pos = margin + checkpoint_idx * (img_width + margin)
                y_pos = y_offset + label_height

                if prompt in results[checkpoint_name]:
                    images = results[checkpoint_name][prompt]
                    if images:
                        # Resize image if needed and paste
                        img = images[0].copy()
                        img.thumbnail((img_width, img_height), Image.LANCZOS)
                        grid.paste(img, (x_pos, y_pos))

            y_offset += img_height + label_height + margin

        return grid

    def compute_metrics(
        self, results: Optional[Dict[str, Dict[str, List[Image.Image]]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each checkpoint.

        Args:
            results: Results dictionary (uses self.results if None)

        Returns:
            Dictionary mapping checkpoint names to metrics
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results available. Run generate_comparisons() first.")

        logger.info("Computing metrics for checkpoints...")

        for checkpoint_name, prompt_results in results.items():
            logger.debug(f"Computing metrics for {checkpoint_name}")

            # Collect all images and prompts for this checkpoint
            all_images = []
            all_prompts = []

            for prompt, images in prompt_results.items():
                all_images.extend(images)
                all_prompts.extend([prompt] * len(images))

            if all_images:
                try:
                    # Compute CLIP score
                    clip_score = self.metrics_computer.compute_clip_score(all_images, all_prompts)

                    # Compute diversity score
                    diversity_score = self.metrics_computer.compute_diversity(all_images)

                    # Store metrics
                    self.metrics[checkpoint_name] = {
                        "clip_score": clip_score,
                        "diversity_score": diversity_score,
                        "num_images": len(all_images),
                        "avg_clip_per_prompt": clip_score / len(set(all_prompts))
                        if all_prompts
                        else 0,
                    }

                    logger.debug(
                        f"Metrics for {checkpoint_name}: CLIP={clip_score:.3f}, Diversity={diversity_score:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to compute metrics for {checkpoint_name}: {e}")
                    self.metrics[checkpoint_name] = {
                        "clip_score": 0.0,
                        "diversity_score": 0.0,
                        "num_images": len(all_images),
                        "error": str(e),
                    }
            else:
                self.metrics[checkpoint_name] = {
                    "clip_score": 0.0,
                    "diversity_score": 0.0,
                    "num_images": 0,
                }

        return self.metrics

    def export_html_report(
        self, output_path: Optional[str] = None, include_individual_images: bool = True
    ) -> str:
        """
        Generate HTML comparison report.

        Args:
            output_path: Path to save HTML report (auto-generated if None)
            include_individual_images: Whether to save individual images

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"checkpoint_comparison_{timestamp}.html"

        output_path = Path(output_path)

        # Ensure we have results and metrics
        if not self.results:
            self.generate_comparisons()
        if not self.metrics:
            self.compute_metrics()

        # Create comparison grid
        grid_image = self.create_comparison_grid()
        grid_path = output_path.with_suffix(".png")
        grid_image.save(grid_path)

        # Save individual images if requested
        image_paths = {}
        if include_individual_images:
            for checkpoint_name, prompt_results in self.results.items():
                image_paths[checkpoint_name] = {}
                checkpoint_dir = self.output_dir / checkpoint_name
                checkpoint_dir.mkdir(exist_ok=True)

                for prompt_idx, (prompt, images) in enumerate(prompt_results.items()):
                    if images:
                        img_path = checkpoint_dir / f"prompt_{prompt_idx:02d}.png"
                        images[0].save(img_path)
                        image_paths[checkpoint_name][prompt] = str(
                            img_path.relative_to(self.output_dir)
                        )

        # Generate HTML
        html_content = self._generate_html(grid_path, image_paths)

        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {output_path}")
        return str(output_path)

    def _generate_html(self, grid_path: Path, image_paths: Dict) -> str:
        """Generate HTML content for the report."""
        checkpoint_names = list(self.results.keys())

        # Create metrics table
        metrics_table = """
        <table border="1" style="border-collapse: collapse; margin: 20px 0;">
            <tr style="background-color: #f0f0f0;">
                <th style="padding: 8px;">Checkpoint</th>
                <th style="padding: 8px;">CLIP Score</th>
                <th style="padding: 8px;">Diversity Score</th>
                <th style="padding: 8px;">Images Generated</th>
            </tr>
        """

        for checkpoint_name in checkpoint_names:
            metrics = self.metrics.get(checkpoint_name, {})
            metrics_table += f"""
            <tr>
                <td style="padding: 8px;">{checkpoint_name}</td>
                <td style="padding: 8px;">{metrics.get("clip_score", 0):.3f}</td>
                <td style="padding: 8px;">{metrics.get("diversity_score", 0):.3f}</td>
                <td style="padding: 8px;">{metrics.get("num_images", 0)}</td>
            </tr>
            """

        metrics_table += "</table>"

        # Create individual images section
        images_section = ""
        if image_paths:
            images_section = "<h2>Individual Images</h2>"
            for checkpoint_name, prompt_images in image_paths.items():
                images_section += (
                    f"<h3>{checkpoint_name}</h3><div style='display: flex; flex-wrap: wrap;'>"
                )
                for prompt, img_path in prompt_images.items():
                    prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
                    images_section += f"""
                    <div style='margin: 10px; text-align: center;'>
                        <img src='{img_path}' style='max-width: 200px; max-height: 200px;'/>
                        <br/><small>{prompt_short}</small>
                    </div>
                    """
                images_section += "</div>"

        # Create relative path for grid image (handle case where paths may not be related)
        try:
            grid_relative_path = grid_path.relative_to(self.output_dir)
        except ValueError:
            grid_relative_path = grid_path.name

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LoRA Checkpoint Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                table {{ width: 100%; }}
                th, td {{ text-align: left; }}
            </style>
        </head>
        <body>
            <h1>LoRA Checkpoint Comparison Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Checkpoints compared:</strong> {len(checkpoint_names)}</p>
                <p><strong>Test prompts:</strong> {len(self.test_prompts)}</p>
                <p><strong>Generation parameters:</strong> {self.num_inference_steps} steps, guidance scale {self.guidance_scale}</p>
                <p><strong>Generated on:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <h2>Comparison Grid</h2>
            <img src="{grid_relative_path}" style="max-width: 100%; height: auto;"/>

            <h2>Metrics</h2>
            {metrics_table}

            {images_section}
        </body>
        </html>
        """

        return html

    def run_full_comparison(
        self, output_path: Optional[str] = None, save_individual_images: bool = True
    ) -> Dict[str, any]:
        """
        Run complete comparison pipeline.

        Args:
            output_path: Path for HTML report
            save_individual_images: Whether to save individual images

        Returns:
            Dictionary with all results
        """
        logger.info("Starting full checkpoint comparison...")

        # Generate comparisons
        results = self.generate_comparisons()

        # Compute metrics
        metrics = self.compute_metrics(results)

        # Export report
        report_path = self.export_html_report(output_path, save_individual_images)

        logger.info("Comparison complete!")

        return {
            "results": results,
            "metrics": metrics,
            "report_path": report_path,
            "grid_path": str(Path(report_path).with_suffix(".png")),
        }
