"""
Validation sampling during training for Flux2-dev LoRA training.

This module provides validation sampling capabilities to generate images during training
for quality assessment and visual progress monitoring.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
import torchvision.transforms as transforms

from ..utils.config_manager import ValidationConfig

logger = logging.getLogger(__name__)


class ValidationSampler:
    """
    Generate validation samples during training.

    Features:
    - Generate images with fixed prompts at regular intervals
    - Create comparison grids showing training progression
    - Log samples to TensorBoard for visualization
    - Memory-efficient sampling without blocking training
    - Configurable sampling parameters
    """

    def __init__(
        self,
        model,
        config: ValidationConfig,
        device: str = "cuda",
        logger=None,
        trigger_word: Optional[str] = None,
    ):
        """
        Initialize validation sampler.

        Args:
            model: Flux2-dev model with LoRA adapters
            config: Validation configuration
            device: Device to run sampling on
            logger: Training logger for TensorBoard integration
            trigger_word: Trigger word to replace [TRIGGER_WORD] in prompts
        """
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        self.trigger_word = trigger_word or "subject"

        # Process prompts with trigger word replacement
        self.prompts = self._process_prompts(config.prompts)

        # Setup sampling parameters
        self.num_inference_steps = config.num_inference_steps
        self.guidance_scale = config.guidance_scale
        self.num_samples = config.num_samples

        # Storage for generated samples
        self.generated_samples: Dict[int, List[Image.Image]] = {}

        logger.info(f"ValidationSampler initialized with {len(self.prompts)} prompts")
        logger.info(f"Sampling every {config.every_n_steps} steps")

    def _process_prompts(self, prompts: List[str]) -> List[str]:
        """
        Process prompts by replacing [TRIGGER_WORD] with actual trigger word.

        Args:
            prompts: List of prompt templates

        Returns:
            Processed prompts
        """
        processed_prompts = []
        for prompt in prompts:
            processed = prompt.replace("[TRIGGER_WORD]", self.trigger_word)
            processed_prompts.append(processed)

        return processed_prompts

    def generate_samples(self, step: int) -> List[Image.Image]:
        """
        Generate validation samples for current training step.

        Args:
            step: Current training step

        Returns:
            List of generated images (one per prompt)
        """
        logger.info(f"Generating validation samples at step {step}")

        start_time = time.time()

        try:
            # Store original model mode
            was_training = self.model.training

            # Switch to eval mode for inference
            self.model.eval()

            generated_images = []

            # Generate image for each prompt
            for prompt in self.prompts:
                try:
                    # Generate image using model's pipeline
                    with torch.no_grad():
                        # Use the model's generate method if available
                        if hasattr(self.model, "generate"):
                            # For FluxPipeline
                            image = self.model.generate(
                                prompt=prompt,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                num_images_per_prompt=self.num_samples,
                                height=1024,  # Flux2-dev default
                                width=1024,
                            )

                            # Handle single image or list
                            if isinstance(image, list):
                                image = image[0] if image else None
                            elif hasattr(image, "images"):
                                image = image.images[0] if image.images else None

                        else:
                            # Fallback: create dummy image for testing
                            logger.warning(
                                "Model does not have generate method, creating dummy image"
                            )
                            image = Image.new("RGB", (512, 512), color="gray")

                    if image is not None:
                        generated_images.append(image)
                    else:
                        # Create placeholder if generation failed
                        logger.warning(f"Failed to generate image for prompt: {prompt}")
                        placeholder = Image.new("RGB", (512, 512), color="red")
                        placeholder.text = f"Failed: {prompt[:30]}..."
                        generated_images.append(placeholder)

                except Exception as e:
                    logger.error(f"Error generating image for prompt '{prompt}': {e}")
                    # Create error placeholder
                    error_img = Image.new("RGB", (512, 512), color="red")
                    generated_images.append(error_img)

            # Store samples for this step
            self.generated_samples[step] = generated_images

            # Log to TensorBoard if logger available
            if self.logger is not None:
                self._log_samples_to_tensorboard(step, generated_images)

            # Restore original model mode
            if was_training:
                self.model.train()

            generation_time = time.time() - start_time
            logger.info(
                f"Generated {len(generated_images)} validation samples in {generation_time:.2f}s"
            )

            return generated_images

        except Exception as e:
            logger.error(f"Validation sampling failed: {e}")
            # Restore model mode
            if "was_training" in locals() and was_training:
                self.model.train()
            return []

    def _log_samples_to_tensorboard(self, step: int, images: List[Image.Image]):
        """
        Log generated samples to TensorBoard.

        Args:
            step: Training step
            images: Generated images
        """
        try:
            if not images:
                return

            # Convert PIL images to tensors for TensorBoard
            image_tensors = []
            for img in images[:8]:  # Limit to 8 images for display
                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Convert to tensor (HWC -> CHW, normalize to [0,1])
                transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
                tensor = transform(img)
                image_tensors.append(tensor)

            if image_tensors:
                # Log individual images
                for i, (img_tensor, prompt) in enumerate(zip(image_tensors, self.prompts)):
                    prompt_short = prompt[:30] + "..." if len(prompt) > 30 else prompt
                    self.logger.log_image(
                        f"validation_samples/step_{step}_prompt_{i}",
                        img_tensor,
                        step,
                        dataformats="CHW",
                    )

                # Log image grid
                self.logger.log_images(
                    f"validation_grid/step_{step}",
                    image_tensors,
                    step,
                    nrow=min(4, len(image_tensors)),
                    dataformats="CHW",
                )

        except Exception as e:
            logger.warning(f"Failed to log validation samples to TensorBoard: {e}")

    def create_comparison_grid(
        self, steps: Optional[List[int]] = None, max_steps: int = 10
    ) -> Optional[Image.Image]:
        """
        Create comparison grid showing progression across training steps.

        Args:
            steps: Specific steps to include (uses most recent if None)
            max_steps: Maximum number of steps to include

        Returns:
            Comparison grid image or None if insufficient data
        """
        try:
            # Get available steps
            available_steps = sorted(self.generated_samples.keys())

            if len(available_steps) < 2:
                logger.warning("Need at least 2 steps of validation samples for comparison")
                return None

            # Select steps to include
            if steps is None:
                # Use most recent steps
                selected_steps = available_steps[-max_steps:]
            else:
                # Use specified steps (filter to available)
                selected_steps = [s for s in steps if s in available_steps]
                if len(selected_steps) < 2:
                    logger.warning("Fewer than 2 specified steps available")
                    return None

            # Get samples for each step
            step_samples = []
            for step in selected_steps:
                samples = self.generated_samples[step]
                if samples:
                    step_samples.append((step, samples[0]))  # Use first sample per step

            if len(step_samples) < 2:
                logger.warning("Insufficient samples for comparison grid")
                return None

            # Create comparison grid
            # Arrange as: rows = prompts, columns = steps
            num_steps = len(step_samples)
            sample_height, sample_width = step_samples[0][1].size[::-1]  # PIL size is (W, H)

            # Create grid image
            grid_width = sample_width * num_steps
            grid_height = sample_height
            grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

            # Place images in grid
            for col, (step, image) in enumerate(step_samples):
                # Resize image if needed
                if image.size != (sample_width, sample_height):
                    image = image.resize((sample_width, sample_height), Image.Resampling.LANCZOS)

                # Paste into grid
                x_offset = col * sample_width
                grid_image.paste(image, (x_offset, 0))

                # Add step label
                from PIL import ImageDraw, ImageFont

                try:
                    draw = ImageDraw.Draw(grid_image)
                    # Try to use default font, fallback to basic
                    try:
                        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
                    except:
                        font = ImageFont.load_default()

                    # Add step number
                    draw.text((x_offset + 10, 10), f"Step {step}", fill="white", font=font)
                except Exception as e:
                    logger.debug(f"Could not add step labels to grid: {e}")

            logger.info(f"Created comparison grid with {num_steps} steps")
            return grid_image

        except Exception as e:
            logger.error(f"Failed to create comparison grid: {e}")
            return None

    def get_samples_for_step(self, step: int) -> List[Image.Image]:
        """
        Get validation samples for a specific step.

        Args:
            step: Training step

        Returns:
            List of images for that step
        """
        return self.generated_samples.get(step, [])

    def get_all_steps(self) -> List[int]:
        """
        Get all steps that have validation samples.

        Returns:
            Sorted list of steps
        """
        return sorted(self.generated_samples.keys())

    def save_samples_to_disk(self, output_dir: Union[str, Path], step: Optional[int] = None):
        """
        Save validation samples to disk.

        Args:
            output_dir: Directory to save samples
            step: Specific step to save (saves all if None)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        steps_to_save = [step] if step is not None else self.get_all_steps()

        for step_num in steps_to_save:
            samples = self.get_samples_for_step(step_num)
            if not samples:
                continue

            step_dir = output_dir / f"step_{step_num}"
            step_dir.mkdir(exist_ok=True)

            for i, (image, prompt) in enumerate(zip(samples, self.prompts)):
                # Create safe filename from prompt
                safe_prompt = "".join(
                    c for c in prompt if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                safe_prompt = safe_prompt.replace(" ", "_")[:50]  # Limit length

                filename = f"{i:02d}_{safe_prompt}.png"
                filepath = step_dir / filename

                try:
                    image.save(filepath)
                    logger.debug(f"Saved sample: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save sample {filepath}: {e}")

    def clear_old_samples(self, keep_recent: int = 10):
        """
        Clear old validation samples to save memory.

        Args:
            keep_recent: Number of most recent steps to keep
        """
        if len(self.generated_samples) <= keep_recent:
            return

        # Keep only the most recent steps
        all_steps = sorted(self.generated_samples.keys())
        steps_to_remove = all_steps[:-keep_recent]

        for step in steps_to_remove:
            del self.generated_samples[step]

        logger.info(f"Cleared {len(steps_to_remove)} old validation samples")


def create_validation_function(
    sampler: ValidationSampler, output_dir: Optional[Union[str, Path]] = None
):
    """
    Create validation function for use with LoRATrainer.

    Args:
        sampler: ValidationSampler instance
        output_dir: Optional directory to save samples

    Returns:
        Validation function that can be passed to trainer
    """

    def validation_fn(model, step: int) -> Dict[str, float]:
        """
        Validation function for trainer integration.

        Args:
            model: Current model (unused, sampler has reference)
            step: Current training step

        Returns:
            Validation metrics
        """
        try:
            # Generate samples
            images = sampler.generate_samples(step)

            # Save to disk if requested
            if output_dir and images:
                sampler.save_samples_to_disk(output_dir, step)

            # Return basic metrics
            return {
                "validation_samples_generated": len(images),
                "validation_step": step,
            }

        except Exception as e:
            logger.error(f"Validation function failed: {e}")
            return {
                "validation_error": str(e),
                "validation_step": step,
            }

    return validation_fn


# Default validation prompts for different use cases
DEFAULT_CHARACTER_PROMPTS = [
    "A photo of [TRIGGER_WORD] smiling",
    "A portrait of [TRIGGER_WORD] in natural lighting",
    "A close-up photo of [TRIGGER_WORD]",
    "[TRIGGER_WORD] wearing a black t-shirt",
    "A candid shot of [TRIGGER_WORD] outdoors",
    "A professional headshot of [TRIGGER_WORD]",
    "[TRIGGER_WORD] with different expressions",
    "[TRIGGER_WORD] in various poses",
]

DEFAULT_STYLE_PROMPTS = [
    "A painting in the style of [TRIGGER_WORD]",
    "Artwork created with [TRIGGER_WORD] aesthetic",
    "An image rendered in [TRIGGER_WORD] style",
    "[TRIGGER_WORD] inspired artwork",
    "A scene with [TRIGGER_WORD] characteristics",
    "Art made in the [TRIGGER_WORD] technique",
    "[TRIGGER_WORD] styled illustration",
    "Visual elements of [TRIGGER_WORD] style",
]

DEFAULT_CONCEPT_PROMPTS = [
    "An example of [TRIGGER_WORD]",
    "A demonstration of [TRIGGER_WORD]",
    "[TRIGGER_WORD] in action",
    "The concept of [TRIGGER_WORD] visualized",
    "[TRIGGER_WORD] represented graphically",
    "An illustration showing [TRIGGER_WORD]",
    "[TRIGGER_WORD] depicted artistically",
    "Visual representation of [TRIGGER_WORD]",
]
