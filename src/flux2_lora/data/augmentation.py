"""
Advanced data augmentation utilities for Flux2-dev LoRA training.

This module provides comprehensive data augmentation capabilities including:
- Image augmentations (geometric, color, noise)
- Text augmentations (synonym replacement, backtranslation)
- Augmentation pipelines with configurable intensity
- Quality preservation checks
- Performance optimizations for training
"""

import random
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    # General settings
    enabled: bool = False
    probability: float = 0.5  # Probability to apply augmentation to each sample

    # Image augmentations
    image_augmentations: Dict[str, Any] = field(default_factory=dict)

    # Text augmentations
    text_augmentations: Dict[str, Any] = field(default_factory=dict)

    # Quality controls
    preserve_quality: bool = True
    max_augmentations_per_sample: int = 3

    def __post_init__(self):
        # Set default image augmentations if not provided
        if not self.image_augmentations:
            self.image_augmentations = {
                "geometric": {
                    "rotation": {"enabled": False, "degrees": 15},
                    "horizontal_flip": {"enabled": True, "probability": 0.5},
                    "vertical_flip": {"enabled": False, "probability": 0.1},
                    "scale": {"enabled": True, "scale_limit": 0.1},
                    "translate": {"enabled": False, "translate_percent": 0.1},
                    "shear": {"enabled": False, "degrees": 5},
                },
                "color": {
                    "brightness": {"enabled": True, "limit": 0.1},
                    "contrast": {"enabled": True, "limit": 0.1},
                    "saturation": {"enabled": False, "limit": 0.1},
                    "hue": {"enabled": False, "limit": 0.1},
                },
                "noise": {
                    "gaussian_noise": {"enabled": False, "var_limit": (10, 50)},
                    "salt_pepper": {"enabled": False, "amount": 0.01},
                    "speckle": {"enabled": False, "var_limit": (0.01, 0.1)},
                },
                "blur": {
                    "gaussian_blur": {"enabled": False, "kernel_size": 3},
                    "motion_blur": {"enabled": False, "kernel_size": 7},
                },
            }

        # Set default text augmentations if not provided
        if not self.text_augmentations:
            self.text_augmentations = {
                "synonym_replacement": {"enabled": True, "probability": 0.1, "max_replacements": 2},
                "random_deletion": {"enabled": False, "probability": 0.05},
                "random_swap": {"enabled": False, "probability": 0.05, "max_swaps": 1},
                "backtranslation": {
                    "enabled": False,
                    "probability": 0.1,
                    "languages": ["fr", "de", "es"],
                },
            }


class ImageAugmenter:
    """Advanced image augmentation utilities."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config

        if ALBUMENTATIONS_AVAILABLE:
            self.transforms = self._build_albumentations_pipeline()
        else:
            self.transforms = None
            logger.warning("Albumentations not available, using basic PIL augmentations")

    def _build_albumentations_pipeline(self) -> Optional[A.Compose]:
        """Build Albumentations augmentation pipeline."""
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        transforms = []

        # Geometric transformations
        geo_config = self.config.get("geometric", {})
        if geo_config.get("horizontal_flip", {}).get("enabled", False):
            transforms.append(
                A.HorizontalFlip(p=geo_config["horizontal_flip"].get("probability", 0.5))
            )

        if geo_config.get("rotation", {}).get("enabled", False):
            transforms.append(A.Rotate(limit=geo_config["rotation"].get("degrees", 15), p=0.5))

        if geo_config.get("scale", {}).get("enabled", False):
            transforms.append(
                A.RandomScale(scale_limit=geo_config["scale"].get("scale_limit", 0.1), p=0.5)
            )

        # Color transformations
        color_config = self.config.get("color", {})
        if color_config.get("brightness", {}).get("enabled", False):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=color_config["brightness"].get("limit", 0.1),
                    contrast_limit=0,
                    p=0.5,
                )
            )

        if color_config.get("contrast", {}).get("enabled", False):
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=0,
                    contrast_limit=color_config["contrast"].get("limit", 0.1),
                    p=0.5,
                )
            )

        # Noise transformations
        noise_config = self.config.get("noise", {})
        if noise_config.get("gaussian_noise", {}).get("enabled", False):
            var_limit = noise_config["gaussian_noise"].get("var_limit", (10, 50))
            transforms.append(A.GaussNoise(var_limit=var_limit, p=0.3))

        return A.Compose(transforms) if transforms else None

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentations to an image.

        Args:
            image: PIL Image to augment

        Returns:
            Augmented PIL Image
        """
        if not self.config:
            return image

        # Use Albumentations if available
        if self.transforms is not None:
            # Convert PIL to numpy array
            image_np = np.array(image)

            # Apply transformations
            augmented = self.transforms(image=image_np)
            augmented_image = augmented["image"]

            # Convert back to PIL
            return Image.fromarray(augmented_image)

        # Fallback to basic PIL augmentations
        return self._apply_basic_augmentations(image)

    def _apply_basic_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply basic PIL-based augmentations."""
        augmented = image.copy()

        # Simple horizontal flip
        if random.random() < 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)

        # Simple brightness/contrast adjustment
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(random.uniform(0.8, 1.2))

        return augmented


class TextAugmenter:
    """Advanced text augmentation utilities."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text augmenter.

        Args:
            config: Text augmentation configuration
        """
        self.config = config

        # Basic synonym dictionary for common augmentation
        self.synonyms = {
            "person": ["individual", "human", "figure", "character"],
            "face": ["facial", "countenance", "visage", "expression"],
            "portrait": ["picture", "photograph", "image", "shot"],
            "beautiful": ["gorgeous", "stunning", "attractive", "lovely"],
            "young": ["youthful", "juvenile", "adolescent", "teenage"],
            "old": ["aged", "elderly", "mature", "ancient"],
            "happy": ["joyful", "pleased", "delighted", "content"],
            "sad": ["unhappy", "sorrowful", "depressed", "melancholy"],
            "bright": ["light", "luminous", "radiant", "vivid"],
            "dark": ["dim", "shadowy", "gloomy", "murky"],
        }

    def augment_caption(self, caption: str) -> str:
        """
        Apply text augmentations to a caption.

        Args:
            caption: Original caption text

        Returns:
            Augmented caption
        """
        if not self.config or not caption:
            return caption

        augmented = caption

        # Apply augmentations in sequence
        augmentations_applied = 0
        max_augmentations = self.config.get("max_augmentations_per_sample", 3)

        # Synonym replacement
        if (
            self.config.get("synonym_replacement", {}).get("enabled", False)
            and augmentations_applied < max_augmentations
        ):
            if random.random() < self.config["synonym_replacement"].get("probability", 0.1):
                augmented = self._synonym_replacement(augmented)
                augmentations_applied += 1

        # Random deletion
        if (
            self.config.get("random_deletion", {}).get("enabled", False)
            and augmentations_applied < max_augmentations
        ):
            if random.random() < self.config["random_deletion"].get("probability", 0.05):
                augmented = self._random_deletion(augmented)
                augmentations_applied += 1

        # Random swap
        if (
            self.config.get("random_swap", {}).get("enabled", False)
            and augmentations_applied < max_augmentations
        ):
            if random.random() < self.config["random_swap"].get("probability", 0.05):
                augmented = self._random_swap(augmented)
                augmentations_applied += 1

        return augmented

    def _synonym_replacement(self, text: str) -> str:
        """Replace random words with synonyms."""
        words = text.split()
        if len(words) < 3:
            return text

        max_replacements = self.config["synonym_replacement"].get("max_replacements", 2)
        replacements_made = 0

        for i, word in enumerate(words):
            if replacements_made >= max_replacements:
                break

            # Remove punctuation for matching
            clean_word = word.lower().strip(".,!?;:")
            if clean_word in self.synonyms:
                synonyms = self.synonyms[clean_word]
                if synonyms:
                    replacement = random.choice(synonyms)
                    # Preserve original capitalization
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    # Preserve punctuation
                    if word != clean_word:
                        punctuation = word[len(clean_word) :]
                        replacement += punctuation

                    words[i] = replacement
                    replacements_made += 1

        return " ".join(words)

    def _random_deletion(self, text: str) -> str:
        """Randomly delete words from text."""
        words = text.split()
        if len(words) < 4:  # Don't delete from very short texts
            return text

        # Keep at least 60% of words
        keep_probability = 0.7
        kept_words = [word for word in words if random.random() < keep_probability]

        # Ensure we keep at least 2 words
        if len(kept_words) < 2:
            kept_words = words[:2]

        return " ".join(kept_words)

    def _random_swap(self, text: str) -> str:
        """Randomly swap word positions."""
        words = text.split()
        if len(words) < 4:  # Need enough words to swap
            return text

        max_swaps = self.config["random_swap"].get("max_swaps", 1)

        for _ in range(max_swaps):
            # Pick two random positions
            pos1, pos2 = random.sample(range(len(words)), 2)
            # Swap them
            words[pos1], words[pos2] = words[pos2], words[pos1]

        return " ".join(words)


class DatasetAugmenter:
    """Main dataset augmentation orchestrator."""

    def __init__(self, config: AugmentationConfig):
        """
        Initialize dataset augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.image_augmenter = (
            ImageAugmenter(config.image_augmentations) if config.enabled else None
        )
        self.text_augmenter = TextAugmenter(config.text_augmentations) if config.enabled else None

        logger.info(f"Dataset augmenter initialized (enabled: {config.enabled})")

    def should_augment(self) -> bool:
        """
        Determine if augmentation should be applied to current sample.

        Returns:
            True if augmentation should be applied
        """
        if not self.config.enabled:
            return False

        return random.random() < self.config.probability

    def augment_sample(self, image: Image.Image, caption: str) -> Tuple[Image.Image, str]:
        """
        Apply augmentations to a dataset sample.

        Args:
            image: Original image
            caption: Original caption

        Returns:
            Tuple of (augmented_image, augmented_caption)
        """
        if not self.should_augment():
            return image, caption

        augmented_image = image
        augmented_caption = caption

        # Apply image augmentations
        if self.image_augmenter is not None:
            try:
                augmented_image = self.image_augmenter.augment_image(image)
            except Exception as e:
                logger.warning(f"Image augmentation failed: {e}")
                augmented_image = image

        # Apply text augmentations
        if self.text_augmenter is not None:
            try:
                augmented_caption = self.text_augmenter.augment_caption(caption)
            except Exception as e:
                logger.warning(f"Text augmentation failed: {e}")
                augmented_caption = caption

        return augmented_image, augmented_caption

    def get_augmentation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about augmentation configuration.

        Returns:
            Dictionary with augmentation statistics
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "Augmentation is disabled"}

        stats = {
            "enabled": True,
            "probability": self.config.probability,
            "image_augmentations": {},
            "text_augmentations": {},
            "quality_controls": {
                "preserve_quality": self.config.preserve_quality,
                "max_augmentations_per_sample": self.config.max_augmentations_per_sample,
            },
        }

        # Count enabled image augmentations
        for category, augmentations in self.config.image_augmentations.items():
            enabled_count = sum(1 for aug in augmentations.values() if aug.get("enabled", False))
            stats["image_augmentations"][category] = enabled_count

        # Count enabled text augmentations
        enabled_text = sum(
            1 for aug in self.config.text_augmentations.values() if aug.get("enabled", False)
        )
        stats["text_augmentations"]["enabled"] = enabled_text

        return stats


def create_augmenter(config: Dict[str, Any]) -> DatasetAugmenter:
    """
    Create a dataset augmenter from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured DatasetAugmenter
    """
    aug_config = AugmentationConfig(**config)
    return DatasetAugmenter(aug_config)


def get_default_augmentation_config() -> Dict[str, Any]:
    """
    Get default augmentation configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "enabled": True,
        "probability": 0.5,
        "preserve_quality": True,
        "max_augmentations_per_sample": 3,
    }


# Quality preservation utilities
def check_image_quality(image: Image.Image) -> Dict[str, float]:
    """
    Check basic quality metrics of an image.

    Args:
        image: PIL Image to check

    Returns:
        Dictionary with quality metrics
    """
    # Convert to grayscale for analysis
    gray = image.convert("L")
    pixels = np.array(gray)

    # Basic metrics
    brightness = np.mean(pixels) / 255.0
    contrast = np.std(pixels) / 128.0  # Normalized std dev
    sharpness = np.var(pixels) / 1000.0  # Rough sharpness measure

    return {"brightness": brightness, "contrast": contrast, "sharpness": sharpness}


def validate_augmentation_quality(
    original_image: Image.Image, augmented_image: Image.Image, threshold: float = 0.7
) -> bool:
    """
    Validate that augmentation preserves acceptable image quality.

    Args:
        original_image: Original image
        augmented_image: Augmented image
        threshold: Quality preservation threshold (0-1)

    Returns:
        True if quality is preserved
    """
    try:
        original_metrics = check_image_quality(original_image)
        augmented_metrics = check_image_quality(augmented_image)

        # Check if key metrics are within acceptable range
        brightness_ok = abs(augmented_metrics["brightness"] - original_metrics["brightness"]) < 0.3
        contrast_ok = augmented_metrics["contrast"] > original_metrics["contrast"] * threshold
        sharpness_ok = augmented_metrics["sharpness"] > original_metrics["sharpness"] * threshold

        return brightness_ok and contrast_ok and sharpness_ok

    except Exception as e:
        logger.warning(f"Quality validation failed: {e}")
        return True  # Default to allowing augmentation if validation fails
