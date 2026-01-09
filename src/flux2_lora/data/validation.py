"""
Dataset validation utilities for LoRA training.

Validates dataset structure, quality, and readiness for training.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .caption_utils import CaptionUtils

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""

    valid: bool
    image_count: int = 0
    valid_pairs: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    min_resolution: Optional[Tuple[int, int]] = None
    max_resolution: Optional[Tuple[int, int]] = None
    avg_caption_length: float = 0.0


def validate_dataset_structure(
    dataset_path: str,
    min_images: int = 5,
    check_captions: bool = True,
    check_resolution: bool = True,
    min_resolution: int = 512,
    caption_sources: Optional[List[str]] = None,
    verbose: bool = False,
) -> DatasetValidationResult:
    """
    Validate dataset structure and quality.

    Args:
        dataset_path: Path to dataset directory
        min_images: Minimum number of images required
        check_captions: Whether to validate captions
        check_resolution: Whether to check image resolutions
        min_resolution: Minimum image resolution (width or height)
        caption_sources: Caption file extensions to check
        verbose: Enable verbose logging

    Returns:
        DatasetValidationResult with validation details
    """
    result = DatasetValidationResult(valid=True)
    dataset_dir = Path(dataset_path)

    # Check if directory exists
    if not dataset_dir.exists():
        result.valid = False
        result.errors.append(f"Dataset directory does not exist: {dataset_path}")
        return result

    if not dataset_dir.is_dir():
        result.valid = False
        result.errors.append(f"Dataset path is not a directory: {dataset_path}")
        return result

    # Find image files
    try:
        image_files = CaptionUtils.find_image_files(dataset_dir)
        result.image_count = len(image_files)

        if verbose:
            logger.info(f"Found {result.image_count} images in {dataset_path}")

    except Exception as e:
        result.valid = False
        result.errors.append(f"Error finding image files: {e}")
        return result

    # Check minimum image count
    if result.image_count < min_images:
        result.valid = False
        result.errors.append(
            f"Insufficient images: found {result.image_count}, need at least {min_images}"
        )
        return result

    # Load captions if checking
    captions = {}
    if check_captions:
        try:
            if caption_sources is None:
                caption_sources = ["txt", "caption", "json", "exif"]

            captions = CaptionUtils.load_dataset_captions(dataset_dir, caption_sources)

            if verbose:
                logger.info(f"Loaded {len(captions)} captions")

        except Exception as e:
            result.warnings.append(f"Error loading captions: {e}")

    # Validate each image
    caption_lengths = []
    resolutions = []

    for image_path in image_files:
        # Check caption if enabled
        if check_captions:
            caption = captions.get(image_path.name)

            if not caption:
                result.warnings.append(f"No caption found for {image_path.name}")
                continue

            # Validate caption quality
            if len(caption.strip()) < 3:
                result.warnings.append(
                    f"Very short caption for {image_path.name}: '{caption}'"
                )
            elif len(caption) > 3000:
                result.warnings.append(
                    f"Very long caption for {image_path.name} ({len(caption)} chars)"
                )
            else:
                caption_lengths.append(len(caption))
                result.valid_pairs += 1

        # Check image resolution if enabled
        if check_resolution:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolutions.append((width, height))

                    # Check minimum resolution
                    if width < min_resolution or height < min_resolution:
                        result.warnings.append(
                            f"Low resolution image {image_path.name}: {width}x{height} "
                            f"(recommended minimum: {min_resolution}x{min_resolution})"
                        )

                    # Check if image is too small
                    if width < 256 or height < 256:
                        result.errors.append(
                            f"Image too small {image_path.name}: {width}x{height} "
                            f"(minimum: 256x256)"
                        )
                        result.valid = False

            except Exception as e:
                result.errors.append(f"Error reading image {image_path.name}: {e}")
                result.valid = False

    # Calculate statistics
    if caption_lengths:
        result.avg_caption_length = sum(caption_lengths) / len(caption_lengths)

    if resolutions:
        result.min_resolution = (
            min(r[0] for r in resolutions),
            min(r[1] for r in resolutions),
        )
        result.max_resolution = (
            max(r[0] for r in resolutions),
            max(r[1] for r in resolutions),
        )

    # Final validation checks
    if check_captions and result.valid_pairs == 0:
        result.valid = False
        result.errors.append("No valid image-caption pairs found")

    if result.valid_pairs < min_images:
        result.valid = False
        result.errors.append(
            f"Insufficient valid pairs: found {result.valid_pairs}, need at least {min_images}"
        )

    # Add warnings for common issues
    if result.valid_pairs < result.image_count:
        missing_captions = result.image_count - result.valid_pairs
        result.warnings.append(
            f"{missing_captions} images have no or invalid captions "
            f"({missing_captions / result.image_count * 100:.1f}% of dataset)"
        )

    if len(result.warnings) > result.image_count * 0.2:
        result.warnings.insert(
            0, f"Many warnings detected ({len(result.warnings)} total) - review dataset quality"
        )

    return result
