"""
Dataset and DataLoader for LoRA training on Flux2-dev.

Supports various image formats and caption sources with efficient preprocessing.
"""

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .caption_utils import CaptionUtils
from .augmentation import DatasetAugmenter, AugmentationConfig

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


class LoRADataset(Dataset):
    """
    Dataset for LoRA training with images and captions.

    Supports:
    - Multiple image formats (jpg, png, bmp, tiff, webp)
    - Multiple caption sources (.txt, .caption, JSON, EXIF)
    - Configurable image preprocessing
    - Data augmentation options
    - Efficient loading with caching
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 1024,
        caption_sources: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        cache_images: bool = False,
        cache_images_threshold: int = 100,
        process_exif: bool = False,
        shuffle_seed: Optional[int] = None,
        validate_captions: bool = True,
        min_caption_length: int = 3,
        max_caption_length: int = 3000,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LoRA dataset.

        Args:
            data_dir: Directory containing images and captions
            resolution: Target image resolution (square)
            caption_sources: Preferred caption sources in order
            transform: Optional custom transform pipeline
            cache_images: Whether to cache images in memory
            cache_images_threshold: Auto-enable caching for datasets smaller than this
            process_exif: Whether to process EXIF orientation (disabled for speed)
            shuffle_seed: Seed for reproducible shuffling
            validate_captions: Whether to validate caption quality
            min_caption_length: Minimum caption length
            max_caption_length: Maximum caption length
            augmentation_config: Configuration for data augmentation
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.caption_sources = caption_sources or ["txt", "caption", "json", "exif"]
        self.cache_images_threshold = cache_images_threshold
        self.process_exif = process_exif
        self.validate_captions = validate_captions
        self.min_caption_length = min_caption_length
        self.max_caption_length = max_caption_length

        # Initialize augmentation
        if augmentation_config is None:
            augmentation_config = {"enabled": False}
        aug_config = AugmentationConfig(**augmentation_config)
        self.augmenter = DatasetAugmenter(aug_config)

        # Validate data directory
        if not self.data_dir.exists():
            raise DatasetValidationError(f"Data directory does not exist: {self.data_dir}")

        # Find images and load captions
        self.image_files = CaptionUtils.find_image_files(self.data_dir)
        if not self.image_files:
            raise DatasetValidationError(f"No image files found in {self.data_dir}")

        self.captions = CaptionUtils.load_dataset_captions(self.data_dir, self.caption_sources)

        # Filter images with valid captions
        self.valid_indices = self._filter_valid_images()

        if not self.valid_indices:
            raise DatasetValidationError(
                f"No valid image-caption pairs found in {self.data_dir}. "
                f"Found {len(self.image_files)} images but none have valid captions."
            )

        # Auto-enable caching for small datasets (improves speed after first epoch)
        if not cache_images and len(self.valid_indices) < self.cache_images_threshold:
            cache_images = True
            logger.info(
                f"Auto-enabling image caching for small dataset "
                f"({len(self.valid_indices)} images < {self.cache_images_threshold} threshold)"
            )

        # Setup image transforms
        self.transform = transform or self._create_default_transform()

        # Image cache
        self.image_cache = {} if cache_images else None
        self.cache_images = cache_images

        # Shuffle if seed provided
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.valid_indices)

        logger.info(
            f"Initialized LoRADataset: {len(self.valid_indices)} valid pairs "
            f"from {len(self.image_files)} images in {self.data_dir}"
        )

    def _filter_valid_images(self) -> List[int]:
        """Filter indices for images with valid captions."""
        valid_indices = []

        for idx, image_path in enumerate(self.image_files):
            caption = self.captions.get(image_path.name)

            if not caption:
                logger.debug(f"No caption found for {image_path.name}")
                continue

            # Validate caption if enabled
            if self.validate_captions:
                if not CaptionUtils.validate_caption(
                    caption, self.min_caption_length, self.max_caption_length
                ):
                    logger.debug(f"Invalid caption for {image_path.name}: '{caption[:50]}...'")
                    continue

                # Clean caption
                caption = CaptionUtils.clean_caption(caption)
                if not caption:
                    continue

                # Update cleaned caption
                self.captions[image_path.name] = caption

            valid_indices.append(idx)

        return valid_indices

    def _create_default_transform(self) -> transforms.Compose:
        """Create default image preprocessing pipeline for Flux2-dev (optimized for speed)."""
        ops = [transforms.Lambda(self._ensure_rgb)]

        # Resize without antialiasing for speed (quality difference minimal for training)
        # Use antialias=True during evaluation if needed
        ops.append(
            transforms.Resize(
                self.resolution, interpolation=InterpolationMode.BILINEAR, antialias=False
            )
        )

        # Crop
        if self.resolution < 1024:
            ops.append(transforms.RandomCrop(self.resolution))
        else:
            ops.append(transforms.CenterCrop(self.resolution))

        # Convert to tensor and normalize to [-1, 1]
        ops.extend([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        return transforms.Compose(ops)

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB format."""
        if image.mode != "RGB":
            if image.mode == "RGBA":
                # Composite on white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                return background
            elif image.mode == "L":
                return image.convert("RGB")
            else:
                return image.convert("RGB")
        return image

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and optionally cache an image."""
        if self.image_cache is not None and image_path in self.image_cache:
            return self.image_cache[image_path]

        try:
            # Load image with PIL
            image = Image.open(image_path)

            # Auto-orient based on EXIF (optional, disabled by default for speed)
            # Most web images are already correctly oriented
            if self.process_exif:
                image = ImageOps.exif_transpose(image)

            # Ensure RGB
            image = self._ensure_rgb(image)

            # Cache if enabled
            if self.image_cache is not None:
                self.image_cache[image_path] = image

            return image

        except Exception as e:
            raise DatasetValidationError(f"Failed to load image {image_path}: {e}")

    def __len__(self) -> int:
        """Return number of valid image-caption pairs."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training example.

        Returns:
            Dictionary containing:
            - image: Preprocessed image tensor
            - caption: Caption text
            - image_path: Original image path
            - image_name: Image filename
            - augmented: Whether this sample was augmented
        """
        # Get actual image index
        actual_idx = self.valid_indices[idx]
        image_path = self.image_files[actual_idx]

        # Load image (PIL format for augmentation)
        image = self._load_image(image_path)

        # Get caption
        caption = self.captions[image_path.name]

        # Apply augmentation
        augmented = False
        if self.augmenter.should_augment():
            try:
                image, caption = self.augmenter.augment_sample(image, caption)
                augmented = True
            except Exception as e:
                logger.warning(f"Augmentation failed for {image_path.name}: {e}")
                # Continue with original data

        # Apply standard transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "image_path": str(image_path),
            "image_name": image_path.name,
            "augmented": augmented,
            "image_id": actual_idx,  # Unique ID for latent caching (not affected by augmentation)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.

        Returns:
            Statistics dictionary
        """
        if not self.valid_indices:
            return {"error": "No valid data"}

        # Image statistics
        resolutions = []
        aspect_ratios = []
        file_sizes = []

        logger.info("Computing image statistics...")
        for idx in self.valid_indices:
            image_path = self.image_files[idx]

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolutions.append((width, height))
                    aspect_ratios.append(width / height)

                    # File size in MB
                    file_size_mb = image_path.stat().st_size / (1024 * 1024)
                    file_sizes.append(file_size_mb)
            except Exception as e:
                logger.warning(f"Failed to analyze {image_path}: {e}")

        # Caption statistics
        captions = [self.captions[self.image_files[idx].name] for idx in self.valid_indices]
        caption_stats = CaptionUtils.analyze_caption_statistics(
            {
                self.image_files[idx].name: self.captions[self.image_files[idx].name]
                for idx in self.valid_indices
            }
        )

        # Compute derived statistics
        widths, heights = zip(*resolutions) if resolutions else ([], [])

        stats = {
            "dataset_path": str(self.data_dir),
            "total_images_found": len(self.image_files),
            "valid_pairs": len(self.valid_indices),
            "invalid_pairs": len(self.image_files) - len(self.valid_indices),
            # Image statistics
            "resolution_stats": {
                "min_width": min(widths) if widths else 0,
                "max_width": max(widths) if widths else 0,
                "min_height": min(heights) if heights else 0,
                "max_height": max(heights) if heights else 0,
                "avg_width": sum(widths) / len(widths) if widths else 0,
                "avg_height": sum(heights) / len(heights) if heights else 0,
                "most_common_resolution": max(set(resolutions), key=resolutions.count)
                if resolutions
                else None,
            },
            "aspect_ratio_stats": {
                "min_aspect_ratio": min(aspect_ratios) if aspect_ratios else 0,
                "max_aspect_ratio": max(aspect_ratios) if aspect_ratios else 0,
                "avg_aspect_ratio": sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 0,
                "square_images": sum(1 for ar in aspect_ratios if 0.9 <= ar <= 1.1),
                "landscape_images": sum(1 for ar in aspect_ratios if ar > 1.1),
                "portrait_images": sum(1 for ar in aspect_ratios if ar < 0.9),
            },
            "file_size_stats": {
                "min_size_mb": min(file_sizes) if file_sizes else 0,
                "max_size_mb": max(file_sizes) if file_sizes else 0,
                "avg_size_mb": sum(file_sizes) / len(file_sizes) if file_sizes else 0,
                "total_size_mb": sum(file_sizes),
            },
            # Caption statistics
            "caption_stats": caption_stats,
            # Dataset configuration
            "config": {
                "resolution": self.resolution,
                "caption_sources": self.caption_sources,
                "cache_images": self.cache_images,
                "validate_captions": self.validate_captions,
            },
        }

        return stats

    def get_sample_images(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample images for preview.

        Args:
            num_samples: Number of samples to return

        Returns:
            List of sample dictionaries
        """
        samples = []
        num_samples = min(num_samples, len(self.valid_indices))

        for i in range(num_samples):
            idx = self.valid_indices[i]
            image_path = self.image_files[idx]
            caption = self.captions.get(image_path.name, "No caption")

            samples.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "caption": caption,
                    "caption_length": len(caption),
                }
            )

        return samples


def create_dataloader(
    dataset: LoRADataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Create optimized DataLoader for LoRA training.

    Args:
        dataset: LoRADataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        Configured DataLoader
    """
    # Optimize number of workers based on dataset size
    if len(dataset) < batch_size * 2:
        num_workers = min(2, num_workers)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for LoRA training batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched data dictionary
    """
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    image_ids = [item.get("image_id", idx) for idx, item in enumerate(batch)]
    augmented = [item.get("augmented", False) for item in batch]

    return {
        "images": images,
        "captions": captions,
        "image_paths": image_paths,
        "image_names": image_names,
        "image_ids": image_ids,
        "augmented": augmented,
    }


def validate_dataset(
    data_dir: Union[str, Path],
    resolution: int = 1024,
    min_caption_length: int = 3,
    max_caption_length: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate a dataset and return detailed report.

    Args:
        data_dir: Directory containing dataset
        resolution: Target resolution for training
        min_caption_length: Minimum caption length
        max_caption_length: Maximum caption length
        verbose: Whether to print detailed information

    Returns:
        Validation report dictionary
    """
    try:
        dataset = LoRADataset(
            data_dir=data_dir,
            resolution=resolution,
            validate_captions=True,
            min_caption_length=min_caption_length,
            max_caption_length=max_caption_length,
        )

        stats = dataset.get_statistics()
        samples = dataset.get_sample_images(5)

        # Add validation status
        stats["validation"] = {"status": "passed", "errors": [], "warnings": []}

        # Check for common issues
        if stats["valid_pairs"] < 10:
            stats["validation"]["warnings"].append(
                f"Very small dataset: only {stats['valid_pairs']} valid pairs"
            )

        if stats["caption_stats"]["short_captions"] > stats["valid_pairs"] * 0.3:
            stats["validation"]["warnings"].append(
                f"Many short captions: {stats['caption_stats']['short_captions']} out of {stats['valid_pairs']}"
            )

        if stats["file_size_stats"]["total_size_mb"] > 1000:
            stats["validation"]["warnings"].append(
                f"Large dataset size: {stats['file_size_stats']['total_size_mb']:.1f} MB"
            )

        if verbose:
            print(f"✅ Dataset validation passed!")
            print(f"   Found {stats['valid_pairs']} valid image-caption pairs")
            print(
                f"   Average caption length: {stats['caption_stats']['avg_length']:.1f} characters"
            )
            print(
                f"   Average image size: {stats['resolution_stats']['avg_width']:.0f}x{stats['resolution_stats']['avg_height']:.0f}"
            )

            if stats["validation"]["warnings"]:
                print("⚠️  Warnings:")
                for warning in stats["validation"]["warnings"]:
                    print(f"   - {warning}")

        return {"dataset": dataset, "statistics": stats, "samples": samples}

    except Exception as e:
        error_msg = f"Dataset validation failed: {e}"
        if verbose:
            print(f"❌ {error_msg}")

        return {
            "dataset": None,
            "statistics": {
                "validation": {"status": "failed", "errors": [error_msg], "warnings": []}
            },
            "samples": [],
        }
