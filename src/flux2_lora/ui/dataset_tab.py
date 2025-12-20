"""
Dataset tools tab for the Gradio interface.

Provides utilities for analyzing and managing training datasets.
"""

import os
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

import gradio as gr

if TYPE_CHECKING:
    from .gradio_app import LoRATrainingApp

from .help_utils import help_system


def load_dataset_for_analysis(app: "LoRATrainingApp", file_obj) -> Tuple[str, Optional[Path]]:
    """
    Load a dataset ZIP file for analysis.

    Args:
        app: Main application instance
        file_obj: Gradio file object

    Returns:
        Tuple of (status_message, dataset_path)
    """
    if not file_obj:
        return "No file uploaded", None

    try:
        # Create temporary directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="flux2_lora_dataset_analysis_"))
        zip_path = Path(file_obj.name)

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Register the uploaded file
        file_id = app.register_uploaded_file(temp_dir, "dataset")

        # Validate dataset structure
        validation_result = validate_dataset_structure(temp_dir)

        if validation_result["valid"]:
            # Calculate basic caption statistics
            avg_caption_len = calculate_basic_caption_stats(temp_dir)
            status = f"✅ Dataset loaded: {validation_result['image_count']} images, {validation_result['caption_count']} captions"
            app.update_training_state("analysis_dataset_path", str(temp_dir))
            app.update_training_state("analysis_dataset_info", validation_result)
            return status, temp_dir
        else:
            # Clean up invalid dataset
            shutil.rmtree(temp_dir, ignore_errors=True)
            return f"❌ Invalid dataset: {validation_result['error']}", None

    except Exception as e:
        return f"❌ Upload failed: {str(e)}", None


def load_dataset_from_path(app: "LoRATrainingApp", dataset_path: str) -> Tuple[str, Optional[Path]]:
    """
    Load a dataset from a directory path for analysis.

    Args:
        app: Main application instance
        dataset_path: Path to dataset directory

    Returns:
        Tuple of (status_message, dataset_path)
    """
    if not dataset_path or not dataset_path.strip():
        return "No dataset path provided", None

    try:
        path = Path(dataset_path.strip())

        if not path.exists():
            return f"❌ Dataset path does not exist: {dataset_path}", None

        if not path.is_dir():
            return f"❌ Path is not a directory: {dataset_path}", None

        # Validate dataset structure
        validation_result = validate_dataset_structure(path)

        if validation_result["valid"]:
            status = f"✅ Dataset loaded: {validation_result['image_count']} images, {validation_result['caption_count']} captions"
            app.update_training_state("analysis_dataset_path", str(path))
            app.update_training_state("analysis_dataset_info", validation_result)
            return status, path
        else:
            return f"❌ Invalid dataset: {validation_result['error']}", None

    except Exception as e:
        return f"❌ Path validation failed: {str(e)}", None


def calculate_basic_caption_stats(dataset_path: Path) -> float:
    """
    Calculate basic caption statistics for quick display.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Average caption length in words
    """
    try:
        caption_extensions = {".txt", ".caption"}
        captions = []

        # Collect all captions
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in caption_extensions:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                            if caption:  # Skip empty captions
                                captions.append(caption)
                    except Exception:
                        continue

        if not captions:
            return 0.0

        # Calculate average word count
        total_words = sum(len(caption.split()) for caption in captions)
        return round(total_words / len(captions), 1)

    except Exception:
        return 0.0


def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
    """
    Validate dataset structure and count files.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Validation results dictionary
    """
    try:
        # Find all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        caption_extensions = {".txt", ".caption"}

        image_files = []
        caption_files = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
                elif file_path.suffix.lower() in caption_extensions:
                    caption_files.append(file_path)

        if not image_files:
            return {"valid": False, "error": "No image files found in dataset"}

        # Basic validation
        result = {
            "valid": True,
            "image_count": len(image_files),
            "caption_count": len(caption_files),
            "image_extensions": list(set(f.suffix.lower() for f in image_files)),
            "caption_extensions": list(set(f.suffix.lower() for f in caption_files)),
        }

        # Check for potential issues
        issues = []

        # Caption coverage
        caption_coverage = len(caption_files) / len(image_files) if image_files else 0
        if caption_coverage < 0.5:
            issues.append(
                f"Low caption coverage: only {caption_coverage:.1%} of images have captions"
            )

        # File size warnings
        total_size = sum(f.stat().st_size for f in image_files)
        avg_size = total_size / len(image_files) if image_files else 0
        if avg_size < 10000:  # Less than 10KB average
            issues.append(
                f"Very small average image size: {avg_size / 1024:.1f}KB - may be too low resolution"
            )

        result["issues"] = issues
        result["caption_coverage"] = caption_coverage
        result["avg_image_size_kb"] = avg_size / 1024

        return result

    except Exception as e:
        return {"valid": False, "error": f"Validation failed: {str(e)}"}


def analyze_dataset_comprehensive(dataset_path: Path) -> Dict[str, Any]:
    """
    Perform comprehensive dataset analysis.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Comprehensive analysis results
    """
    try:
        # Image analysis
        image_analysis = analyze_images(dataset_path)

        # Caption analysis
        caption_analysis = analyze_captions(dataset_path)

        # Basic statistics (simplified version without LoRADataset)
        basic_stats = {
            "dataset_path": str(dataset_path),
            "total_images_found": image_analysis.get("total_images", 0),
            "valid_pairs": min(
                image_analysis.get("total_images", 0), caption_analysis.get("total_captions", 0)
            ),
            "invalid_pairs": 0,  # Simplified
            "caption_stats": {
                "total_captions": caption_analysis.get("total_captions", 0),
                "avg_length": caption_analysis.get("avg_chars_per_caption", 0),
                "min_length": caption_analysis.get("min_chars", 0),
                "max_length": caption_analysis.get("max_chars", 0),
                "empty_captions": 0,  # Simplified
                "short_captions": 0,  # Simplified
                "long_captions": 0,  # Simplified
                "length_distribution": caption_analysis.get("length_distribution", {}),
            },
            "config": {
                "resolution": 1024,
                "caption_sources": ["txt", "caption"],
                "cache_images": False,
                "validate_captions": True,
            },
        }

        # Add image resolution stats
        if "avg_width" in image_analysis and "avg_height" in image_analysis:
            basic_stats["resolution_stats"] = {
                "min_width": image_analysis.get("min_width", 0),
                "max_width": image_analysis.get("max_width", 0),
                "min_height": image_analysis.get("min_height", 0),
                "max_height": image_analysis.get("max_height", 0),
                "avg_width": image_analysis.get("avg_width", 0),
                "avg_height": image_analysis.get("avg_height", 0),
                "most_common_resolution": (
                    image_analysis.get("avg_width", 0),
                    image_analysis.get("avg_height", 0),
                ),
            }
            basic_stats["aspect_ratio_stats"] = {
                "avg_aspect_ratio": image_analysis.get("avg_aspect_ratio", 1.0),
                "square_images": 0,  # Simplified
                "landscape_images": 0,  # Simplified
                "portrait_images": 0,  # Simplified
            }
            basic_stats["file_size_stats"] = {
                "avg_size_mb": image_analysis.get("avg_file_size_kb", 0) / 1024,
                "total_size_mb": image_analysis.get("total_images", 0)
                * image_analysis.get("avg_file_size_kb", 0)
                / 1024,
            }

        # Combine results
        analysis = {
            "basic_stats": basic_stats,
            "image_analysis": image_analysis,
            "caption_analysis": caption_analysis,
            "validation_issues": validate_dataset_quality(dataset_path),
        }

        return analysis

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


def analyze_images(dataset_path: Path) -> Dict[str, Any]:
    """
    Analyze images in the dataset.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Image analysis results
    """
    try:
        from PIL import Image
        import numpy as np

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        image_files = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)

        if not image_files:
            return {"error": "No images found"}

        # Analyze sample of images (first 100 or all if fewer)
        sample_size = min(100, len(image_files))
        sample_files = image_files[:sample_size]

        dimensions = []
        file_sizes = []
        color_spaces = []

        for img_path in sample_files:
            try:
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
                    file_sizes.append(img_path.stat().st_size)
                    color_spaces.append(img.mode)
            except Exception:
                continue  # Skip corrupted images

        if not dimensions:
            return {"error": "Could not analyze any images"}

        # Calculate statistics
        widths, heights = zip(*dimensions)
        aspect_ratios = [w / h for w, h in dimensions]

        # Resolution distribution
        resolution_counts = Counter()
        for w, h in dimensions:
            if w >= 2048 or h >= 2048:
                resolution_counts["4K+"] += 1
            elif w >= 1024 or h >= 1024:
                resolution_counts["1024p+"] += 1
            elif w >= 512 or h >= 512:
                resolution_counts["512p+"] += 1
            else:
                resolution_counts["Low"] += 1

        return {
            "total_images": len(image_files),
            "analyzed_sample": len(dimensions),
            "avg_width": int(np.mean(widths)),
            "avg_height": int(np.mean(heights)),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_aspect_ratio": round(np.mean(aspect_ratios), 2),
            "resolution_distribution": dict(resolution_counts),
            "avg_file_size_kb": round(np.mean(file_sizes) / 1024, 1),
            "color_spaces": dict(Counter(color_spaces)),
        }

    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


def analyze_captions(dataset_path: Path) -> Dict[str, Any]:
    """
    Analyze captions in the dataset.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Caption analysis results
    """
    try:
        caption_extensions = {".txt", ".caption"}
        captions = []

        # Collect all captions
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in caption_extensions:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                            if caption:  # Skip empty captions
                                captions.append(caption)
                    except Exception:
                        continue  # Skip unreadable files

        if not captions:
            return {"error": "No captions found"}

        # Analyze captions
        caption_lengths = [len(caption.split()) for caption in captions]
        char_lengths = [len(caption) for caption in captions]

        # Word frequency analysis (top 20 words)
        all_words = []
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)

        word_freq = Counter(all_words).most_common(20)

        # Caption length distribution
        length_distribution = Counter()
        for length in caption_lengths:
            if length <= 5:
                length_distribution["1-5"] += 1
            elif length <= 10:
                length_distribution["6-10"] += 1
            elif length <= 20:
                length_distribution["11-20"] += 1
            elif length <= 50:
                length_distribution["21-50"] += 1
            else:
                length_distribution["50+"] += 1

        return {
            "total_captions": len(captions),
            "avg_words_per_caption": round(sum(caption_lengths) / len(caption_lengths), 1),
            "avg_chars_per_caption": round(sum(char_lengths) / len(char_lengths), 1),
            "min_words": min(caption_lengths),
            "max_words": max(caption_lengths),
            "min_chars": min(char_lengths) if char_lengths else 0,
            "max_chars": max(char_lengths) if char_lengths else 0,
            "length_distribution": dict(length_distribution),
            "top_words": word_freq[:10],  # Top 10 for UI display
            "unique_words": len(set(all_words)),
            "vocabulary_size": len(set(all_words)),
        }

    except Exception as e:
        return {"error": f"Caption analysis failed: {str(e)}"}


def validate_dataset_quality(dataset_path: Path) -> List[Dict[str, Any]]:
    """
    Validate dataset quality and identify issues.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        List of validation issues
    """
    issues = []

    try:
        # Check for missing captions
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        caption_extensions = {".txt", ".caption"}

        image_stems = set()
        caption_stems = set()

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                stem = file_path.stem

                if file_path.suffix.lower() in image_extensions:
                    image_stems.add(stem)
                elif file_path.suffix.lower() in caption_extensions:
                    caption_stems.add(stem)

        # Find images without captions
        images_without_captions = image_stems - caption_stems
        if images_without_captions:
            issues.append(
                {
                    "type": "missing_captions",
                    "severity": "warning",
                    "description": f"{len(images_without_captions)} images have no corresponding captions",
                    "affected_files": sorted(list(images_without_captions))[:5],  # Show first 5
                    "total_affected": len(images_without_captions),
                }
            )

        # Find captions without images
        captions_without_images = caption_stems - image_stems
        if captions_without_images:
            issues.append(
                {
                    "type": "orphaned_captions",
                    "severity": "info",
                    "description": f"{len(captions_without_images)} captions have no corresponding images",
                    "affected_files": sorted(list(captions_without_images))[:5],
                    "total_affected": len(captions_without_images),
                }
            )

        # Check for very short captions
        short_captions = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in caption_extensions:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            caption = f.read().strip()
                            if len(caption.split()) < 3:
                                short_captions.append(file_path.name)
                    except Exception:
                        continue

        if short_captions:
            issues.append(
                {
                    "type": "short_captions",
                    "severity": "warning",
                    "description": f"{len(short_captions)} captions are very short (< 3 words)",
                    "affected_files": short_captions[:5],
                    "total_affected": len(short_captions),
                }
            )

    except Exception as e:
        issues.append(
            {
                "type": "validation_error",
                "severity": "error",
                "description": f"Validation failed: {str(e)}",
                "affected_files": [],
                "total_affected": 0,
            }
        )

    return issues


def get_image_with_caption(
    dataset_path: Path, index: int
) -> Tuple[Optional[str], Optional[str], int]:
    """
    Get image and caption at specified index.

    Args:
        dataset_path: Path to dataset directory
        index: Image index to retrieve

    Returns:
        Tuple of (image_path, caption, total_images)
    """
    try:
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        image_files = []

        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)

        image_files.sort()  # Sort for consistent ordering

        if not image_files or index < 0 or index >= len(image_files):
            return None, None, len(image_files)

        selected_image = image_files[index]

        # Copy image to temp directory that Gradio can access
        import tempfile
        import shutil

        temp_dir = Path(tempfile.gettempdir()) / "flux2_lora_browser"
        temp_dir.mkdir(exist_ok=True)

        # Create a unique filename to avoid conflicts
        temp_image_path = temp_dir / f"browser_{index}_{selected_image.name}"
        shutil.copy2(selected_image, temp_image_path)
        image_path = str(temp_image_path)

        # Try to find corresponding caption
        caption = None
        stem = selected_image.stem

        # Look for caption file
        for ext in [".txt", ".caption"]:
            caption_file = selected_image.parent / f"{stem}{ext}"
            if caption_file.exists():
                try:
                    with open(caption_file, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                    break
                except Exception:
                    continue

        return image_path, caption, len(image_files)

    except Exception as e:
        return None, f"Error loading image: {str(e)}", 0


def create_dataset_tab(app: "LoRATrainingApp"):
    """
    Create the dataset tools tab interface.

    Args:
        app: Main application instance
    """
    # Help section (collapsible)
    with gr.Accordion("💡 Dataset Tools Help & Tips", open=False):
        gr.Markdown(help_system.get_dataset_help_text())

        # Additional feature explanations
        with gr.Accordion("📊 Dataset Analysis Explained", open=False):
            gr.Markdown(help_system.get_feature_overview()["dataset_analysis"])
            gr.Markdown("""
            ### What Each Analysis Metric Means

            #### Image Statistics
            - **Resolution Distribution**: Shows if images are consistently sized
            - **File Sizes**: Identifies unusually large/small files (potential issues)
            - **Aspect Ratios**: Reveals if images are mostly portrait, landscape, or square
            - **Color Spaces**: Confirms all images use RGB (required for training)

            #### Caption Analysis
            - **Length Distribution**: Average words per caption (aim for 10-20)
            - **Vocabulary Size**: Unique words across all captions
            - **Word Frequency**: Most/least common terms in your dataset
            - **Caption-Image Ratio**: Ensures every image has a caption

            #### Quality Indicators
            - **Missing Captions**: Images without corresponding .txt files
            - **Orphaned Captions**: .txt files without matching images
            - **Corrupt Images**: Files that can't be opened or decoded
            - **Format Consistency**: All images should be JPG/PNG/WebP

            ### Interpreting Your Results

            #### Good Dataset Signs
            - ✅ Consistent resolutions (mostly 1024x1024+)
            - ✅ All images have captions
            - ✅ Captions are 10-25 words on average
            - ✅ Reasonable vocabulary size (50+ unique words)
            - ✅ No missing or corrupt files
            - ✅ Consistent file sizes and aspect ratios

            #### Common Issues & Fixes

            **Problem**: Mixed resolutions (512x512, 1024x1024, 2048x2048)
            **Impact**: Training instability, quality inconsistencies
            **Fix**: Resize all images to consistent dimensions (1024x1024 recommended)

            **Problem**: Short captions ("A cat", "Photo of dog")
            **Impact**: Poor training results, limited LoRA capabilities
            **Fix**: Rewrite captions to be detailed and descriptive

            **Problem**: Missing captions for some images
            **Impact**: Those images won't contribute to training
            **Fix**: Create .txt files for all images with descriptive captions

            **Problem**: Inconsistent naming (img1.jpg, photo_001.jpg, image-01.jpg)
            **Impact**: Minor, but affects organization
            **Fix**: Use consistent naming scheme (image_001.jpg, image_001.txt)

            **Problem**: Very large vocabulary (>500 unique words)
            **Impact**: LoRA may become too complex or unfocused
            **Fix**: Review and standardize terminology in captions

            **Problem**: Repetitive captions across images
            **Impact**: LoRA may overfit to specific phrases
            **Fix**: Vary descriptions while maintaining consistency
            """)

        with gr.Accordion("🖼️ Image Browser Guide", open=False):
            gr.Markdown("""
            ### Using the Image Browser Effectively

            #### Navigation Options
            - **Index Navigation**: Jump directly to any image by number
            - **Sequential**: Previous/Next buttons for systematic review
            - **Random**: Quick sampling to check dataset variety
            - **Thumbnail Grid**: Overview of all images at once

            #### What to Look For
            - **Quality Consistency**: Are all images similar in quality/style?
            - **Subject Clarity**: Is your main subject clear and prominent?
            - **Background Variety**: Do backgrounds support or distract?
            - **Lighting Consistency**: Is lighting similar across images?
            - **Composition**: Do images follow similar framing rules?

            #### Identifying Problems
            - **Outliers**: Images that don't match the overall style
            - **Poor Quality**: Blurry, pixelated, or heavily compressed images
            - **Wrong Subject**: Images that don't contain your target subject
            - **Inconsistent Style**: Photos vs. drawings, different art styles
            - **Technical Issues**: Corrupt files, wrong formats, huge size differences

            #### Caption Review Process
            1. **Read the caption**: Does it accurately describe the image?
            2. **Check detail level**: Is it detailed enough (10+ words)?
            3. **Verify consistency**: Does it use consistent terminology?
            4. **Look for trigger words**: Are your planned trigger words used?
            5. **Assess completeness**: Does it capture all important visual elements?

            #### Cleanup Actions
            - **Remove outliers**: Delete images that don't fit your concept
            - **Edit captions**: Improve short or inaccurate descriptions
            - **Standardize terms**: Use consistent vocabulary across captions
            - **Add variety**: Ensure diverse poses/angles if needed
            - **Fix technical issues**: Convert formats, resize images

            ### Browser Shortcuts & Tips
            - **Quick inspection**: Use random sampling for fast quality checks
            - **Systematic review**: Use sequential navigation for thorough inspection
            - **Problem targeting**: Use index navigation to check specific images
            - **Batch operations**: Note image numbers for later bulk operations
            - **Caption comparison**: Review similar images to check description consistency
            """)

        with gr.Accordion("✅ Validation Report Interpretation", open=False):
            gr.Markdown("""
            ### Understanding Validation Results

            #### Severity Levels
            - **Error (🔴)**: Critical issues that prevent training
            - **Warning (🟡)**: Problems that may affect training quality
            - **Info (🔵)**: Suggestions for optimization (not blocking)

            #### Common Validation Issues

            **Missing Caption Files**
            - **Severity**: Error
            - **Impact**: Images without captions won't be used in training
            - **Fix**: Create .txt files for all images with descriptive captions

            **Corrupt Image Files**
            - **Severity**: Error
            - **Impact**: Training will fail or produce errors
            - **Fix**: Remove or replace corrupted image files

            **Inconsistent Resolutions**
            - **Severity**: Warning
            - **Impact**: May cause training instability or quality issues
            - **Fix**: Resize images to consistent dimensions (1024x1024 recommended)

            **Very Short Captions**
            - **Severity**: Warning
            - **Impact**: Limited training effectiveness, poor LoRA quality
            - **Fix**: Expand captions to 10-20 words with detailed descriptions

            **Large Resolution Variance**
            - **Severity**: Info
            - **Impact**: Minor training inefficiency
            - **Fix**: Consider standardizing resolutions for consistency

            **Low Caption Diversity**
            - **Severity**: Info
            - **Impact**: May limit LoRA's capabilities and generalization
            - **Fix**: Vary descriptions while maintaining subject consistency

            #### Validation Score Interpretation
            - **90-100%**: Excellent dataset, ready for training
            - **75-89%**: Good dataset with minor issues to address
            - **60-74%**: Usable dataset but significant improvements needed
            - **Below 60%**: Major issues require attention before training

            ### Action Priority
            1. **Fix all Errors first** (training-blocking issues)
            2. **Address Warnings** (quality-impacting issues)
            3. **Consider Info items** (optimization opportunities)
            4. **Re-run validation** after fixes to confirm improvements

            ### Automated Fixes Available
            Some tools can automatically fix common issues:
            - Rename files for consistency
            - Remove corrupt files
            - Generate basic captions (though manual review recommended)
            - Convert image formats

            ### When to Re-validate
            - After making any changes to your dataset
            - Before starting training
            - After adding new images or captions
            - If you notice training issues that might stem from data problems
            """)

        with gr.Accordion("🎯 Dataset Optimization Strategies", open=False):
            gr.Markdown("""
            ### Creating Training-Ready Datasets

            #### For Character LoRAs
            - **Subject Focus**: 70%+ of image should show the character clearly
            - **Pose Variety**: Include front, side, back, action poses
            - **Expression Range**: Different emotions and expressions
            - **Angle Diversity**: Various camera angles and distances
            - **Context Variation**: Different backgrounds, lighting, outfits
            - **Consistency**: Maintain similar art style and character design

            #### For Style LoRAs
            - **Style Consistency**: All images should exemplify the target style
            - **Subject Variety**: Different subjects showing the same style
            - **Technique Focus**: Highlight specific artistic techniques
            - **Quality Examples**: Include high-quality examples of the style
            - **Style Elements**: Ensure consistent use of colors, brushes, composition

            #### For Concept LoRAs
            - **Clear Examples**: Object/concept should be prominent in images
            - **Context Variety**: Show the concept in different settings
            - **Scale Variation**: Different sizes and distances
            - **Composition Diversity**: Various arrangements and backgrounds
            - **Quality Range**: Include both simple and complex examples

            #### Caption Optimization
            - **Descriptive Language**: Use specific, descriptive terms
            - **Visual Details**: Mention colors, shapes, textures, lighting
            - **Composition Notes**: Describe positioning and relationships
            - **Style Descriptions**: Note artistic techniques and moods
            - **Trigger Word Integration**: Include planned trigger words naturally

            #### Technical Optimization
            - **Resolution**: 1024x1024 minimum, consistent across dataset
            - **Format**: JPG/PNG/WebP, RGB color space
            - **Compression**: High quality, avoid heavy compression artifacts
            - **Naming**: Consistent scheme (image_001.jpg, image_001.txt)
            - **File Size**: Reasonable sizes, avoid extremes

            ### Dataset Size Guidelines
            - **Minimum**: 10-15 images for basic results
            - **Good**: 30-50 images for reliable training
            - **Excellent**: 100+ images for complex subjects
            - **Quality over Quantity**: Better to have fewer high-quality images

            ### Common Dataset Mistakes
            - **Too Similar**: All images nearly identical (limits generalization)
            - **Poor Quality**: Blurry, low-res, or heavily compressed images
            - **Inconsistent Style**: Mixing different art styles or qualities
            - **Wrong Focus**: Background or secondary elements dominate
            - **Inadequate Captions**: Too short, generic, or inaccurate descriptions
            - **Technical Issues**: Wrong formats, corrupt files, inconsistent sizes

            ### Iterative Improvement
            1. **Initial Assessment**: Run analysis and identify main issues
            2. **Prioritize Fixes**: Address critical errors first
            3. **Test Training**: Train with a subset to check data quality
            4. **Refine Based on Results**: Improve weak areas identified in training
            5. **Expand Gradually**: Add more images as quality improves
            """)

    with gr.Row():
        with gr.Column(scale=1):
            # Left column: Dataset management
            gr.Markdown("## 🗂️ Dataset Management")

            with gr.Group():
                gr.Markdown("### Load Dataset")

                # Dataset source selection
                dataset_source = gr.Radio(
                    choices=["Upload ZIP", "Local Directory"],
                    value="Upload ZIP",
                    label="Dataset Source",
                )

                dataset_upload = gr.File(
                    label="Upload Dataset (ZIP)",
                    file_types=[".zip"],
                    visible=True,
                )

                dataset_path = gr.Textbox(
                    label="Dataset Directory Path",
                    placeholder="/path/to/dataset",
                    visible=False,
                )

                # Load dataset button
                load_dataset_btn = gr.Button(
                    "📂 Load Dataset",
                    variant="primary",
                )

                gr.Markdown("### Dataset Analysis")

                # Analysis options
                with gr.Row():
                    analyze_btn = gr.Button(
                        "📊 Analyze Dataset",
                        variant="secondary",
                    )
                    validate_btn = gr.Button(
                        "✅ Validate Dataset",
                        variant="secondary",
                    )

                # Analysis controls
                with gr.Accordion("🔧 Analysis Options", open=False):
                    check_duplicates = gr.Checkbox(value=True, label="Check for duplicate images")

                    check_corrupt = gr.Checkbox(value=True, label="Check for corrupt images")

                    analyze_captions = gr.Checkbox(value=True, label="Analyze caption quality")

                    sample_size = gr.Number(
                        value=1000, label="Sample size for analysis", minimum=100, maximum=10000
                    )

        with gr.Column(scale=2):
            # Right column: Results and visualization
            gr.Markdown("## 📈 Dataset Analysis")

            # Dataset overview
            with gr.Group():
                dataset_overview = gr.JSON(label="Dataset Overview", value={})

                # Quick stats
                with gr.Row():
                    total_images = gr.Number(label="Total Images", value=0, interactive=False)
                    total_captions = gr.Number(label="Total Captions", value=0, interactive=False)
                    avg_caption_length = gr.Number(
                        label="Avg Caption Length", value=0.0, interactive=False
                    )

            # Image browser
            gr.Markdown("### Image Browser")

            with gr.Group():
                # Image selection controls
                with gr.Row():
                    image_index = gr.Number(
                        label="Image Index (1-based)",
                        value=0,
                        minimum=0,
                        interactive=True,
                    )

                    prev_btn = gr.Button("⬅️ Previous")
                    next_btn = gr.Button("Next ➡️")
                    random_btn = gr.Button("🎲 Random")

                # Current image display
                current_image = gr.Image(label="Current Image", height=300)

                current_caption = gr.Textbox(label="Caption", lines=3, interactive=False)

                # Image gallery (thumbnails)
                image_gallery = gr.Gallery(
                    label="Dataset Thumbnails", columns=6, height=200, allow_preview=False
                )

            # Analysis results
            gr.Markdown("### Analysis Results")

            with gr.Tabs():
                with gr.TabItem("📊 Statistics"):
                    stats_results = gr.JSON(label="Detailed Statistics", value={})

                with gr.TabItem("⚠️ Issues"):
                    issues_results = gr.Dataframe(
                        label="Dataset Issues",
                        headers=["Type", "Severity", "Description", "Affected Files"],
                        value=[],
                    )

                with gr.TabItem("📝 Captions"):
                    caption_analysis = gr.JSON(label="Caption Analysis", value={})

                    caption_samples = gr.Dataframe(
                        label="Caption Samples",
                        headers=["Image", "Caption", "Length", "Quality Score"],
                        value=[],
                    )

                with gr.TabItem("📋 Validation Report"):
                    validation_report = gr.Markdown(value="No validation run yet.")

    # Event handlers
    def update_dataset_visibility(source):
        """Update dataset input visibility based on source selection."""
        if source == "Upload ZIP":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    dataset_source.change(
        fn=update_dataset_visibility,
        inputs=[dataset_source],
        outputs=[dataset_upload, dataset_path],
    )

    # Image navigation handlers
    def update_image_index(current_idx, direction, total_images):
        """Update image index based on navigation."""
        if total_images == 0:
            return current_idx

        if direction == "prev":
            new_idx = max(0, current_idx - 1)
        elif direction == "next":
            new_idx = min(total_images - 1, current_idx + 1)
        elif direction == "random":
            import random

            new_idx = random.randint(0, total_images - 1)
        else:
            new_idx = current_idx

        return new_idx

    # Note: Navigation buttons will be updated dynamically when dataset is loaded
    # The actual total_images is passed through the update_image_display function

    # State variables for UI updates
    loaded_dataset = gr.State(None)
    current_analysis = gr.State({})

    # Event handlers
    def update_dataset_visibility(source):
        """Update dataset input visibility based on source selection."""
        if source == "Upload ZIP":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    dataset_source.change(
        fn=update_dataset_visibility,
        inputs=[dataset_source],
        outputs=[dataset_upload, dataset_path],
    )

    # Dataset upload handler
    def handle_dataset_upload_wrapper(file_obj):
        """Handle dataset upload."""
        if file_obj:
            status, dataset_path = load_dataset_for_analysis(app, file_obj)
            if dataset_path:
                # Get basic info for display
                info = app.get_training_state("analysis_dataset_info", {})
                # Calculate basic caption statistics
                avg_caption_len = calculate_basic_caption_stats(dataset_path)
                total_imgs = info.get("image_count", 0)
                return (
                    info,  # dataset_overview
                    total_imgs,  # total_images
                    info.get("caption_count", 0),  # total_captions
                    avg_caption_len,  # avg_caption_length
                    dataset_path,  # loaded_dataset
                    gr.update(value=1, maximum=total_imgs if total_imgs > 0 else 1),  # image_index
                )
            else:
                return {}, 0, 0, 0, None, gr.update(value=0, maximum=1)
        return {}, 0, 0, 0, None, gr.update(value=0, maximum=1)

    dataset_upload.change(
        fn=handle_dataset_upload_wrapper,
        inputs=[dataset_upload],
        outputs=[
            dataset_overview,
            total_images,
            total_captions,
            avg_caption_length,
            loaded_dataset,
            image_index,
        ],
    )

    # Dataset path handler
    def handle_dataset_path_wrapper(path):
        """Handle dataset path input."""
        if path and path.strip():
            status, dataset_path = load_dataset_from_path(app, path.strip())
            if dataset_path:
                # Get basic info for display
                info = app.get_training_state("analysis_dataset_info", {})
                # Calculate basic caption statistics
                avg_caption_len = calculate_basic_caption_stats(dataset_path)
                total_imgs = info.get("image_count", 0)
                return (
                    info,  # dataset_overview
                    total_imgs,  # total_images
                    info.get("caption_count", 0),  # total_captions
                    avg_caption_len,  # avg_caption_length
                    dataset_path,  # loaded_dataset
                    gr.update(value=1, maximum=total_imgs if total_imgs > 0 else 1),  # image_index
                )
            else:
                return {}, 0, 0, 0, None, gr.update(value=0, maximum=1)
        return {}, 0, 0, 0, None, gr.update(value=0, maximum=1)

    dataset_path.change(
        fn=handle_dataset_path_wrapper,
        inputs=[dataset_path],
        outputs=[
            dataset_overview,
            total_images,
            total_captions,
            avg_caption_length,
            loaded_dataset,
            image_index,
        ],
    )

    # Load dataset button handler
    def load_dataset_handler(source, upload, path):
        """Handle manual dataset loading."""
        if source == "Upload ZIP" and upload:
            return handle_dataset_upload_wrapper(upload)
        elif source == "Local Directory" and path and path.strip():
            return handle_dataset_path_wrapper(path.strip())
        else:
            return {}, 0, 0, 0, None, gr.update(value=0, maximum=1)

    load_dataset_btn.click(
        fn=load_dataset_handler,
        inputs=[dataset_source, dataset_upload, dataset_path],
        outputs=[
            dataset_overview,
            total_images,
            total_captions,
            avg_caption_length,
            loaded_dataset,
            image_index,
        ],
    )

    # Analyze dataset handler
    def analyze_dataset_handler(
        loaded_dataset, check_duplicates, check_corrupt, analyze_captions, sample_size
    ):
        """Handle comprehensive dataset analysis."""
        if not loaded_dataset:
            return {"error": "No dataset loaded"}, [], "No dataset loaded", 0, []

        try:
            dataset_path = Path(loaded_dataset)
            # For now, always do comprehensive analysis
            # TODO: Make analysis conditional based on checkboxes
            analysis = analyze_dataset_comprehensive(dataset_path)

            if "error" in analysis:
                return analysis, [], f"Analysis error: {analysis['error']}", 0, []

            # Format results for display
            basic_stats = analysis.get("basic_stats", {})
            image_analysis = analysis.get("image_analysis", {})
            caption_analysis = analysis.get("caption_analysis", {})

            # Update basic stats display
            app.update_training_state("dataset_analysis", analysis)

            # Get the average caption length for UI update
            avg_caption_length = caption_analysis.get("avg_words_per_caption", 0)

            # Format stats for display
            caption_stats = basic_stats.get("caption_stats", {})
            stats_display = {
                "Basic Statistics": {
                    "Total Images": basic_stats.get("total_images_found", 0),
                    "Total Captions": caption_stats.get("total_captions", 0),
                    "Caption Coverage": f"{len(caption_stats) > 0 and basic_stats.get('valid_pairs', 0) / basic_stats.get('total_images_found', 1) * 100:.1f}%",
                },
                "Image Analysis": {
                    "Average Resolution": f"{image_analysis.get('avg_width', 0)}x{image_analysis.get('avg_height', 0)}",
                    "Resolution Distribution": image_analysis.get("resolution_distribution", {}),
                    "Average File Size": f"{image_analysis.get('avg_file_size_kb', 0):.1f} KB",
                    "Color Spaces": image_analysis.get("color_spaces", {}),
                },
                "Caption Analysis": {
                    "Average Words per Caption": caption_analysis.get("avg_words_per_caption", 0),
                    "Vocabulary Size": caption_analysis.get("vocabulary_size", 0),
                    "Caption Length Distribution": caption_analysis.get("length_distribution", {}),
                    "Top Words": [
                        word for word, count in caption_analysis.get("top_words", [])[:5]
                    ],
                },
            }

            # Format issues for table display
            issues = analysis.get("validation_issues", [])
            issues_table = []
            for issue in issues:
                issues_table.append(
                    [
                        issue.get("type", "unknown"),
                        issue.get("severity", "info"),
                        issue.get("description", ""),
                        ", ".join(issue.get("affected_files", [])[:3]),  # Show first 3 files
                    ]
                )

            # Format caption samples for display
            caption_samples_data = []
            try:
                # Try to get sample captions using direct file reading instead of LoRADataset
                from .dataset_tab import analyze_captions

                caption_analysis = analyze_captions(dataset_path)
                if caption_analysis.get("total_captions", 0) > 0:
                    # Get sample captions by reading files directly
                    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
                    caption_extensions = {".txt", ".caption"}

                    sample_files = []
                    for root, dirs, files in os.walk(dataset_path):
                        for file in files:
                            file_path = Path(root) / file
                            if file_path.suffix.lower() in image_extensions:
                                sample_files.append(file_path)
                                if len(sample_files) >= 10:  # Only need first 10
                                    break
                        if len(sample_files) >= 10:
                            break

                    for image_path in sample_files[:10]:
                        try:
                            # Find corresponding caption
                            caption = None
                            stem = image_path.stem
                            for ext in [".txt", ".caption"]:
                                caption_file = image_path.parent / f"{stem}{ext}"
                                if caption_file.exists():
                                    with open(caption_file, "r", encoding="utf-8") as f:
                                        caption = f.read().strip()
                                    break

                            if caption:
                                image_name = image_path.name
                                length = len(caption.split())
                                quality_score = (
                                    "Good"
                                    if 10 <= length <= 25
                                    else "Short"
                                    if length < 10
                                    else "Long"
                                )
                                caption_samples_data.append(
                                    [image_name, caption, length, quality_score]
                                )
                        except Exception:
                            continue
            except Exception:
                caption_samples_data = [["Error", "Could not load caption samples", 0, "Error"]]

            return (
                stats_display,
                issues_table,
                "Analysis completed successfully",
                avg_caption_length,
                caption_samples_data,
            )

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}, [], f"Analysis failed: {str(e)}", 0, []

    analyze_btn.click(
        fn=analyze_dataset_handler,
        inputs=[loaded_dataset, check_duplicates, check_corrupt, analyze_captions, sample_size],
        outputs=[
            stats_results,
            issues_results,
            validation_report,
            avg_caption_length,
            caption_samples,
        ],
    )

    # Validate dataset handler
    def validate_dataset_handler(loaded_dataset):
        """Handle dataset validation."""
        if not loaded_dataset:
            return {"error": "No dataset loaded"}

        try:
            dataset_path = Path(loaded_dataset)
            issues = validate_dataset_quality(dataset_path)

            # Create validation report
            report = {
                "validation_timestamp": "2025-12-17T12:00:00Z",  # Current timestamp
                "dataset_path": str(dataset_path),
                "total_issues": len(issues),
                "issues_by_severity": {},
                "recommendations": [],
            }

            # Count issues by severity
            severity_counts = {}
            for issue in issues:
                severity = issue.get("severity", "info")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            report["issues_by_severity"] = severity_counts

            # Generate recommendations
            recommendations = []
            if severity_counts.get("error", 0) > 0:
                recommendations.append("Fix critical errors before training")
            if severity_counts.get("warning", 0) > 0:
                recommendations.append("Address warnings to improve training quality")
            if severity_counts.get("info", 0) > 0:
                recommendations.append("Consider optional improvements for better results")

            if not recommendations:
                recommendations.append("Dataset validation passed - ready for training!")

            report["recommendations"] = recommendations

            # Add detailed issue breakdown
            report["detailed_issues"] = issues[:10]  # Show first 10 issues

            return report

        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}

    validate_btn.click(
        fn=validate_dataset_handler, inputs=[loaded_dataset], outputs=[validation_report]
    )

    # Gallery population handler
    def populate_image_gallery(loaded_dataset):
        """Populate the image gallery with thumbnails."""
        if not loaded_dataset:
            return []

        try:
            dataset_path = Path(loaded_dataset)
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
            image_files = []

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in image_extensions:
                        image_files.append(file_path)

            image_files.sort()

            # Limit to first 20 images for gallery
            gallery_images = []
            import tempfile
            import shutil

            temp_dir = Path(tempfile.gettempdir()) / "flux2_lora_gallery"
            temp_dir.mkdir(exist_ok=True)

            for i, img_path in enumerate(image_files[:20]):
                try:
                    temp_path = temp_dir / f"gallery_{i}_{img_path.name}"
                    shutil.copy2(img_path, temp_path)
                    gallery_images.append(str(temp_path))
                except Exception:
                    continue

            return gallery_images

        except Exception:
            return []

    # Update gallery when dataset is loaded
    def update_gallery_on_load(loaded_dataset, *args):
        """Update gallery when dataset is loaded."""
        gallery_images = populate_image_gallery(loaded_dataset)
        return gallery_images

    # Connect gallery update to dataset loading
    loaded_dataset.change(
        fn=update_gallery_on_load,
        inputs=[loaded_dataset],
        outputs=[image_gallery],
    )

    # Image browsing handlers
    def update_image_display(loaded_dataset, index):
        """Update image display based on index."""
        if not loaded_dataset:
            return None, "No dataset loaded", 0

        try:
            dataset_path = Path(loaded_dataset)
            # Convert from 1-based UI index to 0-based array index
            array_index = int(index) - 1
            image_path, caption, total_images = get_image_with_caption(dataset_path, array_index)

            if image_path:
                return image_path, caption or "No caption available", total_images
            else:
                return None, f"No image at index {index}", total_images

        except Exception as e:
            return None, f"Error loading image: {str(e)}", 0

    # Update image when index changes
    def update_image_display_safe(loaded_dataset, index):
        """Safe wrapper for image display update."""
        if not loaded_dataset:
            return None, "No dataset loaded"

        try:
            dataset_path = Path(loaded_dataset)
            # Convert from 1-based UI index to 0-based array index
            array_index = int(index) - 1
            image_path, caption, total_images = get_image_with_caption(dataset_path, array_index)

            if image_path:
                return image_path, caption or "No caption available"
            else:
                return None, f"No image at index {index}"

        except Exception as e:
            return None, f"Error loading image: {str(e)}"

    image_index.change(
        fn=update_image_display_safe,
        inputs=[loaded_dataset, image_index],
        outputs=[current_image, current_caption],
    )

    # Navigation button handlers
    def navigate_image(direction, current_idx, loaded_dataset):
        """Navigate to previous/next/random image."""
        if not loaded_dataset:
            return current_idx

        try:
            dataset_path = Path(loaded_dataset)
            _, _, total_images = get_image_with_caption(dataset_path, 0)  # Just to get total

            if total_images == 0:
                return current_idx

            # Convert from 1-based UI index to 0-based array index for calculation
            array_idx = current_idx - 1

            if direction == "prev":
                new_array_idx = max(0, array_idx - 1)
            elif direction == "next":
                new_array_idx = min(total_images - 1, array_idx + 1)
            elif direction == "random":
                import random

                new_array_idx = random.randint(0, total_images - 1)
            else:
                new_array_idx = array_idx

            # Convert back to 1-based UI index
            return new_array_idx + 1

        except Exception:
            return current_idx

    prev_btn.click(
        fn=lambda idx, ds: navigate_image("prev", idx, ds),
        inputs=[image_index, loaded_dataset],
        outputs=[image_index],
    )

    next_btn.click(
        fn=lambda idx, ds: navigate_image("next", idx, ds),
        inputs=[image_index, loaded_dataset],
        outputs=[image_index],
    )

    random_btn.click(
        fn=lambda idx, ds: navigate_image("random", idx, ds),
        inputs=[image_index, loaded_dataset],
        outputs=[image_index],
    )
