"""
Caption loading and processing utilities for LoRA training.

Supports multiple caption formats:
- Separate .txt files with same basename as image
- .caption files 
- JSON metadata files
- Embedded metadata in image files
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


class CaptionLoadError(Exception):
    """Raised when caption loading fails."""
    pass


class CaptionUtils:
    """Utilities for loading and processing captions from various sources."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    SUPPORTED_CAPTION_EXTENSIONS = {'.txt', '.caption', '.json'}
    
    @staticmethod
    def find_image_files(data_dir: Union[str, Path]) -> List[Path]:
        """
        Find all image files in the given directory.
        
        Args:
            data_dir: Directory to search for images
            
        Returns:
            List of image file paths
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise CaptionLoadError(f"Data directory does not exist: {data_dir}")
        
        image_files = []
        for ext in CaptionUtils.SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(data_dir.glob(f"*{ext}"))
            image_files.extend(data_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    @staticmethod
    def load_caption_from_txt(image_path: Path) -> Optional[str]:
        """
        Load caption from a .txt file with the same basename as the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Caption text or None if not found
        """
        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                return caption if caption else None
            except Exception as e:
                logger.warning(f"Failed to read caption file {txt_path}: {e}")
        return None
    
    @staticmethod
    def load_caption_from_caption_file(image_path: Path) -> Optional[str]:
        """
        Load caption from a .caption file with the same basename as the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Caption text or None if not found
        """
        caption_path = image_path.with_suffix('.caption')
        if caption_path.exists():
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                return caption if caption else None
            except Exception as e:
                logger.warning(f"Failed to read caption file {caption_path}: {e}")
        return None
    
    @staticmethod
    def load_caption_from_json(image_path: Path, json_file: Optional[Path] = None) -> Optional[str]:
        """
        Load caption from a JSON metadata file.
        
        Args:
            image_path: Path to the image file
            json_file: Optional specific JSON file to use
            
        Returns:
            Caption text or None if not found
        """
        # Try specific JSON file if provided
        if json_file and json_file.exists():
            json_paths = [json_file]
        else:
            # Try JSON files with same basename or directory-level metadata
            json_paths = [
                image_path.with_suffix('.json'),
                image_path.parent / 'metadata.json',
                image_path.parent / 'captions.json',
                image_path.parent / '_metadata.json'
            ]
        
        for json_path in json_paths:
            if not json_path.exists():
                continue
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                caption = None
                
                # Single image metadata
                if isinstance(data, dict):
                    # Try common caption field names
                    for field in ['caption', 'text', 'prompt', 'description', 'title']:
                        if field in data and data[field]:
                            caption = str(data[field]).strip()
                            break
                    
                    # Try filename-based lookup
                    if not caption and image_path.name in data:
                        caption = str(data[image_path.name]).strip()
                
                # List of metadata entries
                elif isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict) and 'filename' in entry:
                            if entry['filename'] == image_path.name:
                                for field in ['caption', 'text', 'prompt', 'description']:
                                    if field in entry and entry[field]:
                                        caption = str(entry[field]).strip()
                                        break
                
                if caption:
                    return caption
                    
            except Exception as e:
                logger.warning(f"Failed to parse JSON file {json_path}: {e}")
        
        return None
    
    @staticmethod
    def load_caption_from_exif(image_path: Path) -> Optional[str]:
        """
        Load caption from image EXIF metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Caption text or None if not found
        """
        try:
            with Image.open(image_path) as img:
                exif_data = getattr(img, '_getexif', None)
                if exif_data and callable(exif_data):
                    exif_data = exif_data()
                    if exif_data:
                        for tag_id, value in exif_data.items():
                            tag = TAGS.get(tag_id, tag_id)
                            if tag in ['ImageDescription', 'XPComment', 'XPSubject']:
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode('utf-16le').strip('\x00')
                                    except:
                                        continue
                                caption = str(value).strip()
                                if caption and caption.lower() != 'untitled':
                                    return caption
        except Exception as e:
            logger.debug(f"Failed to read EXIF data from {image_path}: {e}")
        
        return None
    
    @staticmethod
    def load_caption_for_image(
        image_path: Path,
        preferred_sources: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Load caption for an image from various sources.
        
        Args:
            image_path: Path to the image file
            preferred_sources: Ordered list of preferred sources
            
        Returns:
            Caption text or None if not found
        """
        if preferred_sources is None:
            preferred_sources = ['txt', 'caption', 'json', 'exif']
        
        # Define source methods
        source_methods = {
            'txt': CaptionUtils.load_caption_from_txt,
            'caption': CaptionUtils.load_caption_from_caption_file,
            'json': CaptionUtils.load_caption_from_json,
            'exif': CaptionUtils.load_caption_from_exif
        }
        
        # Try each source in preferred order
        for source in preferred_sources:
            if source in source_methods:
                caption = source_methods[source](image_path)
                if caption:
                    return caption
        
        return None
    
    @staticmethod
    def load_dataset_captions(
        data_dir: Union[str, Path],
        preferred_sources: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Load captions for all images in a dataset.

        Args:
            data_dir: Directory containing images and captions
            preferred_sources: Ordered list of preferred caption sources

        Returns:
            Dictionary mapping image filenames to captions
        """
        data_dir = Path(data_dir)
        image_files = CaptionUtils.find_image_files(data_dir)

        print(f"DEBUG: Found {len(image_files)} image files in {data_dir}")
        print(f"DEBUG: preferred_sources = {preferred_sources}")
        if image_files:
            print(f"DEBUG: First 5 images: {[f.name for f in image_files[:5]]}")

        captions = {}
        missing_captions = []

        for image_path in image_files:
            caption = CaptionUtils.load_caption_for_image(image_path, preferred_sources)
            if caption:
                captions[image_path.name] = caption
            else:
                missing_captions.append(image_path.name)
                # Debug: check if txt file exists
                txt_path = image_path.with_suffix('.txt')
                if txt_path.exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        print(f"DEBUG: {image_path.name} -> txt exists but caption is None. Content: '{content[:50]}...' (len={len(content)})")
                    except Exception as e:
                        print(f"DEBUG: {image_path.name} -> txt exists but read failed: {e}")
                else:
                    print(f"DEBUG: {image_path.name} -> no txt file at {txt_path}")

        if missing_captions:
            logger.warning(f"Found {len(missing_captions)} images without captions: {missing_captions[:10]}...")
            print(f"DEBUG: Missing captions for: {missing_captions[:10]}")

        logger.info(f"Loaded captions for {len(captions)}/{len(image_files)} images")
        print(f"DEBUG: Loaded {len(captions)} captions out of {len(image_files)} images")

        return captions
    
    @staticmethod
    def validate_caption(caption: str, min_length: int = 3, max_length: int = 1000) -> bool:
        """
        Validate caption quality.

        Args:
            caption: Caption text to validate
            min_length: Minimum character length
            max_length: Maximum character length

        Returns:
            True if caption is valid
        """
        if not caption or not isinstance(caption, str):
            print(f"DEBUG validate_caption: FAIL - caption is None or not string")
            return False

        caption = caption.strip()

        # Length checks
        if len(caption) < min_length or len(caption) > max_length:
            print(f"DEBUG validate_caption: FAIL - length {len(caption)} not in [{min_length}, {max_length}]")
            return False

        # Skip common placeholder text
        placeholders = [
            'no caption', 'no description', 'untitled', 'placeholder',
            'test image', 'sample image', 'example', 'n/a', 'none'
        ]

        if caption.lower() in placeholders:
            print(f"DEBUG validate_caption: FAIL - is placeholder text")
            return False

        # Skip very repetitive text
        unique_chars = len(set(caption.lower()))
        threshold = len(caption) * 0.3
        if unique_chars < threshold:
            print(f"DEBUG validate_caption: FAIL - too repetitive: {unique_chars} unique chars < {threshold:.1f} threshold")
            return False

        return True
    
    @staticmethod
    def clean_caption(caption: str) -> str:
        """
        Clean and normalize caption text.
        
        Args:
            caption: Raw caption text
            
        Returns:
            Cleaned caption text
        """
        if not caption:
            return ""
        
        caption = str(caption).strip()
        
        # Remove extra whitespace
        caption = ' '.join(caption.split())
        
        # Remove common prefixes/suffixes (case-insensitive)
        prefixes_to_remove = ['photo of ', 'picture of ', 'image of ', 'a photo of ']
        caption_lower = caption.lower()
        for prefix in prefixes_to_remove:
            if caption_lower.startswith(prefix):
                caption = caption[len(prefix):]
                caption = caption.strip()
                break
        
        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]
        
        return caption
    
    @staticmethod
    def analyze_caption_statistics(captions: Dict[str, str]) -> Dict[str, any]:
        """
        Analyze caption statistics for a dataset.
        
        Args:
            captions: Dictionary of image filenames to captions
            
        Returns:
            Statistics dictionary
        """
        if not captions:
            return {
                'total_captions': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'empty_captions': 0,
                'short_captions': 0,
                'long_captions': 0
            }
        
        lengths = [len(cap) for cap in captions.values()]
        empty_count = sum(1 for cap in captions.values() if not cap.strip())
        short_count = sum(1 for cap in captions.values() if len(cap.strip()) < 10)
        long_count = sum(1 for cap in captions.values() if len(cap.strip()) > 200)
        
        return {
            'total_captions': len(captions),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'empty_captions': empty_count,
            'short_captions': short_count,
            'long_captions': long_count,
            'length_distribution': {
                '0-10': sum(1 for l in lengths if l <= 10),
                '11-50': sum(1 for l in lengths if 11 <= l <= 50),
                '51-100': sum(1 for l in lengths if 51 <= l <= 100),
                '101-200': sum(1 for l in lengths if 101 <= l <= 200),
                '200+': sum(1 for l in lengths if l > 200)
            }
        }