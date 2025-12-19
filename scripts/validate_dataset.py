#!/usr/bin/env python3
"""
Dataset validation script for Flux2-dev LoRA training.

This script validates datasets and provides detailed statistics and reports.
"""

import argparse
import json
import sys
from pathlib import Path

from flux2_lora.data import validate_dataset


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate LoRA training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate dataset with default settings
  python validate_dataset.py /path/to/dataset
  
  # Validate with custom resolution and output JSON report
  python validate_dataset.py /path/to/dataset --resolution 1024 --json-report report.json
  
  # Validate with verbose output and custom caption length limits
  python validate_dataset.py /path/to/dataset --verbose --min-caption 5 --max-caption 500
        """
    )
    
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to dataset directory containing images and captions"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target resolution for training (default: 1024)"
    )
    
    parser.add_argument(
        "--min-caption-length",
        type=int,
        default=3,
        help="Minimum caption length in characters (default: 3)"
    )
    
    parser.add_argument(
        "--max-caption-length",
        type=int,
        default=1000,
        help="Maximum caption length in characters (default: 1000)"
    )
    
    parser.add_argument(
        "--json-report",
        type=str,
        help="Path to save JSON validation report"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    result = validate_dataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        min_caption_length=args.min_caption_length,
        max_caption_length=args.max_caption_length,
        verbose=not args.quiet
    )
    
    # Save JSON report if requested
    if args.json_report:
        report_path = Path(args.json_report)
        
        # Prepare report data
        report_data = {
            'validation_config': {
                'data_dir': str(args.data_dir),
                'resolution': args.resolution,
                'min_caption_length': args.min_caption_length,
                'max_caption_length': args.max_caption_length
            },
            'statistics': result['statistics'],
            'sample_images': result['samples']
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        if not args.quiet:
            print(f"\n📄 JSON report saved to: {report_path}")
    
    # Exit with appropriate code
    if result['dataset'] is None:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()