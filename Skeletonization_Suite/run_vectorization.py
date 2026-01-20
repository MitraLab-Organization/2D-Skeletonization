"""
Run Vectorization on DM2D JSON Outputs

Takes DM2D JSON files and runs vectorization to clean up the skeleton.

Usage:
    python run_vectorization.py --input_dir /path/to/dm2d_json --output_dir /path/to/output
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
import json


def json_to_mask(json_path, output_path, lkl_path):
    """Convert JSON to mask image for vectorization input"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    
    # Get dimensions from LKL
    lkl = Image.open(lkl_path)
    image_shape = (lkl.height, lkl.width)
    lkl.close()
    
    # Read JSON
    with open(json_path) as f:
        gj = json.load(f)
    
    # Render skeleton
    skeleton = np.zeros(image_shape, dtype=np.uint8)
    
    features = gj.get('features', [])
    for feat in features:
        geom = feat.get('geometry', {})
        geom_type = geom.get('type')
        coords = geom.get('coordinates', [])
        
        if geom_type == 'LineString' and len(coords) >= 2:
            x1, y1 = int(coords[0][0]), int(-coords[0][1])
            x2, y2 = int(coords[1][0]), int(-coords[1][1])
            cv2.line(skeleton, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
        elif geom_type == 'MultiLineString':
            for segment in coords:
                if len(segment) >= 2:
                    x1, y1 = int(segment[0][0]), int(-segment[0][1])
                    x2, y2 = int(segment[1][0]), int(-segment[1][1])
                    cv2.line(skeleton, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(output_path, skeleton)
    return np.sum(skeleton > 0)


def main():
    parser = argparse.ArgumentParser(description='Run Vectorization on DM2D outputs')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing DM2D JSON files')
    parser.add_argument('--lkl_dir', type=str, required=True,
                        help='Directory containing LKL images (for dimensions)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for vectorized JSON files')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temp mask directory
    mask_dir = os.path.join(args.output_dir, 'mask_temp')
    os.makedirs(mask_dir, exist_ok=True)
    
    # Get JSON files
    json_files = sorted(Path(args.input_dir).glob('*.json'))
    
    print(f"Converting {len(json_files)} JSON files to masks...")
    
    for jf in json_files:
        image_name = jf.stem
        
        # Try multiple extensions
        lkl_path = None
        for ext in ['.tif', '.tiff', '.jpg', '.png']:
            candidate = os.path.join(args.lkl_dir, f"{image_name}{ext}")
            if os.path.exists(candidate):
                lkl_path = candidate
                break
        
        if lkl_path is None:
            print(f"  WARNING: LKL not found for {image_name}, skipping")
            continue
        
        mask_path = os.path.join(mask_dir, f"{image_name}.jpg")
        pixels = json_to_mask(str(jf), mask_path, lkl_path)
        print(f"  {image_name}: {pixels} pixels")
    
    # Run vectorization
    print("\nRunning Vectorization...")
    
    cc_dir = os.path.join(args.output_dir, 'CC')
    json_dir = os.path.join(args.output_dir, 'JSON')
    os.makedirs(cc_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vectorization_script = os.path.join(script_dir, "Vectorization", "script_to_run.py")
    
    cmd = [
        sys.executable, vectorization_script,
        "--input_dir", mask_dir,
        "--output_cc_dir", cc_dir,
        "--output_json_dir", json_dir
    ]
    
    subprocess.check_call(cmd)
    
    # Cleanup
    shutil.rmtree(mask_dir)
    
    print(f"\nVectorization complete! Output: {json_dir}")


if __name__ == '__main__':
    main()
