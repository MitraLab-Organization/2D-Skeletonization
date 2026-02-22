"""
Run DM2D Pipeline on LKL Tiles

Takes pre-cropped LKL tiles and runs DM2D + Vectorization directly on them.
No cropping involved - tiles are processed as-is.

Usage:
    python run_dm2d_tiles.py --lkl_dir /path/to/lkl_tiles --output_dir /path/to/output
"""

import os
import sys
import numpy as np

# Allow large images
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import cv2
from pathlib import Path
from PIL import Image
import shutil
import argparse
import json
from skimage.morphology import remove_small_objects

Image.MAX_IMAGE_PIXELS = None

# Setup paths for DM2D
# This script is already inside Skeletonization_Suite
script_dir = os.path.dirname(os.path.abspath(__file__))
skeletonization_path = script_dir  # We're already in Skeletonization_Suite
dm2d_code_path = os.path.join(skeletonization_path, "DM_2D_code")

os.chdir(skeletonization_path)
sys.path.insert(0, skeletonization_path)
sys.path.insert(0, dm2d_code_path)

from DM2D_Pipeline_Tiled import DM2D_Pipeline


def run_dm2d_on_tile(lkl_path, output_dir, tile_name, ve_persistence=0, et_persistence=0):
    """Run DM2D on a single LKL tile"""
    
    # Read LKL tile
    lkl = cv2.imread(str(lkl_path), cv2.IMREAD_UNCHANGED)
    if lkl is None:
        print(f"  ERROR: Cannot read {lkl_path}")
        return None
    
    if len(lkl.shape) == 3:
        lkl = cv2.cvtColor(lkl, cv2.COLOR_BGR2GRAY)
    
    print(f"  LKL shape: {lkl.shape}")
    
    # Threshold LKL at 40 (matching original pipeline)
    lkl[lkl < 40] = 0
    
    # Binary threshold
    _, lkl_bin = cv2.threshold(lkl, 20, 255, cv2.THRESH_BINARY)
    
    # Setup directories
    json_dir = output_dir
    json_temp_dir = os.path.join(output_dir, "tmp")
    scratch_dir = os.path.join(output_dir, "scratch_1")
    
    os.makedirs(json_temp_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    
    # DM2D parameters - smaller divisions for tiles (1000x1000)
    division_x = 4
    division_y = 4
    ve_persistence_threshold = ve_persistence
    et_persistence_threshold = et_persistence
    
    # Run DM2D
    DM2D_Pipeline(lkl, lkl_bin, division_x, division_y,
                  ve_persistence_threshold, et_persistence_threshold,
                  json_dir, json_temp_dir, scratch_dir)
    
    # Rename output
    merged_json = os.path.join(json_dir, "merged_geojson.json")
    final_json = os.path.join(json_dir, f"{tile_name}.json")
    
    if os.path.exists(merged_json):
        shutil.move(merged_json, final_json)
        return final_json
    else:
        print(f"  WARNING: merged_geojson.json not found")
        return None


def json_to_skeleton(json_path, image_shape):
    """Convert JSON to skeleton image"""
    with open(json_path) as f:
        gj = json.load(f)
    
    skeleton = np.zeros(image_shape, dtype=np.uint8)
    
    features = gj.get('features', [])
    if not features:
        return skeleton
    
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
    
    return skeleton


def main():
    parser = argparse.ArgumentParser(description='Run DM2D on LKL tiles')
    parser.add_argument('--lkl_dir', type=str, required=True,
                        help='Directory containing LKL tiles')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--ve_persistence', type=int, default=0,
                        help='VE persistence threshold (default: 0)')
    parser.add_argument('--et_persistence', type=int, default=0,
                        help='ET persistence threshold (default: 0)')
    parser.add_argument('--min_size', type=int, default=0,
                        help='Minimum connected component size (default: 0, no filtering)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    json_dir = os.path.join(args.output_dir, 'json')
    skel_dir = os.path.join(args.output_dir, 'skeleton')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(skel_dir, exist_ok=True)
    
    # Get LKL tile files
    lkl_files = sorted(Path(args.lkl_dir).glob('*.tif'))
    if not lkl_files:
        lkl_files = sorted(Path(args.lkl_dir).glob('*.jpg'))
    
    print(f"Processing {len(lkl_files)} LKL tiles...")
    print(f"  VE persistence threshold: {args.ve_persistence}")
    print(f"  ET persistence threshold: {args.et_persistence}")
    
    for idx, lkl_file in enumerate(lkl_files):
        tile_name = lkl_file.stem
        print(f"\n[{idx+1}/{len(lkl_files)}] {tile_name}")
        
        # Run DM2D
        json_path = run_dm2d_on_tile(str(lkl_file), json_dir, tile_name, 
                                      ve_persistence=args.ve_persistence,
                                      et_persistence=args.et_persistence)
        
        if json_path and os.path.exists(json_path):
            # Read tile dimensions
            tile = cv2.imread(str(lkl_file), cv2.IMREAD_GRAYSCALE)
            tile_shape = tile.shape
            
            # Convert to skeleton
            skel = json_to_skeleton(json_path, tile_shape)
            
            # Apply min_size filtering if specified
            if args.min_size > 0:
                skel_binary = skel > 0
                skel_filtered = remove_small_objects(skel_binary, min_size=args.min_size, connectivity=2)
                skel = (skel_filtered * 255).astype(np.uint8)
            
            skel_path = os.path.join(skel_dir, f"{tile_name}.tif")
            cv2.imwrite(skel_path, skel)
            print(f"  Skeleton: {np.sum(skel > 0)} pixels")
        else:
            print(f"  No JSON output")
    
    # Cleanup temp directories
    for d in ['tmp', 'scratch_1']:
        path = os.path.join(json_dir, d)
        if os.path.exists(path):
            shutil.rmtree(path)
    
    print(f"\nDM2D tiles complete!")
    print(f"  JSON: {json_dir}")
    print(f"  Skeleton: {skel_dir}")


if __name__ == '__main__':
    main()
