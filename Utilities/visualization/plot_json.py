#!/usr/bin/env python3
"""
Visualize DM2D JSON Output
--------------------------
Reads a vector JSON file (GeoJSON format) and plots the lines on a black background.
Generates:
1. Full resolution image (TIFF) - Ideal for overlay/analysis
2. Compressed preview (JPG) - Ideal for quick visualization

Usage:
    python plot_json.py --json <path/to/file.json> --output_dir <path/to/output> [--width W] [--height H]
"""

import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path
import sys

# Increase CSV/JSON field limit just in case
import csv
csv.field_size_limit(sys.maxsize)

def plot_json(json_path, output_dir, width=None, height=None):
    """
    Plot JSON vectors on a black background.
    
    Args:
        json_path (str): Path to the JSON file
        output_dir (str): Directory to save outputs
        width (int, optional): Image width. If None, inferred from data.
        height (int, optional): Image height. If None, inferred from data.
    """
    print(f"Reading {json_path}...")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    features = data.get('features', [])
    if not features:
        print("No features found in JSON.")
        return

    # If dimensions not provided, infer from data (bounding box)
    if width is None or height is None:
        print("Inferred dimensions from data...")
        max_x, max_y = 0, 0
        for feat in features:
            geom = feat.get('geometry', {})
            coords = geom.get('coordinates', [])
            geom_type = geom.get('type')
            
            if geom_type == 'LineString':
                for pt in coords:
                    max_x = max(max_x, pt[0])
                    max_y = max(max_y, abs(pt[1])) # Y is often negative in these JSONs
            elif geom_type == 'MultiLineString':
                for seg in coords:
                    for pt in seg:
                        max_x = max(max_x, pt[0])
                        max_y = max(max_y, abs(pt[1]))
        
        # Add some padding
        width = int(max_x + 50)
        height = int(max_y + 50)
        print(f"Inferred dimensions: {width}x{height}")
    
    # Initialize black canvas
    print(f"Creating canvas {width}x{height}...")
    canvas = np.zeros((height, width), dtype=np.uint8)
    
    # Draw lines
    print("Drawing vectors...")
    count = 0
    for feat in features:
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        geom_type = geom.get('type')
        
        if geom_type == 'LineString' and len(coords) >= 2:
            x1, y1 = int(coords[0][0]), int(abs(coords[0][1]))
            x2, y2 = int(coords[1][0]), int(abs(coords[1][1]))
            # Clip coordinates to canvas
            x1, y1 = max(0, min(width-1, x1)), max(0, min(height-1, y1))
            x2, y2 = max(0, min(width-1, x2)), max(0, min(height-1, y2))
            cv2.line(canvas, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
            count += 1
            
        elif geom_type == 'MultiLineString':
            for seg in coords:
                if len(seg) >= 2:
                    x1, y1 = int(seg[0][0]), int(abs(seg[0][1]))
                    x2, y2 = int(seg[1][0]), int(abs(seg[1][1]))
                    # Clip coordinates to canvas
                    x1, y1 = max(0, min(width-1, x1)), max(0, min(height-1, y1))
                    x2, y2 = max(0, min(width-1, x2)), max(0, min(height-1, y2))
                    cv2.line(canvas, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
                    count += 1
    
    print(f"Drawn {count} segments.")
    
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(json_path).stem
    if basename == 'merged_geojson':
        basename = 'vectorization'

    # Save Full Resolution (TIFF)
    full_res_path = os.path.join(output_dir, f"{basename}_full.tif")
    cv2.imwrite(full_res_path, canvas)
    print(f"Saved full resolution: {full_res_path}")
    
    # Save Compressed Preview (JPG)
    # Resize if too large for easy viewing (max 4000px dim)
    preview_scale = 1.0
    max_dim = 4000
    if width > max_dim or height > max_dim:
        preview_scale = max_dim / max(width, height)
        print(f"Resizing for preview (scale {preview_scale:.2f})...")
        new_w, new_h = int(width * preview_scale), int(height * preview_scale)
        canvas_preview = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        canvas_preview = canvas
        
    preview_path = os.path.join(output_dir, f"{basename}_preview.jpg")
    cv2.imwrite(preview_path, canvas_preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Saved preview: {preview_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot DM2D JSON vectors')
    parser.add_argument('--json', type=str, required=True, help='Input JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--width', type=int, help='Original image width (optional)')
    parser.add_argument('--height', type=int, help='Original image height (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        print(f"Error: File not found: {args.json}")
        sys.exit(1)
        
    plot_json(args.json, args.output_dir, args.width, args.height)

if __name__ == '__main__':
    main()
