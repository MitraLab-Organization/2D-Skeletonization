import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_bool
import json
import pickle
from graph_analysis import graph_analysis
from cc_to_json import cc_to_json

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Vectorization Suite Script")
    parser.add_argument('--input_dir', type=str, default='mask', help='Input directory containing images')
    parser.add_argument('--output_cc_dir', type=str, default='CC1', help='Output directory for Connected Components (Pickle)')
    parser.add_argument('--output_json_dir', type=str, default='skelJSON1', help='Output directory for GeoJSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--min_size', type=int, default=50, help='Minimum size for connected components')
    
    args = parser.parse_args()

    path = args.input_dir
    out_cc_path = args.output_cc_dir
    out_json_path = args.output_json_dir
    debug = args.debug

    os.makedirs(out_cc_path, exist_ok=True)
    os.makedirs(out_json_path, exist_ok=True)

    image_files = glob.glob(os.path.join(path, "*.jpg"))
    
    if not image_files:
        print(f"No .jpg files found in {path}")
        return

    for i, file_path in enumerate(image_files):
        print(f"Image {i + 1} / {len(image_files)}")
        
        if debug:
            print(f"Loading image: {file_path}")
        
        img = cv2.imread(file_path)
        
        if img is None:
            print(f"Error: Failed to load image {file_path}. Skipping.")
            continue

        if debug:
            print("Converting to grayscale and boolean...")
        
        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bw_img = img_as_bool(gray_img)
            
            if debug:
                print("Skeletonizing (this may take a while)...")
            skeleton = skeletonize(bw_img)
            if debug:
                print("Skeletonization complete.")
            
            if debug:
                print("Starting graph analysis...")
            cc = graph_analysis(skeleton, debug=debug, min_size=args.min_size)
            
            # Save the connected components data
            base_name = os.path.basename(file_path)
            cc_file_name = os.path.splitext(base_name)[0] + ".pkl"
            with open(os.path.join(out_cc_path, cc_file_name), 'wb') as f:
                pickle.dump(cc, f)
                
            data_out = cc_to_json(cc)
            
            # Save the GeoJSON data
            json_file_name = os.path.splitext(base_name)[0] + ".json"
            with open(os.path.join(out_json_path, json_file_name), 'w') as f:
                json.dump(data_out, f)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == '__main__':
    main()
