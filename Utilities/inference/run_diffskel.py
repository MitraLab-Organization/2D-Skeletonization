#!/usr/bin/env python3
"""
Run Differential Skeletonization (gradient-based) on PMD or STP dataset.
Wrapper around tools/skeletonization-for-gradient-based-optimization/batch_process.py

Usage:
    python run_diffskel.py --dataset pmd
    python run_diffskel.py --dataset stp
"""

import os
import sys
import argparse
import subprocess

# Paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject
DIFFSKEL_DIR = os.path.join(BASE_DIR, "tools", "skeletonization-for-gradient-based-optimization")
DIFFSKEL_SCRIPT = os.path.join(DIFFSKEL_DIR, "batch_process.py")

DATASET_CONFIG = {
    "pmd": {
        "lkl_dir": os.path.join(BASE_DIR, "data/pmd/lkl"),
        "output_dir": os.path.join(BASE_DIR, "outputs/pmd/diffskel"),
    },
    "stp": {
        "lkl_dir": os.path.join(BASE_DIR, "data/stp/lkl"),
        "output_dir": os.path.join(BASE_DIR, "outputs/stp/diffskel"),
    }
}

def main():
    parser = argparse.ArgumentParser(description='Run diffskel on PMD or STP dataset')
    parser.add_argument('--dataset', '-d', choices=['pmd', 'stp'], required=True,
                        help='Dataset to process: pmd or stp')
    parser.add_argument('--probabilistic', action='store_true',
                        help='Use probabilistic mode')
    parser.add_argument('--beta', type=float, default=0.33,
                        help='Beta parameter (default: 0.33)')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    
    # Check if script exists
    if not os.path.exists(DIFFSKEL_SCRIPT):
        print(f"ERROR: Diffskel script not found at {DIFFSKEL_SCRIPT}")
        print("Make sure tools/skeletonization-for-gradient-based-optimization exists.")
        sys.exit(1)
    
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    print(f"Running diffskel on {args.dataset.upper()}")
    print(f"  Input:  {cfg['lkl_dir']}")
    print(f"  Output: {cfg['output_dir']}")
    
    cmd = [
        sys.executable, DIFFSKEL_SCRIPT,
        "--input_folder", cfg["lkl_dir"],
        "--output_folder", cfg["output_dir"],
        "--beta", str(args.beta),
    ]
    
    if args.probabilistic:
        cmd.append("--probabilistic")
    
    # Run from the diffskel directory so imports work
    subprocess.run(cmd, check=True, cwd=DIFFSKEL_DIR)
    
    # Post-process outputs to ensure 1-pixel width skeleton
    print("Post-processing diffskel outputs...")
    from skimage.morphology import skeletonize
    import cv2
    import numpy as np
    import glob
    
    files = glob.glob(os.path.join(cfg["output_dir"], "*_skeleton.png"))
    print(f"Found {len(files)} files to post-process")
    
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Threshold (Otsu)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Skeletonize
        skel = skeletonize(thresh > 0)
        skel_img = (skel * 255).astype(np.uint8)
        
        # Save back (replace original)
        cv2.imwrite(f, skel_img)
        
    print("Post-processing complete.")

if __name__ == "__main__":
    main()
