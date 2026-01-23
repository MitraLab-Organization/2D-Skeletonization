#!/usr/bin/env python3
"""
Run VESS (Vesselness) filtering and skeletonization.
Supports both PMD and STP datasets via --dataset flag.
"""

import os
import subprocess
import glob
import argparse
import tifffile
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import shutil

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject
IMAGEJ_DIR = os.environ.get("IMAGEJ_DIR", "/opt/Fiji.app")
FIJI_BIN = os.path.join(IMAGEJ_DIR, "ImageJ-linux64")
PYTHON_BIN = "python"  # Use system python or set PYTHON env var

DATASET_CONFIG = {
    "pmd": {
        "lkl_dir": f"{BASE_DIR}/data/pmd/lkl",
        "gt_dir": f"{BASE_DIR}/data/pmd/GT",
        "output_dir": f"{BASE_DIR}/outputs/pmd/vess",
    },
    "stp": {
        "lkl_dir": f"{BASE_DIR}/data/stp/lkl",
        "gt_dir": f"{BASE_DIR}/data/stp/GT",
        "output_dir": f"{BASE_DIR}/outputs/stp/vess",
    }
}

def main():
    parser = argparse.ArgumentParser(description='Run VESS on PMD or STP dataset')
    parser.add_argument('--dataset', '-d', choices=['pmd', 'stp'], required=True,
                        help='Dataset to process: pmd or stp')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation step')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    lkl_dir = cfg["lkl_dir"]
    gt_dir = cfg["gt_dir"]
    output_dir = cfg["output_dir"]
    macro_path = f"{BASE_DIR}/scripts/inference/run_vess.ijm"
    eval_script = f"{BASE_DIR}/scripts/evaluation/evaluate_model.py"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running VESS on {args.dataset.upper()}/lkl with 16GB RAM...")
    # Use Fiji with --headless flag (Fiji/ImageJ2 properly supports headless mode)
    cmd = [FIJI_BIN, "--headless", "--console", "-macro", macro_path, f"{lkl_dir}/::{output_dir}/"]
    subprocess.run(cmd, check=True)

    # Process vesselness outputs to skeletons
    # Look for _vess.tif in both lkl_dir (new) and output_dir (already processed)
    vess_files = glob.glob(os.path.join(lkl_dir, "*_vess.tif"))
    print(f"Found {len(vess_files)} VESS outputs in lkl_dir.")
    
    for path in vess_files:
        filename = os.path.basename(path)
        img = tifffile.imread(path)
        try:
            binary = img > threshold_otsu(img)
        except:
            binary = img > 0
        skel = (skeletonize(binary) * 255).astype(np.uint8)
        clean_name = filename.replace('.tif_vess.tif', '.tif').replace('_vess.tif', '.tif')
        tifffile.imwrite(os.path.join(output_dir, clean_name), skel)
        # Move vesselness file to output and remove from lkl_dir
        shutil.move(path, os.path.join(output_dir, filename))

    # Evaluation
    if not args.skip_eval:
        print(f"\n--- Evaluating VESS ({args.dataset.upper()}) ---")
        subprocess.run([PYTHON_BIN, eval_script, 
                        "--model_dir", output_dir, 
                        "--model_name", f"vess_{args.dataset}",
                        "--gt_dir", gt_dir, 
                        "--img_dir", lkl_dir], check=True)

if __name__ == "__main__":
    main()
