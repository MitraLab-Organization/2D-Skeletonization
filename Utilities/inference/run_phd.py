#!/usr/bin/env python3
"""
Run PHD (Probability Hypothesis Density) filtering for neuron detection.
Supports both PMD and STP datasets via --dataset flag.
"""

import os
import subprocess
import glob
import argparse
import tifffile
import numpy as np
import cv2
import shutil

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject
IMAGEJ_DIR = os.environ.get("IMAGEJ_DIR", "/opt/Fiji.app")
FIJI_BIN = os.path.join(IMAGEJ_DIR, "ImageJ-linux64")
PYTHON_BIN = "python"

DATASET_CONFIG = {
    "pmd": {
        "lkl_dir": f"{BASE_DIR}/data/pmd/lkl",
        "gt_dir": f"{BASE_DIR}/data/pmd/GT",
        "output_dir": f"{BASE_DIR}/outputs/pmd/phd",
    },
    "stp": {
        "lkl_dir": f"{BASE_DIR}/data/stp/lkl",
        "gt_dir": f"{BASE_DIR}/data/stp/GT",
        "output_dir": f"{BASE_DIR}/outputs/stp/phd",
    }
}

def parse_swc(swc_path):
    nodes = {}
    if not os.path.exists(swc_path): 
        return nodes
    with open(swc_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): 
                continue
            parts = line.split()
            if len(parts) < 7: 
                continue
            try:
                node_id = int(parts[0])
                x, y = float(parts[2]), float(parts[3])
                pid = int(parts[6])
                nodes[node_id] = {'x': x, 'y': y, 'pid': pid}
            except: 
                continue
    return nodes

def draw_swc(nodes, shape=(1000, 1000)):
    img = np.zeros(shape, dtype=np.uint8)
    for node_id, node in nodes.items():
        pid = node['pid']
        if pid != -1 and pid in nodes:
            parent = nodes[pid]
            pt1 = (int(node['x']), int(node['y']))
            pt2 = (int(parent['x']), int(parent['y']))
            cv2.line(img, pt1, pt2, 255, 1)
    return img

def harvest_results(lkl_dir, output_dir):
    """Harvest SWC files created by PHD in lkl subdirectories."""
    swc_files = glob.glob(os.path.join(lkl_dir, "**/*.swc"), recursive=True)
    print(f"Harvesting {len(swc_files)} SWC files...")
    for swc_path in swc_files:
        filename = os.path.basename(swc_path)
        if '.tif' in filename:
            clean_name = filename.split('.tif')[0] + '.tif'
        else:
            clean_name = filename.replace('.swc', '') + '.tif'
        
        out_tif = os.path.join(output_dir, clean_name)
        if not os.path.exists(out_tif):
            try:
                img = draw_swc(parse_swc(swc_path))
                tifffile.imwrite(out_tif, img)
            except Exception as e:
                print(f"Failed to convert {swc_path}: {e}")
        
        out_swc = os.path.join(output_dir, filename)
        if not os.path.exists(out_swc):
            shutil.copy(swc_path, out_swc)

def cleanup_phd_dirs(lkl_dir):
    """Remove PHD intermediate directories from lkl folder after processing."""
    phd_dirs = glob.glob(os.path.join(lkl_dir, "PHD.*"))
    for d in phd_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Cleaned up: {os.path.basename(d)}")

def main():
    parser = argparse.ArgumentParser(description='Run PHD on PMD or STP dataset')
    parser.add_argument('--dataset', '-d', choices=['pmd', 'stp'], required=True,
                        help='Dataset to process: pmd or stp')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation step')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    lkl_dir = cfg["lkl_dir"]
    gt_dir = cfg["gt_dir"]
    output_dir = cfg["output_dir"]
    macro_path = f"{BASE_DIR}/scripts/inference/run_phd_fast.ijm"
    eval_script = f"{BASE_DIR}/scripts/evaluation/evaluate_model.py"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running PHD on {args.dataset.upper()}/lkl with 16GB RAM (Checkpoint enabled)...")
    
    # Pre-harvest existing results
    harvest_results(lkl_dir, output_dir)

    # Use Fiji with --headless flag (Fiji/ImageJ2 properly supports headless mode)
    cmd = [FIJI_BIN, "--headless", "--console", "-macro", macro_path, f"{lkl_dir}/::{output_dir}/"]
    subprocess.run(cmd, check=True)

    # Harvest new results and cleanup
    harvest_results(lkl_dir, output_dir)
    cleanup_phd_dirs(lkl_dir)

    # Evaluation
    if not args.skip_eval:
        print(f"\n--- Evaluating PHD ({args.dataset.upper()}) ---")
        subprocess.run([PYTHON_BIN, eval_script,
                        "--model_dir", output_dir,
                        "--model_name", f"phd_{args.dataset}",
                        "--gt_dir", gt_dir,
                        "--img_dir", lkl_dir], check=True)

if __name__ == "__main__":
    main()

