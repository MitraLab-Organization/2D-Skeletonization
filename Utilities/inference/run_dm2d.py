#!/usr/bin/env python3
"""
Run DM2D skeletonization on PMD or STP dataset.
Wrapper around DM2D_Skeletonization_Vectorization/run_dm2d_tiles.py

Usage:
    python run_dm2d.py --dataset pmd
    python run_dm2d.py --dataset stp
"""

import os
import sys
import argparse
import subprocess

# Paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject
DM2D_DIR = os.path.join(BASE_DIR, "DM2D_Skeletonization_Vectorization")
DM2D_SCRIPT = os.path.join(DM2D_DIR, "run_dm2d_tiles.py")

DATASET_CONFIG = {
    "pmd": {
        "lkl_dir": os.path.join(BASE_DIR, "data/pmd/lkl"),
        "output_dir": os.path.join(BASE_DIR, "outputs/pmd/dm2d"),
        "ve_persistence": 0,
        "et_persistence": 64,  # Optimal for PMD
    },
    "stp": {
        "lkl_dir": os.path.join(BASE_DIR, "data/stp/lkl"),
        "output_dir": os.path.join(BASE_DIR, "outputs/stp/dm2d"),
        "ve_persistence": 0,
        "et_persistence": 32,  # Optimal for STP
    }
}

def main():
    parser = argparse.ArgumentParser(description='Run DM2D on PMD or STP dataset')
    parser.add_argument('--dataset', '-d', choices=['pmd', 'stp'], required=True,
                        help='Dataset to process: pmd or stp')
    parser.add_argument('--ve_persistence', type=int, default=None,
                        help='VE persistence threshold (default: use dataset default)')
    parser.add_argument('--et_persistence', type=int, default=None,
                        help='ET persistence threshold (default: use dataset default)')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    
    # Check if DM2D script exists
    if not os.path.exists(DM2D_SCRIPT):
        print(f"ERROR: DM2D script not found at {DM2D_SCRIPT}")
        print("Make sure DM2D_Skeletonization_Vectorization is in the project root.")
        sys.exit(1)
    
    ve_pers = args.ve_persistence if args.ve_persistence is not None else cfg["ve_persistence"]
    et_pers = args.et_persistence if args.et_persistence is not None else cfg["et_persistence"]
    
    print(f"Running DM2D on {args.dataset.upper()}")
    print(f"  Input:  {cfg['lkl_dir']}")
    print(f"  Output: {cfg['output_dir']}")
    print(f"  VE persistence: {ve_pers}")
    print(f"  ET persistence: {et_pers}")
    
    # Step 1: Run DM2D tiles (skeletonization)
    dm2d_cmd = [
        sys.executable, DM2D_SCRIPT,
        "--lkl_dir", cfg["lkl_dir"],
        "--output_dir", cfg["output_dir"],
        "--ve_persistence", str(ve_pers),
        "--et_persistence", str(et_pers),
    ]
    
    subprocess.run(dm2d_cmd, check=True)
    
    # Step 2: Run Vectorization on JSON outputs
    print("\n=== Running Vectorization ===")
    vectorization_script = os.path.join(DM2D_DIR, "run_vectorization.py")
    json_dir = os.path.join(cfg["output_dir"], "json")
    vectorized_dir = os.path.join(cfg["output_dir"], "vectorized")
    
    if os.path.exists(vectorization_script):
        vec_cmd = [
            sys.executable, vectorization_script,
            "--input_dir", json_dir,
            "--lkl_dir", cfg["lkl_dir"],
            "--output_dir", vectorized_dir,
        ]
        subprocess.run(vec_cmd, check=True)
    else:
        print(f"  WARNING: Vectorization script not found at {vectorization_script}")

if __name__ == "__main__":
    main()
