#!/usr/bin/env python3
"""
Batch skeletonization using neuTube for all images in a directory.
Supports parallel processing using multiprocessing.
"""

import os
import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Setup paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject
NEUTUBE_DIR = os.path.join(BASE_DIR, "tools", "neutube")
MODULE_PATH = os.path.join(NEUTUBE_DIR, "neurolabi/python/module")
SKELETONIZE_SCRIPT = os.path.join(NEUTUBE_DIR, "neurolabi/python/skeletonize.py")
CONFIG_PATH = os.path.join(NEUTUBE_DIR, "neurolabi/json/skeletonize.json")


def run_skeletonize(args):
    """Run skeletonization on a single image (worker function)"""
    input_image, output_swc, idx, total = args
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + MODULE_PATH
    
    cmd = [
        "python3", SKELETONIZE_SCRIPT,
        "-i", str(input_image),
        "-o", str(output_swc),
        "--config", CONFIG_PATH
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        return (True, input_image.name, output_swc.name, elapsed)
    else:
        return (False, input_image.name, result.stderr[:200] if result.stderr else "Unknown error", elapsed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Batch skeletonization using neuTube (parallel)')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Output directory for SWC files')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help=f'Number of parallel workers (default: {cpu_count()})')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIF files
    image_files = sorted(list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff')))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print(f"ERROR: No .tif/.tiff files found in {input_dir}")
        print(f"Please check that the input directory exists and contains images.")
        sys.exit(1)
    
    num_workers = args.workers if args.workers else min(cpu_count(), len(image_files))
    
    print(f"Using {num_workers} parallel workers")
    print("-" * 60)
    
    # Prepare arguments for workers
    work_items = []
    for idx, img_file in enumerate(image_files):
        output_swc = output_dir / f"{img_file.stem}.swc"
        work_items.append((img_file, output_swc, idx, len(image_files)))
    
    # Run in parallel
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_skeletonize, work_items)
    
    total_elapsed = time.time() - start_time
    
    # Report results
    print("-" * 60)
    success_count = 0
    for success, name, output_or_error, elapsed in results:
        if success:
            print(f"✓ {name} -> {output_or_error} ({elapsed:.1f}s)")
            success_count += 1
        else:
            print(f"✗ {name}: {output_or_error}")
    
    print("-" * 60)
    print(f"Completed: {success_count}/{len(image_files)} images")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
