#!/usr/bin/env python3
"""
Run Likelihood -> Skeleton (Entry Point 3)

Runs only the second half of the DM++ whole-image pipeline: threshold a
likelihood image, run DM2D, then vectorize and visualize the result. Takes
a single whole-brain-section-scale likelihood image (e.g. the output of
run_image_to_likelihood.py, or run_whole_image.py's intermediate
lkl/*.jpg) as input -- no raw image or neural network involved.

Usage:
    python run_likelihood_to_skeleton.py --input /path/to/likelihood.jpg --output /path/to/output --mode pmd
    python run_likelihood_to_skeleton.py --input /path/to/likelihood.jpg --output /path/to/output --persistence_threshold 16 --min_size 30
"""

import os
import sys
import argparse
import cv2
from pathlib import Path

skel_suite = Path(__file__).parent.resolve()
sys.path.insert(0, str(skel_suite))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "morse_code"))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "DM_base"))

from run_whole_image import MODE_PRESETS, vectorize_and_visualize


def process_single_likelihood(input_path, output_dir, output_name=None,
                               ve_persistence=0, et_persistence=0, min_size=0):
    """Skeletonize a single likelihood image and vectorize/visualize the result."""
    import pipeline_processDetect_skel_samik_singleChanel as original_pipeline
    import shutil

    print(f"Processing: {input_path}")
    basename = Path(input_path).stem

    if output_name:
        if '_' in output_name:
            parts = output_name.rsplit('_', 1)
            brain_no = parts[0]
            try:
                section_num = int(parts[1])
            except ValueError:
                brain_no = output_name
                section_num = 0
        else:
            brain_no = output_name
            section_num = 0
        print(f"  Using custom output name: {brain_no}_{section_num}")
    else:
        try:
            parts = basename.split('_')
            if len(parts) >= 2:
                section_num = int(parts[-1])
                brain_no = '_'.join(parts[:-1])
            else:
                brain_no = basename
                section_num = 0
        except (ValueError, IndexError):
            brain_no = basename
            section_num = 0
        print(f"  Inferred ID: {brain_no}, Section: {section_num}")

    likelihood_image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if likelihood_image is None:
        print(f"Error: could not read likelihood image {input_path}")
        return None

    json_out_dir = output_dir
    os.makedirs(json_out_dir, exist_ok=True)
    scratch_dir = os.path.join(json_out_dir, "scratch")
    json_out_dir_temp = os.path.join(json_out_dir, "json_tmp")
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(json_out_dir_temp, exist_ok=True)

    success = False
    try:
        original_pipeline.skeletonize_likelihood(
            likelihood_image=likelihood_image,
            json_out_dir=json_out_dir,
            json_out_dir_temp=json_out_dir_temp,
            scratch_dir=scratch_dir,
            brain_no=brain_no,
            section_num=section_num,
            ve_persistence_threshold=ve_persistence,
            et_persistence_threshold=et_persistence
        )
        success = True
    except Exception as e:
        print(f"Error skeletonizing likelihood image: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        shutil.rmtree(json_out_dir_temp, ignore_errors=True)
        for item in os.listdir(json_out_dir):
            item_path = os.path.join(json_out_dir, item)
            if os.path.isdir(item_path) and item.startswith('tmp'):
                shutil.rmtree(item_path, ignore_errors=True)

    if not success:
        return None

    output_json = os.path.join(json_out_dir, f"{brain_no}_{section_num}.json")
    skel_bin_path = os.path.join(json_out_dir, "mask", f"{brain_no}_{section_num}.jpg")
    return vectorize_and_visualize(output_json, skel_bin_path, min_size, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Run Likelihood -> Skeleton (Entry Point 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  pmd       Use PMD parameters (ve_persistence=0, persistence_threshold=64, min_size=40)
  stp       Use STP parameters (ve_persistence=0, persistence_threshold=32, min_size=12)
  custom    Specify custom parameters

Examples:
  %(prog)s --input likelihood.jpg --output out/ --mode pmd
  %(prog)s --input likelihood.jpg --output out/ --persistence_threshold 16 --min_size 30
""")
    parser.add_argument('--input', type=str, required=True, help='A single likelihood image (whole-section scale)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--output_name', type=str, default=None,
                         help='Custom output name (e.g., "MyBrain_001"). If not specified, derived from input filename.')
    parser.add_argument('--mode', type=str, choices=['pmd', 'stp', 'custom'], default='custom',
                         help='Parameter mode: pmd, stp, or custom (default: custom)')
    parser.add_argument('--ve_persistence_threshold', type=int, default=None,
                         help='VE persistence threshold (default: 0 for pmd/stp, required for custom)')
    parser.add_argument('--persistence_threshold', type=int, default=None,
                         help='ET persistence threshold (required for custom mode)')
    parser.add_argument('--min_size', type=int, default=None,
                         help='Minimum connected component size (required for custom mode)')

    args = parser.parse_args()

    if args.mode in MODE_PRESETS:
        preset = MODE_PRESETS[args.mode]
        ve_persistence = preset['ve_persistence']
        et_persistence = preset['et_persistence']
        min_size = preset['min_size']
        if any(v is not None for v in [args.ve_persistence_threshold, args.persistence_threshold, args.min_size]):
            print(f"Warning: --mode {args.mode} overrides all parameter flags")
        print(f"Using {args.mode.upper()} preset: ve_persistence={ve_persistence}, persistence_threshold={et_persistence}, min_size={min_size}")
    else:
        if args.persistence_threshold is None or args.min_size is None:
            parser.error("--persistence_threshold and --min_size are required when using custom mode (default)")
        ve_persistence = args.ve_persistence_threshold if args.ve_persistence_threshold is not None else 0
        et_persistence = args.persistence_threshold
        min_size = args.min_size
        print(f"Using custom parameters: ve_persistence={ve_persistence}, persistence_threshold={et_persistence}, min_size={min_size}")

    process_single_likelihood(args.input, args.output, output_name=args.output_name,
                               ve_persistence=ve_persistence, et_persistence=et_persistence, min_size=min_size)

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
