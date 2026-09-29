#!/usr/bin/env python3
"""
Run Image -> Likelihood (Entry Point 2)

Runs only the first half of the whole-image pipeline: tiling and ALBU
inference. Produces and saves the likelihood image, then stops (no
skeletonization).

Usage:
    python run_image_to_likelihood.py --input /path/to/image.jp2 --output /path/to/output --mode pmd
    python run_image_to_likelihood.py --input_dir /path/to/images/ --output /path/to/output --mode stp
"""

import os
import sys
import argparse
import glob
from pathlib import Path

skel_suite = Path(__file__).parent.resolve()
sys.path.insert(0, str(skel_suite))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "morse_code"))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "DM_base"))

from run_whole_image import load_models, MODE_PRESETS


def process_single_image(input_path, output_dir, models=None, output_name=None,
                          norm_factor=16, likelihood_threshold=40, use_mask=True):
    """Compute and save the likelihood image for a single input image."""
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

    if models is None:
        model_dir = skel_suite / "models"
        print(f"  Loading models from {model_dir}...")
        models = load_models(model_dir)

    json_out_dir = output_dir
    os.makedirs(json_out_dir, exist_ok=True)

    success = False
    try:
        original_pipeline.compute_likelihood(
            input_image_path=str(input_path),
            json_out_dir=json_out_dir,
            brain_no=brain_no,
            section_num=section_num,
            albu_models=models,
            norm_factor=norm_factor,
            likelihood_threshold=likelihood_threshold,
            use_mask=use_mask
        )
        success = True
    except Exception as e:
        print(f"Error computing likelihood: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for item in os.listdir(json_out_dir):
            item_path = os.path.join(json_out_dir, item)
            if os.path.isdir(item_path) and item.startswith('tmp'):
                shutil.rmtree(item_path, ignore_errors=True)

    if not success:
        return None

    lkl_path = os.path.join(json_out_dir, "lkl", f"{brain_no}_{section_num}.jpg")
    print(f"  Likelihood image: {lkl_path}")
    return lkl_path


def main():
    parser = argparse.ArgumentParser(
        description='Run Image -> Likelihood (Entry Point 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  pmd       norm_factor=16, likelihood_threshold=40, mask=true
  stp       norm_factor=256, likelihood_threshold=40, mask=true
  custom    Specify --norm_factor, --likelihood_threshold, and --mask

Examples:
  %(prog)s --input image.jp2 --output out/ --mode pmd
  %(prog)s --input_dir images/ --output out/ --mode stp
""")
    parser.add_argument('--input', type=str, help='Single input image (JP2 or TIFF)')
    parser.add_argument('--input_dir', type=str, help='Directory of input images')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--output_name', type=str, default=None,
                         help='Custom output name (e.g., "MyBrain_001"). If not specified, derived from input filename.')
    parser.add_argument('--mode', type=str, choices=['pmd', 'stp', 'custom'], default='custom',
                         help='Parameter mode: pmd, stp, or custom (default: custom)')
    parser.add_argument('--norm_factor', type=int, default=None,
                         help='Normalization divisor for pixel values (required for custom mode, PMD=16, STP=256)')
    parser.add_argument('--likelihood_threshold', type=int, default=None,
                         help='Likelihood pixels below this are clipped to zero (required for custom mode, PMD=50, STP=60)')
    parser.add_argument('--mask', type=str, default=None, choices=['true', 'false'],
                         help='Restrict processing to tissue via Otsu thresholding (default: true for pmd/stp, required for custom)')

    args = parser.parse_args()

    if args.mode in MODE_PRESETS:
        norm_factor = MODE_PRESETS[args.mode]['norm_factor']
        likelihood_threshold = MODE_PRESETS[args.mode]['likelihood_threshold']
        use_mask = MODE_PRESETS[args.mode]['use_mask']
        if args.mask is not None:
            use_mask = (args.mask == 'true')
        if args.norm_factor is not None or args.likelihood_threshold is not None:
            print(f"Warning: --mode {args.mode} overrides --norm_factor/--likelihood_threshold")
        print(f"Using {args.mode.upper()} preset: norm_factor={norm_factor}, likelihood_threshold={likelihood_threshold}, mask={use_mask}")
    else:
        if args.norm_factor is None or args.likelihood_threshold is None or args.mask is None:
            parser.error("--norm_factor, --likelihood_threshold, and --mask are required when using custom mode (default)")
        norm_factor = args.norm_factor
        likelihood_threshold = args.likelihood_threshold
        use_mask = (args.mask == 'true')
        print(f"Using custom parameters: norm_factor={norm_factor}, likelihood_threshold={likelihood_threshold}, mask={use_mask}")

    if args.input:
        process_single_image(args.input, args.output, output_name=args.output_name,
                              norm_factor=norm_factor, likelihood_threshold=likelihood_threshold, use_mask=use_mask)
    elif args.input_dir:
        if args.output_name:
            print("Warning: --output_name is ignored in batch mode (--input_dir)")
        patterns = ['*.jp2', '*.JP2', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

        print(f"Found {len(files)} images to process")
        for i, f in enumerate(sorted(files)):
            print(f"\n[{i+1}/{len(files)}]")
            process_single_image(f, args.output, norm_factor=norm_factor, likelihood_threshold=likelihood_threshold, use_mask=use_mask)
    else:
        print("Error: Must specify --input or --input_dir")
        sys.exit(1)

    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
