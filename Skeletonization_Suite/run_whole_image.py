#!/usr/bin/env python3
"""
Run Whole-Image DM++ Pipeline

Processes full JP2/TIFF images through the DM++ neural network pipeline.
This is different from the tiled evaluation mode - it runs on full-size images.

Usage:
    python run_whole_image.py --input /path/to/image.jp2 --output /path/to/output --mode pmd
    python run_whole_image.py --input /path/to/image.jp2 --output /path/to/output --mode stp
    python run_whole_image.py --input /path/to/image.jp2 --output /path/to/output --persistence_threshold 16 --min_size 30
    python run_whole_image.py --input_dir /path/to/images/ --output /path/to/output --mode pmd
"""

import os
import sys
import argparse
import glob
import numpy as np
from pathlib import Path

# Add DM2D paths
# run_whole_image.py is directly in Skeletonization_Suite/
skel_suite = Path(__file__).parent.resolve()
sys.path.insert(0, str(skel_suite))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "morse_code"))
sys.path.insert(0, str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "DM_base"))


def load_models(model_dir):
    """Load ALBU models"""
    sys.path.append(str(skel_suite / "DM++" / "Semantic_Segmentation_NMI" / "morse_code"))
    import albu_dingkang
    
    # albu_dingkang.read_model expects a list of paths
    model_paths = [os.path.join(model_dir, f'fold{i}_best.pth') for i in range(4)]
    return albu_dingkang.read_model(model_paths)


# Mode presets (paper reproduction values)
MODE_PRESETS = {
    'pmd': {'ve_persistence': 0, 'et_persistence': 64, 'min_size': 40, 'norm_factor': 16},
    'stp': {'ve_persistence': 0, 'et_persistence': 32, 'min_size': 12, 'norm_factor': 256},
}


def process_single_image(input_path, output_dir, models=None, output_name=None,
                         ve_persistence=0, et_persistence=0, min_size=0, norm_factor=16):
    """Process a single image using the original pipeline script
    
    Args:
        input_path: Path to input image
        output_dir: Output directory
        models: Pre-loaded ALBU models (optional)
        output_name: Custom output name (optional). If provided, used as-is.
                     Format: "BrainName_SectionNum" or just "Name" (section defaults to 0)
        ve_persistence: VE persistence threshold (default: 0)
        et_persistence: ET persistence threshold (default: 0)
        min_size: Minimum connected component size for filtering (default: 0, no filtering)
        norm_factor: Normalization divisor for tile pixel values (PMD=16, STP=256)
    """
    import pipeline_processDetect_skel_samik_singleChanel as original_pipeline
    import shutil
    
    print(f"Processing: {input_path}")
    basename = Path(input_path).stem
    
    # Determine output naming
    if output_name:
        # User specified custom output name
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
        # Parse brain_no and section_num from filename if possible, else defaults
        # Filename format: {brain_no}_{section_num}.jp2
        # brain_no can contain underscores, section_num is the last part
        try:
            parts = basename.split('_')
            if len(parts) >= 2:
                # Try to parse last part as section number
                section_num = int(parts[-1])
                # Everything else is brain_no
                brain_no = '_'.join(parts[:-1])
            else:
                brain_no = basename
                section_num = 0
        except (ValueError, IndexError):
            # If last part isn't a number, use whole basename
            brain_no = basename
            section_num = 0
        print(f"  Inferred ID: {brain_no}, Section: {section_num}")

    # Load Models if not provided
    if models is None:
        model_dir = skel_suite / "models"
        print(f"  Loading models from {model_dir}...")
        models = load_models(model_dir)
        if models is None:
            return None
    
    # Prepare directories
    json_out_dir = output_dir
    os.makedirs(json_out_dir, exist_ok=True)
    
    temp_dir = os.path.join(json_out_dir, "tmp")
    scratch_dir = os.path.join(json_out_dir, "scratch")
    json_out_dir_temp = os.path.join(json_out_dir, "json_tmp")
    
    # Ensure they exist (pipeline script expects them)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)
    os.makedirs(json_out_dir_temp, exist_ok=True)
    
    # Call original pipeline main function
    pipeline_success = False
    try:
        original_pipeline.main(
            input_image_path=str(input_path),
            json_out_dir=json_out_dir,
            brain_no=brain_no,
            section_num=section_num,
            albu_models=models,
            model_dmpp=None,
            temp_dir=temp_dir,
            scratch_dir=scratch_dir,
            json_out_dir_temp=json_out_dir_temp,
            ve_persistence_threshold=ve_persistence,
            et_persistence_threshold=et_persistence,
            norm_factor=norm_factor
        )
        pipeline_success = True
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup temp directories, even on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(scratch_dir, ignore_errors=True)
        shutil.rmtree(json_out_dir_temp, ignore_errors=True)
        
        # Also clean up any stray tmp* directories from multiprocessing
        for item in os.listdir(json_out_dir):
            item_path = os.path.join(json_out_dir, item)
            if os.path.isdir(item_path) and item.startswith('tmp'):
                shutil.rmtree(item_path, ignore_errors=True)
    
    if not pipeline_success:
        return None
    
    # Return output path
    # Original pipeline produces: {json_out_dir}/{brain_no}_{section_num}.json
    output_json = os.path.join(json_out_dir, f"{brain_no}_{section_num}.json")
    
    # Run vectorization on the mask (matching ablation pipeline flow)
    # This does: skeletonize → remove_small_objects(min_size) → graph analysis → clean GeoJSON
    skel_bin_path = os.path.join(json_out_dir, "mask", f"{brain_no}_{section_num}.jpg")
    if os.path.exists(skel_bin_path):
        try:
            import cv2
            from skimage.morphology import skeletonize
            from skimage.util import img_as_bool
            
            # Add parent of Vectorization to path (graph_analysis uses 'from Vectorization.ImageGraphs...')
            vec_parent = str(Path(__file__).parent.parent)
            if vec_parent not in sys.path:
                sys.path.insert(0, vec_parent)
            from Vectorization.graph_analysis import graph_analysis
            from Vectorization.cc_to_json import cc_to_json
            
            print(f"  Running vectorization (min_size={min_size})...")
            
            # 1. Read mask and skeletonize
            mask_img = cv2.imread(skel_bin_path, cv2.IMREAD_GRAYSCALE)
            bw_img = img_as_bool(mask_img)
            skeleton = skeletonize(bw_img)
            
            # 2. Graph analysis with min_size filtering
            effective_min_size = max(min_size, 1)  # 0 means no filtering, but remove_small_objects needs >= 1
            cc = graph_analysis(skeleton, debug=False, min_size=effective_min_size)
            print(f"  Found {cc['NumObjects']} connected components after min_size={effective_min_size} filtering")
            
            # 3. Convert to clean GeoJSON
            vectorized_json = cc_to_json(cc)
            
            # 4. Save vectorized JSON (overwrite raw DM2D JSON)
            import json
            with open(output_json, 'w') as f:
                json.dump(vectorized_json, f)
            print(f"  Vectorized JSON: {len(vectorized_json.get('features', []))} features (was 6M+ raw segments)")
            
            # 5. Re-render skeleton from vectorized JSON for the mask
            skel_out = np.zeros(mask_img.shape, dtype=np.uint8)
            for feat in vectorized_json.get('features', []):
                geom = feat.get('geometry', {})
                geom_type = geom.get('type')
                coords = geom.get('coordinates', [])
                
                if geom_type == 'LineString' and len(coords) >= 2:
                    for j in range(len(coords) - 1):
                        x1, y1 = int(coords[j][0]), int(-coords[j][1])
                        x2, y2 = int(coords[j+1][0]), int(-coords[j+1][1])
                        cv2.line(skel_out, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
                elif geom_type == 'MultiLineString':
                    for segment in coords:
                        if len(segment) >= 2:
                            for j in range(len(segment) - 1):
                                x1, y1 = int(segment[j][0]), int(-segment[j][1])
                                x2, y2 = int(segment[j+1][0]), int(-segment[j+1][1])
                                cv2.line(skel_out, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)
            
            cv2.imwrite(skel_bin_path, skel_out)
            print(f"  Updated mask: {np.sum(skel_out > 0)} skeleton pixels")
            
        except Exception as e:
            print(f"  Warning: Vectorization failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate Visualization
    try:
        print("  Generating visualizations...")
        vis_out_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_out_dir, exist_ok=True)
        
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent.parent / "Utilities" / "visualization" / "plot_json.py"),
            "--json", output_json,
            "--output_dir", vis_out_dir
        ], check=True)
        print(f"  Visualizations saved to: {vis_out_dir}")
    except Exception as e:
        print(f"  Warning: Visualization generation failed: {e}")

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description='Run DM++ whole-image processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  pmd       Use PMD paper parameters (ve_persistence=0, persistence_threshold=64, min_size=40, norm_factor=16)
  stp       Use STP paper parameters (ve_persistence=0, persistence_threshold=32, min_size=12, norm_factor=256)
  custom    Specify custom parameters

Examples:
  %(prog)s --input image.jp2 --output out/ --mode pmd
  %(prog)s --input image.jp2 --output out/ --mode stp
  %(prog)s --input image.jp2 --output out/ --persistence_threshold 16 --min_size 30 --norm_factor 16
  %(prog)s --input_dir images/ --output out/ --mode pmd
""")
    parser.add_argument('--input', type=str, help='Single input image (JP2 or TIFF)')
    parser.add_argument('--input_dir', type=str, help='Directory of input images')
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
    parser.add_argument('--norm_factor', type=int, default=None,
                        help='Normalization divisor for pixel values (required for custom mode, PMD=16, STP=256)')
    
    args = parser.parse_args()
    
    # Resolve mode to parameters
    if args.mode in MODE_PRESETS:
        preset = MODE_PRESETS[args.mode]
        ve_persistence = preset['ve_persistence']
        et_persistence = preset['et_persistence']
        min_size = preset['min_size']
        norm_factor = preset['norm_factor']
        # Warn if user also passed explicit values
        if any(v is not None for v in [args.ve_persistence_threshold, args.persistence_threshold, args.min_size, args.norm_factor]):
            print(f"Warning: --mode {args.mode} overrides all parameter flags")
        print(f"Using {args.mode.upper()} preset: ve_persistence={ve_persistence}, persistence_threshold={et_persistence}, min_size={min_size}, norm_factor={norm_factor}")
    else:
        # Custom mode: require et_persistence, min_size, norm_factor; ve_persistence defaults to 0
        if args.persistence_threshold is None or args.min_size is None or args.norm_factor is None:
            parser.error("--persistence_threshold, --min_size, and --norm_factor are required when using custom mode (default)")
        ve_persistence = args.ve_persistence_threshold if args.ve_persistence_threshold is not None else 0
        et_persistence = args.persistence_threshold
        min_size = args.min_size
        norm_factor = args.norm_factor
        print(f"Using custom parameters: ve_persistence={ve_persistence}, persistence_threshold={et_persistence}, min_size={min_size}, norm_factor={norm_factor}")
    
    if args.input:
        # Single image mode
        process_single_image(args.input, args.output, output_name=args.output_name,
                             ve_persistence=ve_persistence, et_persistence=et_persistence, min_size=min_size, norm_factor=norm_factor)
    elif args.input_dir:
        # Batch mode (output_name not used - each file gets name from its filename)
        if args.output_name:
            print("Warning: --output_name is ignored in batch mode (--input_dir)")
        patterns = ['*.jp2', '*.JP2', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        
        print(f"Found {len(files)} images to process")
        for i, f in enumerate(sorted(files)):
            print(f"\n[{i+1}/{len(files)}]")
            process_single_image(f, args.output, ve_persistence=ve_persistence, et_persistence=et_persistence, min_size=min_size, norm_factor=norm_factor)
    else:
        print("Error: Must specify --input or --input_dir")
        sys.exit(1)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
