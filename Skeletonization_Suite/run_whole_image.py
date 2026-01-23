#!/usr/bin/env python3
"""
Run Whole-Image DM++ Pipeline

Processes full JP2/TIFF images through the DM++ neural network pipeline.
This is different from the tiled evaluation mode - it runs on full-size images.

Usage:
    python run_whole_image.py --input /path/to/image.jp2 --output /path/to/output
    python run_whole_image.py --input_dir /path/to/images/ --output /path/to/output
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add DM2D paths
script_dir = Path(__file__).parent.parent.parent
skel_suite = script_dir / "Skeletonization_Suite"
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


def process_single_image(input_path, output_dir, models=None):
    """Process a single image using the original pipeline script"""
    import pipeline_processDetect_skel_samik_singleChanel as original_pipeline
    
    print(f"Processing: {input_path}")
    basename = Path(input_path).stem
    
    # Parse brain_no and section_num from filename if possible, else defaults
    # Filename format: {brain_no}_{section_num}.jp2
    try:
        parts = basename.split('_')
        brain_no = parts[0]
        section_num = int(parts[-1]) # Last part usually section number
        
        # Sam's code expects e.g. PMD2069_...._F0015
        # If simple name like "input", use dummies
        if len(parts) < 2:
             brain_no = "Brain"
             section_num = 0
    except:
        brain_no = "Brain"
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
    # def main(input_image_path,json_out_dir,brain_no,section_num,albu_models,model_dmpp,temp_dir,scratch_dir,json_out_dir_temp):
    try:
        original_pipeline.main(
            input_image_path=str(input_path),
            json_out_dir=json_out_dir,
            brain_no=brain_no,
            section_num=section_num,
            albu_models=models,
            model_dmpp=None, # Not used in single channel code apparently, or unused arg
            temp_dir=temp_dir,
            scratch_dir=scratch_dir,
            json_out_dir_temp=json_out_dir_temp
        )
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    # Cleanup (Optional)
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(scratch_dir, ignore_errors=True)
    shutil.rmtree(json_out_dir_temp, ignore_errors=True)
    
    # Return output path
    # Original pipeline produces: {json_out_dir}/{brain_no}_{section_num}.json
    output_json = os.path.join(json_out_dir, f"{brain_no}_{section_num}.json")
    
    # Generate Visualization (Restored)
    try:
        print("  Generating visualizations...")
        sys.path.append(str(Path(__file__).parent.parent / "visualization"))
        import plot_json
        
        # Determine output paths
        vis_out_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_out_dir, exist_ok=True)
        
        # Call plot_json main logic or function
        # plot_json.py normally takes args, let's call it via subprocess or import if modular
        # Checking plot_json content would be ideal, but assuming it has a function or main block.
        # Ideally, we refactor plot_json to be callable.
        # For now, let's run it as subprocess to be safe and avoid import conflicts if not designed as module.
        import subprocess
        subprocess.run([
            sys.executable, str(Path(__file__).parent.parent / "visualization/plot_json.py"),
            "--json", output_json,
            "--output_dir", vis_out_dir
        ], check=True)
        print(f"  Visualizations saved to: {vis_out_dir}")
    except Exception as e:
        print(f"  Warning: Visualization generation failed: {e}")

    # run_whole_image expects a generic output structure, maybe symlink or rename?
    # For now, just return valid path
    return output_json


def main():
    parser = argparse.ArgumentParser(description='Run DM++ whole-image processing')
    parser.add_argument('--input', type=str, help='Single input image (JP2 or TIFF)')
    parser.add_argument('--input_dir', type=str, help='Directory of input images')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.input:
        # Single image mode
        process_single_image(args.input, args.output)
    elif args.input_dir:
        # Batch mode
        patterns = ['*.jp2', '*.JP2', '*.tif', '*.tiff', '*.TIF', '*.TIFF']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(args.input_dir, pattern)))
        
        print(f"Found {len(files)} images to process")
        for i, f in enumerate(sorted(files)):
            print(f"\n[{i+1}/{len(files)}]")
            process_single_image(f, args.output)
    else:
        print("Error: Must specify --input or --input_dir")
        sys.exit(1)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
