#!/usr/bin/env python3
"""
Batch Skeletonization Script

Processes a folder of 8-bit grayscale images through the gradient-based
skeletonization pipeline. Supports both 2D images and 3D volumes.

Usage:
    python batch_process.py --input_folder <dir> --output_folder <dir> [options]

Examples:
    # Basic skeletonization
    python batch_process.py --input_folder ./images --output_folder ./skeletons

    # Probabilistic mode (for noisy images)
    python batch_process.py --input_folder ./images --output_folder ./skeletons \\
        --probabilistic --beta 0.33 --tau 1.0

    # Multi-step averaging (gradient-based optimization)
    python batch_process.py --input_folder ./images --output_folder ./skeletons \\
        --probabilistic --beta 0.33 --multi_step 20

Input:  8-bit grayscale images (PNG, JPG, TIFF) or 3D volumes (NPY)
Output: Skeleton images with '_skeleton' suffix (PNG for 2D, NPY for 3D)
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import imageio
import numpy as np
import torch
from PIL import Image

from skeletonize import Skeletonize


def process_2d_image(img_path, output_path, args):
    """Process a single 2D image."""
    # Read image and normalize to [0, 1]
    # Use PIL for TIFF images to handle various compression formats
    if img_path.suffix.lower() in ['.tif', '.tiff']:
        img = np.array(Image.open(img_path))
    else:
        img = imageio.imread(img_path)
    if img.dtype == np.uint8:
        img = img / 255.0
    
    # Convert to tensor with shape [1, 1, H, W]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Initialize skeletonization module
    skeletonization_module = Skeletonize(
        probabilistic=args.probabilistic,
        beta=args.beta,
        tau=args.tau,
        simple_point_detection=args.simple_point_detection,
        num_iter=args.num_iter
    ).to(device)
    
    if args.multi_step > 1:
        # Apply skeletonization multiple times and average
        skeleton_stack = np.zeros_like(img.squeeze())
        for step in range(args.multi_step):
            skeleton_stack = skeleton_stack + skeletonization_module(img).cpu().numpy().squeeze()
        skeleton = (skeleton_stack / args.multi_step).round()
    else:
        # Single pass skeletonization
        skeleton = skeletonization_module(img)
        skeleton = skeleton.cpu().numpy().squeeze()
    
    # Convert back to uint8 and save
    skeleton = (skeleton * 255).astype(np.uint8)
    imageio.imwrite(output_path, skeleton)


def process_3d_volume(img_path, output_path, args):
    """Process a single 3D volume (.npy file)."""
    # Load 3D volume
    img = np.load(img_path)
    
    # Normalize if needed
    if img.dtype == np.uint8:
        img = img / 255.0
    
    # Convert to tensor with shape [1, 1, D, H, W]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Initialize skeletonization module
    skeletonization_module = Skeletonize(
        probabilistic=args.probabilistic,
        beta=args.beta,
        tau=args.tau,
        simple_point_detection=args.simple_point_detection,
        num_iter=args.num_iter
    ).to(device)
    
    # Process volume
    skeleton = skeletonization_module(img)
    skeleton = skeleton.cpu().numpy().squeeze()
    
    # Save as .npy file
    np.save(output_path, skeleton)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process grayscale images through skeletonization pipeline'
    )
    
    # I/O arguments
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing input images')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to folder for output skeletons')
    
    # Skeletonization parameters
    parser.add_argument('--probabilistic', action='store_true',
                        help='Use probabilistic skeletonization (default: False)')
    parser.add_argument('--beta', type=float, default=0.33,
                        help='Beta parameter for probabilistic mode (default: 0.33)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Tau parameter for probabilistic mode (default: 1.0)')
    parser.add_argument('--simple_point_detection', type=str, default='Boolean',
                        choices=['Boolean', 'Euler'],
                        help='Simple point detection method (default: Boolean)')
    parser.add_argument('--num_iter', type=int, default=10,
                        help='Number of iterations for 3D volumes (default: 10)')
    
    # Multi-step processing
    parser.add_argument('--multi_step', type=int, default=1,
                        help='Apply skeletonization multiple times and average (default: 1)')
    
    # File filtering
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'npy'],
                        help='File extensions to process (default: png jpg jpeg tif tiff npy)')
    
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from input folder
    input_folder = Path(args.input_folder)
    image_files = []
    for ext in args.extensions:
        image_files.extend(list(input_folder.glob(f'*.{ext}')))
        image_files.extend(list(input_folder.glob(f'*.{ext.upper()}')))
    
    if len(image_files) == 0:
        print(f"No images found in {input_folder} with extensions {args.extensions}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Configuration:")
    print(f"  Probabilistic: {args.probabilistic}")
    if args.probabilistic:
        print(f"  Beta: {args.beta}")
        print(f"  Tau: {args.tau}")
    print(f"  Simple point detection: {args.simple_point_detection}")
    print(f"  Num iterations: {args.num_iter}")
    if args.multi_step > 1:
        print(f"  Multi-step averaging: {args.multi_step} steps")
    print()
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Determine output filename
            if img_path.suffix.lower() == '.npy':
                output_filename = f"{img_path.stem}_skeleton.npy"
                output_path = output_folder / output_filename
                process_3d_volume(img_path, output_path, args)
            else:
                output_filename = f"{img_path.stem}_skeleton.png"
                output_path = output_folder / output_filename
                process_2d_image(img_path, output_path, args)
                
        except Exception as e:
            print(f"\nError processing {img_path.name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Results saved to: {output_folder}")


if __name__ == '__main__':
    main()
