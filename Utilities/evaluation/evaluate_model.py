"""
Model vs Ground Truth Comparison

Compares any model's skeleton predictions against GT with:
- Overlay visualizations
- All metrics (Precision, Recall, F-Score, Dice, IoU)
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import pandas as pd
import json
import argparse


def get_skeleton_coords(skeleton):
    """Extract coordinates of skeleton pixels"""
    coords = np.argwhere(skeleton > 0)
    return coords


def compute_point_metrics(pred_coords, gt_coords, distance_threshold=5):

    pred_coords = np.asarray(pred_coords)
    gt_coords = np.asarray(gt_coords)

    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return dict(TP=0, FP=0, FN=0, pred_tp=[], gt_fn=[], pred_fp=[])

    if len(pred_coords) == 0:
        return dict(TP=0, FP=0, FN=len(gt_coords), pred_tp=[], gt_fn=gt_coords.tolist(), pred_fp=[])

    if len(gt_coords) == 0:
        return dict(TP=0, FP=len(pred_coords), FN=0, pred_tp=[], gt_fn=[], pred_fp=pred_coords.tolist())

    gt_tree = cKDTree(gt_coords)
    pred_tree = cKDTree(pred_coords)

    # Pred → GT
    d_pred, _ = gt_tree.query(pred_coords, k=1)
    pred_matched = d_pred < distance_threshold

    # GT → Pred
    d_gt, _ = pred_tree.query(gt_coords, k=1)
    gt_matched = d_gt < distance_threshold

    TP = np.count_nonzero(pred_matched)
    FP = np.count_nonzero(~pred_matched)
    FN = np.count_nonzero(~gt_matched)

    return {'TP': TP,
        'FP': FP,
        'FN': FN,
        'pred_tp': pred_coords[pred_matched].tolist(),
        'gt_fn': gt_coords[~gt_matched].tolist(),
        'pred_fp': pred_coords[~pred_matched].tolist()
    }





def calculate_scores(tp, fp, fn):
    """Calculate all metrics"""
    epsilon = 1e-10
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f_score = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = (2 * tp) / (2 * tp + fn + fp + epsilon)
    iou = tp / (tp + fn + fp + epsilon)
    
    return {
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'dice': dice,
        'iou': iou,
        'TP': tp,
        'FP': fp,
        'FN': fn
    }


def create_overlay_image(img, pred_tp, gt_fn, pred_fp, dilate_size=1):
    """
    Create overlay visualization
    - Cyan: True Positives (GT matched with prediction)
    - Magenta: False Negatives (GT not matched)
    - Yellow: False Positives (Prediction not matched)
    """
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create masks
    mask_shape = img.shape[:2]
    mask_tp = np.zeros(mask_shape, dtype=bool)
    mask_fn = np.zeros(mask_shape, dtype=bool)
    mask_fp = np.zeros(mask_shape, dtype=bool)
    
    # Fill masks
    for pt in pred_tp:
        mask_tp[pt[0], pt[1]] = True
    for pt in gt_fn:
        mask_fn[pt[0], pt[1]] = True
    for pt in pred_fp:
        mask_fp[pt[0], pt[1]] = True
    
    # Dilate masks for visibility
    if dilate_size > 0:
        struct_elem = np.ones((3, 3), dtype=bool)
        for _ in range(dilate_size):
            mask_tp = binary_dilation(mask_tp, struct_elem)
            mask_fn = binary_dilation(mask_fn, struct_elem)
            mask_fp = binary_dilation(mask_fp, struct_elem)
    
    # Apply colors
    img_rgb[mask_fn] = [255, 0, 255]  # Magenta for FN
    img_rgb[mask_fp] = [255, 255, 0]  # Yellow for FP
    img_rgb[mask_tp] = [0, 255, 255]  # Cyan for TP
    
    return img_rgb


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model skeleton predictions against ground truth'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model predictions')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model (for logging/output)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth skeletons')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {model_dir}_evaluation)')
    parser.add_argument('--distance_threshold', type=int, default=5,
                        help='Distance threshold for point matching (default: 5)')
    
    args = parser.parse_args()
    
    # Paths
    model_dir = Path(args.model_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir.parent / f"{model_dir.name}_evaluation"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    overlay_dir = output_dir / 'overlays'
    overlay_dir.mkdir(exist_ok=True)
    
    # Get all GT files
    gt_files = sorted(gt_dir.glob('*.tif'))
    
    results = []
    
    print("="*70)
    print(f"{args.model_name.upper()} vs GROUND TRUTH EVALUATION")
    print("="*70)
    print(f"\nModel Directory: {model_dir}")
    print(f"GT Directory:    {gt_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Distance Threshold: {args.distance_threshold} pixels")
    print(f"\nProcessing {len(gt_files)} images...\n")
    
    for idx, gt_file in enumerate(gt_files):
        filename = gt_file.name
        print(f"[{idx+1}/{len(gt_files)}] {filename}")
        
        # Check if files exist - try multiple patterns
        base_name = gt_file.stem  # filename without extension
        model_file = None
        
        # Try different file patterns
        patterns = [
            model_dir / filename,  # exact match
            model_dir / f"{base_name}_skeleton.png",  # _skeleton.png suffix
            model_dir / f"{base_name}_skeleton.tif",  # _skeleton.tif suffix
            model_dir / f"{base_name}.png",  # same name, different extension
        ]
        
        for pattern in patterns:
            if pattern.exists():
                model_file = pattern
                break
        
        img_file = img_dir / filename
        
        if model_file is None:
            print(f"  WARNING: Model prediction not found (tried multiple patterns), skipping")
            continue
        if not img_file.exists():
            print(f"  WARNING: Image file not found, skipping")
            continue
        
        # Load images
        gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE) > 0
        model_pred = cv2.imread(str(model_file), cv2.IMREAD_GRAYSCALE) > 0
        
        # Read original image - handle 16-bit normalization if needed
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint16:
            # Normalize 16-bit to 8-bit for overlay visualization
            img = ((img / img.max()) * 255).astype(np.uint8)
        
        # Ensure RGB 
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            pass # Already BGR (cv2 default)
            
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}")
        
        # Check dimensions match
        if gt.shape != model_pred.shape:
            print(f"  ERROR: Dimension mismatch - GT: {gt.shape}, Model: {model_pred.shape}")
            print(f"  Skipping this image")
            continue
        
        # Get coordinates
        gt_coords = get_skeleton_coords(gt)
        model_coords = get_skeleton_coords(model_pred)
        
        print(f"  GT coords: {len(gt_coords)}, {args.model_name} coords: {len(model_coords)}")
        
        # Compute metrics
        metrics = compute_point_metrics(model_coords, gt_coords, distance_threshold=args.distance_threshold)
        scores = calculate_scores(metrics['TP'], metrics['FP'], metrics['FN'])
        scores['filename'] = filename
        scores['gt_pixels'] = len(gt_coords)
        scores['model_pixels'] = len(model_coords)
        
        print(f"  TP: {scores['TP']}, FP: {scores['FP']}, FN: {scores['FN']}")
        print(f"  Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F-Score: {scores['f_score']:.4f}")
        
        results.append(scores)
        
        # Create overlay
        overlay = create_overlay_image(
            img,
            metrics['pred_tp'],
            metrics['gt_fn'],
            metrics['pred_fp']
        )
        
        # Save overlay
        cv2.imwrite(str(overlay_dir / filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Check if we have any results
    if len(results) == 0:
        print("\n" + "="*70)
        print("ERROR: No images were successfully processed!")
        print("="*70)
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate overall metrics
    total_tp = df['TP'].sum()
    total_fp = df['FP'].sum()
    total_fn = df['FN'].sum()
    
    overall = calculate_scores(total_tp, total_fp, total_fn)
    
    print("\n" + "="*70)
    print(f"{args.model_name.upper()} - OVERALL RESULTS")
    print("="*70)
    print(f"Precision:  {overall['precision']:.4f} ({overall['precision']*100:.2f}%)")
    print(f"Recall:     {overall['recall']:.4f} ({overall['recall']*100:.2f}%)")
    print(f"F-Score:    {overall['f_score']:.4f} ({overall['f_score']*100:.2f}%)")
    print(f"Dice:       {overall['dice']:.4f} ({overall['dice']*100:.2f}%)")
    print(f"IoU:        {overall['iou']:.4f} ({overall['iou']*100:.2f}%)")
    print(f"\nTrue Positives:  {total_tp:,}")
    print(f"False Positives: {total_fp:,}")
    print(f"False Negatives: {total_fn:,}")
    
    # Save results
    df.to_csv(output_dir / 'per_image_metrics.csv', index=False)
    
    summary = {
        'model_name': args.model_name,
        'model_directory': str(model_dir),
        'overall_metrics': {
            'precision': float(overall['precision']),
            'recall': float(overall['recall']),
            'f_score': float(overall['f_score']),
            'dice': float(overall['dice']),
            'iou': float(overall['iou']),
            'TP': int(overall['TP']),
            'FP': int(overall['FP']),
            'FN': int(overall['FN'])
        },
        'total_images': len(results),
        'distance_threshold': args.distance_threshold,
        'color_coding': {
            'cyan': 'True Positives (correctly detected)',
            'magenta': 'False Negatives (GT missed)',
            'yellow': 'False Positives (incorrect detections)'
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n" + "="*70)
    print("RESULTS SAVED")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"- per_image_metrics.csv: Per-image detailed metrics")
    print(f"- summary.json: Overall metrics summary")
    print(f"- overlays/: Visualization images")
    print("="*70)


if __name__ == '__main__':
    main()
