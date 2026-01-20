"""
Min-Size Ablation Comparison Tool

Generates qualitative comparison showing:
- "Without Size Filtering" (min_size=0) vs "With Size Filtering" (min_size=optimal)

Uses the same interactive crop selection and color-coded overlay
visualization as evaluate_model (cyan=TP, magenta=FN, yellow=FP).

PMD: persistence=64, optimal min_size=40
STP: persistence=32, optimal min_size=12

Usage:
    python generate_minsize_ablation.py pmd           # Interactive mode for PMD
    python generate_minsize_ablation.py stp           # Interactive mode for STP
    python generate_minsize_ablation.py pmd --regen   # Regenerate PMD from cache
    python generate_minsize_ablation.py stp --regen   # Regenerate STP from cache
    python generate_minsize_ablation.py pmd --4panel  # Generate combined 4-panel figure
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import tifffile

# Base paths - derive from script location
import os
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = SCRIPT_DIR.parent.parent  # Up to WholeBrainProject

# PMD configuration
PMD_CONFIG = {
    'img_dir': BASE_DIR / 'data' / 'pmd' / 'img',
    'gt_dir': BASE_DIR / 'data' / 'pmd' / 'GT',
    'sweep_dir': BASE_DIR / 'outputs' / 'ablation' / 'final_dm2d_output_pmd_min_size_sweep_persistence64',
    'min_size_optimal': 40,
    'output_dir': BASE_DIR / 'outputs' / 'ablation' / 'minsize_ablation_PMD',
    'image_a': 'PMD1211_115_8201_17701',
    'image_b': 'PMD1211_58_5001_12501',
    # Hardcoded crop coordinates
    'crops': {
        "PMD1211_115_8201_17701": {"x1": 519, "y1": 495, "x2": 669, "y2": 645},
        "PMD1211_58_5001_12501": {"x1": 479, "y1": 848, "x2": 629, "y2": 998},
    },
}

# STP configuration  
STP_CONFIG = {
    'img_dir': BASE_DIR / 'data' / 'stp' / 'img',
    'gt_dir': BASE_DIR / 'data' / 'stp' / 'GT',
    'sweep_dir': BASE_DIR / 'outputs' / 'ablation' / 'final_dm2d_output_stp_min_size_sweep_persistence32',
    'min_size_optimal': 12,
    'output_dir': BASE_DIR / 'outputs' / 'ablation' / 'minsize_ablation_STP',
    'image_a': '190322_74_2301_5401',
    'image_b': '190322_59_3301_5601',
    # Hardcoded crop coordinates
    'crops': {
        "190322_74_2301_5401": {"x1": 300, "y1": 400, "x2": 450, "y2": 550},
        "190322_59_3301_5601": {"x1": 100, "y1": 200, "x2": 250, "y2": 350},
    },
}

# Fixed box size
BOX_SIZE = 150


def load_cache(cache_file):
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def read_image(path):
    """Read image from path."""
    path = str(path)
    
    # Try OpenCV first (handles most formats including compressed TIFF)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.dtype == np.uint16:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        return img
    
    # Fallback to tifffile for special cases
    if path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(path)
            img = np.squeeze(img)
            if img.dtype == np.uint16:
                img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            elif img.dtype == bool:
                img = img.astype(np.uint8) * 255
            return img
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            return None
    
    return None


def create_overlay(img, skeleton, gt):
    """Create color-coded overlay showing TP/FP/FN.
    
    Colors (same as evaluate_model):
    - Cyan: True Positives (skeleton matched with GT)
    - Magenta: False Negatives (GT not matched)
    - Yellow: False Positives (skeleton not matched)
    """
    # Convert to RGB
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get binary masks
    if len(skeleton.shape) == 3:
        skel_mask = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY) > 0
    else:
        skel_mask = skeleton > 0
        
    if len(gt.shape) == 3:
        gt_mask = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) > 0
    else:
        gt_mask = gt > 0
    
    # Calculate TP, FP, FN
    tp = skel_mask & gt_mask
    fp = skel_mask & ~gt_mask
    fn = ~skel_mask & gt_mask
    
    # Apply colors
    img_rgb[fn] = [255, 0, 255]   # Magenta for FN
    img_rgb[fp] = [255, 255, 0]   # Yellow for FP
    img_rgb[tp] = [0, 255, 255]   # Cyan for TP
    
    return img_rgb


class MinSizeAblationTool:
    def __init__(self, config, dataset_name):
        self.config = config
        self.dataset_name = dataset_name
        self.output_dir = config['output_dir']
        self.output_dir.mkdir(exist_ok=True)
        self.cache = config.get('crops', {})  # Use hardcoded crops from config
        self.cache_file = self.output_dir / 'crop_cache.json'  # For saving/loading
        
        self.img_dir = config['img_dir']
        self.gt_dir = config['gt_dir']
        self.sweep_dir = config['sweep_dir']
        self.min_size_optimal = config['min_size_optimal']
        
        # Get available images
        self.images = sorted([f.stem for f in self.img_dir.glob('*.tif')])
        self.current_idx = 0
        self.box_center = None
        self.is_dragging = False
        self.gap = 10
        
    def find_resume_index(self):
        """Find first image without cached crop."""
        for idx, name in enumerate(self.images):
            if name not in self.cache:
                return idx
        return len(self.images)
        
    def load_current_images(self):
        """Load current image, GT, and both skeleton variants."""
        name = self.images[self.current_idx]
        self.current_name = name
        
        # Load original and GT
        self.original_img = read_image(self.img_dir / f"{name}.tif")
        self.gt_img = read_image(self.gt_dir / f"{name}.tif")
        
        if len(self.original_img.shape) == 2:
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2BGR)
        if len(self.gt_img.shape) == 2:
            self.gt_img = cv2.cvtColor(self.gt_img, cv2.COLOR_GRAY2BGR)
        
        # Load skeletons: without filtering (min_size=0) and with filtering (optimal)
        skel_path_no_filter = self.sweep_dir / "min_size_0" / "skeleton" / f"{name}.tif"
        skel_path_filtered = self.sweep_dir / f"min_size_{self.min_size_optimal}" / "skeleton" / f"{name}.tif"
        
        self.skel_no_filter = None
        self.skel_filtered = None
        
        if skel_path_no_filter.exists():
            self.skel_no_filter = read_image(skel_path_no_filter)
            if len(self.skel_no_filter.shape) == 2:
                self.skel_no_filter = cv2.cvtColor(self.skel_no_filter, cv2.COLOR_GRAY2BGR)
        
        if skel_path_filtered.exists():
            self.skel_filtered = read_image(skel_path_filtered)
            if len(self.skel_filtered.shape) == 2:
                self.skel_filtered = cv2.cvtColor(self.skel_filtered, cv2.COLOR_GRAY2BGR)
        
        self.img_height, self.single_width = self.original_img.shape[:2]
    
    def create_side_by_side(self):
        """Create side-by-side view: Without Filtering vs With Filtering."""
        h, w = self.img_height, self.single_width
        
        # Create combined image with gap
        combined = np.zeros((h + 60, w * 2 + self.gap, 3), dtype=np.uint8)
        combined[:, :] = 30
        
        # Create overlays
        if self.skel_no_filter is not None:
            overlay_no_filter = create_overlay(
                cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB),
                self.skel_no_filter, 
                self.gt_img
            )
            overlay_no_filter = cv2.cvtColor(overlay_no_filter, cv2.COLOR_RGB2BGR)
            combined[30:30+h, 0:w] = overlay_no_filter
        
        if self.skel_filtered is not None:
            overlay_filtered = create_overlay(
                cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB),
                self.skel_filtered,
                self.gt_img
            )
            overlay_filtered = cv2.cvtColor(overlay_filtered, cv2.COLOR_RGB2BGR)
            combined[30:30+h, w+self.gap:] = overlay_filtered
        
        return combined
    
    def get_box_rect(self, cx, cy):
        """Get box rectangle from center, clamped to image bounds."""
        half = BOX_SIZE // 2
        cx = max(half, min(self.single_width - half, cx))
        cy = max(half, min(self.img_height - half, cy))
        return (cx - half, cy - half, cx + half, cy + half)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        # Map x to left image coordinates
        if x >= self.single_width + self.gap:
            x = x - self.single_width - self.gap
        elif x >= self.single_width:
            return  # In gap
        
        img_y = y - 30
        if img_y < 0 or img_y >= self.img_height:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.box_center = (x, img_y)
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            self.box_center = (x, img_y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
            self.box_center = (x, img_y)
    
    def draw_ui(self, combined):
        """Draw UI overlay."""
        display = combined.copy()
        h, w = display.shape[:2]
        
        # Labels - NO mention of min_size
        cv2.putText(display, "Without Filtering", (10, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        cv2.putText(display, "With Filtering", (self.single_width + self.gap + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # Progress info
        cached = len(self.cache)
        info = f"[{self.current_idx+1}/{len(self.images)}] {self.current_name} (cached: {cached})"
        cv2.putText(display, info, (w - 400, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw box on both images
        if self.box_center:
            x1, y1, x2, y2 = self.get_box_rect(*self.box_center)
            cv2.rectangle(display, (x1, y1 + 30), (x2, y2 + 30), (0, 255, 0), 2)
            cv2.rectangle(display, (x1 + self.single_width + self.gap, y1 + 30), 
                         (x2 + self.single_width + self.gap, y2 + 30), (0, 255, 0), 2)
            cv2.putText(display, f"{BOX_SIZE}x{BOX_SIZE}", (x1, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Instructions
        cv2.rectangle(display, (0, h - 25), (w, h), (40, 40, 40), -1)
        cv2.putText(display, "ENTER: Save & Next | r: Reset | q: Quit | Click/drag to position box",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        
        return display
    
    def run(self, start_fresh=False):
        """Interactive crop selection."""
        print("="*60)
        print(f"MIN-SIZE ABLATION CROP SELECTION ({self.dataset_name.upper()})")
        print("="*60)
        print(f"Found {len(self.images)} images")
        print(f"Optimal min_size: {self.min_size_optimal}")
        print(f"Cache file: {self.cache_file}")
        print(f"Cached: {len(self.cache)}")
        
        if not start_fresh:
            self.current_idx = self.find_resume_index()
            if self.current_idx >= len(self.images):
                print(f"\n✓ All images already have cached crops!")
                print("Use --regen to regenerate figures.")
                return
            if self.current_idx > 0:
                print(f"\n→ Resuming from image {self.current_idx + 1}")
        
        print("\nControls:")
        print("  - Click/drag to position box")
        print("  - ENTER: Save crop and move to next")
        print("  - r: Reset box")
        print("  - q: Quit")
        print("="*60)
        
        cv2.namedWindow(f'Min-Size Ablation {self.dataset_name.upper()}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Min-Size Ablation {self.dataset_name.upper()}', 1600, 900)
        cv2.setMouseCallback(f'Min-Size Ablation {self.dataset_name.upper()}', self.mouse_callback)
        
        self.load_current_images()
        saved_count = 0
        
        while True:
            combined = self.create_side_by_side()
            display = self.draw_ui(combined)
            cv2.imshow(f'Min-Size Ablation {self.dataset_name.upper()}', display)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 13:  # ENTER
                if self.box_center:
                    x1, y1, x2, y2 = self.get_box_rect(*self.box_center)
                    self.cache[self.current_name] = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    }
                    save_cache(self.cache, self.cache_file)
                    print(f"✓ Saved crop for {self.current_name}")
                    saved_count += 1
                    
                    self.current_idx += 1
                    if self.current_idx >= len(self.images):
                        print("\n✓ All images processed!")
                        break
                    self.load_current_images()
                    self.box_center = None
            elif key == ord('r'):
                self.box_center = None
        
        cv2.destroyAllWindows()
        print(f"\nSaved {saved_count} crops. Total cached: {len(self.cache)}")
    
    def generate_figures(self):
        """Generate comparison figures for all cached images."""
        if not self.cache:
            print(f"No cached crops for {self.dataset_name.upper()}. Run interactive mode first.")
            return
        
        print(f"\nGenerating {self.dataset_name.upper()} ablation figures for {len(self.cache)} images...")
        
        for image_name, coords in self.cache.items():
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Load images
            orig = read_image(self.img_dir / f"{image_name}.tif")
            gt = read_image(self.gt_dir / f"{image_name}.tif")
            
            # Load skeletons
            skel_path_no_filter = self.sweep_dir / "min_size_0" / "skeleton" / f"{image_name}.tif"
            skel_path_filtered = self.sweep_dir / f"min_size_{self.min_size_optimal}" / "skeleton" / f"{image_name}.tif"
            
            if not skel_path_no_filter.exists() or not skel_path_filtered.exists():
                print(f"  Skipping {image_name} - missing skeleton files")
                continue
            
            skel_no_filter = read_image(skel_path_no_filter)
            skel_filtered = read_image(skel_path_filtered)
            
            # Crop all images
            orig_crop = orig[y1:y2, x1:x2]
            gt_crop = gt[y1:y2, x1:x2]
            skel_no_filter_crop = skel_no_filter[y1:y2, x1:x2]
            skel_filtered_crop = skel_filtered[y1:y2, x1:x2]
            
            # Create overlays
            overlay_no_filter = create_overlay(orig_crop.copy(), skel_no_filter_crop, gt_crop)
            overlay_filtered = create_overlay(orig_crop.copy(), skel_filtered_crop, gt_crop)
            
            # Create 2-panel figure
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            
            axes[0].imshow(overlay_no_filter)
            axes[0].set_title('Without Filtering', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(overlay_filtered)
            axes[1].set_title('With Filtering', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout(pad=0.5)
            
            output_path = self.output_dir / f'{self.dataset_name}_{image_name}_ablation.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.05,
                        facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✓ {output_path.name}")
        
        print(f"\nOutput saved to: {self.output_dir}")
    
    def generate_4panel(self):
        """Generate combined 4-panel figure with two image pairs."""
        if 'image_a' not in self.config or 'image_b' not in self.config:
            print(f"4-panel not configured for {self.dataset_name.upper()}")
            return
        
        image_a = self.config['image_a']
        image_b = self.config['image_b']
        
        # Check if both images have cached crops
        if image_a not in self.cache or image_b not in self.cache:
            print(f"Missing cached crops. Run interactive mode first.")
            print(f"  {image_a}: {'✓' if image_a in self.cache else '✗'}")
            print(f"  {image_b}: {'✓' if image_b in self.cache else '✗'}")
            return
        
        def get_overlays(image_name):
            coords = self.cache[image_name]
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            orig = read_image(self.img_dir / f"{image_name}.tif")
            gt = read_image(self.gt_dir / f"{image_name}.tif")
            skel_no_filter = read_image(self.sweep_dir / "min_size_0" / "skeleton" / f"{image_name}.tif")
            skel_filtered = read_image(self.sweep_dir / f"min_size_{self.min_size_optimal}" / "skeleton" / f"{image_name}.tif")
            
            orig_crop = orig[y1:y2, x1:x2]
            gt_crop = gt[y1:y2, x1:x2]
            skel_no_filter_crop = skel_no_filter[y1:y2, x1:x2]
            skel_filtered_crop = skel_filtered[y1:y2, x1:x2]
            
            overlay_no_filter = create_overlay(orig_crop.copy(), skel_no_filter_crop, gt_crop)
            overlay_filtered = create_overlay(orig_crop.copy(), skel_filtered_crop, gt_crop)
            
            return overlay_no_filter, overlay_filtered
        
        # Get overlays for both images
        a_no_filter, a_filtered = get_overlays(image_a)
        b_no_filter, b_filtered = get_overlays(image_b)
        
        # Create 4-panel figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        
        # (a) - Without Size Filtering
        axes[0].imshow(a_no_filter)
        axes[0].set_title('Without Size Filtering', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes, fontsize=14, 
                     fontweight='bold', va='top', ha='left', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        # (a) - With Size Filtering
        axes[1].imshow(a_filtered)
        axes[1].set_title('With Size Filtering', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # (b) - Without Size Filtering
        axes[2].imshow(b_no_filter)
        axes[2].set_title('Without Size Filtering', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        axes[2].text(0.02, 0.98, '(b)', transform=axes[2].transAxes, fontsize=14, 
                     fontweight='bold', va='top', ha='left', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        
        # (b) - With Size Filtering
        axes[3].imshow(b_filtered)
        axes[3].set_title('With Size Filtering', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.tight_layout(pad=0.5)
        
        # Add vertical separator line between (a) and (b) groups
        # Position is roughly at 0.5 in figure coordinates
        from matplotlib.lines import Line2D
        line = Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color="black", linewidth=1, linestyle='-')
        fig.add_artist(line)
        
        output_path = self.output_dir / 'minsize_ablation_combined_4panel.png'
        
        # Add black border around the figure
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)
        
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='black')
        plt.close()
        
        print(f"✓ Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_minsize_ablation.py <pmd|stp> [--regen] [--4panel]")
        sys.exit(1)
    
    dataset = sys.argv[1].lower()
    regen = '--regen' in sys.argv
    fourpanel = '--4panel' in sys.argv
    
    if dataset == 'pmd':
        config = PMD_CONFIG
    elif dataset == 'stp':
        config = STP_CONFIG
    else:
        print(f"Unknown dataset: {dataset}. Use 'pmd' or 'stp'.")
        sys.exit(1)
    
    tool = MinSizeAblationTool(config, dataset)
    
    if fourpanel:
        tool.generate_4panel()
    elif regen:
        tool.generate_figures()
    else:
        tool.run()
        tool.generate_figures()


if __name__ == '__main__':
    main()
