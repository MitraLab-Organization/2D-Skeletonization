"""
Interactive Circle Annotation Tool for Haircut Comparison

Click on images to place green circles highlighting areas of interest.
Circles are saved to a cache for regeneration.

Usage:
    python generate_haircut_4panel.py           # Interactive mode
    python generate_haircut_4panel.py --regen   # Regenerate from cache

Controls:
    - Click to place a green circle
    - Right-click to remove last circle
    - ENTER to save and quit
    - 'c' to clear all circles
    - 'q' to quit without saving
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
from skimage import morphology

# Paths - derive from script location
import os
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = SCRIPT_DIR.parent.parent  # Up to WholeBrainProject
NO_HAIRCUT_DIR = BASE_DIR / 'outputs' / 'no_haircut_skeletons'
HAIRCUT_DIR = BASE_DIR / 'outputs' / 'haircut_skeletons'
IMG_DIR = BASE_DIR / 'data' / 'pmd' / 'img'
OUTPUT_DIR = BASE_DIR / 'outputs' / 'haircut_comparison_plots'
CACHE_FILE = OUTPUT_DIR / 'crop_coordinates.json'
CIRCLES_CACHE_FILE = OUTPUT_DIR / 'circle_annotations.json'

# Skeleton overlay color (cyan)
SKELETON_COLOR_RGB = (0, 255, 255)

# Circle settings
CIRCLE_COLOR_BGR = (0, 255, 0)  # Green in BGR
CIRCLE_COLOR_RGB = (0, 255, 0)  # Green in RGB
CIRCLE_RADIUS = 5
CIRCLE_THICKNESS = 1

# Images to use
IMAGE_A = 'PMD1211_48_8601_16001'   # 8601_16001
IMAGE_B = 'PMD1211_58_5001_12501'   # 5001_12501

# Hardcoded crop coordinates
CROP_COORDS = {
    "PMD1211_48_8601_16001": {"x1": 531, "y1": 125, "x2": 611, "y2": 205},
    "PMD1211_58_5001_12501": {"x1": 757, "y1": 821, "x2": 837, "y2": 901},
}

# Hardcoded circle annotations
CIRCLE_ANNOTATIONS = {'a': [], 'b': []}


def load_cache():
    return CROP_COORDS


def load_circles_cache():
    if CIRCLES_CACHE_FILE.exists():
        with open(CIRCLES_CACHE_FILE, 'r') as f:
            return json.load(f)
    # Only store circles for 'a' and 'b' - they apply to both no_haircut and haircut
    return {'a': [], 'b': []}


def save_circles_cache(circles):
    with open(CIRCLES_CACHE_FILE, 'w') as f:
        json.dump(circles, f, indent=2)


def create_overlay(original_crop, skeleton_crop):
    """Create overlay with 1-pixel wide skeleton."""
    if len(original_crop.shape) == 2:
        overlay = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2RGB)
    else:
        overlay = cv2.cvtColor(original_crop, cv2.COLOR_BGR2RGB)
    
    if len(skeleton_crop.shape) == 3:
        skeleton_gray = cv2.cvtColor(skeleton_crop, cv2.COLOR_BGR2GRAY)
    else:
        skeleton_gray = skeleton_crop
    
    skeleton_mask = skeleton_gray > 0
    skeleton_mask = morphology.skeletonize(skeleton_mask)
    skeleton_mask = morphology.remove_small_objects(skeleton_mask, min_size=40, connectivity=2)
    
    overlay[skeleton_mask] = SKELETON_COLOR_RGB
    return overlay


def load_and_crop(name, cache):
    """Load images and crop according to cache."""
    coords = cache[name]
    x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
    
    no_haircut_img = cv2.imread(str(NO_HAIRCUT_DIR / f"{name}.tif"), cv2.IMREAD_UNCHANGED)
    haircut_img = cv2.imread(str(HAIRCUT_DIR / f"{name}.tif"), cv2.IMREAD_UNCHANGED)
    original_img = cv2.imread(str(IMG_DIR / f"{name}.tif"), cv2.IMREAD_UNCHANGED)
    
    # Crop
    no_haircut_crop = no_haircut_img[y1:y2, x1:x2]
    haircut_crop = haircut_img[y1:y2, x1:x2]
    original_crop = original_img[y1:y2, x1:x2]
    
    # Create overlays
    no_haircut_overlay = create_overlay(original_crop.copy(), no_haircut_crop)
    haircut_overlay = create_overlay(original_crop.copy(), haircut_crop)
    
    return no_haircut_overlay, haircut_overlay


def draw_circles_on_image(img, circles):
    """Draw green circles on the image."""
    result = img.copy()
    for (cx, cy) in circles:
        cv2.circle(result, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR_RGB, CIRCLE_THICKNESS)
    return result


class CircleAnnotationTool:
    def __init__(self, images, circles_cache):
        """
        images: dict with keys 'a_no_haircut', 'a_haircut', 'b_no_haircut', 'b_haircut'
        """
        self.images = images
        self.circles = circles_cache
        self.img_size = images['a_no_haircut'].shape[0]  # Assuming square
        self.gap = 10
        
    def create_combined_view(self):
        """Create a horizontal strip of all 4 images."""
        h = self.img_size
        w = self.img_size
        total_w = 4 * w + 3 * self.gap
        
        combined = np.zeros((h + 40, total_w, 3), dtype=np.uint8)
        combined[:, :] = 40  # Dark gray background
        
        # Draw each image with circles (circles are synced between pairs)
        keys = ['a_no_haircut', 'a_haircut', 'b_no_haircut', 'b_haircut']
        circle_keys = ['a', 'a', 'b', 'b']  # Map to shared circle lists
        
        for i, (img_key, circle_key) in enumerate(zip(keys, circle_keys)):
            img_with_circles = draw_circles_on_image(self.images[img_key], self.circles[circle_key])
            # Convert RGB to BGR for OpenCV display
            img_bgr = cv2.cvtColor(img_with_circles, cv2.COLOR_RGB2BGR)
            x_offset = i * (w + self.gap)
            combined[30:30+h, x_offset:x_offset+w] = img_bgr
        
        # Draw labels
        labels = ['(a) Without', '(a) With', '(b) Without', '(b) With']
        for i, label in enumerate(labels):
            x_offset = i * (w + self.gap) + 5
            cv2.putText(combined, label, (x_offset, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return combined
    
    def get_image_index(self, x):
        """Get which image (0-3) was clicked based on x coordinate."""
        w = self.img_size
        for i in range(4):
            x_start = i * (w + self.gap)
            x_end = x_start + w
            if x_start <= x < x_end:
                return i, x - x_start
        return None, None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        img_y = y - 30
        if img_y < 0 or img_y >= self.img_size:
            return
        
        img_idx, local_x = self.get_image_index(x)
        if img_idx is None:
            return
        
        # Map image index to circle key (0,1 -> 'a', 2,3 -> 'b')
        circle_key = 'a' if img_idx < 2 else 'b'
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add circle (applies to both no_haircut and haircut)
            self.circles[circle_key].append((local_x, img_y))
            print(f"Added circle at ({local_x}, {img_y}) for pair {circle_key}")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last circle
            if self.circles[circle_key]:
                removed = self.circles[circle_key].pop()
                print(f"Removed circle at {removed} from pair {circle_key}")
    
    def run(self):
        """Interactive annotation loop."""
        print("="*60)
        print("CIRCLE ANNOTATION TOOL")
        print("="*60)
        print("Controls:")
        print("  - Left-click: Add a green circle")
        print("  - Right-click: Remove last circle")
        print("  - 'c': Clear all circles")
        print("  - ENTER: Save and generate figure")
        print("  - 'q': Quit without saving")
        print("="*60)
        
        cv2.namedWindow('Annotate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotate', 1200, 200)
        cv2.setMouseCallback('Annotate', self.mouse_callback)
        
        while True:
            combined = self.create_combined_view()
            cv2.imshow('Annotate', combined)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return False
            elif key == 13:  # ENTER
                save_circles_cache(self.circles)
                print("Saved circles to cache.")
                cv2.destroyAllWindows()
                return True
            elif key == ord('c'):
                for k in self.circles:
                    self.circles[k] = []
                print("Cleared all circles.")
        
        cv2.destroyAllWindows()
        return False


def generate_figure(images, circles):
    """Generate the final 4-panel figure with circles using matplotlib (vector graphics)."""
    from matplotlib.patches import Circle
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    
    keys = ['a_no_haircut', 'a_haircut', 'b_no_haircut', 'b_haircut']
    circle_keys = ['a', 'a', 'b', 'b']  # Circles are synced between pairs
    titles = ['Without Haircut', 'With Haircut', 'Without Haircut', 'With Haircut']
    labels = ['(a)', None, '(b)', None]
    
    for i, (img_key, circle_key, title, label) in enumerate(zip(keys, circle_keys, titles, labels)):
        # Display image without circles (circles drawn as vector graphics)
        axes[i].imshow(images[img_key])
        
        # Draw circles using matplotlib patches (vector graphics - stays sharp at any resolution)
        for (cx, cy) in circles[circle_key]:
            circle = Circle((cx, cy), CIRCLE_RADIUS, fill=False, 
                           edgecolor='lime', linewidth=1.0)
            axes[i].add_patch(circle)
        
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')
        
        if label:
            axes[i].text(0.02, 0.98, label, transform=axes[i].transAxes, fontsize=12, 
                         fontweight='bold', va='top', ha='left', color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.tight_layout(pad=0.5)
    
    # Add vertical separator line between (a) and (b) groups
    from matplotlib.lines import Line2D
    line = Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, color="black", linewidth=1, linestyle='-')
    fig.add_artist(line)
    
    output_path = OUTPUT_DIR / 'haircut_comparison_4panel.png'
    
    # Add black border around the figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='black')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    cache = load_cache()
    
    # Load both image pairs
    a_no_haircut, a_haircut = load_and_crop(IMAGE_A, cache)
    b_no_haircut, b_haircut = load_and_crop(IMAGE_B, cache)
    
    images = {
        'a_no_haircut': a_no_haircut,
        'a_haircut': a_haircut,
        'b_no_haircut': b_no_haircut,
        'b_haircut': b_haircut
    }
    
    if '--regen' in sys.argv:
        # Just regenerate from cached circles
        circles = load_circles_cache()
        generate_figure(images, circles)
    else:
        # Interactive mode
        circles = load_circles_cache()
        tool = CircleAnnotationTool(images, circles)
        if tool.run():
            circles = load_circles_cache()
            generate_figure(images, circles)


if __name__ == '__main__':
    main()
