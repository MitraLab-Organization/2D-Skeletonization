import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle, ConnectionPatch
import tifffile
import numpy as np

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject

DATASET_CONFIG = {
    "pmd": {
        "data_dir": f"{BASE_DIR}/data/pmd",
        "outputs_dir": f"{BASE_DIR}/outputs/pmd",
        "output_plot": f"{BASE_DIR}/results/figures/comparison_subplot_pmd.png",
        "crop_output_dir": f"{BASE_DIR}/results/crops/pmd",
        "image_filters": ['12501', '15701', '15501'],
        "crops": {
            "PMD1211_58_5001_12501.tif": {"x": 173, "y": 402, "w": 289, "h": 289},
            "PMD1211_156_10401_15701.tif": {"x": 419, "y": 535, "w": 196, "h": 196},
            "PMD1211_48_8601_16001.tif": {"x": 200, "y": 200, "w": 300, "h": 300},
            "PMD1211_156_11301_15501.tif": {"x": 124, "y": 189, "w": 300, "h": 300},
        },
    },
    "stp": {
        "data_dir": f"{BASE_DIR}/data/stp",
        "outputs_dir": f"{BASE_DIR}/outputs/stp",
        "output_plot": f"{BASE_DIR}/results/figures/comparison_subplot_stp.png",
        "crop_output_dir": f"{BASE_DIR}/results/crops/stp",
        "image_filters": ['5401', '5601', '3301_5601'],
        "crops": {
            "190322_74_2301_5401.tif": {"x": 345, "y": 433, "w": 129, "h": 129},
            "190322_80_2401_5601.tif": {"x": 307, "y": 139, "w": 186, "h": 186},
            "190322_59_2501_4431.tif": {"x": 701, "y": 100, "w": 169, "h": 169},
            "190322_59_3301_5601.tif": {"x": 5, "y": 277, "w": 142, "h": 142},
        },
    }
}

def get_rows(data_dir, outputs_dir):
    return [
        ("Original", os.path.join(data_dir, 'img')),
        ("bwskel", os.path.join(outputs_dir, 'bwskel_evaluation', 'overlays')),
        ("diff. skel", os.path.join(outputs_dir, 'diffskel_evaluation', 'overlays')),
        ("neuTube", os.path.join(outputs_dir, 'neutube_evaluation', 'overlays')),
        ("VESS", os.path.join(outputs_dir, 'vess_evaluation', 'overlays')),
        ("PHDF", os.path.join(outputs_dir, 'phd_evaluation', 'overlays')),
        ("DM2D", os.path.join(outputs_dir, 'dm2d_vectorized_evaluation', 'overlays')),
    ]

def read_image(path):
    try:
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
            img = np.squeeze(img)
            if img.dtype == bool:
                img = img.astype(np.uint8) * 255
            elif img.dtype == np.uint16:
                mi, ma = img.min(), img.max()
                img = ((img - mi) / (ma - mi) * 255.0).astype(np.uint8) if ma > mi else np.zeros_like(img, dtype=np.uint8)
            elif np.issubdtype(img.dtype, np.floating):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    mi, ma = img.min(), img.max()
                    img = ((img - mi) / (ma - mi) * 255.0).astype(np.uint8) if ma > mi else img.astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in [3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 2:
                return Image.fromarray(img).convert("RGB")
            return Image.fromarray(img)
        return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def find_image_file(directory, filename_stem):
    stem = os.path.splitext(filename_stem)[0]
    for cand in [filename_stem, f"{stem}.png", f"{stem}.tif", f"{stem}.tiff"]:
        full = os.path.join(directory, cand)
        if os.path.exists(full):
            return full
    return None

def get_cropped_image(directory, filename, crop_coords):
    img_path = find_image_file(directory, filename)
    if not img_path:
        return None
    img = read_image(img_path)
    if img is None:
        return None
    x, y, w, h = crop_coords['x'], crop_coords['y'], crop_coords['w'], crop_coords['h']
    return img.crop((x, y, x + w, y + h))

def main():
    parser = argparse.ArgumentParser(description='Generate comparison subplot figure')
    parser.add_argument('--dataset', '-d', choices=['pmd', 'stp'], required=True,
                        help='Dataset: pmd or stp')
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    ROWS = get_rows(cfg["data_dir"], cfg["outputs_dir"])
    
    crops = cfg["crops"]  # Use hardcoded crops
    keys = list(crops.keys())

    # Select images based on filters
    selected_images = []
    for f_str in cfg["image_filters"]:
        match = next((k for k in keys if f_str in k), None)
        if match:
            selected_images.append(match)
    
    if len(selected_images) < 3:
        remaining = [k for k in keys if k not in selected_images]
        selected_images.extend(remaining[:3 - len(selected_images)])

    if not selected_images:
        print("No suitable images found in crop log.")
        return

    img_names = selected_images[:3]
    print(f"Selected Images: {img_names}")

    os.makedirs(cfg["crop_output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_plot"]), exist_ok=True)

    num_rows = len(ROWS)
    fig = plt.figure(figsize=(22, 27))
    gs = fig.add_gridspec(num_rows, 6, width_ratios=[1]*6, wspace=0.02, hspace=0.02)
    headers = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for r, (label, path) in enumerate(ROWS):
        for col_pair, img_name in enumerate(img_names):
            if img_name not in crops:
                continue
            
            # Full image column
            ax_full = fig.add_subplot(gs[r, col_pair * 2])
            full_img = read_image(find_image_file(path, img_name)) if find_image_file(path, img_name) else None
            ax_full.imshow(full_img if full_img else np.zeros((10, 10)), cmap='gray' if 'orig' in label.lower() else None)
            ax_full.set_xticks([]); ax_full.set_yticks([])
            if col_pair == 0:
                ax_full.set_ylabel(label, fontsize=24, fontweight='bold')
            if r == 0:
                ax_full.set_title(headers[col_pair * 2], fontsize=24, fontweight='bold', pad=20)

            # Crop column
            ax_crop = fig.add_subplot(gs[r, col_pair * 2 + 1])
            crop_img = get_cropped_image(path, img_name, crops[img_name])
            ax_crop.imshow(crop_img if crop_img else np.zeros((10, 10)), cmap='gray' if 'orig' in label.lower() else None)
            ax_crop.set_xticks([]); ax_crop.set_yticks([])
            if r == 0:
                ax_crop.set_title(headers[col_pair * 2 + 1], fontsize=24, fontweight='bold', pad=20)

            if crop_img:
                crop_filename = f"{label.replace(' ', '_').replace('.', '')}_{img_name.replace('.tif', '')}_crop{col_pair+1}.png"
                crop_img.save(os.path.join(cfg["crop_output_dir"], crop_filename))

            if full_img and crop_img:
                c = crops[img_name]
                ax_full.add_patch(Rectangle((c['x'], c['y']), c['w'], c['h'],
                                            linewidth=2, edgecolor='lime', facecolor='none'))
                ax_crop.add_artist(ConnectionPatch(
                    xyA=(c['x'] + c['w'], c['y']), xyB=(0, 1),
                    coordsA="data", coordsB="axes fraction", axesA=ax_full, axesB=ax_crop, color='lime', linewidth=2))
                ax_crop.add_artist(ConnectionPatch(
                    xyA=(c['x'] + c['w'], c['y'] + c['h']), xyB=(0, 0),
                    coordsA="data", coordsB="axes fraction", axesA=ax_full, axesB=ax_crop, color='lime', linewidth=2))

    plt.savefig(cfg["output_plot"], dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved: {cfg['output_plot']}")

if __name__ == "__main__":
    main()
