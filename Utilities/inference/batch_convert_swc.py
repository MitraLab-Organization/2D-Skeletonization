
import os
import glob
import argparse
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def parse_swc(swc_path):
    try:
        # SWC format: id type x y z r parent
        df = pd.read_csv(swc_path, sep=r'\s+', comment='#', names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
        return df
    except Exception as e:
        print(f"Error parsing {swc_path}: {e}")
        return pd.DataFrame()

def swc_to_image(swc_path, img_shape, out_path):
    df = parse_swc(swc_path)
    img = np.zeros(img_shape, dtype=np.uint8)
    
    if not df.empty:
        # Create a dictionary for node coordinates
        coords = {}
        for _, row in df.iterrows():
            coords[int(row['id'])] = (int(float(row['x'])), int(float(row['y'])))
            
        # Draw lines
        for _, row in df.iterrows():
            pid = int(row['parent'])
            if pid != -1 and pid in coords:
                p1 = coords[int(row['id'])]
                p2 = coords[pid]
                cv2.line(img, p1, p2, 255, 1, lineType=cv2.LINE_AA)
                
    cv2.imwrite(out_path, img)
    return out_path

def process_file(args):
    swc_file, lkl_dir, out_dir = args
    name = os.path.splitext(os.path.basename(swc_file))[0]
    
    # Try to find corresponding LKL file to get dimensions
    lkl_path = os.path.join(lkl_dir, name + ".tif")
    if not os.path.exists(lkl_path):
        lkl_path = os.path.join(lkl_dir, name + ".png")
    if not os.path.exists(lkl_path):
        lkl_path = os.path.join(lkl_dir, name + ".jpg")
        
    if not os.path.exists(lkl_path):
        print(f"Warning: No LKL file found for {name}, skipping")
        return
        
    lkl = cv2.imread(lkl_path, cv2.IMREAD_GRAYSCALE)
    if lkl is None:
        print(f"Warning: Could not read LKL {lkl_path}")
        return
        
    out_path = os.path.join(out_dir, name + ".tif")
    swc_to_image(swc_file, lkl.shape, out_path)
    print(f"Converted {name}.swc -> {name}.tif")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--swc_dir', required=True)
    parser.add_argument('--lkl_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    swc_files = glob.glob(os.path.join(args.swc_dir, "*.swc"))
    tasks = [(f, args.lkl_dir, args.out_dir) for f in swc_files]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(process_file, tasks)

if __name__ == "__main__":
    main()
