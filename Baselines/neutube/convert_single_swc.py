
import cv2
import pandas as pd
import numpy as np
import sys
import os

def parse_swc(swc_path):
    try:
        df = pd.read_csv(swc_path, sep='\s+', comment='#', names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'], header=None)
    except Exception as e:
        print(f"Error reading SWC {swc_path}: {e}")
        return pd.DataFrame()
    return df

def draw_swc(df, shape):
    img = np.zeros(shape, dtype=np.uint8)
    if df.empty:
        return img
    coords = {row['id']: (int(float(row['x'])), int(float(row['y']))) for _, row in df.iterrows()}
    for _, row in df.iterrows():
        pid = row['parent']
        if pid != -1 and pid in coords:
            p1 = coords[row['id']]
            p2 = coords[pid]
            cv2.line(img, p1, p2, 255, 1)
    return img

if __name__ == "__main__":
    swc_path = sys.argv[1]
    ref_img_path = sys.argv[2]
    out_path = sys.argv[3]
    
    # Read ref image for shape
    ref = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        print(f"Error reading reference image {ref_img_path}")
        sys.exit(1)
        
    df = parse_swc(swc_path)
    overlay = draw_swc(df, ref.shape)
    
    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay to {out_path}")
