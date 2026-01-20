#!/bin/bash
#
# Final DM2D Persistence Sweep - STP Dataset
# Varying Persistence Threshold (ET), fixing VE=0
# min_size=12 (Optimal for STP)
#

set -e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BASE_PROJECT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

LKL_DIR="$BASE_PROJECT_DIR/data/stp/lkl"
GT_DIR="$BASE_PROJECT_DIR/data/stp/GT"
IMG_DIR="$BASE_PROJECT_DIR/data/stp/img"
BASE_DIR="$BASE_PROJECT_DIR/outputs/ablation/final_dm2d_output_stp_persistence_sweep"
MIN_SIZE=12

# "Persistence Threshold" in the paper corresponds to ET threshold (with VE fixed at 0)
PERSISTENCE_VALUES=(0 2 4 8 16 32 64 128)
FIXED_VE=0

RESULTS_FILE="$BASE_DIR/persistence_sweep.csv"
mkdir -p "$BASE_DIR"

run_config() {
    local PERSISTENCE=$1 WORK_DIR=$2 NAME=$3
    
    # Check if run already exists
    if [ -f "$WORK_DIR/evaluation/summary.json" ]; then
        return 0
    fi

    mkdir -p "$WORK_DIR/skeleton"
    
    # We map "Persistence Threshold" to et_persistence, and keep ve_persistence=0
    python $BASE_PROJECT_DIR/DM2D_Skeletonization_Vectorization/run_dm2d_tiles.py --lkl_dir "$LKL_DIR" --output_dir "$WORK_DIR/dm2d" \
        --ve_persistence "$FIXED_VE" --et_persistence "$PERSISTENCE"
    
    mkdir -p "$WORK_DIR/vec/CC" "$WORK_DIR/vec/JSON" "$WORK_DIR/mask"
    for f in "$WORK_DIR/dm2d/skeleton"/*.tif; do [ -f "$f" ] && cp "$f" "$WORK_DIR/mask/$(basename "$f" .tif).jpg"; done
    
    python $BASE_PROJECT_DIR/DM2D_Skeletonization_Vectorization/Vectorization/script_to_run.py \
        --input_dir "$WORK_DIR/mask" --output_cc_dir "$WORK_DIR/vec/CC" \
        --output_json_dir "$WORK_DIR/vec/JSON" --min_size "$MIN_SIZE"
    
    python -c "
import os, cv2, json, numpy as np
from pathlib import Path
for jf in sorted(Path('$WORK_DIR/vec/JSON').glob('*.json')):
    lkl = cv2.imread(str(Path('$LKL_DIR')/f'{jf.stem}.tif'), cv2.IMREAD_GRAYSCALE)
    shape = lkl.shape if lkl is not None else (1000,1000)
    with open(jf) as f: gj = json.load(f)
    skel = np.zeros(shape, dtype=np.uint8)
    for feat in gj.get('features', []):
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [])
        if geom.get('type') == 'LineString' and len(coords) >= 2:
            cv2.line(skel, (int(coords[0][0]), int(-coords[0][1])), (int(coords[1][0]), int(-coords[1][1])), 255, 1)
        elif geom.get('type') == 'MultiLineString':
            for seg in coords:
                if len(seg) >= 2: cv2.line(skel, (int(seg[0][0]), int(-seg[0][1])), (int(seg[1][0]), int(-seg[1][1])), 255, 1)
    cv2.imwrite('$WORK_DIR/skeleton/' + jf.stem + '.tif', skel)
"
    rm -rf "$WORK_DIR/mask"
    
    python $BASE_PROJECT_DIR/scripts/evaluation/evaluate_model.py --model_dir "$WORK_DIR/skeleton" --model_name "$NAME" \
        --gt_dir "$GT_DIR" --img_dir "$IMG_DIR" --output_dir "$WORK_DIR/evaluation" 2>/dev/null
}

echo "persistence_threshold,precision,recall,f_score,iou" > "$RESULTS_FILE"

for P in "${PERSISTENCE_VALUES[@]}"; do
    # Folder name reflecting "persistence" concept
    WORK_DIR="$BASE_DIR/persistence_${P}"
    run_config "$P" "$WORK_DIR" "DM2D_Persistence_${P}"
    python -c "import json; m=json.load(open('$WORK_DIR/evaluation/summary.json'))['overall_metrics']; print(f'$P,{m[\"precision\"]},{m[\"recall\"]},{m[\"f_score\"]},{m[\"iou\"]}')" >> "$RESULTS_FILE"
done
