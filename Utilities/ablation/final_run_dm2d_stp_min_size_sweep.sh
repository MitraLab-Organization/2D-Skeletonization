#!/bin/bash
#
# Final DM2D Min Size Sweep - STP Dataset
# Fixed Persistence Threshold, Sweep min_size
#
# Usage: 
#   ./final_run_dm2d_stp_min_size_sweep.sh
#
# Environment variables (optional):
#   DATA_DIR  - Base directory for input data (default: ./data)
#   OUTPUT_DIR - Base directory for outputs (default: ./outputs)
#

set -e

# Use environment variables with sensible defaults (relative to project root)
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

LKL_DIR="${DATA_DIR}/stp/lkl"
GT_DIR="${DATA_DIR}/stp/GT"
IMG_DIR="${DATA_DIR}/stp/img"
BASE_DIR="${OUTPUT_DIR}/stp/dm2d_min_size_sweep"

# Script directory for finding other scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optimal Persistence Parameter for STP
FIXED_PERSISTENCE=32
FIXED_VE=0

MIN_SIZE_VALUES=(0 4 8 12 16 20 24)

RESULTS_FILE="$BASE_DIR/min_size_sweep.csv"
mkdir -p "$BASE_DIR"

# 1. Run DM2D once (since persistence is fixed)
DM2D_OUTPUT="$BASE_DIR/dm2d_fixed_persistence"
if [ ! -d "$DM2D_OUTPUT/skeleton" ]; then
    echo "Running DM2D (persistence=$FIXED_PERSISTENCE)..."
    cd "$PROJECT_ROOT/DM2D_Skeletonization_Vectorization/Skeletonization_Suite"
    python "$PROJECT_ROOT/DM2D_Skeletonization_Vectorization/run_dm2d_tiles.py" \
        --lkl_dir "$LKL_DIR" --output_dir "$DM2D_OUTPUT" \
        --ve_persistence "$FIXED_VE" --et_persistence "$FIXED_PERSISTENCE"
else
    echo "DM2D run already exists at $DM2D_OUTPUT"
fi

# 2. Sweep Min Size
echo "min_size,precision,recall,f_score,iou" > "$RESULTS_FILE"

for SZ in "${MIN_SIZE_VALUES[@]}"; do
    WORK_DIR="$BASE_DIR/min_size_${SZ}"
    
    if [ ! -f "$WORK_DIR/evaluation/summary.json" ]; then
        echo "Running Vectorization (min_size=$SZ)..."
        mkdir -p "$WORK_DIR/skeleton" "$WORK_DIR/vec/CC" "$WORK_DIR/vec/JSON" "$WORK_DIR/mask"
        
        # Prepare masks for vectorization
        for f in "$DM2D_OUTPUT/skeleton"/*.tif; do 
            [ -f "$f" ] && cp "$f" "$WORK_DIR/mask/$(basename "$f" .tif).jpg"
        done
        
        # Run vectorization
        python "$PROJECT_ROOT/DM2D_Skeletonization_Vectorization/Vectorization/script_to_run.py" \
            --input_dir "$WORK_DIR/mask" --output_cc_dir "$WORK_DIR/vec/CC" \
            --output_json_dir "$WORK_DIR/vec/JSON" --min_size "$SZ"
        
        # Reconstruct skeleton from JSON
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
        
        # Evaluate
        python "$PROJECT_ROOT/scripts/evaluation/evaluate_model.py" \
            --model_dir "$WORK_DIR/skeleton" --model_name "DM2D_MinSize_${SZ}" \
            --gt_dir "$GT_DIR" --img_dir "$IMG_DIR" --output_dir "$WORK_DIR/evaluation" 2>/dev/null
    fi
    
    python -c "import json; m=json.load(open('$WORK_DIR/evaluation/summary.json'))['overall_metrics']; print(f'$SZ,{m[\"precision\"]},{m[\"recall\"]},{m[\"f_score\"]},{m[\"iou\"]}')" >> "$RESULTS_FILE"
done

echo "Results saved to: $RESULTS_FILE"
