#!/bin/bash
#
# DM2D Min Size Sweep - PMD Dataset
# Fixed persistence threshold (ET=64, VE=0), sweep min_size
#
# Each value runs run_dm2d_tiles.py with --min_size, i.e. the same
# small-component filter the main pipeline uses, so min_size=40 here
# reproduces the DM2D row of Table 1.
#
# Usage:
#   ./final_run_dm2d_pmd_min_size_sweep.sh
#
# Environment variables (optional):
#   DATA_DIR   - Base directory for input data (default: <project>/data)
#   OUTPUT_DIR - Base directory for outputs (default: <project>/outputs)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DM2D_DIR="$PROJECT_ROOT/Skeletonization_Suite"
EVAL_SCRIPT="$PROJECT_ROOT/Utilities/evaluation/evaluate_model.py"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

LKL_DIR="${DATA_DIR}/pmd/lkl"
GT_DIR="${DATA_DIR}/pmd/GT"
IMG_DIR="${DATA_DIR}/pmd/img"
BASE_DIR="${OUTPUT_DIR}/pmd/dm2d_min_size_sweep"

FIXED_PERSISTENCE=64
FIXED_VE=0
MIN_SIZE_VALUES=(0 10 20 30 40 50 60)

RESULTS_FILE="$BASE_DIR/min_size_sweep.csv"
mkdir -p "$BASE_DIR"

echo "DM2D Min Size Sweep - PMD Dataset (persistence=$FIXED_PERSISTENCE)"
echo "min_size,precision,recall,f_score,iou" > "$RESULTS_FILE"

for SZ in "${MIN_SIZE_VALUES[@]}"; do
    WORK_DIR="$BASE_DIR/min_size_${SZ}"

    if [ ! -f "$WORK_DIR/evaluation/summary.json" ]; then
        echo "  Running min_size=$SZ..."
        cd "$DM2D_DIR"
        python "$DM2D_DIR/run_dm2d_tiles.py" \
            --lkl_dir "$LKL_DIR" --output_dir "$WORK_DIR" \
            --ve_persistence "$FIXED_VE" --et_persistence "$FIXED_PERSISTENCE" \
            --min_size "$SZ"

        python "$EVAL_SCRIPT" \
            --model_dir "$WORK_DIR/skeleton" --model_name "DM2D_MinSize_${SZ}" \
            --gt_dir "$GT_DIR" --img_dir "$IMG_DIR" --output_dir "$WORK_DIR/evaluation" --distance_threshold 5
    else
        echo "  Skipping min_size=$SZ (already done)"
    fi

    python -c "import json; m=json.load(open('$WORK_DIR/evaluation/summary.json'))['overall_metrics']; print(f'$SZ,{m[\"precision\"]},{m[\"recall\"]},{m[\"f_score\"]},{m[\"iou\"]}')" >> "$RESULTS_FILE"
done

echo ""
echo "Results saved to: $RESULTS_FILE"
cat "$RESULTS_FILE"
