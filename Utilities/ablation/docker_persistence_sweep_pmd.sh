#!/bin/bash
#
# DM2D Persistence Sweep - PMD Dataset
# Varying Persistence Threshold (ET), fixing VE=0
# min_size=40 (Optimal for PMD)
#

set -e

# Use environment variables or defaults for Docker
LKL_DIR="${DATA_DIR:-/data}/pmd/lkl"
GT_DIR="${DATA_DIR:-/data}/pmd/GT"
IMG_DIR="${DATA_DIR:-/data}/pmd/img"
BASE_DIR="${OUTPUT_DIR:-/outputs}/pmd/dm2d_persistence_sweep"
MIN_SIZE=40

# Script and executable paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DM2D_DIR="${PROJECT_DIR}/DM2D_Skeletonization_Vectorization"
EVAL_SCRIPT="${PROJECT_DIR}/scripts/evaluation/evaluate_model.py"

# "Persistence Threshold" in the paper corresponds to ET threshold (with VE fixed at 0)
PERSISTENCE_VALUES=(0 2 4 8 16 32 64 128)
FIXED_VE=0

RESULTS_FILE="$BASE_DIR/persistence_sweep.csv"
mkdir -p "$BASE_DIR"

run_config() {
    local PERSISTENCE=$1 WORK_DIR=$2 NAME=$3
    
    # Check if run already exists
    if [ -f "$WORK_DIR/evaluation/summary.json" ]; then
        echo "  Skipping persistence=$PERSISTENCE (already done)"
        return 0
    fi

    echo "  Running persistence=$PERSISTENCE..."
    mkdir -p "$WORK_DIR/skeleton"
    
    # Run DM2D with min_size filtering
    cd "$DM2D_DIR/Skeletonization_Suite"
    python "$DM2D_DIR/run_dm2d_tiles.py" \
        --lkl_dir "$LKL_DIR" \
        --output_dir "$WORK_DIR/dm2d" \
        --ve_persistence "$FIXED_VE" \
        --et_persistence "$PERSISTENCE" \
        --min_size "$MIN_SIZE"
    
    # Copy skeletons to expected location
    cp "$WORK_DIR/dm2d/skeleton"/*.tif "$WORK_DIR/skeleton/" 2>/dev/null || true
    
    # Evaluate
    python "$EVAL_SCRIPT" \
        --model_dir "$WORK_DIR/skeleton" \
        --model_name "$NAME" \
        --gt_dir "$GT_DIR" \
        --img_dir "$IMG_DIR" \
        --output_dir "$WORK_DIR/evaluation" 2>/dev/null || true
}

echo "DM2D Persistence Sweep - PMD Dataset"
echo "====================================="
echo "persistence_threshold,precision,recall,f_score,iou" > "$RESULTS_FILE"

for P in "${PERSISTENCE_VALUES[@]}"; do
    WORK_DIR="$BASE_DIR/persistence_${P}"
    run_config "$P" "$WORK_DIR" "DM2D_Persistence_${P}"
    
    if [ -f "$WORK_DIR/evaluation/summary.json" ]; then
        python -c "import json; m=json.load(open('$WORK_DIR/evaluation/summary.json'))['overall_metrics']; print(f'$P,{m[\"precision\"]},{m[\"recall\"]},{m[\"f_score\"]},{m[\"iou\"]}')" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "Results saved to: $RESULTS_FILE"
cat "$RESULTS_FILE"
