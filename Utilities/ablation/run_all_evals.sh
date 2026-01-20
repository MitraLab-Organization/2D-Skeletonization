#!/bin/bash
# Run all model evaluations for a dataset
# Usage: ./run_all_evals.sh pmd|stp
#
# Environment variables (optional):
#   DATA_DIR  - Base directory for input data (default: ./data relative to project root)
#   OUTPUT_DIR - Base directory for outputs (default: ./outputs relative to project root)

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    echo "  dataset: pmd or stp"
    exit 1
fi

DATASET="$1"

# Script directory for finding project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use environment variables with sensible defaults
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
OUTPUTS_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

EVAL_SCRIPT="$PROJECT_ROOT/scripts/evaluation/evaluate_model.py"
IMG_DIR="$DATA_DIR/$DATASET/img"
GT_DIR="$DATA_DIR/$DATASET/GT"

run_eval() {
    MODEL_DIR="$1"
    MODEL_NAME="$2"
    OUTPUT_DIR="$3"
    
    echo "Running evaluation for $MODEL_NAME..."
    python "$EVAL_SCRIPT" \
        --model_dir "$OUTPUTS_DIR/$DATASET/$MODEL_DIR" \
        --model_name "$MODEL_NAME" \
        --gt_dir "$GT_DIR" \
        --img_dir "$IMG_DIR" \
        --output_dir "$OUTPUTS_DIR/$DATASET/$OUTPUT_DIR"
}

# Run for all methods
run_eval "bwskel" "bwskel" "bwskel_evaluation"
run_eval "diffskel" "diffskel" "diffskel_evaluation"
run_eval "neutube" "neuTube" "neutube_evaluation"
run_eval "vess" "VESS" "vess_evaluation"
run_eval "phd" "PHDF" "phd_evaluation"
run_eval "dm2d/skeleton" "DM2D" "dm2d_evaluation"

echo "All $DATASET evaluations complete."
