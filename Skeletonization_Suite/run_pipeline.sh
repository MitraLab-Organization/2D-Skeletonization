#!/bin/bash
set -e

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
INPUT_FILE=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_FILE" ]] || [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file $INPUT_FILE not found!"
    echo "Usage: $0 --input /path/to/input.jp2 --output /path/to/output"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory not specified!"
    echo "Usage: $0 --input /path/to/input.jp2 --output /path/to/output"
    exit 1
fi

echo "Starting Pipeline with Input: $INPUT_FILE"
echo "Output Directory: $OUTPUT_DIR"

# ----------------------------------------------------------------
# Step 1: Prepare Input for Skeletonization
# ----------------------------------------------------------------
PREP_DIR="$SCRIPT_DIR/tmp_input"
mkdir -p "$PREP_DIR"
rm -f "$PREP_DIR"/*

COMPATIBLE_NAME="Input&Mask_1_Sec_1_F1.jp2"
cp "$INPUT_FILE" "$PREP_DIR/$COMPATIBLE_NAME"

echo "Prepared input at $PREP_DIR/$COMPATIBLE_NAME"

# ----------------------------------------------------------------
# Step 2: Run Skeletonization
# ----------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "Running Skeletonization Script..."
cd "$SCRIPT_DIR/Skeletonization_Suite"

export PYTHONPATH=$PYTHONPATH:.

python3 whole_brain_samik_processDet_skel_singleChanel_mba.py "$PREP_DIR" "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

# ----------------------------------------------------------------
# Step 3: Run Vectorization
# ----------------------------------------------------------------
# The previous step generates masks in $OUTPUT_DIR/mask/
# The inputs for this step are the generated masks.

MASK_DIR="$OUTPUT_DIR/mask"

if [[ ! -d "$MASK_DIR" ]]; then
    echo "Error: Mask directory $MASK_DIR was not generated."
    exit 1
fi

# We might need to handle file extension mismatch. 
# Skeletonization outputs .jpg masks (lines 256, 272 in pipeline script).
# Vectorization looks for .jpg (line 35 in script_to_run.py).
# So it should be fine.

FINAL_CC_DIR="$OUTPUT_DIR/final/CC"
FINAL_JSON_DIR="$OUTPUT_DIR/final/JSON"

mkdir -p "$FINAL_CC_DIR"
mkdir -p "$FINAL_JSON_DIR"

echo "Running Vectorization Script..."
python3 "$SCRIPT_DIR/Vectorization/script_to_run.py" \
    --input_dir "$MASK_DIR" \
    --output_cc_dir "$FINAL_CC_DIR" \
    --output_json_dir "$FINAL_JSON_DIR"

echo "----------------------------------------------------------------"
echo "Pipeline Completed Successfully."
echo "Final Vectorization outputs are in $OUTPUT_DIR/final"
echo "----------------------------------------------------------------"
