#!/bin/bash
# Docker entrypoint for DM2D Pipeline
# Supports: paper reproduction, tiled evaluation, and whole-image processing
set -e

COMMAND="${1:-help}"
ARG2="${2:-}"

DATA_DIR="/data"
OUTPUT_DIR="/outputs"

# Set dataset-specific parameters (supports arbitrary dataset names)
get_dataset_params() {
    local ds="$1"
    case "$ds" in
        pmd)
            DM2D_ET_PERSISTENCE=64
            DM2D_MIN_SIZE=40
            EVAL_DISTANCE=5   # match distance in pixels (Methods 1.4)
            ;;
        stp)
            DM2D_ET_PERSISTENCE=32
            DM2D_MIN_SIZE=12
            # Methods 1.4 says 3 px for STP, but the published Table 1b was computed at 5 px;
            # kept at 5 to reproduce the table (see revision checklist)
            EVAL_DISTANCE=5
            ;;
        *)
            # Default parameters for custom datasets
            DM2D_ET_PERSISTENCE=32
            DM2D_MIN_SIZE=20
            EVAL_DISTANCE=5
            echo "Note: Using default parameters for dataset '$ds' (et_persistence=$DM2D_ET_PERSISTENCE, min_size=$DM2D_MIN_SIZE)"
            ;;
    esac
}

# Run inference on a dataset
run_inference() {
    local DATASET="$1"
    get_dataset_params "$DATASET"
    
    echo "=== Running inference on $DATASET ==="
    
    # neutube
    echo "--- neutube ---"
    python /app/Baselines/neutube/batch_neutube.py \
        -i "$DATA_DIR/$DATASET/lkl" \
        -o "$OUTPUT_DIR/$DATASET/neutube_swc"
    
    echo "Converting neutube SWC to TIF..."
    python /app/Baselines/neutube/batch_convert_swc.py \
        --swc_dir "$OUTPUT_DIR/$DATASET/neutube_swc" \
        --lkl_dir "$DATA_DIR/$DATASET/lkl" \
        --out_dir "$OUTPUT_DIR/$DATASET/neutube"
    
    # VESS
    echo "--- VESS ---"
    python /app/Baselines/vess/run_vess.py --dataset "$DATASET" --skip-eval
    
    # PHD
    echo "--- PHD ---"
    python /app/Baselines/phd/run_phd.py --dataset "$DATASET" --skip-eval
    
    # diffskel
    echo "--- diffskel ---"
    python /app/Baselines/diffskel/batch_process.py \
        --input_folder "$DATA_DIR/$DATASET/lkl" \
        --output_folder "$OUTPUT_DIR/$DATASET/diffskel" \
        --probabilistic --beta 0.33 --tau 1.0 --seed 0
    
    # DM2D
    echo "--- DM2D ---"
    cd /app/Skeletonization_Suite
    python /app/Skeletonization_Suite/run_dm2d_tiles.py \
        --lkl_dir "$DATA_DIR/$DATASET/lkl" \
        --output_dir "$OUTPUT_DIR/$DATASET/dm2d" \
        --ve_persistence 0 --et_persistence $DM2D_ET_PERSISTENCE \
        --min_size $DM2D_MIN_SIZE
}

# Run evaluation on a dataset
run_evaluation() {
    local DATASET="$1"
    get_dataset_params "$DATASET"
    echo "=== Evaluating $DATASET (match distance ${EVAL_DISTANCE} px) ==="
    
    for method in bwskel diffskel neutube vess phd dm2d; do
        if [ -d "$OUTPUT_DIR/$DATASET/$method" ]; then
            echo "--- $method ---"
            if [ "$method" = "dm2d" ]; then
                MODEL_PATH="$OUTPUT_DIR/$DATASET/$method/skeleton"
            else
                MODEL_PATH="$OUTPUT_DIR/$DATASET/$method"
            fi
            python /app/Utilities/evaluation/evaluate_model.py \
                --model_dir "$MODEL_PATH" \
                --model_name "$method" \
                --gt_dir "$DATA_DIR/$DATASET/GT" \
                --img_dir "$DATA_DIR/$DATASET/img" \
                --output_dir "$OUTPUT_DIR/$DATASET/${method}_evaluation" \
                --distance_threshold "$EVAL_DISTANCE"
        fi
    done
}

# Run persistence sweep on a dataset
run_sweep() {
    local DATASET="$1"
    get_dataset_params "$DATASET"
    echo "=== Persistence sweep on $DATASET ==="
    
    export DATA_DIR="$DATA_DIR"
    export OUTPUT_DIR="$OUTPUT_DIR"
    
    if [ "$DATASET" = "pmd" ]; then
        bash /app/Utilities/ablation/docker_persistence_sweep_pmd.sh
    else
        bash /app/Utilities/ablation/docker_persistence_sweep_stp.sh
    fi

    echo "=== Min-size sweep on $DATASET ==="
    bash "/app/Utilities/ablation/final_run_dm2d_${DATASET}_min_size_sweep.sh"
    python /app/Utilities/ablation/generate_minsize_ablation.py "$DATASET" --regen
    python /app/Utilities/ablation/generate_minsize_ablation.py "$DATASET" --4panel
}

# Haircut ablation (Supplementary S.2.2): DM2D on the PMD tiles with and without
# Haircut pruning, then the with/without comparison figure
run_haircut_ablation() {
    get_dataset_params pmd
    echo "=== Haircut ablation (PMD) ==="

    cd /app/Skeletonization_Suite
    for variant in haircut no_haircut; do
        echo "--- $variant ---"
        local extra=""
        if [ "$variant" = "no_haircut" ]; then
            extra="--no_haircut"
        fi
        python /app/Skeletonization_Suite/run_dm2d_tiles.py \
            --lkl_dir "$DATA_DIR/pmd/lkl" \
            --output_dir "$OUTPUT_DIR/haircut_ablation/$variant" \
            --ve_persistence 0 --et_persistence $DM2D_ET_PERSISTENCE \
            --min_size $DM2D_MIN_SIZE $extra
        mkdir -p "$OUTPUT_DIR/${variant}_skeletons"
        cp "$OUTPUT_DIR/haircut_ablation/$variant/skeleton"/*.tif "$OUTPUT_DIR/${variant}_skeletons/"
    done

    python /app/Utilities/visualization/generate_haircut_4panel.py --regen
}

# Generate all plots
run_plots() {
    echo "=== Generating plots ==="
    python /app/Utilities/visualization/generate_figure_subplot.py --dataset pmd
    python /app/Utilities/visualization/generate_figure_subplot.py --dataset stp
    python /app/Utilities/visualization/tabulate_results.py
    python /app/Utilities/visualization/generate_results_barplots.py
    python /app/Utilities/visualization/export_source_data.py
    cp -r /app/results/* "$OUTPUT_DIR/" 2>/dev/null || true
}

case "$COMMAND" in
    # ================================================================
    # PAPER REPRODUCTION (Master mode)
    # ================================================================
    paper)
        echo "========================================================"
        echo "  FULL PAPER REPRODUCTION"
        echo "========================================================"
        echo "This will run everything: inference, evaluation, sweeps, and plots"
        echo ""
        
        # PMD
        echo "########## PMD DATASET ##########"
        run_inference pmd
        run_evaluation pmd
        run_sweep pmd
        
        # STP
        echo ""
        echo "########## STP DATASET ##########"
        run_inference stp
        run_evaluation stp
        run_sweep stp

        # Haircut ablation
        echo ""
        run_haircut_ablation
        
        # Plots
        echo ""
        run_plots
        
        echo ""
        echo "========================================================"
        echo "  PAPER REPRODUCTION COMPLETE!"
        echo "========================================================"
        echo "Results: $OUTPUT_DIR/"
        echo "- figures/: Comparison plots"
        echo "- tables/: CSV result tables"
        echo "- */dm2d_persistence_sweep/, */dm2d_min_size_sweep/: Sweep results"
        echo "- ablation/: Min-size ablation figures"
        echo "- haircut_comparison_plots/: Haircut ablation figure"
        
        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;
    
    # ================================================================
    # QUICK DEMO (One dataset, no sweeps)
    # ================================================================
    demo)
        DATASET="${ARG2:-pmd}"
        echo "========================================================"
        echo "  DEMO: $DATASET dataset"
        echo "========================================================"
        
        run_inference "$DATASET"
        run_evaluation "$DATASET"
        run_plots
        
        echo ""
        echo "Demo complete! Results: $OUTPUT_DIR/"
        
        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;
    
    # ================================================================
    # WHOLE-IMAGE PROCESSING (ALBU neural network)
    # ================================================================
    run-image)
        INPUT_FILE="${ARG2}"
        if [ -z "$INPUT_FILE" ]; then
            echo "Error: No input file specified"
            echo "Usage: run-image <path/to/image.jp2> [--mode pmd|stp|custom] [--ve_persistence_threshold N] [--persistence_threshold N] [--min_size N] [--norm_factor N] [--likelihood_threshold N] [--mask true|false] [--output <name>]"
            exit 1
        fi
        
        echo "========================================================"
        echo "  WHOLE-IMAGE PROCESSING"
        echo "========================================================"
        echo "Input: $INPUT_FILE"
        
        cd /app/Skeletonization_Suite
        
        # Collect all remaining args after command and input file
        shift 2  # Remove command and input file
        EXTRA_ARGS=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --mode)
                    EXTRA_ARGS="$EXTRA_ARGS --mode $2"
                    echo "Mode: $2"
                    shift 2
                    ;;
                --ve_persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --ve_persistence_threshold $2"
                    echo "VE Persistence threshold: $2"
                    shift 2
                    ;;
                --persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --persistence_threshold $2"
                    echo "Persistence threshold: $2"
                    shift 2
                    ;;
                --min_size)
                    EXTRA_ARGS="$EXTRA_ARGS --min_size $2"
                    echo "Min size: $2"
                    shift 2
                    ;;
                --norm_factor)
                    EXTRA_ARGS="$EXTRA_ARGS --norm_factor $2"
                    echo "Norm factor: $2"
                    shift 2
                    ;;
                --likelihood_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --likelihood_threshold $2"
                    echo "Likelihood threshold: $2"
                    shift 2
                    ;;
                --mask)
                    EXTRA_ARGS="$EXTRA_ARGS --mask $2"
                    echo "Mask: $2"
                    shift 2
                    ;;
                --output)
                    EXTRA_ARGS="$EXTRA_ARGS --output_name $2"
                    echo "Output name: $2"
                    shift 2
                    ;;
                *)
                    echo "Warning: Unknown argument '$1' ignored"
                    shift
                    ;;
            esac
        done

        python /app/Skeletonization_Suite/run_whole_image.py \
            --input "$INPUT_FILE" \
            --output "$OUTPUT_DIR/whole_image" \
            $EXTRA_ARGS
        
        echo ""
        echo "Processing complete! Results: $OUTPUT_DIR/whole_image/"
        
        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;
    
    run-folder)
        INPUT_DIR="${ARG2}"
        if [ -z "$INPUT_DIR" ]; then
            echo "Error: No input directory specified"
            echo "Usage: run-folder <path/to/images/> [--mode pmd|stp|custom] [--ve_persistence_threshold N] [--persistence_threshold N] [--min_size N] [--norm_factor N] [--likelihood_threshold N] [--mask true|false]"
            exit 1
        fi
        
        echo "========================================================"
        echo "  BATCH WHOLE-IMAGE PROCESSING"
        echo "========================================================"
        echo "Input directory: $INPUT_DIR"
        
        cd /app/Skeletonization_Suite
        
        # Collect all remaining args after command and input dir
        shift 2  # Remove command and input dir
        EXTRA_ARGS=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --mode)
                    EXTRA_ARGS="$EXTRA_ARGS --mode $2"
                    echo "Mode: $2"
                    shift 2
                    ;;
                --ve_persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --ve_persistence_threshold $2"
                    echo "VE Persistence threshold: $2"
                    shift 2
                    ;;
                --persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --persistence_threshold $2"
                    echo "Persistence threshold: $2"
                    shift 2
                    ;;
                --min_size)
                    EXTRA_ARGS="$EXTRA_ARGS --min_size $2"
                    echo "Min size: $2"
                    shift 2
                    ;;
                --norm_factor)
                    EXTRA_ARGS="$EXTRA_ARGS --norm_factor $2"
                    echo "Norm factor: $2"
                    shift 2
                    ;;
                --likelihood_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --likelihood_threshold $2"
                    echo "Likelihood threshold: $2"
                    shift 2
                    ;;
                --mask)
                    EXTRA_ARGS="$EXTRA_ARGS --mask $2"
                    echo "Mask: $2"
                    shift 2
                    ;;
                *)
                    echo "Warning: Unknown argument '$1' ignored"
                    shift
                    ;;
            esac
        done

        python /app/Skeletonization_Suite/run_whole_image.py \
            --input_dir "$INPUT_DIR" \
            --output "$OUTPUT_DIR/whole_image" \
            $EXTRA_ARGS
        
        echo ""
        echo "Batch processing complete! Results: $OUTPUT_DIR/whole_image/"

        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;

    # ================================================================
    # ENTRY POINT 2: IMAGE -> LIKELIHOOD ONLY
    # ================================================================
    image-to-likelihood)
        INPUT_FILE="${ARG2}"
        if [ -z "$INPUT_FILE" ]; then
            echo "Error: No input file specified"
            echo "Usage: image-to-likelihood <path/to/image.jp2> [--mode pmd|stp|custom] [--norm_factor N] [--likelihood_threshold N] [--mask true|false] [--output <name>]"
            exit 1
        fi

        echo "========================================================"
        echo "  IMAGE -> LIKELIHOOD"
        echo "========================================================"
        echo "Input: $INPUT_FILE"

        cd /app/Skeletonization_Suite

        # Collect all remaining args after command and input file
        shift 2  # Remove command and input file
        EXTRA_ARGS=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --mode)
                    EXTRA_ARGS="$EXTRA_ARGS --mode $2"
                    echo "Mode: $2"
                    shift 2
                    ;;
                --norm_factor)
                    EXTRA_ARGS="$EXTRA_ARGS --norm_factor $2"
                    echo "Norm factor: $2"
                    shift 2
                    ;;
                --likelihood_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --likelihood_threshold $2"
                    echo "Likelihood threshold: $2"
                    shift 2
                    ;;
                --mask)
                    EXTRA_ARGS="$EXTRA_ARGS --mask $2"
                    echo "Mask: $2"
                    shift 2
                    ;;
                --output)
                    EXTRA_ARGS="$EXTRA_ARGS --output_name $2"
                    echo "Output name: $2"
                    shift 2
                    ;;
                *)
                    echo "Warning: Unknown argument '$1' ignored"
                    shift
                    ;;
            esac
        done

        python /app/Skeletonization_Suite/run_image_to_likelihood.py \
            --input "$INPUT_FILE" \
            --output "$OUTPUT_DIR/likelihood" \
            $EXTRA_ARGS

        echo ""
        echo "Processing complete! Results: $OUTPUT_DIR/likelihood/"

        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;

    # ================================================================
    # ENTRY POINT 3: LIKELIHOOD -> SKELETON ONLY
    # ================================================================
    likelihood-to-skeleton)
        INPUT_FILE="${ARG2}"
        if [ -z "$INPUT_FILE" ]; then
            echo "Error: No input file specified"
            echo "Usage: likelihood-to-skeleton <path/to/likelihood.jpg> [--mode pmd|stp|custom] [--ve_persistence_threshold N] [--persistence_threshold N] [--min_size N] [--output <name>]"
            exit 1
        fi

        echo "========================================================"
        echo "  LIKELIHOOD -> SKELETON"
        echo "========================================================"
        echo "Input: $INPUT_FILE"

        cd /app/Skeletonization_Suite

        # Collect all remaining args after command and input file
        shift 2  # Remove command and input file
        EXTRA_ARGS=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --mode)
                    EXTRA_ARGS="$EXTRA_ARGS --mode $2"
                    echo "Mode: $2"
                    shift 2
                    ;;
                --ve_persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --ve_persistence_threshold $2"
                    echo "VE Persistence threshold: $2"
                    shift 2
                    ;;
                --persistence_threshold)
                    EXTRA_ARGS="$EXTRA_ARGS --persistence_threshold $2"
                    echo "Persistence threshold: $2"
                    shift 2
                    ;;
                --min_size)
                    EXTRA_ARGS="$EXTRA_ARGS --min_size $2"
                    echo "Min size: $2"
                    shift 2
                    ;;
                --output)
                    EXTRA_ARGS="$EXTRA_ARGS --output_name $2"
                    echo "Output name: $2"
                    shift 2
                    ;;
                *)
                    echo "Warning: Unknown argument '$1' ignored"
                    shift
                    ;;
            esac
        done

        python /app/Skeletonization_Suite/run_likelihood_to_skeleton.py \
            --input "$INPUT_FILE" \
            --output "$OUTPUT_DIR/skeleton" \
            $EXTRA_ARGS

        echo ""
        echo "Processing complete! Results: $OUTPUT_DIR/skeleton/"

        # Fix permissions
        echo "Fixing permissions..."
        chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true
        ;;

    # ================================================================
    # INDIVIDUAL STEPS
    # ================================================================
    inference)
        DATASET="${ARG2:-pmd}"
        run_inference "$DATASET"
        ;;
    
    evaluate)
        DATASET="${ARG2:-pmd}"
        run_evaluation "$DATASET"
        ;;
    
    plots)
        run_plots
        ;;

    haircut-ablation)
        run_haircut_ablation
        ;;
    
    # ================================================================
    # HELP
    # ================================================================
    help|*)
        echo "DM2D Docker Pipeline"
        echo ""
        echo "USAGE: docker run -v /data:/data -v /outputs:/outputs dm2d <command>"
        echo ""
        echo "PAPER REPRODUCTION (Tiled Evaluation):"
        echo "  paper              Full paper reproduction (both datasets + sweeps + plots)"
        echo "  demo <pmd|stp>     Quick demo on one dataset (no sweeps)"
        echo "  haircut-ablation   With/without Haircut comparison figure (Supplementary S.2.2)"
        echo ""
        echo "WHOLE-IMAGE PROCESSING (Entry Point 1: image -> skeleton):"
        echo "  run-image <file> [options]     Process single JP2/TIFF through ALBU inference"
        echo "  run-folder <dir> [options]     Process all images in a folder"
        echo ""
        echo "  Options for run-image / run-folder:"
        echo "    --mode <pmd|stp|custom>              Parameter preset (default: custom)"
        echo "    --ve_persistence_threshold <N>       VE persistence threshold (default: 0, optional for custom)"
        echo "    --persistence_threshold <N>          ET persistence threshold (required for custom)"
        echo "    --min_size <N>                       Min component size (required for custom)"
        echo "    --norm_factor <N>                    Pixel normalization divisor (required for custom)"
        echo "    --likelihood_threshold <N>           Likelihood background clip-to-zero cutoff (required for custom)"
        echo "    --mask <true|false>                  Restrict processing to tissue via Otsu thresholding (default: true for pmd/stp, required for custom)"
        echo "    --output <name>                      Custom output name (run-image only)"
        echo ""
        echo "  Presets:"
        echo "    pmd:  ve_persistence=0, persistence_threshold=64, min_size=40, norm_factor=16, likelihood_threshold=40, mask=true"
        echo "    stp:  ve_persistence=0, persistence_threshold=32, min_size=12, norm_factor=256, likelihood_threshold=40, mask=true"
        echo ""
        echo "STAGED PROCESSING (split image -> skeleton into two steps):"
        echo "  image-to-likelihood <file> [options]     Entry Point 2: image -> likelihood only"
        echo "                                            (ALBU inference + background threshold)"
        echo "    Options: --mode, --norm_factor, --likelihood_threshold, --mask, --output"
        echo ""
        echo "  likelihood-to-skeleton <file> [options]  Entry Point 3: likelihood -> skeleton only"
        echo "                                            (no raw image or neural network needed)"
        echo "    Options: --mode, --ve_persistence_threshold, --persistence_threshold, --min_size, --output"
        echo ""
        echo "  Chaining these two reproduces run-image's output:"
        echo "    dm2d image-to-likelihood /input.jp2 --mode pmd --output MyBrain_001"
        echo "    dm2d likelihood-to-skeleton /outputs/likelihood/lkl/MyBrain_001_0.jpg --mode pmd --output MyBrain_001"
        echo ""
        echo "INDIVIDUAL STEPS:"
        echo "  inference <dataset>   Run inference only"
        echo "  evaluate <dataset>    Run evaluation only"
        echo "  plots                 Generate plots only"
        echo ""
        echo "DATA STRUCTURE:"
        echo "  /data/<dataset>/lkl/  Input likelihood images"
        echo "  /data/<dataset>/GT/   Ground truth skeletons"
        echo "  /data/<dataset>/img/  Original images"
        echo ""
        echo "EXAMPLES:"
        echo "  docker run ... dm2d paper"
        echo "  docker run ... dm2d demo pmd"
        echo "  docker run -v /my/image.jp2:/input.jp2 ... dm2d run-image /input.jp2 --mode pmd"
        echo "  docker run -v /my/image.jp2:/input.jp2 ... dm2d run-image /input.jp2 --persistence_threshold 16 --min_size 30 --norm_factor 16"
        echo "  docker run -v /my/image.jp2:/input.jp2 ... dm2d run-image /input.jp2 --mode stp --output MyBrain_001"
        ;;
esac
