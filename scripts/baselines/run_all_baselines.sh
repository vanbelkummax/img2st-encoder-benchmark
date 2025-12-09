#!/bin/bash
#
# Unified Run Script for External Baselines
#
# This script runs all external baselines (Loki, FmH2ST) on the Visium HD benchmark.
#
# Prerequisites:
# 1. Loki cloned at ~/Loki with checkpoint.pt (7.2GB)
# 2. FmH2ST cloned at ~/FmH2ST with downloaded checkpoints
# 3. Conda environment 'loki_env' with all dependencies
# 4. Either real Visium HD images or synthetic test images
#
# Usage:
#   ./run_all_baselines.sh [--synthetic] [--skip-loki] [--skip-fmh2st]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$BASE_DIR/data/bin_level"
IMAGES_DIR="$BASE_DIR/data/images"
RESULTS_DIR="$BASE_DIR/results"
CONDA_ENV="loki_env"

# Parse arguments
SYNTHETIC=false
SKIP_LOKI=false
SKIP_FMH2ST=false
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --synthetic)
            SYNTHETIC=true
            shift
            ;;
        --skip-loki)
            SKIP_LOKI=true
            shift
            ;;
        --skip-fmh2st)
            SKIP_FMH2ST=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "External Baselines Evaluation Pipeline"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Data directory:    $DATA_DIR"
echo "  Images directory:  $IMAGES_DIR"
echo "  Results directory: $RESULTS_DIR"
echo "  Device:            $DEVICE"
echo "  Synthetic mode:    $SYNTHETIC"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Loki
if [ ! -f "$HOME/Loki/checkpoint.pt" ]; then
    echo "WARNING: Loki checkpoint not found at ~/Loki/checkpoint.pt"
    SKIP_LOKI=true
fi

# Check FmH2ST
if [ ! -f "$HOME/FmH2ST/checkpoint/A2/1000_checkpoint.pth" ]; then
    echo "WARNING: FmH2ST checkpoint not found"
    SKIP_FMH2ST=true
fi

# Check conda environment
if ! mamba env list | grep -q "$CONDA_ENV"; then
    echo "ERROR: Conda environment '$CONDA_ENV' not found"
    echo "Create it with: mamba create -n $CONDA_ENV python=3.9"
    exit 1
fi

# Create/verify images directory
if [ ! -d "$IMAGES_DIR" ] || [ ! -f "$IMAGES_DIR/patch_index.json" ]; then
    if [ "$SYNTHETIC" = true ]; then
        echo ""
        echo "Creating synthetic test images..."
        mamba run -n $CONDA_ENV python "$SCRIPT_DIR/extract_visium_hd_images.py" \
            --synthetic \
            --bin_level_dir "$DATA_DIR" \
            --output_dir "$IMAGES_DIR"
    else
        echo ""
        echo "WARNING: No images found and --synthetic not specified"
        echo "Run with --synthetic to create placeholder images for testing"
        echo "Or provide real Visium HD images in $IMAGES_DIR"
    fi
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo ""
echo "================================================"
echo "Running Baselines"
echo "================================================"

# Run Loki baseline
if [ "$SKIP_LOKI" = false ]; then
    echo ""
    echo "--- Running Loki Baseline ---"
    mamba run -n $CONDA_ENV python "$SCRIPT_DIR/run_loki_visium_hd.py" \
        --data_dir "$DATA_DIR" \
        --loki_dir "$HOME/Loki" \
        --output_dir "$RESULTS_DIR/loki" \
        --device "$DEVICE"
else
    echo ""
    echo "--- Skipping Loki (checkpoint not found or --skip-loki) ---"
fi

# Run FmH2ST baseline
if [ "$SKIP_FMH2ST" = false ]; then
    echo ""
    echo "--- Running FmH2ST Baseline ---"
    mamba run -n $CONDA_ENV python "$SCRIPT_DIR/run_fmh2st_visium_hd.py" \
        --data_dir "$DATA_DIR" \
        --fmh2st_dir "$HOME/FmH2ST" \
        --output_dir "$RESULTS_DIR/fmh2st" \
        --device "$DEVICE"
else
    echo ""
    echo "--- Skipping FmH2ST (checkpoint not found or --skip-fmh2st) ---"
fi

echo ""
echo "================================================"
echo "Results Summary"
echo "================================================"

# Print results summary
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

if [ -f "$RESULTS_DIR/loki/loki_metrics.json" ]; then
    echo "Loki Results:"
    python3 -c "
import json
with open('$RESULTS_DIR/loki/loki_metrics.json') as f:
    m = json.load(f)
print(f'  Mean PCC: {m[\"pcc_mean\"]:.4f}')
print(f'  N genes:  {m[\"n_genes\"]}')
"
fi

if [ -f "$RESULTS_DIR/fmh2st/fmh2st_metrics.json" ]; then
    echo ""
    echo "FmH2ST Results:"
    python3 -c "
import json
with open('$RESULTS_DIR/fmh2st/fmh2st_metrics.json') as f:
    m = json.load(f)
print(f'  Mean PCC (all genes):     {m[\"pcc_mean_all_genes\"]:.4f}')
print(f'  Mean PCC (overlap genes): {m[\"pcc_mean_overlap_genes\"]:.4f}')
print(f'  N overlap genes:          {m[\"n_genes_overlap\"]}/50')
"
fi

echo ""
echo "================================================"
echo "Evaluation Complete"
echo "================================================"
