#!/bin/bash
# ARG-Expert Optimized Training Script
# Features: Mixed precision, optimized batch sizes, parallel data loading
#
# Usage:
#   Foreground: ./scripts/train_optimized.sh
#   Background: nohup ./scripts/train_optimized.sh &
#   View logs:  tail -f logs/training_*.log

set -e

# Create output directories
mkdir -p output
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# Redirect all stdout/stderr to log file
exec > "$LOG_FILE" 2>&1

echo "=================================================="
echo "ARG-Expert Optimized Training"
echo "=================================================="

# Activate conda environment if needed
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "gene_pred" ]; then
    echo "Activating conda environment: gene_pred"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gene_pred
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Environment variables for performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Record start time
START_TIME=$(date +%s)

echo "Starting training at $(date)"
echo "Logs will be saved to: $LOG_FILE"
echo ""

# Run training from repo root
cd "$(dirname "$0")/.."
python model/arg_transformer_complete.py

# Compute elapsed time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=================================================="
echo "Training Complete!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=================================================="

# Display final results
if [ -f "output/evaluation_results.json" ]; then
    echo ""
    echo "Final Results:"
    cat output/evaluation_results.json | python -m json.tool 2>/dev/null || cat output/evaluation_results.json
fi
