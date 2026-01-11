#!/bin/bash
set -e

# Activate venv if exists, or assume in path
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH=$PYTHONPATH:.

# echo "========================================"
# echo "Creating Datasets (Splits & Augmentation)"
# echo "========================================"
# $PYTHON src/create_dataset.py

# echo ""
# echo "========================================"
# echo "Processing Data-Inside (All Splits)"
# echo "========================================"
# $PYTHON src/pipeline.py --config configs/config_inside.yaml --all-splits

# echo ""
# echo "========================================"
# echo "Processing Data-Inside-Zoom (All Splits)"
# echo "========================================"
# $PYTHON src/pipeline.py --config configs/config_inside_zoom.yaml --all-splits

echo ""
echo "========================================"
echo "Processing Data-Outside (All Splits)"
echo "========================================"
$PYTHON src/pipeline.py --config configs/config_outside.yaml --all-splits

echo ""
echo "========================================"
echo "All processing completed successfully!"
echo "========================================"
