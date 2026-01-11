#!/bin/bash
#
# Fishometry - Run All Predictions
#
# This script trains all models (Linear Regression, XGBoost, MLP, CNN)
# for both global (all fish) and per-fish-type configurations.
#
# Usage:
#   ./run_predictions.sh                    # Run everything
#   ./run_predictions.sh --dataset data-outside  # Specific dataset
#   ./run_predictions.sh --skip-cnn         # Skip CNN training (slow)
#   ./run_predictions.sh --only-per-type    # Only per-type models
#   ./run_predictions.sh --only-global      # Only global models
#

set -e # Exit on error

# Default values
DATASET="data-outside"
SKIP_CNN=true
ONLY_PER_TYPE=false
ONLY_GLOBAL=false
EPOCHS_MLP=200
EPOCHS_CNN=100

# Parse arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	--dataset)
		DATASET="$2"
		shift 2
		;;
	--skip-cnn)
		SKIP_CNN=true
		shift
		;;
	--only-per-type)
		ONLY_PER_TYPE=true
		shift
		;;
	--only-global)
		ONLY_GLOBAL=true
		shift
		;;
	--epochs-mlp)
		EPOCHS_MLP="$2"
		shift 2
		;;
	--epochs-cnn)
		EPOCHS_CNN="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --dataset NAME      Dataset to use (default: data-outside)"
		echo "  --skip-cnn          Skip CNN training (GPU intensive)"
		echo "  --only-per-type     Only train per-fish-type models"
		echo "  --only-global       Only train global models (all fish)"
		echo "  --epochs-mlp N      MLP epochs (default: 200)"
		echo "  --epochs-cnn N      CNN epochs (default: 100)"
		echo "  -h, --help          Show this help message"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

echo "============================================================"
echo "  Fishometry - Training All Models"
echo "============================================================"
echo ""
echo "  Dataset:        $DATASET"
echo "  Skip CNN:       $SKIP_CNN"
echo "  Only Per-Type:  $ONLY_PER_TYPE"
echo "  Only Global:    $ONLY_GLOBAL"
echo "  MLP Epochs:     $EPOCHS_MLP"
echo "  CNN Epochs:     $EPOCHS_CNN"
echo ""
echo "============================================================"

# Function to run a training command
run_training() {
	local desc="$1"
	shift
	echo ""
	echo ">>> $desc"
	echo "    Command: uv run python $@"
	echo ""
	uv run python "$@"
}

# ============================================================
# GLOBAL MODELS (trained on all fish types combined)
# ============================================================

if [ "$ONLY_PER_TYPE" = false ]; then
	echo ""
	echo "============================================================"
	echo "  PHASE 1: Global Models (All Fish Types)"
	echo "============================================================"

	# Pipeline 0: Baseline
	run_training "Baseline Model" \
		-m src.train_baseline --dataset "$DATASET"

	# Pipeline 1: coords features
	run_training "Linear Regression (coords)" \
		-m src.train_regression --dataset "$DATASET" --feature-set coords

	run_training "XGBoost (coords)" \
		-m src.train_xgboost --dataset "$DATASET" --feature-set coords

	run_training "MLP (coords)" \
		-m src.train_mlp --dataset "$DATASET" --feature-set coords --epochs "$EPOCHS_MLP"

	# Pipeline 3: scaled features
	run_training "Linear Regression (scaled)" \
		-m src.train_regression --dataset "$DATASET" --feature-set scaled

	run_training "XGBoost (scaled)" \
		-m src.train_xgboost --dataset "$DATASET" --feature-set scaled

	run_training "MLP (scaled)" \
		-m src.train_mlp --dataset "$DATASET" --feature-set scaled --epochs "$EPOCHS_MLP"

	# Pipeline 4: coords + depth v2
	run_training "Linear Regression (coords + depth v2)" \
		-m src.train_regression --dataset "$DATASET" --feature-set coords --depth

	run_training "XGBoost (coords + depth v2)" \
		-m src.train_xgboost --dataset "$DATASET" --feature-set coords --depth

	run_training "MLP (coords + depth v2)" \
		-m src.train_mlp --dataset "$DATASET" --feature-set coords --depth --epochs "$EPOCHS_MLP"

	# Pipeline 5: scaled + depth v2
	run_training "Linear Regression (scaled + depth v2)" \
		-m src.train_regression --dataset "$DATASET" --feature-set scaled --depth

	run_training "XGBoost (scaled + depth v2)" \
		-m src.train_xgboost --dataset "$DATASET" --feature-set scaled --depth

	run_training "MLP (scaled + depth v2)" \
		-m src.train_mlp --dataset "$DATASET" --feature-set scaled --depth --epochs "$EPOCHS_MLP"

	# Pipeline 6: CNN (optional)
	if [ "$SKIP_CNN" = false ]; then
		run_training "CNN (scaled + depth v2)" \
			-m src.train_cnn --dataset "$DATASET" --feature-set scaled --depth --epochs "$EPOCHS_CNN"
	else
		echo ""
		echo ">>> Skipping CNN training (--skip-cnn flag set)"
	fi
fi

# ============================================================
# PER-FISH-TYPE MODELS
# ============================================================

if [ "$ONLY_GLOBAL" = false ]; then
	echo ""
	echo "============================================================"
	echo "  PHASE 2: Per-Fish-Type Models"
	echo "============================================================"

	# Per-type with coords features
	run_training "Per-Fish-Type (coords)" \
		-m src.train_per_fishtype --dataset "$DATASET" --feature-set coords --epochs "$EPOCHS_MLP"

	# Per-type with scaled features
	run_training "Per-Fish-Type (scaled)" \
		-m src.train_per_fishtype --dataset "$DATASET" --feature-set scaled --epochs "$EPOCHS_MLP"

	# Per-type with coords + depth v2
	run_training "Per-Fish-Type (coords + depth v2)" \
		-m src.train_per_fishtype --dataset "$DATASET" --feature-set coords --depth --epochs "$EPOCHS_MLP"

	# Per-type with scaled + depth v2
	run_training "Per-Fish-Type (scaled + depth v2)" \
		-m src.train_per_fishtype --dataset "$DATASET" --feature-set scaled --depth --epochs "$EPOCHS_MLP"
fi

# ============================================================
# ANALYSIS
# ============================================================

echo ""
echo "============================================================"
echo "  PHASE 3: Generate Analysis Charts"
echo "============================================================"

run_training "Generate Analysis Charts" \
	-m src.analyze_results --dataset "$DATASET"

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "  Results saved to:"
echo "    - Predictions: data/$DATASET/predictions/"
echo "    - Models:      checkpoints/$DATASET/"
echo "    - Analysis:    data/$DATASET/processed/analysis/"
echo ""
echo "  To view results:"
echo "    uv run streamlit run src/app.py"
echo ""
echo "============================================================"
