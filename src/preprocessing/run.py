"""
Preprocessing Pipeline Runner

Runs the image preprocessing pipeline including:
- YOLO detection (initial + rotated)
- Image rotation
- Depth estimation
- Segmentation
- Blackout
- Feature extraction

Usage:
    python -m src.preprocessing.run --config configs/config_inside.yaml
    python -m src.preprocessing.run --config configs/config_inside.yaml --all-splits
"""

import argparse
import os
import polars as pl
from src.utils.io import load_config, load_csv, save_csv
from src.preprocessing.steps.yolo import YoloStep
from src.preprocessing.steps.rotate import RotateStep
from src.preprocessing.steps.depth import DepthStep
from src.preprocessing.steps.segment import SegmentStep
from src.preprocessing.steps.blackout import BlackoutStep
from src.preprocessing.steps.feature import FeatureStep


def run_pipeline(config, in_data_path=None, out_data_path=None, steps=None):
    # Determine paths
    if in_data_path is None:
        in_data_path = config["paths"]["in_data"]
    if out_data_path is None:
        out_data_path = config["paths"]["out_data"]

    print(f"Starting pipeline processing...")
    print(f"Input: {in_data_path}")
    print(f"Output: {out_data_path}")

    # Initialize data
    try:
        df = load_csv(in_data_path)
    except Exception:
        print(
            f"Data file not found at {in_data_path}, creating new one from raw images..."
        )
        df = pl.DataFrame({"name": []})  # Placeholder

    # Define steps
    if steps is None:
        steps = [
            YoloStep(config, stage="initial"),
            RotateStep(config),
            YoloStep(config, stage="rotated"),
            DepthStep(config),
            SegmentStep(config),
            BlackoutStep(config),
            FeatureStep(config),
        ]

    # Run pipeline
    for step in steps:
        print(f"Running {step.__class__.__name__}...")
        df = step.process(df)

    # Save final result
    save_csv(df, out_data_path)
    print(f"Saved results to {out_data_path}")


def main():
    parser = argparse.ArgumentParser(description="Fishometry Preprocessing Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of images to process"
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Process all splits (train, val, test)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply limit if specified
    if args.limit:
        config["params"]["limit"] = args.limit

    if args.all_splits:
        dataset_name = config.get("dataset_name", "data-inside")
        # Ensure output directory exists
        base_out_dir = config["paths"]["output"]
        if not os.path.exists(base_out_dir):
            os.makedirs(base_out_dir)

        # Set train path for FeatureStep
        config["paths"]["train_split"] = f"data/{dataset_name}/splits/train.csv"

        splits = ["train", "val", "test"]
        for split in splits:
            in_path = f"data/{dataset_name}/splits/{split}.csv"
            out_path = f"data/{dataset_name}/processed/processed_{split}.csv"

            if not os.path.exists(in_path):
                print(f"Split file {in_path} does not exist. Skipping.")
                continue

            print(f"\nProcessing split: {split}")
            run_pipeline(config, in_path, out_path)
    else:
        # Single run, assume config has paths or use default?
        # If running mostly for dev, we might not have train_split set?
        # Let's try to set it if not present.
        if "train_split" not in config["paths"]:
            dataset_name = config.get("dataset_name", "data-inside")
            config["paths"]["train_split"] = f"data/{dataset_name}/splits/train.csv"

        run_pipeline(config)


if __name__ == "__main__":
    main()
