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

import os
import polars as pl
import typer
from src.config import get_config
from src.preprocessing.steps.split import SplitStep
from src.preprocessing.steps.yolo import YoloStep
from src.preprocessing.steps.rotate import RotateStep
from src.preprocessing.steps.depth import DepthStep
from src.preprocessing.steps.segment import SegmentStep
from src.preprocessing.steps.blackout import BlackoutStep
from src.preprocessing.steps.feature import FeatureStep

app = typer.Typer(add_completion=False, help="Run the preprocessing pipeline.")


def run_pipeline(config):
    print(f"Starting preprocessing pipeline for {config.name}...")
    config.output_dir.mkdir(exist_ok=True)

    # Initialize data
    df = pl.read_csv(config.input_csv_path).drop_nulls()
    steps = [
        SplitStep(config),
        # YoloStep(config),
        # RotateStep(config),
        # YoloStep(config),
        # DepthStep(config),
        # SegmentStep(config),
        # BlackoutStep(config),
        # FeatureStep(config),
    ]

    # Run pipeline
    for step in steps:
        print(f"Running {step.__class__.__name__}...")
        df = step.process(df)

    # Save final result
    df.write_csv(config.output_csv_path)
    print(f"Preprocessing pipeline for {config.name} finished.")
    print(f"Saved processed data to {config.output_csv_path}")

@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Path to config file"),
):
    config = get_config(dataset_name)

    run_pipeline(config)


import polars as pl
if __name__ == "__main__":
    app()
