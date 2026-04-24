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
    python -m src.preprocessing.run --dataset-name data-inside
"""
from src.preprocessing.steps.vlm import VLMStep

import polars as pl
import typer
from src.config import get_config, Config
from src.preprocessing.steps.yolo import YoloStep
from src.preprocessing.steps.rotate import RotateStep
from src.preprocessing.steps.depth import DepthStep
from src.preprocessing.steps.segment import SegmentStep
from src.preprocessing.steps.blackout import BlackoutStep
from src.preprocessing.steps.feature import FeatureStep

app = typer.Typer(add_completion=False, help="Run the preprocessing pipeline.")


def run_pipeline(config: Config):
    print(f"Starting preprocessing pipeline for {config.dataset.name}...")

    if not config.dataset.split_csv_path.exists():
        raise FileNotFoundError(
            f"Run create data module first. {config.dataset.split_csv_path} not found."
        )

    config.dataset.output_dir.mkdir(exist_ok=True)
    rotated = config.dataset.rotate

    # Initialize data
    df = pl.read_csv(config.dataset.split_csv_path).drop_nulls()
    steps = [
        YoloStep(config),
        RotateStep(config),
        YoloStep(config, rotated=rotated),
        DepthStep(config, rotated=rotated),
        SegmentStep(config, rotated=rotated),
        BlackoutStep(config, rotated=rotated),
        VLMStep(config, rotated=rotated),
        FeatureStep(config, rotated=rotated),
    ]

    # Run pipeline
    for step in steps:
        print(f"Running {step.__class__.__name__}...")
        df = step.process(df)

    # Save final result
    df.write_csv(config.dataset.output_csv_path)
    print(f"Preprocessing pipeline for {config.dataset.name} finished.")
    print(f"Saved processed data to {config.dataset.output_csv_path}")


@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Path to config file"),
):
    config = get_config(dataset_name)

    run_pipeline(config)


if __name__ == "__main__":
    app()
