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

from datetime import datetime, timezone
import os
from typing import Protocol

import polars as pl
import typer

from src.artifacts import atomic_write_json
from src.config import Config, get_config
from src.preprocessing.steps.blackout import BlackoutStep
from src.preprocessing.steps.depth import DepthStep
from src.preprocessing.steps.feature import FeatureStep
from src.preprocessing.steps.rotate import RotateStep
from src.preprocessing.steps.segment import SegmentStep
from src.preprocessing.steps.vlm import VLMStep
from src.preprocessing.steps.yolo import YoloStep

app = typer.Typer(add_completion=False, help="Run the preprocessing pipeline.")


class PipelineStep(Protocol):
    def process(self, df: pl.DataFrame) -> pl.DataFrame: ...


def _validate_split(df: pl.DataFrame, config: Config) -> None:
    required = {"name", "length", "is_train", "is_val", "is_test"}
    if config.dataset.fish_type_available:
        required.add("fish_type")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Split data is missing required columns: " + ", ".join(missing)
        )
    null_columns = sorted(column for column in required if df[column].null_count() > 0)
    if null_columns:
        raise ValueError(
            "Split data contains null required values in: " + ", ".join(null_columns)
        )
    if df["name"].n_unique() != df.height:
        raise ValueError("Split image names must be unique")
    memberships = pl.sum_horizontal(
        [pl.col(column).cast(pl.Int8) for column in ("is_train", "is_val", "is_test")]
    )
    if df.filter(memberships != 1).height:
        raise ValueError("Every split row must belong to exactly one split")


def _pipeline_steps(config: Config) -> list[PipelineStep]:
    steps: list[PipelineStep] = [
        YoloStep(config, initial=True),
        RotateStep(config),
        YoloStep(config),
    ]
    if any(config.dataset.depth):
        steps.append(DepthStep(config))
    steps.extend([SegmentStep(config), BlackoutStep(config)])
    if "features" in config.dataset.feature_sets:
        steps.append(VLMStep(config))
    steps.append(FeatureStep(config))
    return steps


def run_pipeline(config: Config):
    print(f"Starting preprocessing pipeline for {config.dataset.name}...")

    if not config.dataset.split_csv_path.exists():
        raise FileNotFoundError(
            f"Run create data module first. {config.dataset.split_csv_path} not found."
        )

    config.dataset.output_dir.mkdir(exist_ok=True)

    df = pl.read_csv(config.dataset.split_csv_path)
    _validate_split(df, config)
    steps = _pipeline_steps(config)
    stage_reports = []

    for step in steps:
        step_name = step.__class__.__name__
        print(f"Running {step_name}...")
        input_names = set(df["name"].to_list())
        input_count = df.height
        df = step.process(df)
        output_names = set(df["name"].to_list())
        stage_reports.append(
            {
                "stage": step_name,
                "input_count": input_count,
                "output_count": df.height,
                "dropped_count": input_count - df.height,
                "dropped_names": sorted(input_names - output_names),
            }
        )

    temporary_csv = config.dataset.output_csv_path.with_suffix(".csv.tmp")
    df.write_csv(temporary_csv)
    os.replace(temporary_csv, config.dataset.output_csv_path)
    atomic_write_json(
        config.dataset.output_dir / "preprocessing_report.json",
        {
            "schema": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": config.dataset.name,
            "input_count": stage_reports[0]["input_count"]
            if stage_reports
            else df.height,
            "output_count": df.height,
            "stages": stage_reports,
        },
    )
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
