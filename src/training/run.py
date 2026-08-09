"""
Training Orchestrator

Runs preprocessing and training pipelines for different configurations.

Usage:
    python -m src.training.run --pipeline 1 --dataset data-inside
    python -m src.training.run  # Runs default tasks
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import random
import shutil
from typing import Callable
from uuid import uuid4

import numpy as np
import polars as pl
import torch
import typer

from src.artifacts import atomic_write_json, file_sha256
from src.config import Config, get_config
from src.training.models import (
    train_baseline,
    train_cnn_model,
    train_linear_model,
    train_mlp_model,
    train_xgboost_model,
    train_efficientnet_ridge_model,
    train_dino_ridge_model,
)
from src.training.data_loader import get_feature_names_and_desc

app = typer.Typer(add_completion=False, help="Training orchestrator.")

RANDOM_SEED = 42
MIN_PER_TYPE_TRAIN_ROWS = 10


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and Torch so MLP/CNN runs are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


pipeline_function = Callable[..., pl.DataFrame]


def _invoke_task(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool,
    per_type: bool,
    checkpoint_dir: Path | None,
) -> pl.DataFrame:
    arguments = (df, config, feature_set, depth, per_type)
    if checkpoint_dir is None:
        return task(*arguments)
    return task(*arguments, checkpoint_dir)


def run_pipeline(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_sets: list[str],
    depth_flag: list[bool],
    pred_df: pl.DataFrame,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred = _invoke_task(
                task,
                df,
                config,
                feature_set,
                depth,
                False,
                checkpoint_dir,
            )
            pred_df = pred_df.join(pred, on="name", how="left")
    return pred_df


def run_per_fish_task(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool,
    pred_df: pl.DataFrame,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    per_fish_pred = []
    for fish_type in sorted(df["fish_type"].unique().to_list()):
        data = df.filter(pl.col("fish_type") == fish_type)
        train_rows = data.filter(pl.col("is_train")).height
        if train_rows < MIN_PER_TYPE_TRAIN_ROWS:
            print(
                f"Skipping per-type model for {fish_type}: "
                f"{train_rows} train rows (< {MIN_PER_TYPE_TRAIN_ROWS})"
            )
            continue
        per_fish_pred.append(
            _invoke_task(
                task,
                data,
                config,
                feature_set,
                depth,
                True,
                checkpoint_dir,
            )
        )

    if not per_fish_pred:
        return pred_df
    return pred_df.join(pl.concat(per_fish_pred), on="name", how="left")


def run_per_fish_pipeline(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_sets: list[str],
    depth_flag: list[bool],
    pred_df: pl.DataFrame,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred_df = run_per_fish_task(
                task,
                df,
                config,
                feature_set,
                depth,
                pred_df,
                checkpoint_dir,
            )
    return pred_df


def save_metrics(pred_df: pl.DataFrame, split: str, output_dir: Path) -> Path:
    """Write per-model metrics for one split into an immutable run directory."""
    path = output_dir / f"{split[3:]}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pred_df.filter(pl.col(split))

    target = pl.col("length")
    error = pl.selectors.numeric().exclude("length") - target
    # Total sum of squares of the target, shared by every model column.
    sstot = ((target - target.mean()) ** 2).sum()

    metrics = {
        "mae": error.abs().mean(),
        "mape": (error / target).abs().mean() * 100,
        "rmse": (error**2).mean().sqrt(),
        # Written as `-SSres / SStot + 1` so the model column names survive.
        "r2": -(error**2).sum() / sstot + 1,
    }

    report = None
    for metric, expr in metrics.items():
        values = data.select(expr).unpivot(variable_name="model", value_name=metric)
        report = values if report is None else report.join(values, on="model")

    if report is None:
        raise ValueError(f"No prediction columns available for {split}")
    report.sort("mape").with_columns(pl.selectors.numeric().round(2)).write_csv(path)
    return path


def validate_training_frame(df: pl.DataFrame, config: Config) -> None:
    required = {"name", "length", "is_train", "is_val", "is_test"}
    if config.dataset.fish_type_available:
        required.add("fish_type")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Processed data is missing required columns: " + ", ".join(missing)
        )
    if df["name"].n_unique() != df.height:
        raise ValueError("Processed image names must be unique")

    for column in ("is_train", "is_val", "is_test"):
        if df.schema[column] != pl.Boolean:
            raise ValueError(f"{column} must be a boolean column")
    split_count = pl.sum_horizontal(
        [pl.col(column).cast(pl.Int8) for column in ("is_train", "is_val", "is_test")]
    )
    if df.filter(split_count != 1).height:
        raise ValueError("Every processed row must belong to exactly one split")
    for column in ("is_train", "is_val", "is_test"):
        if df.filter(pl.col(column)).is_empty():
            raise ValueError(f"Processed data has no rows for {column}")

    selected_columns = set(required)
    for feature_set in config.dataset.feature_sets:
        for depth in config.dataset.depth:
            expressions, _ = get_feature_names_and_desc(
                "validation", feature_set, depth
            )
            selected_columns.update(df.select(expressions).columns)
    null_columns = sorted(
        column for column in selected_columns if df[column].null_count() > 0
    )
    if null_columns:
        raise ValueError(
            "Configured training columns contain null values: "
            + ", ".join(null_columns)
        )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Specific dataset"),
):
    seed_everything()
    config = get_config(dataset_name)
    df = pl.read_csv(config.dataset.output_csv_path)
    validate_training_frame(df, config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    checkpoint_root = Path("checkpoints") / dataset_name
    checkpoint_run_dir = checkpoint_root / "runs" / run_id
    report_root = Path("reports") / dataset_name
    report_run_dir = report_root / "runs" / run_id
    prediction_run_dir = config.dataset.dataset_dir / "runs" / run_id
    checkpoint_run_dir.mkdir(parents=True, exist_ok=False)
    prediction_run_dir.mkdir(parents=True, exist_ok=False)

    cols = ["name", "length", "is_train", "is_val", "is_test"]
    if config.dataset.fish_type_available:
        cols.insert(1, "fish_type")

    tasks = [
        train_linear_model,
        train_xgboost_model,
        train_mlp_model,
        train_cnn_model,
    ]
    feature_sets = config.dataset.feature_sets
    depth_flags = config.dataset.depth

    pred_df = train_baseline(df, config, "", False)
    for task in tasks:
        pred_df = run_pipeline(
            task,
            df,
            config,
            feature_sets,
            depth_flags,
            pred_df,
            checkpoint_run_dir,
        )

    if config.dataset.fish_type_available:
        for task in tasks:
            pred_df = run_per_fish_pipeline(
                task,
                df,
                config,
                feature_sets,
                depth_flags,
                pred_df,
                checkpoint_run_dir,
            )

    if dataset_name == "data-outside":
        for image_task in (
            train_efficientnet_ridge_model,
            train_dino_ridge_model,
        ):
            pred_df = run_pipeline(
                image_task,
                df,
                config,
                feature_sets,
                depth_flags,
                pred_df,
                checkpoint_run_dir,
            )

    pred_df = df.select(cols).join(pred_df, on="name", how="left")
    prediction_run_path = prediction_run_dir / "predictions.csv"
    pred_df.write_csv(prediction_run_path)

    metric_paths = [
        save_metrics(pred_df, split, report_run_dir)
        for split in ("is_train", "is_val", "is_test")
    ]
    checkpoint_paths = sorted(
        path for path in checkpoint_run_dir.iterdir() if path.is_file()
    )
    manifest = {
        "schema": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "config": config.model_dump(mode="json"),
        "processed": {
            "path": str(config.dataset.output_csv_path),
            "sha256": file_sha256(config.dataset.output_csv_path),
        },
        "predictions": {
            "path": str(prediction_run_path),
            "sha256": file_sha256(prediction_run_path),
        },
        "checkpoints": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in checkpoint_paths
        ],
        "reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in metric_paths
        ],
    }
    atomic_write_json(checkpoint_run_dir / "manifest.json", manifest)

    _atomic_copy(
        prediction_run_path,
        config.dataset.dataset_dir / "predictions.csv",
    )
    for path in metric_paths:
        _atomic_copy(path, report_root / path.name)
    atomic_write_json(checkpoint_root / "current.json", manifest)


if __name__ == "__main__":
    app()
