"""
Training Orchestrator

Runs preprocessing and training pipelines for different configurations.

Usage:
    python -m src.training.run --pipeline 1 --dataset data-inside
    python -m src.training.run  # Runs default tasks
"""
from genericpath import exists

import os
import random
from typing import Callable

import numpy as np
import polars as pl
import torch
import typer

from src.config import Config, get_config
from src.training.models import (
    train_baseline,
    train_cnn_model,
    train_linear_model,
    train_mlp_model,
    train_xgboost_model,
    train_efficientnet_ridge_model,
)

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


pipeline_function = Callable[[pl.DataFrame, Config, str, bool, bool], pl.DataFrame]


def run_pipeline(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_sets: list[str],
    depth_flag: list[bool],
    pred_df: pl.DataFrame,
) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred = task(df, config, feature_set, depth, False)
            pred_df = pred_df.join(pred, on="name", how="left")
    return pred_df


def run_per_fish_task(
    task: pipeline_function,
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool,
    pred_df: pl.DataFrame,
) -> pl.DataFrame:
    per_fish_pred = []
    for fish_type in df["fish_type"].unique():
        data = df.filter(pl.col("fish_type") == fish_type)
        train_rows = data.filter(pl.col("is_train")).height
        if train_rows < MIN_PER_TYPE_TRAIN_ROWS:
            print(
                f"Skipping per-type model for {fish_type}: "
                f"{train_rows} train rows (< {MIN_PER_TYPE_TRAIN_ROWS})"
            )
            continue
        per_fish_pred.append(task(data, config, feature_set, depth, True))

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
) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred_df = run_per_fish_task(task, df, config, feature_set, depth, pred_df)
    return pred_df


def save_metrics(pred_df: pl.DataFrame, dataset_name: str, split: str) -> None:
    """Write per-model mae/mape/rmse/r2 for one split to reports/<split>.csv."""
    path = os.path.join("reports", dataset_name, split[3:] + ".csv")
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

    report.sort("mape").with_columns(pl.selectors.numeric().round(2)).write_csv(path)


@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Specific dataset"),
):
    seed_everything()
    config = get_config(dataset_name)
    df = pl.read_csv(config.dataset.output_csv_path).drop_nulls()

    pred_path = config.dataset.dataset_dir / "predictions.csv"
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
        pred_df = run_pipeline(task, df, config, feature_sets, depth_flags, pred_df)

    if config.dataset.fish_type_available:
        for task in tasks:
            pred_df = run_per_fish_pipeline(
                task, df, config, feature_sets, depth_flags, pred_df
            )

    if "data-outside" == dataset_name:
        pred_df = run_pipeline(
            train_efficientnet_ridge_model,
            df,
            config,
            feature_sets,
            depth_flags,
            pred_df,
        )

    pred_df = df.select(cols).join(pred_df, on="name", how="left")
    pred_df.write_csv(pred_path)

    os.makedirs("reports", exist_ok=True)
    os.makedirs(os.path.join("reports", dataset_name), exist_ok=True)
    for split in ("is_train", "is_val", "is_test"):
        save_metrics(pred_df, dataset_name, split)


if __name__ == "__main__":
    app()
