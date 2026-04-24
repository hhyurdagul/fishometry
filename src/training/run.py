"""
Training Orchestrator

Runs preprocessing and training pipelines for different configurations.

Usage:
    python -m src.training.run --pipeline 1 --dataset data-inside
    python -m src.training.run  # Runs default tasks
"""
from typing import Callable

import polars as pl
import typer

from src.config import Config, get_config
from src.training.models import (
    train_baseline,
    train_cnn_model,
    train_linear_model,
    train_mlp_model,
    train_xgboost_model,
)

app = typer.Typer(add_completion=False, help="Training orchestrator.")


pipeline_function = Callable[[pl.DataFrame, Config, str, bool, bool], pl.DataFrame]

def run_pipeline(task: pipeline_function, df: pl.DataFrame, config: Config, feature_sets: list[str], depth_flag: list[bool], pred_df: pl.DataFrame) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred = task(df, config, feature_set, depth, False)
            pred_df = pred_df.join(pred, on="name", how="left")
    return pred_df


def run_per_fish_task(task: pipeline_function, df: pl.DataFrame, config: Config, feature_set: str, depth: bool, pred_df: pl.DataFrame) -> pl.DataFrame:
    per_fish_pred = []
    for fish_type in df["fish_type"].unique():
        data = df.filter(pl.col("fish_type") == fish_type)
        per_fish_pred.append(
            task(data, config, feature_set, depth, True)
        )
    return pred_df.join(pl.concat(per_fish_pred), on="name", how="left")


def run_per_fish_pipeline(task: pipeline_function, df: pl.DataFrame, config: Config, feature_sets: list[str], depth_flag: list[bool], pred_df: pl.DataFrame) -> pl.DataFrame:
    for feature_set in feature_sets:
        for depth in depth_flag:
            pred_df = run_per_fish_task(task, df, config, feature_set, depth, pred_df)
    return pred_df

@app.command()
def main(
    dataset_name: str = typer.Option(None, help="Specific dataset"),
):
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
            pred_df = run_per_fish_pipeline(task, df, config, feature_sets, depth_flags, pred_df)

    df.select(cols).join(pred_df, on="name", how="left").write_csv(pred_path)
 

if __name__ == "__main__":
    app()
