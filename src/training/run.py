"""
Training Orchestrator

Runs preprocessing and training pipelines for different configurations.

Usage:
    python -m src.training.run --pipeline 1 --dataset data-inside
    python -m src.training.run  # Runs default tasks
"""
from src.training.models import (
    train_baseline,
    train_linear_model,
    train_mlp_model,
    train_xgboost_model,
    train_cnn_model,
)
from src.config import get_config

import sys
import subprocess
import polars as pl
import typer

app = typer.Typer(add_completion=False, help="Training orchestrator.")

@app.command()
def main(
    dataset_name: str = typer.Option(None, help="Specific dataset"),
):
    config = get_config(dataset_name)
    df = pl.read_csv(config.dataset.output_csv_path)

    pred_path = config.dataset.dataset_dir / "predictions.csv"

    if not pred_path.exists():
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
    depth = config.dataset.depth

    from itertools import product

    tasks_product = list(product(tasks, (df, ), (config,), feature_sets, depth))

    pred_df = train_baseline(df, config, "", False)
    for task, df, config, feature_set, depth_model in tasks_product:
        pred = task(df, config, feature_set, depth_model)
        pred_df.join(pred, on="name", how="left")

    df.select(cols).join(pred_df, on="name", how="left").write_csv(pred_path)
 

if __name__ == "__main__":
    app()
