"""
Baseline Model Training

Trains a simple baseline model using mean prediction per fish type.
"""

import polars as pl
from src.config import Config

# feature_set and depth are for compatibility with other models
def train_baseline(df: pl.DataFrame, config: Config, feature_set: str="", depth: bool=False) -> None:
    print("Training Baseline Regression Model (Mean)")

    pred_name = "mean_regression"
    if config.fish_type_available:
        mean_stats = (
            df.filter(pl.col("is_train"))
            .group_by("fish_type")
            .agg(
                pl.col("length").mean().round(2).alias(pred_name),
            )
        )

        pred_df = df.join(mean_stats, on="fish_type", how="left").select("name", pred_name)
    else:
        global_mean = df.filter(pl.col("is_train"))["length"].mean()
        pred_df = df.with_columns(pl.lit(global_mean).alias(pred_name).round(2)).select("name", pred_name)

    pred_path = config.dataset_dir / "predictions.csv"
    saved_df = pl.read_csv(pred_path).drop(pred_name, strict=False)
    saved_df.join(pred_df, on="name", how="left").write_csv(pred_path)

