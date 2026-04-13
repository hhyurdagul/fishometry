"""
Baseline Model Training

Trains a simple baseline model using mean prediction per fish type.
"""

import polars as pl
from src.config import Config

# feature_set and depth are for compatibility with other models
def train_baseline(df: pl.DataFrame, config: Config, feature_set: str="", depth: bool=False) -> pl.DataFrame:
    print("Training Baseline Regression Model (Mean)")

    pred_name = "mean_regression"
    if config.dataset.fish_type_available:
        mean_stats = (
            df.filter(pl.col("is_train"))
            .group_by("fish_type")
            .agg(
                pl.col("length").mean().round(2).alias(pred_name),
            )
        )

        pred = df.join(mean_stats, on="fish_type", how="left").select("name", pred_name)
    else:
        global_mean = df.filter(pl.col("is_train"))["length"].mean()
        pred = df.with_columns(pl.lit(global_mean).alias(pred_name).round(2)).select("name", pred_name)

    return pred

