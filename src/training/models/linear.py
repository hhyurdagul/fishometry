"""
Linear Regression Training

Trains a linear regression model with StandardScaler.

Usage:
    python -m src.training.linear --dataset data-inside --feature-set coords
    python -m src.training.linear --dataset data-outside --feature-set scaled --depth
"""

import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from src.training.data_loader import get_feature_names_and_desc
from src.config import Config


def train_linear_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool=False,
):
    features, feature_desc = get_feature_names_and_desc("linear", feature_set, depth)

    print(f"Training Regression Model: {feature_desc}")

    X_train = df.filter(pl.col("is_train")).select(features).to_numpy()
    y_train = df.filter(pl.col("is_train")).select("length").to_numpy().ravel()

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    y_pred = model.predict(df.select(features).to_numpy())
    pred_df = df.with_columns(pl.lit(y_pred).round(2).alias(feature_desc)).select("name", feature_desc)

    pred_path = config.dataset_dir / "predictions.csv"
    saved_df = pl.read_csv(pred_path).drop(feature_desc, strict=False)
    saved_df.join(pred_df, on="name", how="left").write_csv(pred_path)

    # Eval & Save Predictions
    model_dir = os.path.join("checkpoints", config.name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, feature_desc + ".joblib"))

