"""
Linear Regression Training

Trains a linear regression model with StandardScaler.

Usage:
    python -m src.training.regression --dataset data-inside --feature-set coords
    python -m src.training.regression --dataset data-outside --feature-set scaled --depth
"""

import argparse
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

from src.training.data_loader import load_data, get_feature_description


def train_and_eval(
    train_path, val_path, test_path, feature_sets, dataset_name, depth_model=None
):
    feature_desc = get_feature_description(feature_sets, depth_model)

    print(f"Training Regression Model (Features: {feature_desc})")

    try:
        X_train, y_train, names_train = load_data(train_path, feature_sets, depth_model)
        X_val, y_val, names_val = load_data(val_path, feature_sets, depth_model)
        X_test, y_test, names_test = load_data(test_path, feature_sets, depth_model)
    except ValueError as e:
        print(f"Skipping training due to data issue: {e}")
        return

    print(f"Train size: {len(X_train)}")

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

    model.fit(X_train, y_train)

    # Eval & Save Predictions

    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    for split_name, X, y, names in [
        ("train", X_train, y_train, names_train),
        ("val", X_val, y_val, names_val),
        ("test", X_test, y_test, names_test),
    ]:
        if len(X) == 0:
            print(f"{split_name}: No samples")
            continue

        pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, pred)
        print(f"{split_name.capitalize()} MAPE: {mape:.4f}")

        # Save output
        res_df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(y, 1),
                "pred_length": np.round(pred, 1),
            }
        )

        filename = f"{feature_desc}_{split_name}.csv"
        res_df.write_csv(os.path.join(pred_dir, filename))

    # Save Model
    save_name = f"regression_{feature_desc}.joblib"
    joblib.dump(model, os.path.join(model_dir, save_name))
    print(f"Model saved to {os.path.join(model_dir, save_name)}")


def main():
    parser = argparse.ArgumentParser(description="Train linear regression model")
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name e.g. data-inside"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        nargs="+",
        choices=["coords", "scaled", "eye"],
        required=True,
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Use depth v2 features",
    )
    args = parser.parse_args()

    base_dir = f"data/{args.dataset}/processed"
    train_path = f"{base_dir}/processed_train.csv"
    val_path = f"{base_dir}/processed_val.csv"
    test_path = f"{base_dir}/processed_test.csv"

    train_and_eval(
        train_path,
        val_path,
        test_path,
        args.feature_set,
        args.dataset,
        depth_model="depth" if args.depth else None,
    )


if __name__ == "__main__":
    main()
