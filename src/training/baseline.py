"""
Baseline Model Training

Trains a simple baseline model using mean prediction per fish type.

Usage:
    python -m src.training.baseline --dataset data-inside
"""

import argparse
import polars as pl
import numpy as np
import os
from sklearn.metrics import mean_absolute_percentage_error


def train_baseline(train_path, val_path, test_path, dataset_name):
    print("Training Baseline Regression Model (Mean)")

    # Load training data
    df_train = pl.read_csv(train_path)

    # Check for fish_type
    has_fish_type = "fish_type" in df_train.columns

    model_stats = {}

    if has_fish_type:
        print("Feature 'fish_type' found. Calculating mean per type.")
        # Calculate mean per group
        means = df_train.group_by("fish_type").agg(
            pl.col("length").mean().alias("mean_length")
        )
        # Handle case where a type in test might not be in train? Fallback to global mean.
        global_mean = df_train["length"].mean()
        model_stats = {
            "type": "grouped",
            "means": means,  # Polars DF
            "global_mean": global_mean,
        }
    else:
        print("Feature 'fish_type' not found. Calculating global mean.")
        global_mean = df_train["length"].mean()
        model_stats = {"type": "global", "mean": global_mean}

    # Predict and Save
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for split_name, path in [
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ]:
        df = pl.read_csv(path)
        if df.height == 0:
            continue

        names = df["name"]
        y_true = df["length"].to_numpy()

        preds = []
        if model_stats["type"] == "global":
            val = model_stats["mean"]
            preds = np.full(len(df), val)
        else:
            # Grouped prediction
            # Join with means
            # If fish_type missing in split (unlikely if in train), handle it.
            if "fish_type" not in df.columns:
                print(
                    f"Warning: fish_type missing in {split_name} but model is grouped. Using global mean."
                )
                preds = np.full(len(df), model_stats["global_mean"])
            else:
                # Join
                means_df = model_stats["means"]
                # We need to preserve order, so we join and then restore order or iterate
                # Join is safer
                # df with preds
                df_with_pred = df.join(means_df, on="fish_type", how="left")
                # Fill nulls (unseen types) with global mean
                df_with_pred = df_with_pred.with_columns(
                    pl.col("mean_length")
                    .fill_null(model_stats["global_mean"])
                    .alias("pred_length")
                )
                preds = df_with_pred["pred_length"].to_numpy()

        # Metric
        mape = mean_absolute_percentage_error(y_true, preds)
        print(f"{split_name.capitalize()} Baseline MAPE: {mape:.4f}")

        # Save
        res_df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(y_true, 1),
                "pred_length": np.round(preds, 1),
            }
        )
        res_df.write_csv(os.path.join(pred_dir, f"baseline_{split_name}.csv"))


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    base_dir = f"data/{args.dataset}/processed"
    train_path = f"{base_dir}/processed_train.csv"
    val_path = f"{base_dir}/processed_val.csv"
    test_path = f"{base_dir}/processed_test.csv"

    train_baseline(train_path, val_path, test_path, args.dataset)


if __name__ == "__main__":
    main()
