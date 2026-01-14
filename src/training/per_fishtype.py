"""
Per-Fish-Type Training Script

Trains Linear Regression, XGBoost, and MLP models for each fish_type separately,
plus generates merged predictions combining per-type model outputs.

Usage:
    python -m src.training.per_fishtype --dataset data-outside --feature-set scaled --depth --epochs 200
"""

import argparse
import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import joblib

from src.training.data_loader import load_data, get_feature_description
from src.training.mlp import FishMLP, TabularDataset


MIN_SAMPLES_WARNING = 30


def get_fish_types(dataset_name):
    """Get unique fish types from the dataset."""
    base_dir = f"data/{dataset_name}/processed"
    train_path = f"{base_dir}/processed_train.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pl.read_csv(train_path)

    if "fish_type" not in df.columns:
        raise ValueError("Dataset does not contain 'fish_type' column")

    fish_types = df["fish_type"].unique().sort().to_list()
    return fish_types


def filter_data_by_fishtype(X, y, names, df, fish_type):
    """Filter data arrays by fish_type."""
    # Get indices where fish_type matches
    mask = df["fish_type"] == fish_type
    indices = [i for i, m in enumerate(mask.to_list()) if m]

    X_filtered = X[indices]
    y_filtered = y[indices]
    names_filtered = [names[i] for i in indices]

    return X_filtered, y_filtered, names_filtered


def load_data_with_fishtype(split_path, feature_sets, depth_model=None):
    """Load data and also return the full dataframe for filtering."""
    df = pl.read_csv(split_path)
    X, y, names = load_data(split_path, feature_sets, depth_model)

    # We need to filter df to match X, y, names (after null drops)
    # Rebuild df with same filtering logic
    required = []
    for fs in feature_sets:
        if fs == "coords":
            required.extend(
                ["Fish_w", "Fish_h", "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"]
            )
        elif fs == "eye":
            required.extend(["Eye_w", "Eye_h", "Fish_w", "Fish_h"])
        elif fs == "scaled":
            required.extend(["Fish_w_scaled", "Fish_h_scaled"])

    if depth_model:
        required.append("head_center_depth")
        required.append("tail_center_depth")

    aux_cols = [
        c for c in df.columns if c.startswith("fish_type_") or c.startswith("species_")
    ]
    required.extend(aux_cols)

    # Filter same way as load_data
    required = [c for c in required if c in df.columns]
    df_filtered = df.drop_nulls(subset=required)

    return X, y, names, df_filtered


def train_linear_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    names_train,
    names_val,
    names_test,
    dataset_name,
    feature_desc,
    fish_type=None,
):
    """Train linear regression model."""
    suffix = f"_{fish_type}" if fish_type else ""
    model_name = f"linear{suffix}_{feature_desc}"

    if len(X_train) == 0:
        print(f"  [LR] No training samples for {fish_type or 'all'}")
        return None

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)

    # Save model
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"{model_name}.joblib"))

    # Save predictions
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    results = {}
    for split_name, X, y, names in [
        ("train", X_train, y_train, names_train),
        ("val", X_val, y_val, names_val),
        ("test", X_test, y_test, names_test),
    ]:
        if len(X) == 0:
            continue

        pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, pred)

        df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(y, 1),
                "pred_length": np.round(pred, 1),
            }
        )

        # Only save individual predictions for non-fish-type-specific models
        if fish_type is None:
            filename = f"linear{suffix}_{feature_desc}_{split_name}.csv"
            df.write_csv(os.path.join(pred_dir, filename))
        results[split_name] = {"mape": mape, "predictions": df}

    if "val" in results:
        print(f"  [LR] {fish_type or 'all'}: Val MAPE = {results['val']['mape']:.4f}")

    return results


def train_xgboost_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    names_train,
    names_val,
    names_test,
    dataset_name,
    feature_desc,
    fish_type=None,
):
    """Train XGBoost model."""
    suffix = f"_{fish_type}" if fish_type else ""
    model_name = f"xgboost{suffix}_{feature_desc}"

    if len(X_train) == 0:
        print(f"  [XGB] No training samples for {fish_type or 'all'}")
        return None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else None
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else None

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10,
    )

    eval_set = (
        [(X_val_scaled, y_val)] if X_val_scaled is not None and len(y_val) > 0 else None
    )
    model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

    # Save model and scaler
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, f"{model_name}.json"))
    joblib.dump(scaler, os.path.join(model_dir, f"{model_name}_scaler.joblib"))

    # Save predictions
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    results = {}
    for split_name, X_scaled, y, names in [
        ("train", X_train_scaled, y_train, names_train),
        ("val", X_val_scaled, y_val, names_val),
        ("test", X_test_scaled, y_test, names_test),
    ]:
        if X_scaled is None or len(X_scaled) == 0:
            continue

        pred = model.predict(X_scaled)
        mape = mean_absolute_percentage_error(y, pred)

        df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(y, 1),
                "pred_length": np.round(pred, 1),
            }
        )

        # Only save individual predictions for non-fish-type-specific models
        if fish_type is None:
            filename = f"xgboost{suffix}_{feature_desc}_{split_name}.csv"
            df.write_csv(os.path.join(pred_dir, filename))
        results[split_name] = {"mape": mape, "predictions": df}

    if "val" in results:
        print(f"  [XGB] {fish_type or 'all'}: Val MAPE = {results['val']['mape']:.4f}")

    return results


def train_mlp_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    names_train,
    names_val,
    names_test,
    dataset_name,
    feature_desc,
    fish_type=None,
    epochs=200,
):
    """Train MLP model."""
    suffix = f"_{fish_type}" if fish_type else ""
    model_name = f"mlp{suffix}_{feature_desc}"

    if len(X_train) == 0:
        print(f"  [MLP] No training samples for {fish_type or 'all'}")
        return None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else np.array([])
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.array([])

    # Create datasets
    train_set = TabularDataset(X_train_scaled, y_train, names_train)
    val_set = TabularDataset(X_val_scaled, y_val, names_val) if len(X_val) > 0 else None
    test_set = (
        TabularDataset(X_test_scaled, y_test, names_test) if len(X_test) > 0 else None
    )

    batch_size = min(16, len(X_train))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size) if val_set else None
    test_loader = DataLoader(test_set, batch_size=batch_size) if test_set else None

    # Build model
    input_dim = X_train_scaled.shape[1]
    model = FishMLP(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Save paths
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{model_name}.pth")

    best_mape = float("inf")

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).flatten()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        if val_loader:
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for X_batch, y_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).flatten()
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(y_batch.numpy())

            val_mape = mean_absolute_percentage_error(val_targets, val_preds)

            if val_mape < best_mape:
                best_mape = val_mape
                torch.save(model.state_dict(), save_path)

    # Load best model
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, f"{model_name}_scaler.joblib"))

    # Save predictions
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    results = {}
    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        if not loader:
            continue

        preds, targets, batch_names = [], [], []
        with torch.no_grad():
            for X_batch, y_batch, n_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).flatten()
                preds.extend(outputs.cpu().numpy())
                targets.extend(y_batch.numpy())
                batch_names.extend(n_batch)

        mape = mean_absolute_percentage_error(targets, preds)

        df = pl.DataFrame(
            {
                "name": batch_names,
                "gt_length": np.round(targets, 1),
                "pred_length": np.round(preds, 1),
            }
        )

        # Only save individual predictions for non-fish-type-specific models
        if fish_type is None:
            filename = f"mlp{suffix}_{feature_desc}_{split_name}.csv"
            df.write_csv(os.path.join(pred_dir, filename))
        results[split_name] = {"mape": mape, "predictions": df}

    if "val" in results:
        print(f"  [MLP] {fish_type or 'all'}: Val MAPE = {results['val']['mape']:.4f}")

    return results


def generate_merged_predictions(
    dataset_name, feature_desc, fish_types, model_type, all_results
):
    """Merge per-fish-type predictions into a single file.

    Args:
        dataset_name: Name of the dataset
        feature_desc: Feature description string
        fish_types: List of fish types
        model_type: Type of model (linear, xgboost, mlp)
        all_results: Dict mapping fish_type -> results dict from training
    """
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        merged_dfs = []

        for fish_type in fish_types:
            if fish_type in all_results and all_results[fish_type] is not None:
                results = all_results[fish_type]
                if split in results:
                    df = results[split]["predictions"]
                    merged_dfs.append(df)

        if merged_dfs:
            merged_df = pl.concat(merged_dfs)
            merged_df = merged_df.select(["name", "gt_length", "pred_length"])

            output_filename = f"{model_type}_merged_{feature_desc}_{split}.csv"
            merged_df.write_csv(os.path.join(pred_dir, output_filename))

            mape = mean_absolute_percentage_error(
                merged_df["gt_length"].to_numpy(), merged_df["pred_length"].to_numpy()
            )
            print(
                f"  [{model_type.upper()} MERGED] {split}: MAPE = {mape:.4f} ({len(merged_df)} samples)"
            )


def train_per_fishtype(dataset_name, feature_sets, depth_model=None, epochs=200):
    """Main function to train models per fish type."""
    feature_desc = get_feature_description(feature_sets, depth_model)

    print(f"\n{'=' * 60}")
    print(f"Per-Fish-Type Training: {dataset_name}")
    print(f"Features: {feature_desc}")
    print(f"{'=' * 60}\n")

    # Get fish types
    fish_types = get_fish_types(dataset_name)
    print(f"Found {len(fish_types)} fish types: {', '.join(fish_types)}\n")

    # Load full data
    base_dir = f"data/{dataset_name}/processed"

    try:
        X_train, y_train, names_train, df_train = load_data_with_fishtype(
            f"{base_dir}/processed_train.csv", feature_sets, depth_model
        )
        X_val, y_val, names_val, df_val = load_data_with_fishtype(
            f"{base_dir}/processed_val.csv", feature_sets, depth_model
        )
        X_test, y_test, names_test, df_test = load_data_with_fishtype(
            f"{base_dir}/processed_test.csv", feature_sets, depth_model
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    # Train on ALL data first
    print("\n--- Training on ALL data ---")
    train_linear_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        names_train,
        names_val,
        names_test,
        dataset_name,
        feature_desc,
        fish_type=None,
    )
    train_xgboost_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        names_train,
        names_val,
        names_test,
        dataset_name,
        feature_desc,
        fish_type=None,
    )
    train_mlp_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        names_train,
        names_val,
        names_test,
        dataset_name,
        feature_desc,
        fish_type=None,
        epochs=epochs,
    )

    # Train per fish type and collect results
    linear_results = {}
    xgboost_results = {}
    mlp_results = {}

    for fish_type in fish_types:
        print(f"\n--- Training for: {fish_type} ---")

        X_train_ft, y_train_ft, names_train_ft = filter_data_by_fishtype(
            X_train, y_train, names_train, df_train, fish_type
        )
        X_val_ft, y_val_ft, names_val_ft = filter_data_by_fishtype(
            X_val, y_val, names_val, df_val, fish_type
        )
        X_test_ft, y_test_ft, names_test_ft = filter_data_by_fishtype(
            X_test, y_test, names_test, df_test, fish_type
        )

        total_samples = len(X_train_ft) + len(X_val_ft) + len(X_test_ft)
        print(
            f"  Samples: Train={len(X_train_ft)}, Val={len(X_val_ft)}, Test={len(X_test_ft)}"
        )

        if total_samples < MIN_SAMPLES_WARNING:
            print(
                f"  WARNING: Low sample count ({total_samples}). Results may be unreliable."
            )

        if len(X_train_ft) < 5:
            print(f"  SKIPPING: Not enough training samples ({len(X_train_ft)} < 5)")
            continue

        linear_results[fish_type] = train_linear_regression(
            X_train_ft,
            y_train_ft,
            X_val_ft,
            y_val_ft,
            X_test_ft,
            y_test_ft,
            names_train_ft,
            names_val_ft,
            names_test_ft,
            dataset_name,
            feature_desc,
            fish_type=fish_type,
        )
        xgboost_results[fish_type] = train_xgboost_model(
            X_train_ft,
            y_train_ft,
            X_val_ft,
            y_val_ft,
            X_test_ft,
            y_test_ft,
            names_train_ft,
            names_val_ft,
            names_test_ft,
            dataset_name,
            feature_desc,
            fish_type=fish_type,
        )
        mlp_results[fish_type] = train_mlp_model(
            X_train_ft,
            y_train_ft,
            X_val_ft,
            y_val_ft,
            X_test_ft,
            y_test_ft,
            names_train_ft,
            names_val_ft,
            names_test_ft,
            dataset_name,
            feature_desc,
            fish_type=fish_type,
            epochs=epochs,
        )

    # Generate merged predictions
    print(f"\n--- Generating Merged Predictions ---")
    generate_merged_predictions(
        dataset_name, feature_desc, fish_types, "linear", linear_results
    )
    generate_merged_predictions(
        dataset_name, feature_desc, fish_types, "xgboost", xgboost_results
    )
    generate_merged_predictions(
        dataset_name, feature_desc, fish_types, "mlp", mlp_results
    )

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train models per fish type")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name e.g. data-outside"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        nargs="+",
        choices=["coords", "scaled", "eye"],
        required=True,
    )
    parser.add_argument("--depth", action="store_true", help="Use depth v2 features")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs for MLP"
    )
    args = parser.parse_args()

    train_per_fishtype(
        args.dataset,
        feature_sets=args.feature_set,
        depth_model="depth" if args.depth else None,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
