"""
XGBoost Training

Trains an XGBoost regressor with early stopping.

Usage:
    python -m src.training.xgboost --dataset data-inside --feature-set coords
    python -m src.training.xgboost --dataset data-outside --feature-set scaled --depth
"""

import os
import numpy as np
import polars as pl
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import typer

from src.training.data_loader import load_data, get_feature_description

app = typer.Typer(add_completion=False, help="Train XGBoost model.")
VALID_FEATURE_SETS = {"coords", "scaled", "eye"}


def train_xgboost(
    dataset_name,
    feature_sets=["coords"],
    depth_model=None,
    include_stats=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
):
    feature_desc = get_feature_description(feature_sets, depth_model, include_stats)

    print(f"Training XGBoost on {dataset_name} with features: {feature_desc}...")

    base_dir = f"data/{dataset_name}/processed"

    # Load Data
    try:
        X_train, y_train, names_train = load_data(
            f"{base_dir}/processed_train.csv", feature_sets, depth_model, include_stats
        )
        X_val, y_val, names_val = load_data(
            f"{base_dir}/processed_val.csv", feature_sets, depth_model, include_stats
        )
        X_test, y_test, names_test = load_data(
            f"{base_dir}/processed_test.csv", feature_sets, depth_model, include_stats
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    print(
        f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}"
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else None
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else None

    # Train XGBoost with early stopping
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=10,
    )

    eval_set = (
        [(X_val_scaled, y_val)] if X_val_scaled is not None and len(y_val) > 0 else None
    )

    model.fit(
        X_train_scaled,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration}")

    # Save model and scaler
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"xgboost_{feature_desc}.json")
    model.save_model(model_path)

    scaler_path = os.path.join(model_dir, f"xgboost_scaler_{feature_desc}.joblib")
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")

    # Evaluation and save predictions
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for split_name, X_scaled, y, names in [
        ("train", X_train_scaled, y_train, names_train),
        ("val", X_val_scaled, y_val, names_val),
        ("test", X_test_scaled, y_test, names_test),
    ]:
        if X_scaled is None or len(X_scaled) == 0:
            print(f"{split_name}: No samples")
            continue

        pred = model.predict(X_scaled)
        mape = mean_absolute_percentage_error(y, pred)
        print(f"{split_name.capitalize()} MAPE: {mape:.4f}")

        df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(y, 1),
                "pred_length": np.round(pred, 1),
            }
        )
        df.write_csv(os.path.join(pred_dir, f"xgboost_{feature_desc}_{split_name}.csv"))


@app.command()
def main(
    dataset: str = typer.Option(..., help="Dataset name e.g. data-outside"),
    feature_set: list[str] = typer.Option(..., "--feature-set", help="Feature set"),
    depth: bool = typer.Option(False, help="Use depth v2 features"),
    stats: bool = typer.Option(False, help="Include species stats features"),
    n_estimators: int = typer.Option(100, "--n-estimators", help="Number of boosting rounds"),
    max_depth: int = typer.Option(6, help="Maximum tree depth"),
    learning_rate: float = typer.Option(0.1, help="Learning rate"),
):
    invalid = sorted(set(feature_set) - VALID_FEATURE_SETS)
    if invalid:
        raise typer.BadParameter(
            f"Unsupported feature set(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(sorted(VALID_FEATURE_SETS))}."
        )

    train_xgboost(
        dataset,
        feature_sets=feature_set,
        depth_model="depth" if depth else None,
        include_stats=stats,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    app()
