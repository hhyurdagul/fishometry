"""Data loading utilities with Streamlit caching."""

import os
import streamlit as st
import polars as pl


PREDICTION_ID_COLUMNS = {"name", "fish_type", "length", "is_train", "is_val", "is_test"}


@st.cache_data
def get_datasets():
    """Get list of available datasets from data directory."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


@st.cache_data
def load_dataset_metadata(dataset):
    """Load and concatenate metadata from all splits for a dataset."""
    base_dir = f"data/{dataset}/processed"
    dfs = []

    single_file = f"data/{dataset}/processed.csv"
    if os.path.exists(single_file):
        try:
            full_df = pl.read_csv(single_file)
            names = full_df["name"].to_list() if "name" in full_df.columns else []
            return full_df, names
        except Exception as e:
            print(f"Error reading {single_file}: {e}")

    for split in ["train", "val", "test"]:
        path = f"{base_dir}/processed_{split}.csv"
        if os.path.exists(path):
            try:
                df = pl.read_csv(path)
                df = df.with_columns(pl.lit(split).alias("split"))
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if not dfs:
        return None, []

    full_df = pl.concat(dfs, how="diagonal")
    names = full_df["name"].to_list()
    return full_df, names


@st.cache_data
def load_prediction_df(dataset):
    """Load the wide prediction CSV for a dataset."""
    path = f"data/{dataset}/predictions.csv"
    if not os.path.exists(path):
        return None
    return pl.read_csv(path)


def get_prediction_model_columns(df_pred):
    """Return model columns from a wide prediction dataframe."""
    if df_pred is None:
        return []
    return [c for c in df_pred.columns if c not in PREDICTION_ID_COLUMNS]


def get_prediction_fish_types(df_pred):
    """Return available fish types if the prediction CSV has them."""
    if df_pred is None or "fish_type" not in df_pred.columns:
        return []
    return sorted(df_pred["fish_type"].drop_nulls().unique().to_list())


def get_prediction_splits(df_pred):
    """Return available split labels from boolean split columns."""
    if df_pred is None:
        return []
    splits = []
    for split in ["train", "val", "test"]:
        col = f"is_{split}"
        if col in df_pred.columns and df_pred.filter(pl.col(col)).height > 0:
            splits.append(split)
    return splits


def add_prediction_errors(df):
    """Add standard regression error columns."""
    return df.with_columns(
        [
            (pl.col("pred_length") - pl.col("gt_length")).alias("residual"),
            (pl.col("pred_length") - pl.col("gt_length")).abs().alias("abs_error"),
            (
                (pl.col("pred_length") - pl.col("gt_length")).abs()
                / pl.col("gt_length")
                * 100
            ).alias("mape"),
        ]
    )


def calculate_metrics(df):
    """Calculate MAE, MAPE, and R2 from normalized prediction data."""
    if df is None or df.height == 0:
        return None

    gt = df["gt_length"].to_numpy()
    pred = df["pred_length"].to_numpy()
    abs_error = abs(pred - gt)
    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()

    return {
        "mae": float(abs_error.mean()),
        "mape": float((abs_error / gt * 100).mean()),
        "r2": float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0,
        "samples": df.height,
    }


def normalize_predictions(df_pred, model_columns=None, split=None, fish_types=None):
    """Convert wide predictions into name/model/gt/pred rows."""
    if df_pred is None:
        return None

    required_cols = {"name", "length"}
    if not required_cols.issubset(df_pred.columns):
        return None

    if model_columns is None:
        model_columns = get_prediction_model_columns(df_pred)
    model_columns = [c for c in model_columns if c in df_pred.columns]
    if not model_columns:
        return None

    df = df_pred
    if split:
        split_col = f"is_{split}"
        if split_col in df.columns:
            df = df.filter(pl.col(split_col))

    if fish_types and "fish_type" in df.columns:
        df = df.filter(pl.col("fish_type").is_in(fish_types))

    id_vars = [c for c in ["name", "fish_type", "length"] if c in df.columns]
    df_long = df.unpivot(
        index=id_vars,
        on=model_columns,
        variable_name="model",
        value_name="pred_length",
    ).rename({"length": "gt_length"})

    if "fish_type" not in df_long.columns:
        df_long = df_long.with_columns(pl.lit(None).cast(pl.Utf8).alias("fish_type"))

    return add_prediction_errors(
        df_long.with_columns(
            [
                pl.col("gt_length").cast(pl.Float64),
                pl.col("pred_length").cast(pl.Float64),
            ]
        ).drop_nulls(["gt_length", "pred_length"])
    )


@st.cache_data
def load_all_predictions_for_image(dataset, image_name):
    """Load all model predictions for one image from the wide prediction CSV."""
    df_pred = load_prediction_df(dataset)
    if df_pred is None or "name" not in df_pred.columns:
        return {}

    row = df_pred.filter(pl.col("name") == image_name)
    if row.height == 0:
        return {}

    data = row.to_dicts()[0]
    return {model: data[model] for model in get_prediction_model_columns(df_pred)}
