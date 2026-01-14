"""Data loading utilities with Streamlit caching."""

import os
import glob
import streamlit as st
import polars as pl


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
def load_predictions_list(dataset):
    """Returns list of prediction CSV files available."""
    paths = [f"data/{dataset}/predictions", f"checkpoints/{dataset}/predictions"]
    files = []
    for p in paths:
        if os.path.exists(p):
            files.extend(glob.glob(os.path.join(p, "*.csv")))

    return sorted(list(set(files)))


@st.cache_data
def load_prediction_df(file_path):
    """Load a single prediction CSV file."""
    return pl.read_csv(file_path)


@st.cache_data
def load_all_predictions_for_image(dataset, image_name):
    """Loads all predictions for a single image across all prediction CSVs."""
    pred_files = load_predictions_list(dataset)
    preds = {}
    for f in pred_files:
        try:
            df = pl.read_csv(f)
            if "name" in df.columns and "pred_length" in df.columns:
                row = df.filter(pl.col("name") == image_name)
                if row.height > 0:
                    model_base = os.path.basename(f).replace(".csv", "")
                    preds[model_base] = row["pred_length"][0]
        except Exception:
            pass
    return preds


@st.cache_data
def get_depth_models(dataset):
    """Get list of available depth models for a dataset."""
    base_dir = f"data/{dataset}/processed/depth"
    if not os.path.exists(base_dir):
        return []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(dirs)
