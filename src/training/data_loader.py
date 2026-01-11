"""
Shared data loading utilities for training scripts.

This module provides the load_data function used by all training scripts.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Optional


def load_data(
    split_path: str,
    feature_sets: List[str] = ["coords"],
    depth_model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare data for training.

    Args:
        split_path: Path to the processed split CSV
        feature_sets: List of feature sets to include ("coords", "scaled", "eye")
        depth_model: If provided, includes depth features

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        names: List of image names

    Raises:
        ValueError: If required columns are missing
    """
    df = pl.read_csv(split_path)

    required = []

    # Base Features
    for fs in feature_sets:
        if fs == "coords":
            required.extend(
                ["Fish_w", "Fish_h", "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"]
            )
        elif fs == "eye":
            required.extend(["Eye_w", "Eye_h", "Fish_w", "Fish_h"])
        elif fs == "scaled":
            required.extend(["Fish_w_scaled", "Fish_h_scaled"])

    # Depth Features
    if depth_model:
        required.append("head_center_depth")
        required.append("tail_center_depth")

    # Auxiliary Features (Stats & Fish Type)
    # Auto-detect if they exist
    aux_cols = [
        c for c in df.columns if c.startswith("fish_type_") or c.startswith("species_")
    ]
    required.extend(aux_cols)

    # Drop rows missing features
    # Check if columns exist
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        # Filter out missing auxiliary columns from requirement if they prefer?
        # But for base features, we must fail.
        # Actually, for reliability, let's fail if base features missing, but warn/skip for aux?
        # "scaled" is a base feature now.

        # Simplify: Check base features.
        base_req = [c for c in required if not (c.startswith("fish_type_"))]
        missing_base = [c for c in base_req if c not in df.columns]
        if missing_base:
            raise ValueError(f"Missing required base columns: {missing_base}")

        # For aux, if missing, we just don't include them?
        # But we added them to `required` because we found them in `df.columns`!
        # Wait, logic above: `aux_cols` comes from `df.columns`. So they exist.
        # So `missing_cols` should only contain missing BASE features.
        pass

    df = df.drop_nulls(subset=required)

    X = df.select(required).to_numpy()
    y = df.select("length").to_numpy().ravel()
    names = df.select("name").to_series().to_list()

    return X, y, names


def get_feature_description(
    feature_sets: List[str], depth_model: Optional[str] = None
) -> str:
    """
    Generate a feature description string for model naming.

    Args:
        feature_sets: List of feature sets used
        depth_model: Depth model name if used

    Returns:
        Feature description string (e.g., "coords_scaled_depth")
    """
    feature_desc = "_".join(sorted(feature_sets))
    if depth_model:
        feature_desc += f"_{depth_model}"
    return feature_desc
