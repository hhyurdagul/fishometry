"""
Shared data loading utilities for training scripts.

This module provides the load_data function used by all training scripts.
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Optional


def get_feature_names_and_desc(
    model_name: str = "linear",
    feature_set: str = "coords",
    depth: bool = False,
) -> tuple[list[pl.Expr], str]:
    """
    Load and prepare data for training.

    Args:
        feature_set: Feature set name
        depth: If depth model is used

    Returns:
        features: List if feature selectors
        feature_desc: Feature description string (e.g., "coords_depth")

    Raises:
        ValueError: If required columns are missing
    """

    features = [pl.selectors.starts_with("fish_type_")]
    feature_desc = f"{model_name}_{feature_set}"
    # Base Features
    if feature_set == "coords":
        features.append(
            pl.col([
                "img_w","img_h", "Fish_w", "Fish_h", "Fish_w_scaled", 
                "Fish_h_scaled", "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"
            ])
        )
    if feature_set == "scaled":
        features.append(
            pl.col([
                "img_w","img_h", "Fish_w", "Fish_h",
                "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"
            ])
        )
    elif feature_set == "eye":
        features.append(pl.col(["Eye_w", "Eye_h", "Fish_w", "Fish_h"]))

    # Depth Features
    if depth:
        features.append(pl.col(["head_center_depth", "fish_center_depth", "tail_center_depth"]))
        feature_desc = f"{feature_desc}_depth"

    return features, feature_desc

