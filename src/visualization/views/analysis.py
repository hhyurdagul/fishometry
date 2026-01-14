"""Error Analysis view for investigating prediction errors."""

import os
import streamlit as st
import polars as pl
import numpy as np

from data_loading import load_predictions_list, load_prediction_df
from image_processing import process_images


def render_analysis(dataset, df_meta, all_image_names, depth_model):
    """Render the Error Analysis view."""
    st.header(f"Error Analysis: {dataset}")

    pred_files = load_predictions_list(dataset)
    if not pred_files:
        st.warning("No prediction CSV files found in `data/{dataset}/predictions`.")
        return

    # File Selector
    pred_file_map = {os.path.basename(f): f for f in pred_files}
    selected_file_name = st.selectbox(
        "Select Prediction Model", list(pred_file_map.keys())
    )
    selected_file_path = pred_file_map[selected_file_name]

    # Load and Compute Metrics
    df_pred = load_prediction_df(selected_file_path)

    required_cols = ["name", "gt_length", "pred_length"]
    if not all(col in df_pred.columns for col in required_cols):
        st.error(f"Selected CSV must contain columns: {required_cols}")
        st.dataframe(df_pred.head())
        return

    # Calculate Errors
    df_pred = df_pred.with_columns(
        [pl.col("gt_length").cast(pl.Float64), pl.col("pred_length").cast(pl.Float64)]
    )

    df_pred = df_pred.with_columns(
        [
            (pl.col("gt_length") - pl.col("pred_length")).abs().alias("abs_error"),
            (
                (pl.col("gt_length") - pl.col("pred_length")).abs()
                / pl.col("gt_length")
                * 100
            ).alias("pct_error"),
        ]
    )

    # Metrics Summary
    mae = df_pred["abs_error"].mean()
    mape = df_pred["pct_error"].mean()

    # Calculate R²
    gt = df_pred["gt_length"].to_numpy()
    pred = df_pred["pred_length"].to_numpy()
    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("MAPE", f"{mape:.2f}%")
    m3.metric("R²", f"{r2:.4f}")
    m4.metric("Samples", df_pred.height)

    # Specific Image Analysis
    st.subheader("Deep Dive")

    # Sort options
    sort_by = st.radio(
        "Sort By",
        ["Percentage Error (Descending)", "Absolute Error (Descending)"],
        horizontal=True,
    )

    if "Percentage" in sort_by:
        df_sorted = df_pred.sort("pct_error", descending=True)
    else:
        df_sorted = df_pred.sort("abs_error", descending=True)

    # Display interactive table
    st.dataframe(
        df_sorted.select(
            ["name", "gt_length", "pred_length", "abs_error", "pct_error"]
        ),
        width="stretch",
        hide_index=True,
    )

    top_n_names = df_sorted["name"].to_list()

    st.markdown("---")
    st.markdown("### Visualize Specific Image")

    selected_err_img = st.selectbox("Select Image (Sorted by Error)", top_n_names)

    # Retrieve specific error stats for this image
    img_row = df_sorted.filter(pl.col("name") == selected_err_img).to_dicts()[0]

    st.info(
        f"GT: {img_row['gt_length']:.2f} | Pred: {img_row['pred_length']:.2f} | Error: {img_row['abs_error']:.2f} ({img_row['pct_error']:.1f}%)"
    )

    row_data = {}
    if df_meta is not None:
        r = df_meta.filter(pl.col("name") == selected_err_img)
        if r.height > 0:
            row_data = r.to_dicts()[0]
            row_data = {k: v for k, v in row_data.items() if v is not None}

    row_data["Specific Prediction"] = {
        "Model": selected_file_name,
        "GT": img_row["gt_length"],
        "Pred": img_row["pred_length"],
    }

    img_raw, img_rot, img_depth, img_seg = process_images(
        dataset, selected_err_img, row_data, depth_model
    )

    # Simplified Grid for Analysis
    c1, c2, c3 = st.columns(3)
    with c1:
        if img_raw is not None:
            st.image(img_raw, caption="Raw", width="stretch")
        else:
            st.info("No raw image found")
    with c2:
        if img_rot is not None:
            st.image(img_rot, caption="Rotated/Annotated", width="stretch")
        else:
            st.info("No rotated image found")
    with c3:
        if img_depth is not None:
            st.image(img_depth, caption="Depth", width="stretch")
        else:
            st.info("No depth map found")
