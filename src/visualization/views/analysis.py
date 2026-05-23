"""Error Analysis view for investigating prediction errors."""

import streamlit as st
import polars as pl

from src.visualization.data_loading import (
    calculate_metrics,
    get_prediction_fish_types,
    get_prediction_model_columns,
    get_prediction_splits,
    load_prediction_df,
    normalize_predictions,
)
from src.visualization.image_processing import process_images


def render_analysis(dataset, df_meta, all_image_names, depth_model):
    """Render the Error Analysis view."""
    st.header(f"Error Analysis: {dataset}")

    df_pred = load_prediction_df(dataset)
    if df_pred is None:
        st.warning(f"No prediction CSV found at `data/{dataset}/predictions.csv`.")
        return

    models = get_prediction_model_columns(df_pred)
    if not models:
        st.error("No model prediction columns found.")
        return

    splits = get_prediction_splits(df_pred)
    fish_types = get_prediction_fish_types(df_pred)

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_model = st.selectbox("Model", models)
    with c2:
        selected_split = st.selectbox("Split", ["all"] + splits)
    with c3:
        selected_fish_types = (
            st.multiselect("Fish Type", fish_types, default=[]) if fish_types else []
        )

    df = normalize_predictions(
        df_pred,
        model_columns=[selected_model],
        split=None if selected_split == "all" else selected_split,
        fish_types=selected_fish_types or None,
    )
    if df is None or df.height == 0:
        st.warning("No prediction rows match the selected filters.")
        return

    metrics = calculate_metrics(df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics['mae']:.2f}")
    m2.metric("MAPE", f"{metrics['mape']:.2f}%")
    m3.metric("R2", f"{metrics['r2']:.4f}")
    m4.metric("Samples", metrics["samples"])

    st.subheader("Deep Dive")
    sort_by = st.radio(
        "Sort By",
        ["Percentage Error (Descending)", "Absolute Error (Descending)"],
        horizontal=True,
    )
    df_sorted = df.sort(
        "mape" if "Percentage" in sort_by else "abs_error", descending=True
    )

    st.dataframe(
        df_sorted.select(
            ["name", "fish_type", "gt_length", "pred_length", "abs_error", "mape"]
        ).with_columns(pl.selectors.numeric().round(3)),
        width="stretch",
        hide_index=True,
    )

    top_names = df_sorted["name"].to_list()
    if not top_names:
        return

    st.markdown("---")
    st.markdown("### Visualize Specific Image")
    selected_err_img = st.selectbox("Select Image (Sorted by Error)", top_names)
    img_row = df_sorted.filter(pl.col("name") == selected_err_img).to_dicts()[0]

    st.info(
        f"GT: {img_row['gt_length']:.2f} | Pred: {img_row['pred_length']:.2f} | "
        f"Error: {img_row['abs_error']:.2f} ({img_row['mape']:.1f}%)"
    )

    row_data = {}
    if df_meta is not None:
        r = df_meta.filter(pl.col("name") == selected_err_img)
        if r.height > 0:
            row_data = {k: v for k, v in r.to_dicts()[0].items() if v is not None}

    row_data["Specific Prediction"] = {
        "Model": selected_model,
        "GT": img_row["gt_length"],
        "Pred": img_row["pred_length"],
    }
    if img_row.get("fish_type") is not None:
        row_data["Specific Prediction"]["Fish Type"] = img_row["fish_type"]

    img_raw, img_rot, img_depth, _ = process_images(
        dataset, selected_err_img, row_data, depth_model
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_raw, caption="Raw", width="stretch")
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
