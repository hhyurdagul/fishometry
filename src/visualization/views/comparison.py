"""Model Comparison view for comparing prediction models."""

import os
import streamlit as st
import polars as pl
import numpy as np

from data_loading import load_predictions_list, load_prediction_df


def render_comparison(dataset):
    """Render the Model Comparison view."""
    st.header(f"Model Comparison: {dataset}")

    pred_files = load_predictions_list(dataset)
    if not pred_files:
        st.warning("No prediction files found.")
        return

    # User selects split
    split_choice = st.selectbox("Select Split to Compare", ["val", "test", "train"])

    # Filter files for this split
    relevant_files = [f for f in pred_files if f.endswith(f"_{split_choice}.csv")]

    if not relevant_files:
        st.info(f"No prediction files found for split '{split_choice}'")
        return

    results = []

    for f in relevant_files:
        base_name = os.path.basename(f)
        model_name = base_name.replace(f"_{split_choice}.csv", "")

        try:
            df = load_prediction_df(f)
            if "gt_length" not in df.columns or "pred_length" not in df.columns:
                continue

            df = df.with_columns(
                [
                    pl.col("gt_length").cast(pl.Float64),
                    pl.col("pred_length").cast(pl.Float64),
                ]
            )

            mae = (df["gt_length"] - df["pred_length"]).abs().mean()
            mape = (
                (df["gt_length"] - df["pred_length"]).abs() / df["gt_length"] * 100
            ).mean()

            # Calculate R²
            gt = df["gt_length"].to_numpy()
            pred = df["pred_length"].to_numpy()
            ss_res = ((pred - gt) ** 2).sum()
            ss_tot = ((gt - gt.mean()) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            results.append(
                {
                    "Model": model_name,
                    "MAPE (%)": mape,
                    "MAE": mae,
                    "R²": r2,
                    "Samples": df.height,
                }
            )
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not results:
        st.warning("Could not extract metrics from files.")
        return

    df_res = pl.DataFrame(results).sort("MAPE (%)").with_columns(pl.selectors.numeric().round(3))

    st.subheader("Leaderboard")
    st.dataframe(df_res, width="stretch", hide_index=True)

    st.subheader("Visualization")

    chart_metric = st.radio("Metric to Visualize", ["MAPE (%)", "MAE"], horizontal=True)

    st.bar_chart(df_res.to_pandas().set_index("Model")[chart_metric], width="stretch")
