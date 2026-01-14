"""Prediction Visualization view for comparing real vs predicted values."""

import os
import streamlit as st
import polars as pl
import altair as alt
import numpy as np

from data_loading import load_predictions_list, load_prediction_df


def render_prediction_viz(dataset):
    """Render the Prediction Visualization view."""
    st.header(f"Prediction Visualization: {dataset}")

    pred_files = load_predictions_list(dataset)
    if not pred_files:
        st.warning("No prediction CSV files found.")
        return

    # File Selector
    pred_file_map = {os.path.basename(f): f for f in pred_files}
    selected_file_name = st.selectbox(
        "Select Prediction Model", list(pred_file_map.keys())
    )
    selected_file_path = pred_file_map[selected_file_name]

    # Load predictions
    df_pred = load_prediction_df(selected_file_path)

    required_cols = ["name", "gt_length", "pred_length"]
    if not all(col in df_pred.columns for col in required_cols):
        st.error(f"Selected CSV must contain columns: {required_cols}")
        return

    # Prepare data
    df = (
        df_pred.with_columns(
            [
                pl.col("gt_length").cast(pl.Float64),
                pl.col("pred_length").cast(pl.Float64),
            ]
        )
        .with_columns(
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
        .with_row_index("index")
    )

    pandas_df = df.to_pandas()

    # --- Metrics Summary ---
    # Calculate R²
    ss_res = ((pandas_df["pred_length"] - pandas_df["gt_length"]) ** 2).sum()
    ss_tot = ((pandas_df["gt_length"] - pandas_df["gt_length"].mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(pandas_df))
    col2.metric("MAE", f"{pandas_df['abs_error'].mean():.2f}")
    col3.metric("R²", f"{r2:.4f}")
    col4.metric("MAPE", f"{pandas_df['mape'].mean():.1f}%")

    st.markdown("---")

    # --- 1. Scatter Plot: Predicted vs Actual ---
    st.subheader("Predicted vs Actual (Scatter)")

    min_val = min(pandas_df["gt_length"].min(), pandas_df["pred_length"].min())
    max_val = max(pandas_df["gt_length"].max(), pandas_df["pred_length"].max())

    scatter = (
        alt.Chart(pandas_df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X(
                "gt_length:Q",
                title="Actual Length",
                scale=alt.Scale(domain=[min_val * 0.9, max_val * 1.1]),
            ),
            y=alt.Y(
                "pred_length:Q",
                title="Predicted Length",
                scale=alt.Scale(domain=[min_val * 0.9, max_val * 1.1]),
            ),
            tooltip=["name", "gt_length", "pred_length", "mape"],
            color=alt.Color(
                "mape:Q",
                scale=alt.Scale(scheme="reds"),
                legend=alt.Legend(title="MAPE (%)"),
            ),
        )
        .properties(width=600, height=500, title="Predicted vs Actual Length")
        .interactive()
    )

    # Perfect prediction line (y = x)
    line_df = pl.DataFrame(
        {"x": [min_val * 0.9, max_val * 1.1], "y": [min_val * 0.9, max_val * 1.1]}
    ).to_pandas()
    perfect_line = (
        alt.Chart(line_df)
        .mark_line(color="green", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )

    st.altair_chart(scatter + perfect_line, width="stretch")

    st.markdown("---")

    # --- 2. Line Chart: Sorted Comparison ---
    st.subheader("Sorted Comparison (Line Chart)")

    sort_option = st.radio(
        "Sort by",
        ["Percentage Error", "Actual Length"],
        horizontal=True,
    )

    if sort_option == "Percentage Error":
        sorted_df = pandas_df.sort_values("mape", ascending=False).reset_index(
            drop=True
        )
    else:
        sorted_df = pandas_df.sort_values("gt_length").reset_index(drop=True)

    sorted_df["sorted_index"] = range(len(sorted_df))

    # Melt for line chart
    melted = sorted_df.melt(
        id_vars=["sorted_index", "name"],
        value_vars=["gt_length", "pred_length"],
        var_name="Type",
        value_name="Length",
    )
    melted["Type"] = melted["Type"].map(
        {"gt_length": "Actual", "pred_length": "Predicted"}
    )

    line_chart = (
        alt.Chart(melted)
        .mark_line(point=True, opacity=0.7)
        .encode(
            x=alt.X("sorted_index:Q", title="Sample Index (Sorted)"),
            y=alt.Y("Length:Q", title="Length"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Actual", "Predicted"], range=["#1f77b4", "#ff7f0e"]
                ),
            ),
            tooltip=["name", "Type", "Length"],
        )
        .properties(width=700, height=400, title="Actual vs Predicted (Sorted)")
        .interactive()
    )

    st.altair_chart(line_chart, width="stretch")

    st.markdown("---")

    # --- 3. Error Distribution ---
    st.subheader("Error Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Absolute error histogram
        hist_abs = (
            alt.Chart(pandas_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("abs_error:Q", bin=alt.Bin(maxbins=30), title="Absolute Error"),
                y=alt.Y("count()", title="Count"),
                tooltip=["count()"],
            )
            .properties(width=350, height=300, title="Absolute Error Distribution")
        )
        st.altair_chart(hist_abs, width="stretch")

    with col2:
        # MAPE histogram
        hist_mape = (
            alt.Chart(pandas_df)
            .mark_bar(opacity=0.7, color="orange")
            .encode(
                x=alt.X("mape:Q", bin=alt.Bin(maxbins=30), title="MAPE (%)"),
                y=alt.Y("count()", title="Count"),
                tooltip=["count()"],
            )
            .properties(width=350, height=300, title="MAPE Distribution")
        )
        st.altair_chart(hist_mape, width="stretch")

    st.markdown("---")

    # --- 5. Error by Length Bins ---
    st.subheader("Error by Length Range")

    n_bins = st.slider("Number of bins", 3, 10, 5)

    # Create bins
    pandas_df["length_bin"] = pd.cut(
        pandas_df["gt_length"],
        bins=n_bins,
        labels=[f"Bin {i + 1}" for i in range(n_bins)],
    )

    bin_stats = (
        pandas_df.groupby("length_bin", observed=True)
        .agg(
            {
                "mape": ["mean", "std", "count"],
                "gt_length": ["min", "max"],
            }
        )
        .reset_index()
    )
    bin_stats.columns = ["Bin", "MAPE", "Std", "Count", "Min Length", "Max Length"]
    bin_stats["Range"] = bin_stats.apply(
        lambda r: f"{r['Min Length']:.1f} - {r['Max Length']:.1f}", axis=1
    )

    # Bar chart of MAPE by bin
    bin_chart = (
        alt.Chart(bin_stats)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("Range:N", title="Length Range", sort=None),
            y=alt.Y("MAPE:Q", title="Mean Absolute Percentage Error (%)"),
            color=alt.Color("MAPE:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Range", "MAPE", "Std", "Count"],
        )
        .properties(width=600, height=350, title="MAPE by Length Range")
    )

    st.altair_chart(bin_chart, width="stretch")

    # Show table
    st.dataframe(
        bin_stats[["Range", "Count", "MAPE", "Std"]].round(2),
        hide_index=True,
        width="stretch",
    )


# Need pandas for pd.cut
import pandas as pd
