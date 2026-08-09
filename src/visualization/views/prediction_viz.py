"""Prediction Visualization view for wide prediction CSVs."""

import altair as alt
import pandas as pd
import streamlit as st

from src.visualization.data_loading import (
    calculate_metrics,
    get_prediction_fish_types,
    get_prediction_model_columns,
    get_prediction_splits,
    load_prediction_df,
    normalize_predictions,
)


def render_prediction_viz(dataset):
    """Render prediction charts for one selected model."""
    st.header(f"Prediction Visualization: {dataset}")

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

    pandas_df = df.to_pandas()
    metrics = calculate_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", metrics["samples"])
    col2.metric("MAE", f"{metrics['mae']:.2f}")
    col3.metric("R2", f"{metrics['r2']:.4f}")
    col4.metric("MAPE", f"{metrics['mape']:.1f}%")

    st.markdown("---")
    st.subheader("Predicted vs Actual")

    min_val = min(pandas_df["gt_length"].min(), pandas_df["pred_length"].min())
    max_val = max(pandas_df["gt_length"].max(), pandas_df["pred_length"].max())

    scatter = (
        alt.Chart(pandas_df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("gt_length:Q", title="Actual Length"),
            y=alt.Y("pred_length:Q", title="Predicted Length"),
            color=alt.Color("mape:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["name", "fish_type", "gt_length", "pred_length", "mape"],
        )
        .properties(height=500)
        .interactive()
    )
    line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
    perfect_line = (
        alt.Chart(line_df)
        .mark_line(color="green", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart(scatter + perfect_line, width="stretch")

    st.markdown("---")
    st.subheader("Sorted Comparison")
    sort_option = st.radio(
        "Sort by", ["Percentage Error", "Actual Length"], horizontal=True
    )
    sorted_df = pandas_df.sort_values(
        "mape" if sort_option == "Percentage Error" else "gt_length",
        ascending=sort_option != "Percentage Error",
    ).reset_index(drop=True)
    sorted_df["sorted_index"] = range(len(sorted_df))

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
            x=alt.X("sorted_index:Q", title="Sample Index"),
            y=alt.Y("Length:Q"),
            color="Type:N",
            tooltip=["name", "Type", "Length"],
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(line_chart, width="stretch")

    st.markdown("---")
    st.subheader("Error Distribution")
    col1, col2 = st.columns(2)
    with col1:
        hist_abs = (
            alt.Chart(pandas_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("abs_error:Q", bin=alt.Bin(maxbins=30), title="Absolute Error"),
                y=alt.Y("count()", title="Count"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist_abs, width="stretch")
    with col2:
        hist_mape = (
            alt.Chart(pandas_df)
            .mark_bar(opacity=0.7, color="orange")
            .encode(
                x=alt.X("mape:Q", bin=alt.Bin(maxbins=30), title="MAPE (%)"),
                y=alt.Y("count()", title="Count"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist_mape, width="stretch")

    st.markdown("---")
    st.subheader("Error by Length Range")
    n_bins = st.slider("Number of bins", 3, 10, 5)
    pandas_df["length_bin"] = pd.cut(
        pandas_df["gt_length"],
        bins=n_bins,
        labels=[f"Bin {i + 1}" for i in range(n_bins)],
    )
    bin_stats = (
        pandas_df.groupby("length_bin", observed=True)
        .agg({"mape": ["mean", "std", "count"], "gt_length": ["min", "max"]})
        .reset_index()
    )
    bin_stats.columns = ["Bin", "MAPE", "Std", "Count", "Min Length", "Max Length"]
    bin_stats["Range"] = bin_stats.apply(
        lambda r: f"{r['Min Length']:.1f} - {r['Max Length']:.1f}", axis=1
    )
    bin_chart = (
        alt.Chart(bin_stats)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X("Range:N", title="Length Range", sort=None),
            y=alt.Y("MAPE:Q", title="Mean Absolute Percentage Error (%)"),
            color=alt.Color("MAPE:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Range", "MAPE", "Std", "Count"],
        )
        .properties(height=400)
    )
    st.altair_chart(bin_chart, width="stretch")
    st.dataframe(
        bin_stats[["Range", "Count", "MAPE", "Std"]].round(2),
        hide_index=True,
        width="stretch",
    )
