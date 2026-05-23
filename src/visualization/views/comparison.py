"""Model and fish-type comparison views."""

import altair as alt
import pandas as pd
import polars as pl
import streamlit as st

from src.visualization.data_loading import (
    calculate_metrics,
    get_prediction_fish_types,
    get_prediction_model_columns,
    get_prediction_splits,
    load_prediction_df,
    normalize_predictions,
)


def _metric_rows(df_pred, models, split=None, fish_types=None):
    rows = []
    for model in models:
        df = normalize_predictions(
            df_pred, model_columns=[model], split=split, fish_types=fish_types
        )
        metrics = calculate_metrics(df)
        if metrics:
            rows.append(
                {
                    "Model": model,
                    "MAPE (%)": metrics["mape"],
                    "MAE": metrics["mae"],
                    "R2": metrics["r2"],
                    "Samples": metrics["samples"],
                }
            )
    return rows


def render_comparison(dataset):
    """Render model comparison for the wide prediction CSV."""
    st.header(f"Model Comparison: {dataset}")

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

    c1, c2 = st.columns(2)
    with c1:
        selected_split = st.selectbox("Split", ["all"] + splits)
    with c2:
        selected_fish_types = (
            st.multiselect("Fish Type", fish_types, default=[]) if fish_types else []
        )

    rows = _metric_rows(
        df_pred,
        models,
        split=None if selected_split == "all" else selected_split,
        fish_types=selected_fish_types or None,
    )
    if not rows:
        st.warning("No results match the selected filters.")
        return

    df_res = (
        pl.DataFrame(rows)
        .sort("MAPE (%)")
        .with_columns(pl.selectors.numeric().round(3))
    )
    st.subheader("Leaderboard")
    st.dataframe(df_res, width="stretch", hide_index=True)

    st.subheader("Visualization")
    chart_metric = st.radio("Metric to Visualize", ["MAPE (%)", "MAE", "R2"], horizontal=True)
    chart = (
        alt.Chart(df_res.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort="-y"),
            y=alt.Y(f"{chart_metric}:Q"),
            color=alt.Color(
                f"{chart_metric}:Q", scale=alt.Scale(scheme="blues"), legend=None
            ),
            tooltip=["Model", "MAPE (%)", "MAE", "R2", "Samples"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, width="stretch")


def render_fish_type_comparison(dataset):
    """Compare one selected model across fish types."""
    st.header(f"Fish Type Comparison: {dataset}")

    df_pred = load_prediction_df(dataset)
    if df_pred is None:
        st.warning(f"No prediction CSV found at `data/{dataset}/predictions.csv`.")
        return

    models = get_prediction_model_columns(df_pred)
    if not models:
        st.error("No model prediction columns found.")
        return
    fish_types = get_prediction_fish_types(df_pred)
    if not fish_types:
        st.info("This dataset does not contain fish types.")
        return
    splits = get_prediction_splits(df_pred)

    c1, c2 = st.columns(2)
    with c1:
        selected_model = st.selectbox("Model", models)
    with c2:
        selected_split = st.selectbox("Split", ["all"] + splits)

    split = None if selected_split == "all" else selected_split
    rows = []
    all_df = normalize_predictions(df_pred, [selected_model], split=split)
    all_metrics = calculate_metrics(all_df)
    if all_metrics:
        rows.append(
            {
                "Fish Type": "All",
                "MAPE (%)": all_metrics["mape"],
                "MAE": all_metrics["mae"],
                "R2": all_metrics["r2"],
                "Samples": all_metrics["samples"],
            }
        )

    for fish_type in fish_types:
        df = normalize_predictions(
            df_pred, [selected_model], split=split, fish_types=[fish_type]
        )
        metrics = calculate_metrics(df)
        if metrics:
            rows.append(
                {
                    "Fish Type": fish_type,
                    "MAPE (%)": metrics["mape"],
                    "MAE": metrics["mae"],
                    "R2": metrics["r2"],
                    "Samples": metrics["samples"],
                }
            )

    df_res = pl.DataFrame(rows).with_columns(pl.selectors.numeric().round(3))
    st.subheader("Metrics by Fish Type")
    st.dataframe(df_res, width="stretch", hide_index=True)

    chart_metric = st.radio("Metric to Visualize", ["MAPE (%)", "MAE", "R2"], horizontal=True)
    chart_df = df_res.filter(pl.col("Fish Type") != "All").to_pandas()
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Fish Type:N", sort="-y"),
            y=alt.Y(f"{chart_metric}:Q"),
            color=alt.Color(
                f"{chart_metric}:Q", scale=alt.Scale(scheme="reds"), legend=None
            ),
            tooltip=["Fish Type", "MAPE (%)", "MAE", "R2", "Samples"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, width="stretch")


def render_model_fish_type_heatmap(dataset):
    """Render model x fish type metric heatmap."""
    st.header(f"Model x Fish Type Heatmap: {dataset}")

    df_pred = load_prediction_df(dataset)
    if df_pred is None:
        st.warning(f"No prediction CSV found at `data/{dataset}/predictions.csv`.")
        return

    models = get_prediction_model_columns(df_pred)
    if not models:
        st.error("No model prediction columns found.")
        return
    fish_types = get_prediction_fish_types(df_pred)
    if not fish_types:
        st.info("This dataset does not contain fish types.")
        return
    splits = get_prediction_splits(df_pred)

    selected_split = st.selectbox("Split", ["all"] + splits)
    selected_models = st.multiselect("Models", models, default=models)
    split = None if selected_split == "all" else selected_split

    heatmap_data = []
    for model in selected_models:
        for fish_type in fish_types:
            df = normalize_predictions(df_pred, [model], split=split, fish_types=[fish_type])
            metrics = calculate_metrics(df)
            if metrics:
                heatmap_data.append(
                    {
                        "Model": model,
                        "Fish Type": fish_type,
                        "MAPE": round(metrics["mape"], 2),
                        "MAE": round(metrics["mae"], 2),
                        "R2": round(metrics["r2"], 4),
                        "Samples": metrics["samples"],
                    }
                )

    if not heatmap_data:
        st.warning("No data available for heatmap.")
        return

    heatmap_df = pd.DataFrame(heatmap_data)
    metric = st.radio("Metric", ["MAPE", "MAE", "R2"], horizontal=True)
    color_scheme = "blues" if metric == "R2" else "reds"

    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("Fish Type:N", title=None),
            y=alt.Y("Model:N", title=None),
            color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme=color_scheme)),
            tooltip=["Model", "Fish Type", "MAPE", "MAE", "R2", "Samples"],
        )
        .properties(height=max(300, len(selected_models) * 40))
    )
    text = (
        alt.Chart(heatmap_df)
        .mark_text(fontSize=11)
        .encode(
            x=alt.X("Fish Type:N"),
            y=alt.Y("Model:N"),
            text=alt.Text(f"{metric}:Q", format=".1f" if metric != "R2" else ".3f"),
        )
    )
    st.altair_chart(heatmap + text, width="stretch")
    st.dataframe(
        heatmap_df.pivot(index="Model", columns="Fish Type", values=metric),
        width="stretch",
    )
