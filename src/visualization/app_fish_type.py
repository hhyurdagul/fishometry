"""Fish Type Analysis App - Streamlit app for per-fish-type result analysis."""

import os
import streamlit as st
import polars as pl
import pandas as pd
import altair as alt
import numpy as np
import glob

st.set_page_config(layout="wide", page_title="Fish Type Analysis")

DATASET = "data-outside"


# --- Data Loading ---
@st.cache_data
def load_metadata():
    """Load metadata with fish_type information."""
    base_dir = f"data/{DATASET}/processed"
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
        return None

    return pl.concat(dfs, how="diagonal")


@st.cache_data
def get_fish_types(df_meta):
    """Get unique fish types from metadata."""
    if df_meta is None or "fish_type" not in df_meta.columns:
        return []
    return sorted(df_meta["fish_type"].unique().to_list())


@st.cache_data
def load_predictions_list():
    """Returns list of prediction CSV files available."""
    paths = [f"data/{DATASET}/predictions", f"checkpoints/{DATASET}/predictions"]
    files = []
    for p in paths:
        if os.path.exists(p):
            files.extend(glob.glob(os.path.join(p, "*.csv")))
    return sorted(list(set(files)))


@st.cache_data
def load_prediction_df(file_path):
    """Load a single prediction CSV file."""
    return pl.read_csv(file_path)


def filter_by_fish_type(df, df_meta, fish_type):
    """Filter prediction dataframe by fish type using metadata."""
    if fish_type == "All" or df_meta is None:
        return df

    # Get names for this fish type
    fish_names = df_meta.filter(pl.col("fish_type") == fish_type)["name"].to_list()
    return df.filter(pl.col("name").is_in(fish_names))


def calculate_metrics(df):
    """Calculate MAE, MAPE, R² from a prediction dataframe."""
    if df.height == 0:
        return None, None, None

    gt = df["gt_length"].to_numpy()
    pred = df["pred_length"].to_numpy()

    mae = np.abs(gt - pred).mean()
    mape = (np.abs(gt - pred) / gt * 100).mean()

    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return mae, mape, r2


# --- Views ---
def render_overview(df_meta, fish_type, pred_files):
    """Overview of all models for selected fish type."""
    st.header(f"Model Overview: {fish_type}")

    if not pred_files:
        st.warning("No prediction files found.")
        return

    # Select split
    split_choice = st.selectbox("Select Split", ["val", "test", "train"])
    relevant_files = [f for f in pred_files if f.endswith(f"_{split_choice}.csv")]

    if not relevant_files:
        st.info(f"No prediction files found for split '{split_choice}'")
        return

    results = []
    for f in relevant_files:
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

            # Filter by fish type
            df_filtered = filter_by_fish_type(df, df_meta, fish_type)

            if df_filtered.height == 0:
                continue

            mae, mape, r2 = calculate_metrics(df_filtered)

            model_name = os.path.basename(f).replace(f"_{split_choice}.csv", "")
            results.append(
                {
                    "Model": model_name,
                    "MAPE (%)": round(mape, 2),
                    "MAE": round(mae, 2),
                    "R²": round(r2, 4),
                    "Samples": df_filtered.height,
                }
            )
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not results:
        st.warning("No results available for this fish type.")
        return

    df_res = pl.DataFrame(results).sort("MAPE (%)")

    st.subheader("Leaderboard")
    st.dataframe(df_res, width="stretch", hide_index=True)

    # Visualization
    st.subheader("Visualization")
    chart_metric = st.radio("Metric", ["MAPE (%)", "MAE", "R²"], horizontal=True)

    chart = (
        alt.Chart(df_res.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort="-y"),
            y=alt.Y(f"{chart_metric}:Q"),
            color=alt.Color(
                f"{chart_metric}:Q", scale=alt.Scale(scheme="blues"), legend=None
            ),
            tooltip=["Model", "MAPE (%)", "MAE", "R²", "Samples"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, width="stretch")


def render_comparison_by_fish_type(df_meta, pred_files):
    """Compare performance across all fish types for a single model."""
    st.header("Comparison by Fish Type")

    if not pred_files:
        st.warning("No prediction files found.")
        return

    # File selector
    pred_file_map = {os.path.basename(f): f for f in pred_files}
    selected_file = st.selectbox("Select Model", list(pred_file_map.keys()))

    df = load_prediction_df(pred_file_map[selected_file])

    if "gt_length" not in df.columns or "pred_length" not in df.columns:
        st.error("Invalid prediction file format.")
        return

    df = df.with_columns(
        [
            pl.col("gt_length").cast(pl.Float64),
            pl.col("pred_length").cast(pl.Float64),
        ]
    )

    fish_types = get_fish_types(df_meta)

    results = []

    # Calculate for "All"
    mae, mape, r2 = calculate_metrics(df)
    results.append(
        {
            "Fish Type": "All",
            "MAPE (%)": round(mape, 2),
            "MAE": round(mae, 2),
            "R²": round(r2, 4),
            "Samples": df.height,
        }
    )

    # Calculate per fish type
    for ft in fish_types:
        df_filtered = filter_by_fish_type(df, df_meta, ft)
        if df_filtered.height > 0:
            mae, mape, r2 = calculate_metrics(df_filtered)
            results.append(
                {
                    "Fish Type": ft,
                    "MAPE (%)": round(mape, 2),
                    "MAE": round(mae, 2),
                    "R²": round(r2, 4),
                    "Samples": df_filtered.height,
                }
            )

    df_res = pl.DataFrame(results)

    st.subheader("Metrics by Fish Type")
    st.dataframe(df_res, width="stretch", hide_index=True)

    # Bar chart
    chart_metric = st.radio(
        "Metric to Visualize", ["MAPE (%)", "MAE", "R²"], horizontal=True
    )

    # Exclude "All" from chart for cleaner comparison
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
            tooltip=["Fish Type", "MAPE (%)", "MAE", "R²", "Samples"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, width="stretch")


def render_detailed_analysis(df_meta, fish_type, pred_files):
    """Detailed prediction analysis for selected fish type."""
    st.header(f"Detailed Analysis: {fish_type}")

    if not pred_files:
        st.warning("No prediction files found.")
        return

    # File selector
    pred_file_map = {os.path.basename(f): f for f in pred_files}
    selected_file = st.selectbox("Select Model", list(pred_file_map.keys()))

    df = load_prediction_df(pred_file_map[selected_file])

    required_cols = ["name", "gt_length", "pred_length"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        return

    df = df.with_columns(
        [
            pl.col("gt_length").cast(pl.Float64),
            pl.col("pred_length").cast(pl.Float64),
        ]
    )

    # Filter by fish type
    df_filtered = filter_by_fish_type(df, df_meta, fish_type)

    if df_filtered.height == 0:
        st.warning(f"No data available for fish type: {fish_type}")
        return

    # Add error columns
    df_filtered = df_filtered.with_columns(
        [
            (pl.col("pred_length") - pl.col("gt_length")).abs().alias("abs_error"),
            (
                (pl.col("pred_length") - pl.col("gt_length")).abs()
                / pl.col("gt_length")
                * 100
            ).alias("mape"),
        ]
    )

    pandas_df = df_filtered.to_pandas()

    # Metrics
    mae, mape, r2 = calculate_metrics(df_filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(pandas_df))
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MAPE", f"{mape:.1f}%")
    col4.metric("R²", f"{r2:.4f}")

    st.markdown("---")

    # Scatter plot
    st.subheader("Predicted vs Actual")

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
            color=alt.Color(
                "mape:Q",
                scale=alt.Scale(scheme="reds"),
                legend=alt.Legend(title="MAPE (%)"),
            ),
            tooltip=["name", "gt_length", "pred_length", "mape"],
        )
        .properties(height=500)
        .interactive()
    )

    # Perfect prediction line
    line_df = pd.DataFrame(
        {"x": [min_val * 0.9, max_val * 1.1], "y": [min_val * 0.9, max_val * 1.1]}
    )
    perfect_line = (
        alt.Chart(line_df)
        .mark_line(color="green", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )

    st.altair_chart(scatter + perfect_line, width="stretch")

    st.markdown("---")

    # Error distribution
    st.subheader("MAPE Distribution")

    hist = (
        alt.Chart(pandas_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("mape:Q", bin=alt.Bin(maxbins=25), title="MAPE (%)"),
            y=alt.Y("count()", title="Count"),
        )
        .properties(height=300)
    )
    st.altair_chart(hist, width="stretch")

    st.markdown("---")

    # Sorted table
    st.subheader("Samples Sorted by Error")

    df_sorted = df_filtered.sort("mape", descending=True).select(
        ["name", "gt_length", "pred_length", "abs_error", "mape"]
    )

    st.dataframe(df_sorted, width="stretch", hide_index=True)


def render_multi_model_fish_type(df_meta, pred_files):
    """Compare multiple models across fish types in a heatmap."""
    st.header("Model x Fish Type Heatmap")

    if not pred_files:
        st.warning("No prediction files found.")
        return

    # Select split
    split_choice = st.selectbox("Select Split", ["val", "test", "train"])
    relevant_files = [f for f in pred_files if f.endswith(f"_{split_choice}.csv")]

    if not relevant_files:
        st.info(f"No prediction files found for split '{split_choice}'")
        return

    fish_types = get_fish_types(df_meta)

    if not fish_types:
        st.warning("No fish types found in metadata.")
        return

    # Build heatmap data
    heatmap_data = []

    for f in relevant_files:
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

            model_name = os.path.basename(f).replace(f"_{split_choice}.csv", "")

            for ft in fish_types:
                df_filtered = filter_by_fish_type(df, df_meta, ft)
                if df_filtered.height > 0:
                    mae, mape, r2 = calculate_metrics(df_filtered)
                    heatmap_data.append(
                        {
                            "Model": model_name,
                            "Fish Type": ft,
                            "MAPE": round(mape, 2),
                            "MAE": round(mae, 2),
                            "R²": round(r2, 4),
                            "Samples": df_filtered.height,
                        }
                    )
        except Exception as e:
            print(f"Error: {e}")

    if not heatmap_data:
        st.warning("No data available for heatmap.")
        return

    heatmap_df = pd.DataFrame(heatmap_data)

    metric = st.radio("Metric", ["MAPE", "MAE", "R²"], horizontal=True)

    # For R², higher is better, so reverse the color scheme
    color_scheme = "blues" if metric == "R²" else "reds"

    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("Fish Type:N", title=None),
            y=alt.Y("Model:N", title=None),
            color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme=color_scheme)),
            tooltip=["Model", "Fish Type", "MAPE", "MAE", "R²", "Samples"],
        )
        .properties(height=max(300, len(relevant_files) * 40))
    )

    text = (
        alt.Chart(heatmap_df)
        .mark_text(fontSize=11)
        .encode(
            x=alt.X("Fish Type:N"),
            y=alt.Y("Model:N"),
            text=alt.Text(f"{metric}:Q", format=".1f" if metric != "R²" else ".3f"),
            color=alt.condition(
                alt.datum[metric] > heatmap_df[metric].median(),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    st.altair_chart(heatmap + text, width="stretch")

    # Also show as table
    st.subheader("Data Table")
    pivot_df = heatmap_df.pivot(index="Model", columns="Fish Type", values=metric)
    st.dataframe(pivot_df, width="stretch")


# --- Main ---
def main():
    st.sidebar.title("Fish Type Analysis")
    st.sidebar.markdown(f"**Dataset:** {DATASET}")

    # Load data
    df_meta = load_metadata()

    if df_meta is None:
        st.error(f"Could not load metadata for {DATASET}")
        return

    if "fish_type" not in df_meta.columns:
        st.error("No 'fish_type' column found in metadata.")
        return

    fish_types = get_fish_types(df_meta)
    pred_files = load_predictions_list()

    st.sidebar.markdown("---")

    # Mode selection
    mode = st.sidebar.radio(
        "View",
        [
            "Model Overview",
            "Fish Type Comparison",
            "Detailed Analysis",
            "Model x Fish Type Heatmap",
        ],
    )

    # Fish type selector (for relevant modes)
    if mode in ["Model Overview", "Detailed Analysis"]:
        st.sidebar.markdown("---")
        fish_type = st.sidebar.selectbox(
            "Fish Type",
            ["All"] + fish_types,
        )
    else:
        fish_type = "All"

    # Render selected view
    if mode == "Model Overview":
        render_overview(df_meta, fish_type, pred_files)
    elif mode == "Fish Type Comparison":
        render_comparison_by_fish_type(df_meta, pred_files)
    elif mode == "Detailed Analysis":
        render_detailed_analysis(df_meta, fish_type, pred_files)
    elif mode == "Model x Fish Type Heatmap":
        render_multi_model_fish_type(df_meta, pred_files)


if __name__ == "__main__":
    main()
