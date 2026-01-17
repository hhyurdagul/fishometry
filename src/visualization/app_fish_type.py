"""Fish Type Analysis App - Streamlit app for per-fish-type result analysis."""

import os
import streamlit as st
import polars as pl
import pandas as pd
import altair as alt
import numpy as np
import glob
import cv2

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


def filter_by_fish_type(df, df_meta, fish_types):
    """Filter prediction dataframe by fish type(s) using metadata."""
    if not fish_types or df_meta is None:
        return df

    # Get names for selected fish types
    fish_names = df_meta.filter(pl.col("fish_type").is_in(fish_types))["name"].to_list()
    return df.filter(pl.col("name").is_in(fish_names))


def filter_metadata_by_fish_type(df_meta, fish_types):
    """Filter metadata dataframe by fish type(s)."""
    if not fish_types or df_meta is None:
        return df_meta
    return df_meta.filter(pl.col("fish_type").is_in(fish_types))


@st.cache_data
def get_depth_models():
    """Get list of available depth models."""
    base_dir = f"data/{DATASET}/processed/depth"
    if not os.path.exists(base_dir):
        return []
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(dirs)


def get_image_paths(image_name, depth_model=None):
    """Resolve paths for raw, rotated, depth, and segmentation images."""
    found_path = None
    possible_paths = [
        f"data/{DATASET}/raw/{image_name}",
        f"data/{DATASET}/splits/{image_name}",
        f"data/{DATASET}/{image_name}",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    rot_path = f"data/{DATASET}/processed/rotated/{image_name}"

    depth_path = None
    if depth_model:
        base_depth_dir = f"data/{DATASET}/processed/depth/{depth_model}"
        p = f"{base_depth_dir}/{image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
        if not os.path.exists(p):
            p = f"{base_depth_dir}/{image_name}.npy"
        if os.path.exists(p):
            depth_path = p

    if not depth_path:
        p = f"data/{DATASET}/processed/depth/{image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
        if not os.path.exists(p):
            p = f"data/{DATASET}/processed/depth/{image_name}.npy"
        if os.path.exists(p):
            depth_path = p

    seg_path = f"data/{DATASET}/processed/segment/{image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
    if not os.path.exists(seg_path):
        seg_path = f"data/{DATASET}/processed/segment/{image_name}.npy"

    return found_path, rot_path, depth_path, seg_path


def process_images(image_name, data_row, depth_model=None):
    """Load and process all image variants for display."""
    raw_path, rot_path, depth_path, seg_path = get_image_paths(image_name, depth_model)

    img_raw = None
    if raw_path and os.path.exists(raw_path):
        img_raw = cv2.imread(raw_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_raw = np.zeros((200, 200, 3), dtype=np.uint8)

    img_rot = None
    if os.path.exists(rot_path):
        img_rot = cv2.imread(rot_path)
        img_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)

        if data_row:
            if "Fish_x1" in data_row and data_row["Fish_x1"] is not None:
                pt1 = (int(data_row["Fish_x1"]), int(data_row["Fish_y1"]))
                pt2 = (int(data_row["Fish_x2"]), int(data_row["Fish_y2"]))
                cv2.rectangle(img_rot, pt1, pt2, (0, 120, 255), 2)
                cv2.putText(
                    img_rot,
                    "Fish",
                    pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 120, 255),
                    2,
                )

            if "Head_x1" in data_row and data_row["Head_x1"] is not None:
                cx = int((data_row["Head_x1"] + data_row["Head_x2"]) / 2)
                cy = int((data_row["Head_y1"] + data_row["Head_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (255, 0, 0), -1)

            if "Tail_x1" in data_row and data_row["Tail_x1"] is not None:
                cx = int((data_row["Tail_x1"] + data_row["Tail_x2"]) / 2)
                cy = int((data_row["Tail_y1"] + data_row["Tail_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (0, 255, 0), -1)

    img_depth = None
    if depth_path and os.path.exists(depth_path):
        depth = np.load(depth_path)
        norm_depth = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        img_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB)

    img_seg = None
    if seg_path and os.path.exists(seg_path):
        mask = np.load(seg_path)
        if img_rot is not None:
            overlay = img_rot.copy()
            overlay[mask > 0] = [0, 255, 0]
            img_seg = cv2.addWeighted(img_rot, 0.7, overlay, 0.3, 0)
        else:
            img_seg = (mask * 255).astype(np.uint8)
            img_seg = cv2.cvtColor(img_seg, cv2.COLOR_GRAY2RGB)

    return img_raw, img_rot, img_depth, img_seg


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
def render_overview(df_meta, fish_types, pred_files):
    """Overview of all models for selected fish type(s)."""
    fish_label = (
        ", ".join(fish_types) if len(fish_types) <= 3 else f"{len(fish_types)} types"
    )
    st.header(f"Model Overview: {fish_label}")

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

            # Filter by fish type(s)
            df_filtered = filter_by_fish_type(df, df_meta, fish_types)

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
        df_filtered = filter_by_fish_type(df, df_meta, [ft])
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


def render_detailed_analysis(df_meta, fish_types, pred_files):
    """Detailed prediction analysis for selected fish type(s)."""
    fish_label = (
        ", ".join(fish_types) if len(fish_types) <= 3 else f"{len(fish_types)} types"
    )
    st.header(f"Detailed Analysis: {fish_label}")

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

    # Filter by fish type(s)
    df_filtered = filter_by_fish_type(df, df_meta, fish_types)

    if df_filtered.height == 0:
        st.warning(f"No data available for selected fish types")
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
                df_filtered = filter_by_fish_type(df, df_meta, [ft])
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


def render_data_explorer(df_meta, fish_types, depth_model):
    """Data explorer view with fish type filtering."""
    fish_label = (
        ", ".join(fish_types) if len(fish_types) <= 3 else f"{len(fish_types)} types"
    )
    st.header(f"Data Explorer: {fish_label}")

    # Filter metadata by fish type(s)
    df_filtered = filter_metadata_by_fish_type(df_meta, fish_types)

    if df_filtered is None or df_filtered.height == 0:
        st.warning("No images found for selected fish types")
        return

    all_image_names = df_filtered["name"].to_list()

    st.info(f"Showing {len(all_image_names)} images for {len(fish_types)} fish type(s)")

    # Image Selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_img = st.selectbox("Select Image", all_image_names)

    # Get Data
    row_data = {}
    row = df_filtered.filter(pl.col("name") == selected_img)
    if row.height > 0:
        row_data = row.to_dicts()[0]
        row_data = {k: v for k, v in row_data.items() if v is not None}

    # Get Predictions for this image
    pred_files = load_predictions_list()
    preds = {}
    for f in pred_files:
        try:
            df = pl.read_csv(f)
            if "name" in df.columns and "pred_length" in df.columns:
                pred_row = df.filter(pl.col("name") == selected_img)
                if pred_row.height > 0:
                    model_base = os.path.basename(f).replace(".csv", "")
                    preds[model_base] = pred_row["pred_length"][0]
        except Exception:
            pass

    if preds:
        row_data["Predictions"] = preds

    # Process Images
    img_raw, img_rot, img_depth, img_seg = process_images(
        selected_img, row_data, depth_model
    )

    # Display Grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Raw Image")
        if img_raw is not None:
            st.image(img_raw, width="stretch")
        else:
            st.info("No raw image found")

    with c2:
        st.markdown("### Rotated & Annotated")
        if img_rot is not None:
            st.image(img_rot, width="stretch")
        else:
            st.info("No rotated image found")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Depth Map")
        if img_depth is not None:
            st.image(img_depth, width="stretch")
        else:
            st.info("No depth map found")

    with c4:
        st.markdown("### Segmentation Overlay")
        if img_seg is not None:
            st.image(img_seg, width="stretch")
        else:
            st.info("No segmentation found")

    # Metadata Expander
    with st.expander("See Metadata JSON", expanded=True):
        st.json(row_data)


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
    depth_models = get_depth_models()

    st.sidebar.markdown("---")

    # Mode selection
    mode = st.sidebar.radio(
        "View",
        [
            "Data Explorer",
            "Model Overview",
            "Fish Type Comparison",
            "Detailed Analysis",
            "Model x Fish Type Heatmap",
        ],
    )

    # Fish type selector (for relevant modes) - multiselect
    if mode in ["Data Explorer", "Model Overview", "Detailed Analysis"]:
        st.sidebar.markdown("---")
        selected_fish_types = st.sidebar.multiselect(
            "Fish Types (leave empty for all)",
            fish_types,
            default=[],
            help="Select one or more fish types to filter. Leave empty to show all.",
        )
        # If empty, use all fish types
        if not selected_fish_types:
            selected_fish_types = fish_types
    else:
        selected_fish_types = fish_types

    # Depth model selector (for Data Explorer)
    depth_model = None
    if mode == "Data Explorer" and depth_models:
        depth_model = st.sidebar.selectbox("Depth Model", depth_models)

    # Render selected view
    if mode == "Data Explorer":
        render_data_explorer(df_meta, selected_fish_types, depth_model)
    elif mode == "Model Overview":
        render_overview(df_meta, selected_fish_types, pred_files)
    elif mode == "Fish Type Comparison":
        render_comparison_by_fish_type(df_meta, pred_files)
    elif mode == "Detailed Analysis":
        render_detailed_analysis(df_meta, selected_fish_types, pred_files)
    elif mode == "Model x Fish Type Heatmap":
        render_multi_model_fish_type(df_meta, pred_files)


if __name__ == "__main__":
    main()
