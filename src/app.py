import streamlit as st
import os
import cv2
import numpy as np
import polars as pl
import glob

st.set_page_config(layout="wide", page_title="Fishometry Explorer")

# --- Helper Functions (Cached) ---

@st.cache_data
def get_datasets():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return []
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

@st.cache_data
def load_dataset_metadata(dataset):
    base_dir = f"data/{dataset}/processed"
    dfs = []
    
    # Try loading from standard splits
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
        # Fallback: check if there's just a single processed.csv or similar if split files missing?
        # For now, return empty if standard structure missing, or maybe the user just has raw images? 
        # But gui.py relied on metadata.
        return None, []

    full_df = pl.concat(dfs, how="diagonal")
    names = full_df["name"].to_list()
    return full_df, names

@st.cache_data
def load_predictions_list(dataset):
    """Returns list of prediction CSV files available."""
    # Look in data/{dataset}/predictions AND checkpoints/{dataset}/predictions
    paths = [
        f"data/{dataset}/predictions",
        f"checkpoints/{dataset}/predictions"
    ]
    files = []
    for p in paths:
        if os.path.exists(p):
            files.extend(glob.glob(os.path.join(p, "*.csv")))
            
    # return unique basenames or full paths? 
    # Let's return full paths but display basenames
    return sorted(list(set(files)))

@st.cache_data
def load_prediction_df(file_path):
    return pl.read_csv(file_path)

@st.cache_data
def load_all_predictions_for_image(dataset, image_name):
    """Loads all predictions for a single image across all CSVs (for Explorer view)"""
    # This is slightly expensive to do per image if we iterate files, 
    # but okay for single viewing if cached or if we pre-load all.
    # To mimic gui.py, we scan all CSVs. 
    pred_files = load_predictions_list(dataset)
    preds = {}
    for f in pred_files:
        try:
            df = pl.read_csv(f)
            # Filter for this image
            if "name" in df.columns and "pred_length" in df.columns:
                row = df.filter(pl.col("name") == image_name)
                if row.height > 0:
                    model_base = os.path.basename(f).replace(".csv", "")
                    preds[model_base] = row["pred_length"][0]
        except:
            pass
    return preds

@st.cache_data
def get_depth_models(dataset):
    base_dir = f"data/{dataset}/processed/depth"
    if not os.path.exists(base_dir):
        return []
    # Return subdirectories
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(dirs)

# --- Image Processing ---

    return found_path, rot_path, depth_path, seg_path

def get_image_paths(dataset, image_name, depth_model=None):
    # Try finding image in raw or splits
    found_path = None
    possible_paths = [
        f"data/{dataset}/raw/{image_name}",
        f"data/{dataset}/splits/{image_name}",
        f"data/{dataset}/{image_name}"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    rot_path = f"data/{dataset}/processed/rotated/{image_name}"
    
    # Depth extension check
    depth_path = None
    if depth_model:
         # Check inside the specific model folder
         base_depth_dir = f"data/{dataset}/processed/depth/{depth_model}"
         p = f"{base_depth_dir}/{image_name.replace('.jpg','.npy').replace('.jpeg', '.npy')}"
         if not os.path.exists(p):
             p = f"{base_depth_dir}/{image_name}.npy"
         
         if os.path.exists(p):
             depth_path = p
    
    # Fallback to root depth folder if not found or no model selected (legacy support)
    if not depth_path:
        p = f"data/{dataset}/processed/depth/{image_name.replace('.jpg','.npy').replace('.jpeg', '.npy')}"
        if not os.path.exists(p): 
             p = f"data/{dataset}/processed/depth/{image_name}.npy"
        if os.path.exists(p):
            depth_path = p

    # Segment extension check
    seg_path = f"data/{dataset}/processed/segment/{image_name.replace('.jpg','.npy').replace('.jpeg', '.npy')}"
    if not os.path.exists(seg_path):
        seg_path = f"data/{dataset}/processed/segment/{image_name}.npy"
        
    return found_path, rot_path, depth_path, seg_path

def process_images(dataset, image_name, data_row, depth_model=None):
    raw_path, rot_path, depth_path, seg_path = get_image_paths(dataset, image_name, depth_model)
    
    img_raw = None
    if raw_path and os.path.exists(raw_path):
        img_raw = cv2.imread(raw_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_raw = np.zeros((200, 200, 3), dtype=np.uint8) # Placeholder

    img_rot = None
    if os.path.exists(rot_path):
        img_rot = cv2.imread(rot_path)
        img_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
        
        # Annotations
        if data_row:
            # Fish BBox
            if "Fish_x1" in data_row and data_row["Fish_x1"] is not None:
                pt1 = (int(data_row["Fish_x1"]), int(data_row["Fish_y1"]))
                pt2 = (int(data_row["Fish_x2"]), int(data_row["Fish_y2"]))
                cv2.rectangle(img_rot, pt1, pt2, (0, 120, 255), 2)
                cv2.putText(img_rot, "Fish", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)

            # Head
            if "Head_x1" in data_row and data_row["Head_x1"] is not None:
                cx = int((data_row["Head_x1"] + data_row["Head_x2"]) / 2)
                cy = int((data_row["Head_y1"] + data_row["Head_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (255, 0, 0), -1)

            # Tail
            if "Tail_x1" in data_row and data_row["Tail_x1"] is not None:
                cx = int((data_row["Tail_x1"] + data_row["Tail_x2"]) / 2)
                cy = int((data_row["Tail_y1"] + data_row["Tail_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (0, 255, 0), -1)
    
    img_depth = None
    if depth_path and os.path.exists(depth_path):
        depth = np.load(depth_path)
        norm_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
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


# --- Main App Interface ---

def main():
    st.sidebar.title("🐟 Fishometry")
    
    # 1. Select Dataset
    datasets = get_datasets()
    if not datasets:
        st.error("No datasets found in `data/` directory.")
        return
        
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    # Load Metadata for dataset
    df_meta, all_image_names = load_dataset_metadata(selected_dataset)
    
    if df_meta is None:
        st.sidebar.warning(f"Could not load metadata (processed_*.csv) for {selected_dataset}.")
    
    # Load Depth Models
    depth_models = get_depth_models(selected_dataset)
    selected_depth_model = None
    if depth_models:
        selected_depth_model = st.sidebar.selectbox("Depth Model", depth_models)
    
    # 2. Select Mode
    mode = st.sidebar.radio("Mode", ["Data Explorer", "Error Analysis", "Model Comparison"])
    
    if mode == "Data Explorer":
        render_explorer(selected_dataset, df_meta, all_image_names, selected_depth_model)
    elif mode == "Error Analysis":
        render_analysis(selected_dataset, df_meta, all_image_names, selected_depth_model)
    elif mode == "Model Comparison":
        render_comparison(selected_dataset)

def render_explorer(dataset, df_meta, all_image_names, depth_model):
    st.header(f"Explorer: {dataset}")
    
    if not all_image_names:
        st.warning("No images found in metadata.")
        return

    # Image Selector
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_img = st.selectbox("Select Image", all_image_names)
    
    # Get Data
    row_data = {}
    if df_meta is not None:
        row = df_meta.filter(pl.col("name") == selected_img)
        if row.height > 0:
            row_data = row.to_dicts()[0]
            # Clean up None values
            row_data = {k: v for k, v in row_data.items() if v is not None}
            
    # Get Predictions
    preds = load_all_predictions_for_image(dataset, selected_img)
    if preds:
        row_data["Predictions"] = preds
    
    # Process Images
    img_raw, img_rot, img_depth, img_seg = process_images(dataset, selected_img, row_data, depth_model)
    
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


def render_analysis(dataset, df_meta, all_image_names, depth_model):
    st.header(f"Error Analysis: {dataset}")
    
    pred_files = load_predictions_list(dataset)
    if not pred_files:
        st.warning("No prediction CSV files found in `data/{dataset}/predictions`.")
        return
        
    # File Selector
    pred_file_map = {os.path.basename(f): f for f in pred_files}
    selected_file_name = st.selectbox("Select Prediction Model", list(pred_file_map.keys()))
    selected_file_path = pred_file_map[selected_file_name]
    
    # Load and Compute Metrics
    df_pred = load_prediction_df(selected_file_path)
    
    required_cols = ["name", "gt_length", "pred_length"]
    if not all(col in df_pred.columns for col in required_cols):
        st.error(f"Selected CSV must contain columns: {required_cols}")
        st.dataframe(df_pred.head())
        return
        
    # Calculate Errors
    # Ensure numeric types
    df_pred = df_pred.with_columns([
        pl.col("gt_length").cast(pl.Float64),
        pl.col("pred_length").cast(pl.Float64)
    ])
    
    df_pred = df_pred.with_columns([
        (pl.col("gt_length") - pl.col("pred_length")).abs().alias("abs_error"),
        ((pl.col("gt_length") - pl.col("pred_length")).abs() / pl.col("gt_length") * 100).alias("pct_error")
    ])
    
    # Metrics Summary
    mae = df_pred["abs_error"].mean()
    mape = df_pred["pct_error"].mean()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Abs Error", f"{mae:.2f}")
    m2.metric("Mean % Error", f"{mape:.2f}%")
    m3.metric("Total Samples", df_pred.height)
    
    # Specific Image Analysis
    st.subheader("Deep Dive")
    
    # Sort options
    sort_by = st.radio("Sort By", ["Absolute Error (Descending)", "Percentage Error (Descending)"], horizontal=True)
    
    if "Absolute" in sort_by:
        df_sorted = df_pred.sort("abs_error", descending=True)
    else:
        df_sorted = df_pred.sort("pct_error", descending=True)
        
    # Display interactive table
    st.dataframe(
        df_sorted.select(["name", "gt_length", "pred_length", "abs_error", "pct_error"]),
        width="stretch",
        hide_index=True
    )
    
    # Selector to view specific image from this list
    # Because we want to support "Next/Prev" or simple top-k investigation, 
    # let's just use a selectbox populated by the sorted dataframe.
    
    top_n_names = df_sorted["name"].to_list()
    
    st.markdown("---")
    st.markdown("### Visualize Specific Image")
    
    # Optional: Filter/Search
    
    selected_err_img = st.selectbox("Select Image (Sorted by Error)", top_n_names)
    
    # Retrieve specific error stats for this image
    img_row = df_sorted.filter(pl.col("name") == selected_err_img).to_dicts()[0]
    
    st.info(f"GT: {img_row['gt_length']:.2f} | Pred: {img_row['pred_length']:.2f} | Error: {img_row['abs_error']:.2f} ({img_row['pct_error']:.1f}%)")
    
    # Reuse Explorer view logic for this single image
    # Note: We can reuse render_explorer logic but we need to pass just this image.
    # Simpler to just call the processing function and display.
    
    row_data = {}
    if df_meta is not None:
        r = df_meta.filter(pl.col("name") == selected_err_img)
        if r.height > 0:
            row_data = r.to_dicts()[0]
            row_data = {k: v for k, v in row_data.items() if v is not None}
    
    row_data["Specific Prediction"] = {
        "Model": selected_file_name,
        "GT": img_row['gt_length'],
        "Pred": img_row['pred_length']
    }
            
    
    img_raw, img_rot, img_depth, img_seg = process_images(dataset, selected_err_img, row_data, depth_model)
    
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


def render_comparison(dataset):
    st.header(f"Model Comparison: {dataset}")
    
    pred_files = load_predictions_list(dataset)
    if not pred_files:
        st.warning("No prediction files found.")
        return
        
    # User selects split
    split_choice = st.selectbox("Select Split to Compare", ["val", "test", "train"])
    
    # Filter files for this split
    # Assumption: filename ends with _{split}.csv
    relevant_files = [f for f in pred_files if f.endswith(f"_{split_choice}.csv")]
    
    if not relevant_files:
        st.info(f"No prediction files found for split '{split_choice}'")
        return
        
    results = []
    
    for f in relevant_files:
        base_name = os.path.basename(f)
        # Model Name: remove _{split}.csv
        model_name = base_name.replace(f"_{split_choice}.csv", "")
        
        try:
            df = load_prediction_df(f)
            if "gt_length" not in df.columns or "pred_length" not in df.columns:
                continue
                
            df = df.with_columns([
                pl.col("gt_length").cast(pl.Float64),
                pl.col("pred_length").cast(pl.Float64)
            ])
            
            mae = (df["gt_length"] - df["pred_length"]).abs().mean()
            mape = ((df["gt_length"] - df["pred_length"]).abs() / df["gt_length"] * 100).mean()
            
            results.append({
                "Model": model_name,
                "MAE": mae,
                "MAPE (%)": mape,
                "Samples": df.height
            })
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if not results:
        st.warning("Could not extract metrics from files.")
        return
        
    df_res = pl.DataFrame(results).sort("MAE")
    
    st.subheader("Leaderboard")
    st.dataframe(df_res, width="stretch", hide_index=True)
    
    st.subheader("Visualization")
    
    chart_metric = st.radio("Metric to Visualize", ["MAE", "MAPE (%)"], horizontal=True)
    
    st.bar_chart(
        df_res.to_pandas().set_index("Model")[chart_metric],
        width="stretch"
    ) 


if __name__ == "__main__":
    main()
