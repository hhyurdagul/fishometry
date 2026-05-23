"""Fishometry Explorer - Streamlit visualization app."""

import streamlit as st

from src.visualization.data_loading import (
    get_datasets,
    get_depth_models,
    get_prediction_fish_types,
    load_prediction_df,
    load_dataset_metadata,
)
from src.visualization.views import (
    render_explorer,
    render_analysis,
    render_comparison,
    render_fish_type_comparison,
    render_model_fish_type_heatmap,
    render_prediction_viz,
)

st.set_page_config(layout="wide", page_title="Fishometry Explorer")


def main():
    st.sidebar.title("Fishometry")

    # 1. Select Dataset
    datasets = get_datasets()
    if not datasets:
        st.error("No datasets found in `data/` directory.")
        return

    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)

    # Load Metadata for dataset
    df_meta, all_image_names = load_dataset_metadata(selected_dataset)

    if df_meta is None:
        st.sidebar.warning(
            f"Could not load metadata for {selected_dataset}."
        )

    # Load Depth Models
    depth_models = get_depth_models(selected_dataset)
    selected_depth_model = None
    if depth_models:
        selected_depth_model = st.sidebar.selectbox("Depth Model", depth_models)

    df_pred = load_prediction_df(selected_dataset)
    has_fish_types = bool(get_prediction_fish_types(df_pred))
    modes = [
        "Data Explorer",
        "Prediction Visualization",
        "Error Analysis",
        "Model Comparison",
    ]
    if has_fish_types:
        modes.extend(["Fish Type Comparison", "Model x Fish Type Heatmap"])

    # 2. Select Mode
    mode = st.sidebar.radio(
        "Mode",
        modes,
    )

    if mode == "Data Explorer":
        render_explorer(
            selected_dataset, df_meta, all_image_names, selected_depth_model
        )
    elif mode == "Prediction Visualization":
        render_prediction_viz(selected_dataset)
    elif mode == "Error Analysis":
        render_analysis(
            selected_dataset, df_meta, all_image_names, selected_depth_model
        )
    elif mode == "Model Comparison":
        render_comparison(selected_dataset)
    elif mode == "Fish Type Comparison":
        render_fish_type_comparison(selected_dataset)
    elif mode == "Model x Fish Type Heatmap":
        render_model_fish_type_heatmap(selected_dataset)


if __name__ == "__main__":
    main()
