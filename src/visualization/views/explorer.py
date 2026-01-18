"""Data Explorer view for browsing images and metadata."""

import streamlit as st
import polars as pl

from data_loading import load_all_predictions_for_image
from image_processing import process_images


def render_explorer(dataset, df_meta, all_image_names, depth_model):
    """Render the Data Explorer view."""
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
            row_data = {k: v for k, v in row_data.items() if v is not None}

    # Get Predictions
    preds = load_all_predictions_for_image(dataset, selected_img)
    if preds:
        row_data["Predictions"] = preds

    # Process Images
    img_raw, img_rot, img_depth, img_blackout = process_images(
        dataset, selected_img, row_data, depth_model
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
        st.markdown("### Blackout")
        if img_blackout is not None:
            st.image(img_blackout, width="stretch")
        else:
            st.info("No blackout image found")

    # Metadata Expander
    with st.expander("See Metadata JSON", expanded=True):
        st.json(row_data)
