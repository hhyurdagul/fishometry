"""Correlation Analysis view for exploring feature correlations."""

import streamlit as st
import polars as pl
import pandas as pd
import altair as alt


def render_correlation(dataset, df_meta):
    """Render the Correlation Analysis view."""
    st.header(f"Correlation Analysis: {dataset}")

    if df_meta is None or df_meta.height == 0:
        st.warning("No metadata available for correlation analysis.")
        return

    # Define feature groups
    feature_groups = {
        "Bounding Box": [
            "Fish_w",
            "Fish_h",
            "Fish_x1",
            "Fish_x2",
            "Fish_y1",
            "Fish_y2",
        ],
        "Scaled": ["Fish_w_scaled", "Fish_h_scaled"],
        "Head/Tail": ["Head_w", "Head_h", "Tail_w", "Tail_h"],
        "Depth": [c for c in df_meta.columns if "depth" in c.lower()],
        "Species Stats": [c for c in df_meta.columns if c.startswith("species_")],
    }

    # Filter to only existing columns
    available_features = []
    for group, cols in feature_groups.items():
        existing = [c for c in cols if c in df_meta.columns]
        if existing:
            available_features.extend(existing)

    if not available_features:
        st.warning("No numeric features found for correlation analysis.")
        return

    # Feature selection - default to all features
    st.subheader("Select Features")
    selected_features = st.multiselect(
        "Features to analyze",
        available_features,
        default=available_features,
    )

    if not selected_features:
        st.info("Please select at least one feature.")
        return

    # Check if length column exists
    if "length" not in df_meta.columns:
        st.error("Target column 'length' not found in metadata.")
        return

    # Prepare data for correlation
    analysis_cols = selected_features + ["length"]
    df_numeric = df_meta.select([c for c in analysis_cols if c in df_meta.columns])

    # Drop nulls for clean correlation
    df_clean = df_numeric.drop_nulls()

    if df_clean.height < 10:
        st.warning(
            f"Only {df_clean.height} samples available after removing nulls. Need at least 10."
        )
        return

    st.info(
        f"Analyzing {df_clean.height} samples with {len(selected_features)} features."
    )

    # --- 1. Correlation Table with Length ---
    st.subheader("Correlation with Length (Target)")

    correlations = []
    for feat in selected_features:
        if feat in df_clean.columns:
            corr = df_clean.select([feat, "length"]).to_pandas().corr().iloc[0, 1]
            correlations.append({"Feature": feat, "Correlation": round(corr, 4)})

    corr_df = pl.DataFrame(correlations).sort("Correlation", descending=True)
    st.dataframe(corr_df, hide_index=True, width="stretch")

    # --- 2. Correlation Heatmap ---
    st.subheader("Feature Correlation Heatmap")

    # Calculate full correlation matrix
    pandas_df = df_clean.to_pandas()
    corr_matrix = pandas_df.corr()

    # Prepare data for altair heatmap
    corr_data = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            corr_data.append(
                {
                    "Feature1": str(col1),
                    "Feature2": str(col2),
                    "Correlation": float(corr_matrix.iloc[i, j]),
                }
            )

    heatmap_pandas = pd.DataFrame(corr_data)

    # Calculate dynamic size based on number of features
    n_features = len(selected_features) + 1  # +1 for length
    cell_size = max(40, min(60, 400 // n_features))
    chart_size = cell_size * n_features + 100

    heatmap = (
        alt.Chart(heatmap_pandas)
        .mark_rect()
        .encode(
            x=alt.X("Feature1:N", title=None, sort=None),
            y=alt.Y("Feature2:N", title=None, sort=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
            ),
            tooltip=[
                "Feature1",
                "Feature2",
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(width=chart_size, height=chart_size)
    )

    text = (
        alt.Chart(heatmap_pandas)
        .mark_text(fontSize=9)
        .encode(
            x=alt.X("Feature1:N", sort=None),
            y=alt.Y("Feature2:N", sort=None),
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                (alt.datum.Correlation > 0.5) | (alt.datum.Correlation < -0.5),
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    st.altair_chart(heatmap + text, width="stretch")

    # --- 3. Scatter Plots ---
    st.subheader("Scatter Plot: Feature vs Length")

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("X-axis Feature", selected_features, index=0)
    with col2:
        color_by = st.selectbox(
            "Color by",
            ["None"] + (["fish_type"] if "fish_type" in df_meta.columns else []),
            index=0,
        )

    # Prepare scatter data
    scatter_cols = [x_feature, "length"]
    if color_by != "None" and color_by in df_meta.columns:
        scatter_cols.append(color_by)
    if "name" in df_meta.columns:
        scatter_cols.append("name")

    scatter_df = df_meta.select(
        [c for c in scatter_cols if c in df_meta.columns]
    ).drop_nulls()

    # Build scatter plot
    if color_by != "None" and color_by in scatter_df.columns:
        tooltip_fields = [x_feature, "length", color_by]
        if "name" in scatter_df.columns:
            tooltip_fields = ["name"] + tooltip_fields
        scatter = (
            alt.Chart(scatter_df.to_pandas())
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{x_feature}:Q", title=x_feature),
                y=alt.Y("length:Q", title="Length (Target)"),
                color=alt.Color(f"{color_by}:N", legend=alt.Legend(title=color_by)),
                tooltip=tooltip_fields,
            )
            .properties(width=700, height=400, title=f"{x_feature} vs Length")
            .interactive()
        )
    else:
        tooltip_fields = [x_feature, "length"]
        if "name" in scatter_df.columns:
            tooltip_fields = ["name"] + tooltip_fields
        scatter = (
            alt.Chart(scatter_df.to_pandas())
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{x_feature}:Q", title=x_feature),
                y=alt.Y("length:Q", title="Length (Target)"),
                color=alt.value("steelblue"),
                tooltip=tooltip_fields,
            )
            .properties(width=700, height=400, title=f"{x_feature} vs Length")
            .interactive()
        )

    # Add regression line
    regression = scatter.transform_regression(x_feature, "length").mark_line(
        color="red", strokeDash=[5, 5]
    )

    st.altair_chart(scatter + regression, width="stretch")

    # --- 4. Summary Statistics ---
    st.subheader("Feature Summary Statistics")

    summary_data = []
    for feat in selected_features:
        if feat in df_clean.columns:
            col_data = df_clean[feat]
            summary_data.append(
                {
                    "Feature": feat,
                    "Mean": round(col_data.mean(), 2),
                    "Std": round(col_data.std(), 2),
                    "Min": round(col_data.min(), 2),
                    "Max": round(col_data.max(), 2),
                    "Corr w/ Length": round(
                        pandas_df[[feat, "length"]].corr().iloc[0, 1], 3
                    ),
                }
            )

    summary_df = pl.DataFrame(summary_data)
    st.dataframe(summary_df, hide_index=True, width="stretch")
