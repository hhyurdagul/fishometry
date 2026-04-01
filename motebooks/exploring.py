import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl

    return alt, mo, pl


@app.cell
def _(pl):
    df = (
        pl.read_csv("data/data-outside/raw.csv")
        .with_columns(
            pl.int_range(0, pl.len()).alias("row_id"),
            pl.col("fish_type").cast(pl.Utf8).str.strip_chars(),
            pl.col("name").cast(pl.Utf8).str.strip_chars(),
            pl.col("length").cast(pl.Int64),
        )
    )
    return (df,)


@app.cell
def _(alt, pl):
    def add_absolute_percentage_error(data):
        base_data = data.drop(
            [
                column_name
                for column_name in [
                    "median_length_by_type",
                    "absolute_error",
                    "absolute_percentage_error",
                ]
                if column_name in data.columns
            ]
        )
        medians = base_data.group_by("fish_type").agg(
            pl.col("length").median().alias("median_length_by_type")
        )
        return base_data.join(medians, on="fish_type", how="left").with_columns(
            (pl.col("length") - pl.col("median_length_by_type")).abs().alias("absolute_error"),
            pl.when(pl.col("median_length_by_type") == 0)
            .then(None)
            .otherwise(
                ((pl.col("length") - pl.col("median_length_by_type")).abs() / pl.col("median_length_by_type") * 100)
            )
            .round(2)
            .alias("absolute_percentage_error")
        )

    def list_iqr_outliers(data):
        bounds = (
            data.group_by("fish_type")
            .agg(
                pl.col("length").quantile(0.25).alias("q1"),
                pl.col("length").quantile(0.75).alias("q3"),
            )
            .with_columns((pl.col("q3") - pl.col("q1")).alias("iqr"))
        )
        return (
            add_absolute_percentage_error(data)
            .join(bounds, on="fish_type", how="left")
            .filter(
                (pl.col("length") < pl.col("q1") - 1.5 * pl.col("iqr"))
                | (pl.col("length") > pl.col("q3") + 1.5 * pl.col("iqr"))
            )
            .select(
                "row_id",
                "fish_type",
                "name",
                "length",
                "median_length_by_type",
                "absolute_error",
                "absolute_percentage_error",
            )
            .sort(["fish_type", "length"])
        )

    def remove_iqr_outliers(data):
        current_outliers = list_iqr_outliers(data).select("row_id")
        if current_outliers.height == 0:
            return add_absolute_percentage_error(data)
        return add_absolute_percentage_error(data).join(current_outliers, on="row_id", how="anti")

    def add_mad_scores(data):
        ape_data = add_absolute_percentage_error(data)
        mad_stats = ape_data.group_by("fish_type").agg(
            pl.col("length").median().alias("median_length_by_type"),
            (pl.col("length") - pl.col("length").median()).abs().median().alias("mad_length_by_type"),
        )
        base_data = ape_data.drop(
            [
                column_name
                for column_name in ["mad_length_by_type", "robust_z_score"]
                if column_name in ape_data.columns
            ]
        )
        return base_data.drop(
            [column_name for column_name in ["median_length_by_type"] if column_name in base_data.columns]
        ).join(mad_stats, on="fish_type", how="left").with_columns(
            pl.when(pl.col("mad_length_by_type") == 0)
            .then(None)
            .otherwise(
                0.6745 * (pl.col("length") - pl.col("median_length_by_type")) / pl.col("mad_length_by_type")
            )
            .round(3)
            .alias("robust_z_score")
        )

    def list_mad_outliers(data, threshold=2):
        return (
            add_mad_scores(data)
            .filter(pl.col("robust_z_score").abs() > threshold)
            .select(
                "row_id",
                "fish_type",
                "name",
                "length",
                "median_length_by_type",
                "mad_length_by_type",
                "robust_z_score",
                "absolute_error",
                "absolute_percentage_error",
            )
            .sort(["fish_type", "length"])
        )

    def remove_mad_outliers(data, threshold=2):
        current_outliers = list_mad_outliers(data, threshold=threshold).select("row_id")
        if current_outliers.height == 0:
            return add_mad_scores(data)
        return add_mad_scores(data).join(current_outliers, on="row_id", how="anti")

    def iterate_outlier_removal(data, iteration_count, method="iqr", threshold=2):
        current_data = add_absolute_percentage_error(data)
        for _ in range(iteration_count):
            if method == "mad":
                current_outliers = list_mad_outliers(current_data, threshold=threshold)
            else:
                current_outliers = list_iqr_outliers(current_data)
            if current_outliers.height == 0:
                break
            if method == "mad":
                current_data = remove_mad_outliers(current_data, threshold=threshold)
            else:
                current_data = remove_iqr_outliers(current_data)
        return add_mad_scores(current_data) if method == "mad" else add_absolute_percentage_error(current_data)

    def max_outlier_iterations(data, method="iqr", threshold=2):
        max_iterations = 0
        probe_df = add_absolute_percentage_error(data)
        while True:
            if method == "mad":
                probe_outliers = list_mad_outliers(probe_df, threshold=threshold)
            else:
                probe_outliers = list_iqr_outliers(probe_df)
            if probe_outliers.height == 0:
                break
            next_probe_df = (
                remove_mad_outliers(probe_df, threshold=threshold)
                if method == "mad"
                else remove_iqr_outliers(probe_df)
            )
            if next_probe_df.height == probe_df.height:
                break
            probe_df = next_probe_df
            max_iterations += 1
        return max_iterations

    def boxplot_by_fish_type(data, title):
        return (
            alt.Chart(add_absolute_percentage_error(data))
            .mark_boxplot(size=35)
            .encode(
                x=alt.X("fish_type:N", title="Fish type", sort="-y"),
                y=alt.Y("length:Q", title="Length"),
                color=alt.Color("fish_type:N", legend=None),
                tooltip=[
                    alt.Tooltip("fish_type:N", title="Fish type"),
                    alt.Tooltip("name:N", title="Image"),
                    alt.Tooltip("length:Q", title="Length"),
                    alt.Tooltip("absolute_percentage_error:Q", title="APE (%)"),
                ],
            )
            .properties(title=title, width=700, height=340)
        )

    return (
        add_absolute_percentage_error,
        boxplot_by_fish_type,
        iterate_outlier_removal,
        list_iqr_outliers,
        list_mad_outliers,
        max_outlier_iterations,
    )


@app.cell
def _(df, max_outlier_iterations):
    iqr_max_iterations = max_outlier_iterations(df, method="iqr")
    mad_max_iterations = max_outlier_iterations(df, method="mad")
    return iqr_max_iterations, mad_max_iterations


@app.cell
def _(iqr_max_iterations, mad_max_iterations, mo):
    iqr_iteration_slider = mo.ui.slider(
        start=0,
        stop=iqr_max_iterations,
        value=0,
        step=1,
        label="IQR iteration",
        full_width=True,
    )
    mad_iteration_slider = mo.ui.slider(
        start=0,
        stop=mad_max_iterations,
        value=0,
        step=1,
        label="MAD iteration",
        full_width=True,
    )
    mo.hstack([iqr_iteration_slider, mad_iteration_slider], widths=[1, 1])
    return iqr_iteration_slider, mad_iteration_slider


@app.cell
def _(
    add_absolute_percentage_error,
    df,
    iqr_iteration_slider,
    iterate_outlier_removal,
    list_iqr_outliers,
    list_mad_outliers,
    mad_iteration_slider,
):
    base_df = add_absolute_percentage_error(df)
    iqr_current_df = iterate_outlier_removal(df, iqr_iteration_slider.value, method="iqr")
    mad_current_df = iterate_outlier_removal(df, mad_iteration_slider.value, method="mad")
    iqr_current_outliers = list_iqr_outliers(iqr_current_df)
    mad_current_outliers = list_mad_outliers(mad_current_df)
    return (
        base_df,
        iqr_current_df,
        iqr_current_outliers,
        mad_current_df,
        mad_current_outliers,
    )


@app.cell
def _(
    base_df,
    iqr_current_df,
    iqr_current_outliers,
    iqr_iteration_slider,
    mad_current_df,
    mad_current_outliers,
    mad_iteration_slider,
    mo,
    pl,
):
    iqr_summary = (
        iqr_current_df.group_by("fish_type")
        .agg(
            pl.len().alias("count"),
            pl.col("length").median().alias("median_length"),
            pl.col("length").mean().round(2).alias("mean_length"),
            pl.col("absolute_percentage_error").mean().round(2).alias("mean_ape"),
        )
        .sort("count", descending=True)
    )
    mad_summary = (
        mad_current_df.group_by("fish_type")
        .agg(
            pl.len().alias("count"),
            pl.col("length").median().alias("median_length"),
            pl.col("length").mean().round(2).alias("mean_length"),
            pl.col("absolute_percentage_error").mean().round(2).alias("mean_ape"),
        )
        .sort("count", descending=True)
    )
    mo.tabs(
        {
            "IQR": mo.vstack([
                mo.md(
                    f"""
                    # IQR outlier removal

                    - Iteration: **{iqr_iteration_slider.value} / {iqr_iteration_slider.stop}**
                    - Rows shown: **{iqr_current_df.height}**
                    - Rows removed so far: **{base_df.height - iqr_current_df.height}**
                    - Current outliers found: **{iqr_current_outliers.height}**
                    """
                ),
                mo.ui.table(iqr_summary, selection=None),
            ]),
            "MAD": mo.vstack([
                mo.md(
                    f"""
                    # MAD outlier removal

                    - Iteration: **{mad_iteration_slider.value} / {mad_iteration_slider.stop}**
                    - Rows shown: **{mad_current_df.height}**
                    - Rows removed so far: **{base_df.height - mad_current_df.height}**
                    - Current outliers found: **{mad_current_outliers.height}**
                    """
                ),
                mo.ui.table(mad_summary, selection=None),
            ]),
        }
    )
    return


@app.cell
def _(
    boxplot_by_fish_type,
    iqr_current_df,
    iqr_iteration_slider,
    mad_current_df,
    mad_iteration_slider,
    mo,
):
    mo.hstack(
        [
            boxplot_by_fish_type(
                iqr_current_df,
                f"IQR boxplot after iteration {iqr_iteration_slider.value}",
            ),
            boxplot_by_fish_type(
                mad_current_df,
                f"MAD boxplot after iteration {mad_iteration_slider.value}",
            ),
        ],
        widths=[1, 1],
    )
    return


@app.cell
def _(iqr_current_outliers, mad_current_outliers, mo):
    mo.tabs(
        {
            "IQR Outliers": mo.ui.table(iqr_current_outliers, selection=None),
            "MAD Outliers": mo.ui.table(mad_current_outliers, selection=None),
        }
    )
    return


@app.cell
def _(iqr_current_df, mad_current_df, mo):
    mo.tabs(
        {
            "IQR Data": mo.ui.table(iqr_current_df, selection=None),
            "MAD Data": mo.ui.table(mad_current_df, selection=None),
        }
    )
    return


if __name__ == "__main__":
    app.run()
