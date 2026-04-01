import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from sklearn.model_selection import train_test_split

    return (
        ConfusionMatrixDisplay,
        LogisticRegression,
        Path,
        classification_report,
        mo,
        pl,
        train_test_split,
    )


@app.cell
def _(Path, pl):
    candidate_paths = [
        Path("../data/data-outside/processed/processed_train.csv"),
        Path("data/data-outside/processed/processed_train.csv"),
    ]
    train_path = next((path for path in candidate_paths if path.exists()), None)
    if train_path is None:
        raise FileNotFoundError("Could not find processed_train.csv in the expected data folders.")

    train_df = pl.read_csv(train_path)
    train_df
    return (train_df,)


@app.cell
def _(pl, train_df):
    species_quantiles = train_df.group_by("fish_type").agg(
        pl.col("length").quantile(0.1).alias("species_q10_length"),
        pl.col("length").quantile(0.2).alias("species_q20_length"),
        pl.col("length").quantile(0.3).alias("species_q30_length"),
        pl.col("length").quantile(0.4).alias("species_q40_length"),
        pl.col("length").quantile(0.5).alias("species_q50_length"),
        pl.col("length").quantile(0.6).alias("species_q60_length"),
        pl.col("length").quantile(0.7).alias("species_q70_length"),
        pl.col("length").quantile(0.8).alias("species_q80_length"),
        pl.col("length").quantile(0.9).alias("species_q90_length"),
    )

    enriched_df = train_df.join(species_quantiles, on="fish_type", how="left").with_columns(
        (pl.col("length") > pl.col("species_mean_length")).cast(pl.Int8).alias("is_above_species_mean"),
        pl.when(pl.col("length") <= pl.col("species_q10_length"))
        .then(pl.lit(0))
        .when(pl.col("length") <= pl.col("species_q20_length"))
        .then(pl.lit(1))
        .when(pl.col("length") <= pl.col("species_q30_length"))
        .then(pl.lit(2))
        .when(pl.col("length") <= pl.col("species_q40_length"))
        .then(pl.lit(3))
        .when(pl.col("length") <= pl.col("species_q50_length"))
        .then(pl.lit(4))
        .when(pl.col("length") <= pl.col("species_q60_length"))
        .then(pl.lit(5))
        .when(pl.col("length") <= pl.col("species_q70_length"))
        .then(pl.lit(6))
        .when(pl.col("length") <= pl.col("species_q80_length"))
        .then(pl.lit(7))
        .when(pl.col("length") <= pl.col("species_q90_length"))
        .then(pl.lit(8))
        .otherwise(pl.lit(9))
        .alias("species_length_quartile"),
    )

    enriched_df.select(
        "fish_type",
        "length",
        "species_mean_length",
        "species_q10_length",
        "species_q20_length",
        "species_q30_length",
        "species_q40_length",
        "species_q50_length",
        "species_q60_length",
        "species_q70_length",
        "species_q80_length",
        "species_q90_length",
        "is_above_species_mean",
        "species_length_quartile",
    )
    return (enriched_df,)


@app.cell
def _(enriched_df):
    leakage_columns = [
        "name",
        "fish_type",
        "length",
        "species_q10_length",
        "species_q20_length",
        "species_q30_length",
        "species_q40_length",
        "species_q50_length",
        "species_q60_length",
        "species_q70_length",
        "species_q80_length",
        "species_q90_length",
        "is_above_species_mean",
        "species_length_quartile",
    ]

    X = enriched_df.drop(*leakage_columns).to_pandas()
    y_above_mean = enriched_df["is_above_species_mean"].to_pandas()
    y_quartile = enriched_df["species_length_quartile"].to_pandas()
    return X, y_above_mean, y_quartile


@app.cell
def _(X, train_test_split, y_above_mean, y_quartile):
    (
        X_train,
        X_test,
        y_above_train,
        y_above_test,
        y_quartile_train,
        y_quartile_test,
    ) = train_test_split(
        X,
        y_above_mean,
        y_quartile,
        test_size=0.25,
        random_state=42,
        stratify=y_above_mean,
    )
    return X_test, X_train, y_above_test, y_above_train, y_quartile_train


@app.cell
def _(LogisticRegression, X_test, X_train, y_above_train):
    above_mean_model = LogisticRegression(
        max_iter=2000,
        class_weight={0: 3, 1: 1},
    )
    above_mean_model.fit(X_train, y_above_train)

    positive_threshold = 0.70
    above_mean_scores = above_mean_model.predict_proba(X_test)[:, 1]
    above_mean_pred = (above_mean_scores >= positive_threshold).astype(int)
    return above_mean_model, above_mean_pred, positive_threshold


@app.cell
def _(
    ConfusionMatrixDisplay,
    above_mean_pred,
    positive_threshold,
    y_above_test,
):
    ConfusionMatrixDisplay.from_predictions(y_above_test, above_mean_pred).ax_.set_title(
        f"Above-mean classification at threshold={positive_threshold:.2f}"
    )
    return


@app.cell
def _(
    above_mean_pred,
    classification_report,
    positive_threshold,
    y_above_test,
):
    print("Target: 1 = above species mean, 0 = at-or-below species mean")
    print(f"Using a stricter positive threshold of {positive_threshold:.2f} to reduce false positives.")
    print(classification_report(y_above_test, above_mean_pred, digits=3))
    return


@app.cell
def _(
    LogisticRegression,
    X,
    X_train,
    above_mean_model,
    enriched_df,
    mo,
    pl,
    y_quartile_train,
):
    quartile_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    quartile_model.fit(X_train, y_quartile_train)

    quartile_feature_names = [
        "p_length_q1",
        "p_length_q2",
        "p_length_q3",
        "p_length_q4",
        "p_length_q5",
        "p_length_q6",
        "p_length_q7",
        "p_length_q8",
        "p_length_q9",
        "p_length_q10",
    ]

    above_mean_probabilities = above_mean_model.predict_proba(X)[:, 1]
    quartile_probabilities = quartile_model.predict_proba(X)

    feature_message = mo.md(
        """
        ## How to use this for exact length estimation

        Use the classifier outputs as soft features, not hard labels:

        - `p_above_species_mean`: probability the fish is above its species mean
        - `p_length_q1` to `p_length_q4`: probabilities of being in each species quartile

        These features help a regressor avoid collapsing too hard toward the species mean.
        """
    )

    quartile_probability_df = pl.DataFrame(quartile_probabilities, schema=quartile_feature_names)

    regression_ready_features = pl.concat(
        [
            enriched_df.select("fish_type", "length", "species_mean_length", "species_median_length"),
            pl.DataFrame({"p_above_species_mean": above_mean_probabilities}),
            quartile_probability_df,
        ],
        how="horizontal",
    )

    mo.vstack([feature_message, regression_ready_features.head(10)])
    return (quartile_probability_df,)


@app.cell
def _(enriched_df, pl, quartile_probability_df):
    data = pl.concat([enriched_df, quartile_probability_df], how="horizontal").drop(pl.selectors.starts_with("species_", "is_above"))
    return (data,)


@app.cell
def _(data):
    data
    return


@app.cell
def _():
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_predict

    return XGBRegressor, cross_val_predict


@app.cell
def _(XGBRegressor, cross_val_predict, data):
    y = data["length"]
    pred = cross_val_predict(XGBRegressor(), data.drop("fish_type", "name", "length"), y)
    return pred, y


@app.cell
def _(pred, y):
    (abs(y - pred) / y).mean()
    return


if __name__ == "__main__":
    app.run()
