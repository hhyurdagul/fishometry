import polars as pl

from src.config import Config


class FeatureStep:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if config.dataset.rotate
            else config.dataset.input_dir
        )

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        print("Count:", len(df))
        return (
            df.pipe(self._create_geometric_features)
            .pipe(self._one_hot_encode_fish_type)
            .pipe(self._encode_vlm_features_if_available)
        )

    def _create_geometric_features(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            relative_w=pl.col("Fish_w") / pl.col("Image_w"),
            relative_h=pl.col("Fish_h") / pl.col("Image_h"),
            relative_area=(
                (pl.col("Fish_w") * pl.col("Fish_h"))
                / (pl.col("Image_w") * pl.col("Image_h"))
            ),
            fish_aspect=pl.col("Fish_w") / pl.col("Fish_h"),
            fish_area=(pl.col("Fish_w") * pl.col("Fish_h")).sqrt(),
        )

    def _one_hot_encode_fish_type(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.config.dataset.fish_type_available:
            return df.to_dummies("fish_type").with_columns(df["fish_type"])
        return df

    def _encode_vlm_features_if_available(self, df: pl.DataFrame) -> pl.DataFrame:
        expr = []
        columns = df.columns
        if "background_depth" in columns:
            expr.append(
                pl.col("background_depth")
                .replace_strict({"far": 1, "close": 0})
                .cast(int)
            )
        if "has_other_objects" in columns:
            expr.append(pl.col("has_other_objects").cast(int))
        if "is_in_fishnet" in columns:
            expr.append(pl.col("is_in_fishnet").cast(int))

        dummies = list(
            filter(
                lambda x: x in columns,
                ["fish_placement", "fish_orientation", "lightning_condition"],
            )
        )
        if dummies:
            df = df.to_dummies(dummies)

        return df.with_columns(expr)
