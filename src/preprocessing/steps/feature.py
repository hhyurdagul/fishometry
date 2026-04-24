import polars as pl

from src.config import Config


class FeatureStep:
    def __init__(self, config: Config, rotated: bool = False):
        self.config = config
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if rotated
            else config.dataset.input_dir
        )

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.pipe(self._scale_length)
            .pipe(self._one_hot_encode)
            .pipe(self._encode_vlm_features)
        )

    def _scale_length(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            relative_w=pl.col("Fish_w") / pl.col("Image_w"),
            relative_h=pl.col("Fish_h") / pl.col("Image_h"),
            relative_area=(
                (pl.col("Fish_w") * pl.col("Fish_h"))
                / (pl.col("Image_w") * pl.col("Image_h"))
            ),
        )

    def _one_hot_encode(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.config.dataset.fish_type_available:
            return df.to_dummies("fish_type").with_columns(df["fish_type"])
        return df

    def _encode_vlm_features(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            return df.with_columns(
                pl.col("background_depth").replace_strict({"far": 1, "close": 0}),
                pl.col("has_other_objects").cast(int),
                pl.col("is_in_fishnet").cast(int),
            ).to_dummies(["fish_placement", "fish_orientation", "lightning_condition"])
        except Exception:
            return df
