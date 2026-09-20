import polars as pl

from src.config import Config
from src.context_features import (
    VLM_BOOLEAN_COLUMNS,
    VLM_CATEGORICAL_VALUES,
    VLM_INTEGER_COLUMNS,
)


class FeatureStep:
    def __init__(self, config: Config, *, encode_vlm_features: bool = True):
        self.config = config
        self.encode_vlm_features = encode_vlm_features
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if config.dataset.rotate
            else config.dataset.input_dir
        )

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        print("Count:", len(df))
        df = df.pipe(self._create_geometric_features).pipe(
            self._one_hot_encode_fish_type
        )
        if self.encode_vlm_features:
            df = self._encode_vlm_features_if_available(df)
        return df

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
        numeric_columns = [
            name
            for name in VLM_BOOLEAN_COLUMNS + VLM_INTEGER_COLUMNS
            if name in df.columns
        ]
        categorical_columns = [
            name for name in VLM_CATEGORICAL_VALUES if name in df.columns
        ]
        df = df.with_columns(pl.col(numeric_columns).cast(pl.Int64))
        if categorical_columns:
            df = df.to_dummies(categorical_columns)
        return df
