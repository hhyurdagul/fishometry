from src.config import Config
import polars as pl
import numpy as np
import os
import json
from PIL import Image
from .base import PipelineStep


class FeatureStep(PipelineStep):
    def __init__(self, config: Config, rotated: bool=False):
        super().__init__(config)
        self.input_dir = config.dataset.output_dir / "rotated" if rotated else config.dataset.input_dir

    def _scale_length(self, df: pl.DataFrame):
        return df.with_columns(
            relative_w = pl.col("Fish_w") / pl.col("img_w"),
            relative_h = pl.col("Fish_h") / pl.col("img_h"),
            relative_area = (
                    (pl.col("Fish_w") * pl.col("Fish_h")) / 
                    (pl.col("img_w") * pl.col("img_h"))
            ),
        )

    def _one_hot_encode(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.config.dataset.fish_type_available:
            return df.to_dummies("fish_type")
        return df

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._scale_length).pipe(self._one_hot_encode)
