from src.config import Config
import polars as pl
import numpy as np
import os
import json
from .base import PipelineStep


class FeatureStep(PipelineStep):
    def __init__(self, config: Config, rotated: bool=False):
        super().__init__(config)
        self.input_dir = config.output_dir / "rotated" if rotated else config.input_dir

    def _get_max_dims(self, names):
        # This might be slow. Is there a better way?
        # Maybe we can just store the max dim in a config or cache?
        # For now, let's assume standard maxes or scan.
        # To avoid being too slow, let's limit the scan or use a known max if possible.
        # But precise scaling requires precise max.

        # Let's use `identify` if available (Linux) for speed?
        # Or just PIL Image.open(path).size (lazy load).
        from PIL import Image

        max_w, max_h = 0, 0
        # Check first 100 to guess? No, unsafe.
        # Let's checks all.

        cache_path = self.config.output_dir / "cache" / "max_dims.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                d = json.load(f)
                return d["w"], d["h"]

        print("Calculating max image dimensions from the data...")
        for name in names:
            p = self.input_dir / name
            if p.exists():
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                        max_w = max(max_w, w)
                        max_h = max(max_h, h)
                except Exception:
                    pass

        # Cache it
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"w": max_w, "h": max_h}, f)

        return max_w, max_h

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. Get Max Dims
        max_w, max_h = self._get_max_dims(df["name"].to_list())

        # 2. Add Scaled Features
        # Requires img_w, img_h in df
        if "img_w" in df.columns and "img_h" in df.columns:
            # fish_w_scaled = fish_w * (max_x / x)
            # fish_h_scaled = fish_h * (max_y / y)

            df = df.with_columns(
                [
                    (max_w / pl.col("img_w") * pl.col("Fish_w")).alias("Fish_w_scaled"),
                    (max_h / pl.col("img_h") * pl.col("Fish_h")).alias("Fish_h_scaled"),
                ]
            )

        # 3. Fish Type Encoding & Stats
        if self.config.fish_type_available:
            # One-hot encoding
            # We use self.fish_types to determine columns.
            for ft in df["fish_type"].unique():
                # col name: fish_type_Salmon
                df = df.with_columns(
                    (pl.col("fish_type") == ft).cast(pl.Int8).alias(f"fish_type_{ft}")
                )

        return df
