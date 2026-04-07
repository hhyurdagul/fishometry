import os
import polars as pl
import numpy as np
from src.config import Config
from src.create_data.steps.base import PipelineStep


class SplitStep(PipelineStep):
    def __init__(self, config: Config):
        super().__init__(config)
        np.random.seed(42)

    def __split_frame(self, df: pl.DataFrame, train_ratio: float, val_ratio: float) -> pl.DataFrame:

        train_size = int(len(df) * train_ratio)
        val_size = int(round(len(df) * val_ratio))
        df_sampled = df.sample(
            fraction=1, shuffle=True
        ).with_columns(
            pl.lit(False).alias("is_train"), 
            pl.lit(False).alias("is_val"), 
            pl.lit(False).alias("is_test")
        )

        train_df = df_sampled[:train_size].with_columns(pl.lit(True).alias("is_train"))
        val_df = df_sampled[train_size : train_size + val_size].with_columns(pl.lit(True).alias("is_val"))
        test_df = df_sampled[train_size + val_size :].with_columns(pl.lit(True).alias("is_test"))

        return pl.concat([train_df, val_df, test_df])


    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, Config]:
        train_ratio = self.config.params.train_ratio
        val_ratio = self.config.params.val_ratio

        if self.config.fish_type_available:
            fish_types = df["fish_type"].unique().to_list()
            data = []
            for fish_type in fish_types:
                data.append(
                    self.__split_frame(
                        df.filter(pl.col("fish_type") == fish_type),
                        train_ratio,
                        val_ratio,
                    )
                )
            df = pl.concat(data)
        else:
            df = self.__split_frame(df, train_ratio, val_ratio)

        return df.sample(fraction=1, shuffle=True), self.config
