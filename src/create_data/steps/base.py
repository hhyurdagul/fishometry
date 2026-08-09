from src.config import Config
from abc import ABC, abstractmethod
import polars as pl


class PipelineStep(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, Config]:
        pass
