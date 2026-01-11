from abc import ABC, abstractmethod
from typing import Any, Dict
import polars as pl

class PipelineStep(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        pass
