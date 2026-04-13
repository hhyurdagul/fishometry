from pathlib import Path
from pydantic import (
    BaseModel,
    computed_field,
    model_validator,
    field_validator
)

import json

DATA_ROOT = Path("data")
CONFIG_ROOT = Path("configs")


class ModelConfig(BaseModel):
    yolo: str
    sam: str
    depth: str


class ParamConfig(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    @field_validator("train_ratio", "val_ratio", "test_ratio")
    @classmethod
    def check_range(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate(self):
        if self.train_ratio + self.val_ratio + self.test_ratio > 1.0:
            raise ValueError("Train, val, and test ratios must sum to less than 1.0")
        return self


class DatasetConfig(BaseModel):
    name: str
    rotate: bool = True
    fish_type_available: bool = False
    feature_sets: list[str] = ["coords", "scaled"]
    depth: list[bool] = [True, False]

    @computed_field()
    @property
    def dataset_dir(self) -> Path:
        return DATA_ROOT / self.name

    @computed_field
    @property
    def input_csv_path(self) -> Path:
        return self.dataset_dir / "raw.csv"

    @computed_field
    @property
    def split_csv_path(self) -> Path:
        return self.dataset_dir / "split.csv"

    @computed_field
    @property
    def output_csv_path(self) -> Path:
        return self.dataset_dir / "processed.csv"

    @computed_field
    @property
    def input_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @computed_field
    @property
    def output_dir(self) -> Path:
        return self.dataset_dir / "processed"

    @model_validator(mode="after")
    def validate(self):
        if not (DATA_ROOT / self.name).exists():
            raise ValueError(f"Dataset `{self.name}` does not exist")
        if not self.input_dir.exists():
            raise ValueError(f"Directory `raw` does not exist")
        if not self.input_csv_path.exists():
            raise ValueError(f"File `raw.csv` does not exist")
        return self


class Config(BaseModel):
    models: ModelConfig
    params: ParamConfig
    dataset: DatasetConfig

    def __repr__(self):
        return f"Config(dataset={self.dataset.name})"

    def __str__(self):
        return self.__repr__()


def get_valid_configs() -> list[str]:
    configs = []
    for path in CONFIG_ROOT.iterdir():
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                try:
                    Config(**json.load(f))
                    configs.append(path.stem)
                except Exception:
                    continue

    return sorted(configs)

def get_config(config_name: str) -> Config:
    config_path = (CONFIG_ROOT / config_name).with_suffix(".json")
    if not config_path.exists():
        raise ValueError(f"Config `{config_name}` does not exist")
    with open(config_path, "r", encoding="utf-8") as f:
        return Config(**json.load(f))

if __name__ == "__main__":
    print(get_valid_configs())

