from pathlib import Path
from pydantic import (
    BaseModel,
    computed_field,
    model_validator,
)

import yaml

DATA_ROOT = Path("data")
CONFIG_ROOT = Path("configs")


class ModelConfig(BaseModel):
    yolo: str
    sam: str
    depth: str


class ParamConfig(BaseModel):
    yolo_conf: float


class Config(BaseModel):
    name: str
    models: ModelConfig
    params: ParamConfig

    @computed_field
    @property
    def raw_csv_path(self) -> Path:
        return DATA_ROOT / self.name / "raw.csv"

    @computed_field
    @property
    def raw_dir(self) -> Path:
        return DATA_ROOT / self.name / "raw"

    @computed_field
    @property
    def processed_dir(self) -> Path:
        return DATA_ROOT / self.name / "processed"

    @model_validator(mode="after")
    def validate(self):
        if not (DATA_ROOT / self.name).exists():
            raise ValueError(f"Dataset `{self.name}` does not exist")
        if not self.raw_dir.exists():
            raise ValueError(f"Directory `raw` does not exist")
        if not self.raw_csv_path.exists():
            raise ValueError(f"File `raw.csv` does not exist")
        return self


def get_valid_configs() -> list[str]:
    configs = []
    for path in CONFIG_ROOT.iterdir():
        if path.suffix == ".yaml":
            with open(path, "r", encoding="utf-8") as f:
                try:
                    Config(**yaml.safe_load(f))
                    configs.append(path.stem)
                except Exception:
                    continue

    return sorted(configs)


if __name__ == "__main__":
    print(get_valid_configs())

# print(load_config("data-inside"))
