import yaml
import polars as pl
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_image(image_path: str):
    """Load image using cv2 (lazy import)."""
    import cv2

    return cv2.imread(image_path)


def save_image(image, output_path: str):
    """Save image using cv2 (lazy import)."""
    import cv2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)


def load_csv(csv_path: str) -> pl.DataFrame:
    return pl.read_csv(csv_path).drop_nulls()


def save_csv(df: pl.DataFrame, csv_path: str):
    df.write_csv(csv_path)


def get_raw_name(name: str) -> str:
    return name[::-1].split(".", 1)[1][::-1]
