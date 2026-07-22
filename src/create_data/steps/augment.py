import os
import random
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from src.config import Config
from src.create_data.steps.base import PipelineStep

RANDOM_SEED = 42


def resolve_image_path(root: Path, name: str) -> Path:
    """Resolve a metadata image name while keeping it inside the dataset root."""
    if not isinstance(name, str) or not name:
        raise ValueError("Image names must be non-empty strings")

    relative_path = Path(name)
    if (
        not relative_path.name
        or relative_path.is_absolute()
        or any(part in {".", ".."} for part in relative_path.parts)
    ):
        raise ValueError(f"Image name must be a normalized relative path: {name}")

    resolved_root = root.resolve()
    resolved_path = (root / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(
            f"Image name resolves outside the dataset raw directory: {name}"
        )
    return resolved_path


def augment_zoom(
    image: np.ndarray, zoom_type: str = "in", magnitude: float = 0.1
) -> np.ndarray:
    """
    Apply zoom augmentation to an image.

    Args:
        image: Input image (numpy array)
        zoom_type: "in" for zoom in, "out" for zoom out
        magnitude: Zoom magnitude (0.0-0.9)

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]

    # Validation
    magnitude = max(0.0, min(0.9, magnitude))  # Clamp to safe range
    ratio = 1.0 - magnitude

    if zoom_type == "in":
        # Zoom in: Crop center region of size (h*ratio, w*ratio) and resize to (h, w)
        # Ratio 0.8 means we keep 80% of the image (0.2 magnitude zoom)
        nh, nw = max(1, int(h * ratio)), max(1, int(w * ratio))

        # Top left corner
        y = (h - nh) // 2
        x = (w - nw) // 2

        cropped = image[y : y + nh, x : x + nw]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    elif zoom_type == "out":
        # Zoom out: Simply resize image to (h*ratio, w*ratio)
        # Ratio 0.8 means result is 80% of original size (0.2 magnitude zoom)
        nh, nw = max(1, int(h * ratio)), max(1, int(w * ratio))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized

    return image


class AugmentStep(PipelineStep):
    def __init__(self, config: Config, target_config: Config):
        super().__init__(config)
        self.target_config = target_config

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to write image {path}")

    def _build_plan(
        self, df: pl.DataFrame
    ) -> list[tuple[dict, float, str, float, str]]:
        rng = random.Random(RANDOM_SEED)
        planned_paths: set[Path] = set()
        plan = []

        for row in df.to_dicts():
            name = row["name"]
            base_name, ext = os.path.splitext(name)
            zoom_in = rng.uniform(0.2, 0.5)
            zoom_in_name = f"{base_name}-zin-{int(zoom_in * 100)}{ext}"
            zoom_out = rng.uniform(0.2, 0.5)
            zoom_out_name = f"{base_name}-zout-{int(zoom_out * 100)}{ext}"

            for output_name in (name, zoom_in_name, zoom_out_name):
                output_path = resolve_image_path(
                    self.target_config.dataset.input_dir, output_name
                )
                if output_path in planned_paths:
                    raise ValueError(
                        f"Augmentation would create a duplicate image name: {output_name}"
                    )
                planned_paths.add(output_path)

            plan.append((row, zoom_in, zoom_in_name, zoom_out, zoom_out_name))

        return plan

    def __create_data(self, df: pl.DataFrame) -> tuple[pl.DataFrame, Config]:
        plan = self._build_plan(df)
        self.target_config.dataset.input_dir.mkdir(parents=True, exist_ok=True)
        new_rows = []

        for row, zoom_in, zoom_in_name, zoom_out, zoom_out_name in plan:
            name = row["name"]
            base_row = {k: v for k, v in row.items() if k != "name"}

            src_img_path = resolve_image_path(self.config.dataset.input_dir, name)
            if not src_img_path.exists():
                print(f"Image {name} not found, skipping.")
                continue

            img = cv2.imread(str(src_img_path))
            if img is None:
                print(f"Failed to read {src_img_path}")
                continue

            dest_img_path = resolve_image_path(
                self.target_config.dataset.input_dir, name
            )
            self._write_image(dest_img_path, img)
            new_rows.append({"name": name, **base_row})

            img_in = augment_zoom(img, "in", zoom_in)
            zoom_in_path = resolve_image_path(
                self.target_config.dataset.input_dir, zoom_in_name
            )
            self._write_image(zoom_in_path, img_in)
            new_rows.append({"name": zoom_in_name, **base_row})

            img_out = augment_zoom(img, "out", zoom_out)
            zoom_out_path = resolve_image_path(
                self.target_config.dataset.input_dir, zoom_out_name
            )
            self._write_image(zoom_out_path, img_out)
            new_rows.append({"name": zoom_out_name, **base_row})

        if new_rows:
            augmented_df = pl.DataFrame(new_rows, schema=df.schema).select(df.columns)
        else:
            augmented_df = df.clear()
        return augmented_df, self.target_config

    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, Config]:
        df, config = self.__create_data(df)
        df.write_csv(config.dataset.input_csv_path)
        return df, config
