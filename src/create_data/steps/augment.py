import os
import polars as pl
import numpy as np
import random
import cv2
from pathlib import Path
from src.config import Config
from src.create_data.steps.base import PipelineStep

np.random.seed(42)

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
        nh, nw = int(h * ratio), int(w * ratio)

        # Top left corner
        y = (h - nh) // 2
        x = (w - nw) // 2

        cropped = image[y : y + nh, x : x + nw]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    elif zoom_type == "out":
        # Zoom out: Simply resize image to (h*ratio, w*ratio)
        # Ratio 0.8 means result is 80% of original size (0.2 magnitude zoom)
        nh, nw = int(h * ratio), int(w * ratio)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized

    return image


class AugmentStep(PipelineStep):
    def __init__(self, config: Config):
        super().__init__(config)

    def __create_data(self, df: pl.DataFrame):
        new_data_dir = Path(str(self.config.input_dir).replace(self.config.name, self.config.name + "-zoom"))
        new_data_dir.mkdir(parents=True, exist_ok=True)
        with open(new_data_dir.with_suffix(".csv"), "w") as f:
            f.write("")
        

        new_config = Config(
            name=self.config.name + "-zoom",
            models=self.config.models,
            params=self.config.params,
            fish_type_available=self.config.fish_type_available,
        )

        new_rows = []
        for row in df.to_dicts():
            name = row["name"]
            base_name, ext = os.path.splitext(name)
            # Copy other columns (like length)
            base_row = {k: v for k, v in row.items() if k != "name"}

            src_img_path = self.config.input_dir / name
            if not src_img_path.exists():
                print(f"Image {name} not found, skipping.")
                continue

            img = cv2.imread(src_img_path)
            if img is None:
                print(f"Failed to read {src_img_path}")
                continue

            # 1. Original
            dest_img_path = new_config.input_dir / name
            if not dest_img_path.exists():
                cv2.imwrite(dest_img_path, img)
            new_rows.append({"name": name, **base_row})

            # 2. Zoom In
            # Generate exactly one zoom-in
            # Magnitude 0.2 to 0.5 (Mild to moderate)
            rin = random.uniform(0.2, 0.5)
            img_in = augment_zoom(img, "in", rin)

            # Naming: name-zin-{magnitude}
            suffix = int(rin * 100)
            name_in = f"{base_name}-zin-{suffix}{ext}"

            if not (new_config.input_dir / name_in).exists():
                cv2.imwrite(str(new_config.input_dir / name_in), img_in)
            new_rows.append({"name": name_in, **base_row})

            # 3. Zoom Out
            # Generate exactly one zoom-out
            # Magnitude 0.2 to 0.5
            rout = random.uniform(0.2, 0.5)
            img_out = augment_zoom(img, "out", rout)

            suffix = int(rout * 100)
            name_out = f"{base_name}-zout-{suffix}{ext}"

            if not (new_config.input_dir / name_out).exists():
                cv2.imwrite(str(new_config.input_dir / name_out), img_out)
            new_rows.append({"name": name_out, **base_row})

        with open(f"configs/{new_config.name}.json", "w") as f:
            import json
            model_dump = new_config.model_dump()
            repr_dict = {
                "name": model_dump["name"],
                "rotate": model_dump["rotate"],
                "fish_type_available": model_dump["fish_type_available"],
                "models": model_dump["models"],
                "params": model_dump["params"],
            }
            json.dump(repr_dict, f)

        return pl.DataFrame(new_rows), new_config


    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, Config]:
        df, config = self.__create_data(df)
        df.write_csv(config.input_csv_path)
        return df, config
