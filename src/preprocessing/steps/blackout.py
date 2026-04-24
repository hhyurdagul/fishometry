import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import Config


class BlackoutStep:
    def __init__(self, config: Config, rotated: bool = False, canvas_size=(224, 224)):
        self.config = config
        self.canvas_size = canvas_size

        self.image_dir = (
            config.dataset.output_dir / "rotated"
            if rotated
            else config.dataset.input_dir
        )
        self.mask_dir = config.dataset.output_dir / "segment"
        self.output_dir = config.dataset.output_dir / "blackout"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images)

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].drop_nulls().to_list()

        for name in tqdm(names, desc="Blackout & Center"):
            image_path = self.image_dir / name
            mask_path = self.mask_dir / (name + ".npy")
            output_path = self.output_dir / name

            if not image_path.exists() or not mask_path.exists() or output_path.exists():
                continue

            try:
                # Load
                img = cv2.imread(image_path)
                mask = np.load(mask_path)

                if img is None:
                    continue

                # Resize mask to img if needed (SAM output matches input size usually)
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # Apply blackout
                # mask is boolean or 0/1.
                img_masked = img.copy()
                img_masked[mask == 0] = 0

                # Find bbox of mask
                coords = np.column_stack(np.where(mask > 0))
                if coords.size == 0:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # Crop fish
                fish_crop = img_masked[y_min : y_max + 1, x_min : x_max + 1]

                # Resize to fit within canvas while maintaining aspect ratio?
                # Or just put in center?
                # "All images should have to be same size here".
                # Usually means we resize the crop to fit the canvas, or at least scale it.
                # Let's resize longest side to canvas size (with some padding) and pad the rest.

                h, w = fish_crop.shape[:2]
                target_h, target_w = self.canvas_size

                scale = min(target_w / w, target_h / h)
                # Let's upscale or downscale
                new_w = int(w * scale)
                new_h = int(h * scale)

                resized_fish = cv2.resize(
                    fish_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

                # Place on black canvas
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2

                canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized_fish

                # Save
                cv2.imwrite(output_path, canvas)

            except Exception as e:
                print(f"Error processing {name}: {e}")

        return df
