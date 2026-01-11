import os
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
from .base import PipelineStep


class BlackoutStep(PipelineStep):
    def __init__(self, config, canvas_size=(224, 224)):
        super().__init__(config)
        self.image_dir = os.path.join(config["paths"]["output"], "rotated")
        self.mask_dir = os.path.join(config["paths"]["output"], "segment")
        self.output_dir = os.path.join(config["paths"]["output"], "blackout")
        os.makedirs(self.output_dir, exist_ok=True)
        self.canvas_size = canvas_size

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].to_list()

        for name in tqdm(names, desc="Blackout & Center"):
            # Paths
            img_path = os.path.join(self.image_dir, name)

            # Standardize naming: append .npy
            mask_name = name + ".npy"
            mask_path = os.path.join(self.mask_dir, mask_name)

            output_path = os.path.join(self.output_dir, name)

            if os.path.exists(output_path):
                continue

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            try:
                # Load
                img = cv2.imread(img_path)
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
                cv2.imwrite(os.path.join(self.output_dir, name), canvas)

            except Exception as e:
                print(f"Error processing {name}: {e}")

        # We don't necessarily update DF unless we want to link new path?
        # But for regression we just need the image file to exist.
        return df
