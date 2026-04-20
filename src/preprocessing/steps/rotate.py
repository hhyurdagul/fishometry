import math
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from src.config import Config
from src.preprocessing.steps.utils import FISH_COORDINATE_FEATURES 


class RotateStep():
    def __init__(self, config: Config):
        self.config = config
        self.input_dir = config.dataset.input_dir
        self.output_dir = config.dataset.output_dir / "rotated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images).drop_nulls()

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(FISH_COORDINATE_FEATURES).rows(named=True)  # type: list[dict]

        for row in tqdm(rows, desc="Segmentation"):
            if not all(row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")

            name = row["name"]
            image_path = self.input_dir / name
            output_path = self.output_dir / name

            if not image_path.exists() or output_path.exists():
                continue

            try:
                rotated_image, _ = self._rotate_and_crop(image_path, row)
                cv2.imwrite(output_path, rotated_image)
            except Exception as e:
                print(f"Error rotating {name}: {e}")

        return df

    def _rotate_and_crop(self, image_path: Path, data: dict) -> tuple[np.ndarray, dict]:
        image = cv2.imread(image_path)
        h, w = image.shape[:2] # type: ignore

        # 1. Calculate Angle (Tail to Head)
        # We want Tail on Left, Head on Right.
        # Vector T->H should point +X (0 degrees).

        hx = (data["Head_x1"] + data["Head_x2"]) / 2
        hy = (data["Head_y1"] + data["Head_y2"]) / 2
        tx = (data["Tail_x1"] + data["Tail_x2"]) / 2
        ty = (data["Tail_y1"] + data["Tail_y2"]) / 2

        dx = hx - tx  # Vector from Tail to Head
        dy = hy - ty
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Rotate by -angle to align T->H to +X
        rotation_angle = angle_deg

        # 2. Rotate Image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_image = cv2.warpAffine(image, M, (new_w, new_h)) # type: ignore

        # Check for upside down (if we rotated > 90 degrees)
        # If we rotated by > 90, the fish was facing Left-ish.
        # Rotating it 180 makes it upside down. We need to flip vertical.
        if abs(rotation_angle) > 90:
            rotated_image = cv2.flip(rotated_image, 0)
            # Update M to reflect the flip
            # y' = new_h - y
            # M[1, :] = -M[1, :]
            # M[1, 2] += new_h (but we need to be careful with the affine transform math)
            # y_new = -(M10*x + M11*y + M12) + new_h
            #       = -M10*x - M11*y + (new_h - M12)
            M[1, :] = M[1, :] * -1
            M[1, 2] += new_h

        # 3. Crop Black Borders (Largest Inscribed Rectangle)
        # We want to crop the black areas introduced by rotation.
        # We use a heuristic or geometric solution.
        # Geometric solution for largest axis-aligned rectangle inside a rotated rectangle.

        crop_x, crop_y, crop_w, crop_h = self._largest_rotated_rect(
            w, h, math.radians(rotation_angle)
        )

        # Adjust crop to be centered in the new image
        # The geometric formula gives dimensions centered at origin.
        # We need to map to new_w, new_h center.

        cx, cy = new_w // 2, new_h // 2
        min_x = int(cx - crop_w / 2)
        max_x = int(cx + crop_w / 2)
        min_y = int(cy - crop_h / 2)
        max_y = int(cy + crop_h / 2)

        # Clip to image bounds
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(new_w, max_x)
        max_y = min(new_h, max_y)

        cropped_image = rotated_image[min_y:max_y, min_x:max_x]

        # We do NOT update coordinates anymore as per user request.
        # YoloStep (rotated) will handle detection on the new image.

        return (
            cropped_image,
            data,
        )  # Return original attr (or empty dict since we don't use it)

    def _largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle',
        computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_h) if w < h else math.atan2(bb_h, bb_w)

        delta = math.pi - alpha - gamma

        length = h if w < h else w
        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return x, y, bb_w - 2 * x, bb_h - 2 * y
