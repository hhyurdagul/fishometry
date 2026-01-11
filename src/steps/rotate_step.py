import cv2
import numpy as np
import polars as pl
import os
import math
from tqdm import tqdm
from src.steps.base_step import PipelineStep
from src.utils.io import load_image, save_image

class RotateStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.input_dir = config["paths"]["raw"]
        self.output_dir = os.path.join(config["paths"]["output"], "rotated")
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].to_list()
        
        for name in tqdm(names, desc="Rotating Images"):
            image_path = os.path.join(self.input_dir, name)
            output_path = os.path.join(self.output_dir, name)
            
            if os.path.exists(output_path):
                continue

            if not os.path.exists(image_path):
                continue
                
            # Get coordinates for this image
            row = df.filter(pl.col("name") == name)
            if row.height == 0:
                continue
                
            image_attr = row.to_dicts()[0]
            
            required_keys = ["Head_x1", "Head_x2", "Head_y1", "Head_y2", "Tail_x1", "Tail_x2", "Tail_y1", "Tail_y2"]
            if any(k not in image_attr or image_attr[k] is None for k in required_keys):
                continue

            try:
                rotated_image, _ = self.rotate_and_crop(image_path, image_attr)
                save_image(rotated_image, os.path.join(self.output_dir, name))
            except Exception as e:
                print(f"Error rotating {name}: {e}")
                pass
        
        return df

    def rotate_and_crop(self, image_path: str, attr: dict) -> tuple[np.ndarray, dict]:
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # 1. Calculate Angle (Tail to Head)
        # We want Tail on Left, Head on Right.
        # Vector T->H should point +X (0 degrees).
        
        hx = (attr["Head_x1"] + attr["Head_x2"]) / 2
        hy = (attr["Head_y1"] + attr["Head_y2"]) / 2
        tx = (attr["Tail_x1"] + attr["Tail_x2"]) / 2
        ty = (attr["Tail_y1"] + attr["Tail_y2"]) / 2
        
        dx = hx - tx # Vector from Tail to Head
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
        
        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
        
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
            M[1, :] *= -1
            M[1, 2] += new_h

        # 3. Crop Black Borders (Largest Inscribed Rectangle)
        # We want to crop the black areas introduced by rotation.
        # We use a heuristic or geometric solution.
        # Geometric solution for largest axis-aligned rectangle inside a rotated rectangle.
        
        crop_x, crop_y, crop_w, crop_h = self.largest_rotated_rect(w, h, math.radians(rotation_angle))
        
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
        
        return cropped_image, attr # Return original attr (or empty dict since we don't use it)

    def largest_rotated_rect(self, w, h, angle):
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
