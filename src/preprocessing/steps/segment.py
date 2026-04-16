import torch
import os
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from .base import PipelineStep
from src.config import Config


class SegmentStep(PipelineStep):
    def __init__(self, config: Config, rotated: bool=False):
        super().__init__(config)
        self.predictor_loaded = False
        self.input_dir = config.dataset.output_dir / "rotated" if rotated else config.dataset.input_dir
        self.output_dir = config.dataset.output_dir / "segment"
        os.makedirs(self.output_dir, exist_ok=True)


    def _get_segmentation_model(self, model_path: str) -> SamPredictor:
        sam = sam_model_registry["vit_l"](checkpoint=model_path)
        sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return SamPredictor(sam)


    def _get_mask(self, image_path: str, points: np.ndarray) -> np.ndarray:

        if not self.predictor_loaded:
            # To kill the overhead of loading the model if all the images are cached
            self.predictor = self._get_segmentation_model(self.config.models.sam)
            self.predictor_loaded = True

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        # points shape: (N, 2)
        # labels: (N,) -> 1 for foreground
        labels = np.ones(len(points))

        masks, _, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=False,
        )
        return masks[0]


    def _extract_geometric_features(self, mask: np.ndarray) -> dict:
        # Mask is binary
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # 2. Find the contours (the boundary lines) of the mask
        # RETR_EXTERNAL ensures we only get the outer boundary, ignoring any holes inside the fish
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Safety check in case SAM failed to generate a mask
        if not contours:
            return {"area": 0, "perimeter": 0, "major_axis": 0, "minor_axis": 0, "solidity": 0}
            
        # Grab the largest contour (to ignore any small noise artifacts SAM might have picked up)
        fish_contour = max(contours, key=cv2.contourArea) # type: ignore
        
        # --- Feature Calculations ---
        
        # Mask Area
        area = cv2.contourArea(fish_contour)
        
        # Perimeter (True means the contour is closed)
        perimeter = cv2.arcLength(fish_contour, True)
        
        # Major and Minor Axes
        # cv2.fitEllipse requires at least 5 points to fit an ellipse mathematically
        if len(fish_contour) >= 5:
            # returns: (center(x, y), (minor_axis, major_axis), angle_of_rotation)
            _, (minor_axis, major_axis), _ = cv2.fitEllipse(fish_contour)
        else:
            major_axis, minor_axis = 0, 0
            
        # Solidity
        # First, find the convex hull (the tightest polygon wrapped around the contour)
        hull = cv2.convexHull(fish_contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity is the ratio of the actual area to the hull area
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            "mask_area": area,
            "mask_perimeter": perimeter,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "solidity": solidity
        }


    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(
            "name",
            "Head_x1",
            "Head_x2",
            "Head_y1",
            "Head_y2",
            "Tail_x1",
            "Tail_x2",
            "Tail_y1",
            "Tail_y2",
        ).rows(named=True) # type: dict


        data = []
        for row in tqdm(rows, desc="Segmentation"):
            if not all(row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")

            name = row["name"]
            image_path = self.input_dir / name
            output_path = self.output_dir / name + ".npy"

            if not image_path.exists():
                continue

            try:
                if output_path.exists():
                    mask = np.load(output_path)
                else:
                    # Calculate centers
                    h_cx = (row["Head_x1"] + row["Head_x2"]) / 2
                    h_cy = (row["Head_y1"] + row["Head_y2"]) / 2
                    t_cx = (row["Tail_x1"] + row["Tail_x2"]) / 2
                    t_cy = (row["Tail_y1"] + row["Tail_y2"]) / 2

                    points = np.array([[h_cx, h_cy], [t_cx, t_cy]])
                    mask = self._get_mask(image_path, points)
                    np.save(output_path, mask)

                features = self._extract_geometric_features(mask)
                features["name"] = name
                data.append(features)

            except Exception as e:
                print(f"Error segmenting {name}: {e}")
                continue
            

        return df.join(pl.DataFrame(data), on="name", how="left")


    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images)

