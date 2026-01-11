import os
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from .base import PipelineStep


class SegmentStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.input_dir = os.path.join(config["paths"]["output"], "rotated")
        self.output_dir = os.path.join(config["paths"]["output"], "segment")
        os.makedirs(self.output_dir, exist_ok=True)
        self.model_path = config["models"]["sam"]
        self.device = config["params"]["device"]

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        sam = sam_model_registry["vit_h"](checkpoint=self.model_path)
        sam.to(self.device)
        predictor = SamPredictor(sam)

        names = df["name"].to_list()

        for name in tqdm(names, desc="Segmentation"):
            image_path = os.path.join(self.input_dir, name)

            # Standardize naming: append .npy
            output_name = name + ".npy"
            output_path = os.path.join(self.output_dir, output_name)

            if os.path.exists(output_path):
                continue

            if not os.path.exists(image_path):
                continue

            # Get bbox from DF (rotated)
            row = df.filter(pl.col("name") == name)
            if row.height == 0:
                continue

            # Assuming columns are present.
            # We need the Fish bbox.
            # Fish_x1, Fish_y1, ...

            try:
                data = row.to_dicts()[0]

                # Check for required Head/Tail cols
                required = [
                    "Head_x1",
                    "Head_x2",
                    "Head_y1",
                    "Head_y2",
                    "Tail_x1",
                    "Tail_x2",
                    "Tail_y1",
                    "Tail_y2",
                ]
                if any(r not in data or data[r] is None for r in required):
                    # Fallback to Fish center if head/tail missing?
                    # User requested Head/Tail. If missing, maybe skip or fallback.
                    # Let's skip for now or try Fish center as backup?
                    # Prompt implies strict requirement. Let's skip if missing.
                    print(f"Skipping {name} due to missing Head/Tail coordinates.")
                    continue

                # Calculate centers
                h_cx = (data["Head_x1"] + data["Head_x2"]) / 2
                h_cy = (data["Head_y1"] + data["Head_y2"]) / 2

                t_cx = (data["Tail_x1"] + data["Tail_x2"]) / 2
                t_cy = (data["Tail_y1"] + data["Tail_y2"]) / 2

                points = np.array([[h_cx, h_cy], [t_cx, t_cy]])

                mask = self.get_mask(predictor, image_path, points)

                # Save explicitly
                np.save(output_path, mask)
            except Exception as e:
                print(f"Error segmenting {name}: {e}")

        return df

    def get_mask(self, predictor, image_path, points):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        # points shape: (N, 2)
        # labels: (N,) -> 1 for foreground
        labels = np.ones(len(points))

        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=None,
            multimask_output=False,
        )
        return masks[0]
