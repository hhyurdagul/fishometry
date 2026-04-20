import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from src.config import Config
from src.preprocessing.steps.utils import FISH_COORDINATE_FEATURES, get_center_coord

# V2 repo path
v2_path = "third_party/Depth-Anything-V2"
if v2_path not in sys.path:
    sys.path.append(v2_path)

from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore


class DepthModel:
    def __init__(self, model_path: Path, download: bool = True):
        if not model_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"DepthAnythingV2 model not found at path: {model_path}"
                )
            print(
                f"Warning: DepthAnythingV2 weights not found at {self.model_path}, downloading..."
            )

            torch.hub.download_url_to_file(
                "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
                str(model_path),
            )

        self.model: DepthAnythingV2
        self.model_initialized = False
        self.model_path = model_path

    def _get_depth_model(self) -> DepthAnythingV2:
        model = DepthAnythingV2(
            {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            }
        )

        model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model

    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        if not self.model_loaded:
            # To kill the overhead of loading the model if all the images are cached
            self.model = self._get_depth_model()
            self.model_loaded = True

        if self.model is None:
            raise ValueError("Could not load DepthAnythingV2 model")

        h, w = image.shape[:2]
        depth = self.model.infer_image(image)

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth


class DepthStep:
    def __init__(self, config: Config, rotated: bool = False):
        self.config = config
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if rotated
            else config.dataset.input_dir
        )
        self.output_dir = config.dataset.output_dir / "depth"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.depth_model = DepthModel(config.model_path.depth)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images).drop_nulls()

    def _get_depth_map(self, image_path: Path, output_path: Path) -> np.ndarray:
        if output_path.exists():
            depth = np.load(output_path)
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            depth = self.depth_model.get_depth_map(image)
            np.save(output_path, depth)
        return depth

    def _get_robust_depth(
        self, depth_map: np.ndarray, x: int, y: int, patch_size: int = 5
    ) -> float:
        """
        Extracts the median depth around a specific coordinate to avoid single-pixel noise.
        """
        h, w = depth_map.shape
        half_patch = patch_size // 2

        # Define bounding box for the patch, ensuring it stays within image boundaries
        y_min = max(0, y - half_patch)
        y_max = min(h, y + half_patch + 1)
        x_min = max(0, x - half_patch)
        x_max = min(w, x + half_patch + 1)

        # Extract the patch from the depth map
        patch = depth_map[y_min:y_max, x_min:x_max]

        # Return the median value of the patch
        return np.median(patch)

    def _extract_metrics(self, data: dict, depth: np.ndarray) -> dict:
        head_cx, head_cy = get_center_coord(data, "Head")
        body_cx, body_cy = get_center_coord(data, "Body")
        tail_cx, tail_cy = get_center_coord(data, "Tail")

        head_depth = self._get_robust_depth(depth, head_cx, head_cy, 9)
        body_depth = self._get_robust_depth(depth, body_cx, body_cy, 9)
        tail_depth = self._get_robust_depth(depth, tail_cx, tail_cy, 9)
        depth_gradient = head_depth = tail_depth

        return {
            "name": data["name"],
            "head_depth": head_depth,
            "body_depth": body_depth,
            "tail_depth": tail_depth,
            "depth_gradient_raw": depth_gradient,
            "depth_gradient_abs": abs(depth_gradient),
        }

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(FISH_COORDINATE_FEATURES).rows(named=True)  # type: ignore

        data = []
        for row in tqdm(rows, desc="Depth Estimation"):
            if not all(row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")

            name = row["name"]
            image_path = self.input_dir / name
            output_path = self.output_dir / (name + ".npy")

            if not image_path.exists():
                continue

            try:
                depth = self._get_depth_map(image_path, output_path)
                features = self._extract_metrics(row, depth)
                data.append(features)

            except Exception as e:
                print(f"Error processing depth for {name}: {e}")
                continue

        return df.join(pl.DataFrame(data), on="name", how="left") if data else df
