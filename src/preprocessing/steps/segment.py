from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

from src.artifacts import (
    atomic_save_numpy,
    build_signature,
    cache_matches,
    write_manifest,
)
from src.config import Config
from src.preprocessing.steps.utils import (
    FISH_COORDINATE_FEATURES,
    clear_unused_gpu_memory,
    get_center_coord,
)


class SegmentModel:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"SegmentAnything model not found at path: {model_path}"
            )
        self.model: SamPredictor
        self.model_initialized = False
        self.model_path = model_path

    def _get_segmentation_model(self) -> SamPredictor:
        sam = sam_model_registry["vit_l"](checkpoint=self.model_path)
        return SamPredictor(sam)

    def release(self) -> None:
        if self.model_initialized:
            del self.model
            self.model_initialized = False
            clear_unused_gpu_memory()

    @torch.no_grad()
    def _predict_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        predictor = self.model
        predictor.model.to(device)
        predictor.set_image(image)

        # Match SamPredictor's point-prompt inference, but expand the small
        # decoder logits on CPU. Original-resolution masks can exhaust VRAM.
        coords = predictor.transform.apply_coords(points, predictor.original_size)
        point_coords = torch.as_tensor(coords, dtype=torch.float, device=device)[None]
        point_labels = torch.as_tensor(labels, dtype=torch.int, device=device)[None]
        sparse, dense = predictor.model.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        low_res_masks, _ = predictor.model.mask_decoder(
            image_embeddings=predictor.features,
            image_pe=predictor.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        masks = predictor.model.postprocess_masks(
            low_res_masks.cpu(), predictor.input_size, predictor.original_size
        )
        return (masks[0, 0] > predictor.model.mask_threshold).numpy()

    def get_mask(
        self, image: np.ndarray, points: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        if not self.model_initialized:
            self.model = self._get_segmentation_model()
            self.model_initialized = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            try:
                return self._predict_mask(image, points, labels, device)
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                print("SAM CUDA memory exhausted; retrying this image on CPU.")

            # Leave the exception handler first so its traceback no longer
            # holds failed CUDA intermediates alive during the retry.
            self.model.reset_image()
            self.model.model.to(torch.device("cpu"))
            clear_unused_gpu_memory()
            return self._predict_mask(image, points, labels, torch.device("cpu"))
        finally:
            self.model.reset_image()


class SegmentStep:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if config.dataset.rotate
            else config.dataset.input_dir
        )
        self.output_dir = config.dataset.output_dir / "segment"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segment_model = SegmentModel(config.model_path.sam)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            return self._process_images(df)
        finally:
            self.segment_model.release()

    def _get_segment_mask(
        self, data: dict, image_path: Path, output_path: Path
    ) -> np.ndarray:
        head_cx, head_cy = get_center_coord(data, "Head")
        tail_cx, tail_cy = get_center_coord(data, "Tail")
        signature = build_signature(
            inputs={
                "image": image_path,
                "checkpoint": self.config.model_path.sam,
            },
            parameters={
                "step": "segment",
                "version": 1,
                "model": "vit_l",
                "positive_points": [
                    [head_cx, head_cy],
                    [tail_cx, tail_cy],
                ],
            },
        )
        if cache_matches(output_path, signature):
            return np.load(output_path)

        points = np.array([[head_cx, head_cy], [tail_cx, tail_cy]])
        labels = np.ones(len(points))

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.segment_model.get_mask(image, points, labels)
        atomic_save_numpy(output_path, mask)
        write_manifest(output_path, signature)
        return mask

    def _extract_geometric_features(self, name: str, mask: np.ndarray) -> dict:
        # Mask is binary
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        # 2. Find the contours (the boundary lines) of the mask
        # RETR_EXTERNAL ensures we only get the outer boundary, ignoring any holes inside the fish
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Safety check in case SAM failed to generate a mask
        if not contours:
            return {}

        # Grab the largest contour (to ignore any small noise artifacts SAM might have picked up)
        fish_contour = max(contours, key=cv2.contourArea)  # type: ignore

        # --- Feature Calculations ---

        # Mask Area
        area = cv2.contourArea(fish_contour)

        # Perimeter (True means the contour is closed)
        perimeter = cv2.arcLength(fish_contour, True)

        # Major and Minor Axes
        # cv2.fitEllipse requires at least 5 points to fit an ellipse mathematically
        if len(fish_contour) >= 5:
            _, axes, _ = cv2.fitEllipse(fish_contour)
            major_axis = max(axes)
            minor_axis = min(axes)
        else:
            major_axis, minor_axis = 0, 0

        # Solidity
        # First, find the convex hull (the tightest polygon wrapped around the contour)
        hull = cv2.convexHull(fish_contour)
        hull_area = cv2.contourArea(hull)

        # Solidity is the ratio of the actual area to the hull area
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            "name": name,
            "mask_area": area,
            "mask_perimeter": perimeter,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "solidity": solidity,
        }

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(FISH_COORDINATE_FEATURES).rows(named=True)  # type: dict

        data = []
        for row in tqdm(rows, desc="Segmentation"):
            if any(value is None for value in row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")
                continue

            name = row["name"]
            image_path = self.input_dir / name
            output_path = self.output_dir / (name + ".npy")

            if not image_path.exists():
                continue

            try:
                mask = self._get_segment_mask(row, image_path, output_path)
                features = self._extract_geometric_features(name, mask)
                data.append(features)

            except Exception as e:
                print(f"Error segmenting {name}: {e}")
                continue

        if not data:
            return df.clear()
        feature_columns = [
            "mask_area",
            "mask_perimeter",
            "major_axis",
            "minor_axis",
            "solidity",
        ]
        result = df.drop(feature_columns, strict=False).join(
            pl.DataFrame(data), on="name", how="left"
        )
        return result.drop_nulls(feature_columns)
