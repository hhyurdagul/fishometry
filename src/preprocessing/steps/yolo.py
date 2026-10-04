import json
from pathlib import Path

import polars as pl
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from src.artifacts import (
    atomic_write_json,
    build_signature,
    cache_matches,
    write_manifest,
)
from src.config import Config
from src.preprocessing.steps.utils import clear_unused_gpu_memory

# A landmark box counts as belonging to the selected fish when at least this
# share of its area lies inside the whole-fish box ("best" selection only).
LANDMARK_INSIDE_FISH = 0.5


class YoloModel:
    def __init__(self, model_path: Path, confidence: float, imgsz: int | None = None):
        if not model_path.exists():
            raise FileNotFoundError(f"Yolo model not found at path: {model_path}")

        self.model: YOLO
        self.model_initialized = False
        self.model_path = model_path
        self.confidence = confidence
        self.imgsz = imgsz

    def _get_yolo_model(self) -> YOLO:
        return YOLO(self.model_path)

    def release(self) -> None:
        if self.model_initialized:
            del self.model
            self.model_initialized = False
            clear_unused_gpu_memory()

    def predict(
        self, image_path: Path
    ) -> tuple[Boxes, int, int] | tuple[None, None, None]:
        """Return every box at or above the confidence cutoff (no filtering)."""
        if not self.model_initialized:
            self.model = self._get_yolo_model()
            self.model_initialized = True
        kwargs = {"conf": self.confidence, "verbose": False}
        if getattr(self, "imgsz", None) is not None:
            kwargs["imgsz"] = self.imgsz
        results = self.model.predict(str(image_path), **kwargs)
        if not results:
            return None, None, None

        prediction = results[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return None, None, None

        image_height, image_width = prediction.orig_shape
        return boxes, image_height, image_width


def _xyxy(box) -> list[float]:
    return [float(v) for v in box.xyxy[0].tolist()]


def _share_inside(inner: list[float], outer: list[float]) -> float:
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return ix * iy / area if area > 0 else 0.0


def select_boxes(boxes, classes: list[str], selection: str) -> dict | None:
    """Map detector boxes to exactly one box per configured semantic class.

    Class IDs are positional: IDs below len(classes) - 1 take that label, every
    other ID takes the final label (the whole fish). Returns None when the image
    cannot provide one unambiguous box per class.

    - "unique": any semantic class detected more than once rejects the image.
    - "best":   the highest-confidence box per class is kept; the image is
                rejected unless every landmark box lies mostly inside the fish box.
    """
    if boxes is None or len(boxes) == 0:
        return None
    name_map = dict(zip(range(len(classes) - 1), classes[:-1]))
    grouped: dict[str, list] = {label: [] for label in classes}
    for box in boxes:
        grouped[name_map.get(int(box.cls.item()), classes[-1])].append(box)
    if any(not grouped[label] for label in classes):
        return None
    if selection == "unique":
        if any(len(grouped[label]) != 1 for label in classes):
            return None
        return {label: grouped[label][0] for label in classes}
    if selection != "best":
        raise ValueError(f"Unknown yolo_selection: {selection}")
    chosen = {
        label: max(grouped[label], key=lambda b: float(b.conf.item()))
        for label in classes
    }
    fish = _xyxy(chosen[classes[-1]])
    if any(
        _share_inside(_xyxy(chosen[label]), fish) < LANDMARK_INSIDE_FISH
        for label in classes[:-1]
    ):
        return None
    return chosen


class YoloStep:
    def __init__(self, config: Config, initial: bool = False):
        self.config = config
        rotate = config.dataset.rotate
        if initial:
            rotate = False

        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if rotate
            else config.dataset.input_dir
        )
        self.output_dir = (
            config.dataset.output_dir
            / "cache"
            / f"yolo_{'rotated' if rotate else 'initial'}"
        )
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_variant = "rotated" if rotate else "initial"
        self.yolo_model = YoloModel(
            config.model_path.yolo,
            config.params.yolo_confidence,
            getattr(config.params, "yolo_imgsz", None),
        )

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            return self._process_images(df)
        finally:
            self.yolo_model.release()

    def _get_xxyywh(self, label: str, box: Boxes):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        _, _, w, h = box.xywh[0].int().tolist()
        return {
            f"{label}_x1": x1,
            f"{label}_x2": x2,
            f"{label}_y1": y1,
            f"{label}_y2": y2,
            f"{label}_w": w,
            f"{label}_h": h,
        }

    def _signature(self, image_path: Path):
        params = self.config.params
        imgsz = getattr(params, "yolo_imgsz", None)
        selection = getattr(params, "yolo_selection", "unique")
        parameters = {
            "step": "yolo",
            "version": 1,
            "variant": self.cache_variant,
            "confidence": params.yolo_confidence,
            "classes": params.yolo_classes,
        }
        # Non-default settings get their own cache version so that existing
        # caches for the default behaviour stay valid.
        if imgsz is not None or selection != "unique":
            parameters.update({"version": 3, "imgsz": imgsz, "selection": selection})
        return build_signature(
            inputs={"image": image_path, "checkpoint": self.config.model_path.yolo},
            parameters=parameters,
        )

    def _get_yolo_data(self, name: str, image_path: Path, output_path: Path) -> dict:
        signature = self._signature(image_path)
        if cache_matches(output_path, signature):
            with output_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        boxes, image_h, image_w = self.yolo_model.predict(image_path)
        chosen = select_boxes(
            boxes,
            self.config.params.yolo_classes,
            getattr(self.config.params, "yolo_selection", "unique"),
        )
        if chosen is None:
            return {}

        data = {"name": name, "Image_w": image_w, "Image_h": image_h}
        for label, box in chosen.items():
            data.update(self._get_xxyywh(label, box))

        atomic_write_json(output_path, data)
        write_manifest(output_path, signature)
        return data

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].drop_nulls().to_list()

        data = []
        for name in tqdm(names, desc="YOLO Object Detection"):
            image_path = self.input_dir / name
            output_path = self.output_dir / f"{name}.json"
            if not image_path.is_file():
                continue

            features = self._get_yolo_data(name, image_path, output_path)
            if features:
                data.append(features)

        cols_to_drop = ["Image_w", "Image_h"]
        for label in self.config.params.yolo_classes:
            for suffix in ["_x1", "_x2", "_y1", "_y2", "_w", "_h"]:
                cols_to_drop.append(f"{label}{suffix}")

        if not data:
            return df.clear()

        result = df.drop(cols_to_drop, strict=False).join(
            pl.DataFrame(data), on="name", how="left"
        )
        return result.drop_nulls(cols_to_drop)
