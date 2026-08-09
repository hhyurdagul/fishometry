import json
from pathlib import Path

import polars as pl
import torch
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


class YoloModel:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Yolo model not found at path: {model_path}")

        self.model: YOLO
        self.model_initialized = False
        self.model_path = model_path

    def _get_yolo_model(self) -> YOLO:
        return YOLO(self.model_path)

    def predict(
        self, image_path: Path
    ) -> tuple[Boxes, int, int] | tuple[None, None, None]:
        if not self.model_initialized:
            self.model = self._get_yolo_model()
            self.model_initialized = True
        results = self.model.predict(str(image_path), conf=0.8, verbose=False)
        if not results:
            return None, None, None

        prediction = results[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return None, None, None

        if len(torch.unique(boxes.cls)) != len(boxes.cls):
            return None, None, None

        image_height, image_width = prediction.orig_shape
        return boxes, image_height, image_width


class YoloStep:
    CONFIDENCE = 0.8

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
        self.yolo_model = YoloModel(config.model_path.yolo)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._process_images(df)

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

    def _get_yolo_data(self, name: str, image_path: Path, output_path: Path) -> dict:
        signature = build_signature(
            inputs={
                "image": image_path,
                "checkpoint": self.config.model_path.yolo,
            },
            parameters={
                "step": "yolo",
                "version": 1,
                "variant": self.cache_variant,
                "confidence": self.CONFIDENCE,
                "classes": self.config.params.yolo_classes,
            },
        )
        if cache_matches(output_path, signature):
            with output_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        boxes, image_h, image_w = self.yolo_model.predict(image_path)
        if boxes is None:
            return {}

        classes = self.config.params.yolo_classes
        default_item = classes[-1]
        name_map = dict(zip(range(len(classes[:-1])), classes[:-1]))

        data = {"name": name, "Image_w": image_w, "Image_h": image_h}
        for box in boxes:
            label = name_map.get(box.cls.item(), default_item)
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
