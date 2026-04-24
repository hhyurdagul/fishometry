import json
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

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

        results = self.model.predict(image_path, conf=0.8, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None, None, None

        prediction = results[0]
        if len(torch.unique(prediction.boxes.cls)) != len(prediction.boxes.cls):
            return None, None, None

        image_height, image_width = prediction.orig_shape

        return prediction.boxes, image_height, image_width


class YoloStep:
    def __init__(self, config: Config, rotated: bool = False):
        self.config = config
        self.rotated = rotated

        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if rotated
            else config.dataset.input_dir
        )
        self.output_dir = (
            config.dataset.output_dir
            / "cache"
            / f"yolo_{'rotated' if self.rotated else 'initial'}"
        )
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.yolo_model = YoloModel(config.model_path.yolo)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images).drop_nulls()

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
        if output_path.exists():
            with open(output_path, "r") as f:
                return json.load(f)

        boxes, image_w, image_h = self.yolo_model.predict(image_path)
        if boxes is None:
            return {}

        classes = self.config.params.yolo_classes
        default_item = classes[-1]
        name_map = dict(zip(range(len(classes[:-1])), classes[:-1]))

        data = {"name": name, "Image_w": image_w, "Image_h": image_h}
        for box in boxes:
            label = name_map.get(box.cls.item(), default_item)
            data.update(self._get_xxyywh(label, box))

        with open(output_path, "w") as f:
            json.dump(data, f)

        return data

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].drop_nulls().to_list()

        data = []
        for name in tqdm(names, desc="YOLO Object Detection"):
            image_path = self.input_dir / name
            output_path = self.output_dir / (name + ".json")
            if not image_path.exists():
                continue

            features = self._get_yolo_data(name, image_path, output_path)
            data.append(features)

        return df.join(pl.DataFrame(data), on="name", how="left") if data else df
