import json
import time
from pathlib import Path

import polars as pl
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

from src.artifacts import (
    atomic_write_json,
    build_signature,
    cache_matches,
    write_manifest,
)
from src.config import Config
from src.context_features import VLM_FEATURE_COLUMNS, VLM_SCHEMA
from src.preprocessing.steps.utils import FISH_COORDINATE_FEATURES

MODEL_NAME = "gemma-4-31b-it"
VLM_PROMPT = (
    "Analyze the original image and describe the primary fish and its scene using "
    "every field in the response schema. Use only the allowed category values. "
    "Describe actual visibility, truncation, occlusion, curvature, and view angle; "
    "do not assume the fish is fully visible. Orientation is relative to the original "
    "image frame. Count all visible fish in num_fish and set is_multiple_fish "
    "accordingly. Use none for absent holding methods, measuring devices, or nets."
)


class VLM:
    def __init__(self) -> None:
        self.model_config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=VLM_SCHEMA
        )
        with open(".env.json", "r") as f:
            api_key = json.load(f)["GEMINI_API_KEY"]

        self.client = genai.Client(api_key=api_key)

    def extract_fish_dataset_metadata(self, image: Image.Image) -> dict:
        time.sleep(5)
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                VLM_PROMPT,
                image,
            ],
            config=self.model_config,
        )

        return json.loads(response.text)  # type: ignore


class VLMStep:
    def __init__(self, config: Config):
        self.config = config
        self.input_dir = config.dataset.input_dir
        self.output_dir = config.dataset.output_dir / "cache" / "vlm"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vlm = VLM()

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        result = self._process_images(df)
        return result if result.is_empty() else result.drop_nulls(VLM_FEATURE_COLUMNS)

    def _get_features(
        self, name: str, image_path: Path, output_path: Path
    ) -> dict | None:
        signature = build_signature(
            inputs={"image": image_path},
            parameters={
                "step": "vlm",
                "version": 2,
                "model": MODEL_NAME,
                "feature_columns": VLM_FEATURE_COLUMNS,
                "schema": VLM_SCHEMA,
                "prompt": VLM_PROMPT,
            },
        )
        if cache_matches(output_path, signature):
            with output_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        with Image.open(image_path) as source:
            features = self.vlm.extract_fish_dataset_metadata(source)

        features["name"] = name
        atomic_write_json(output_path, features)
        write_manifest(output_path, signature)
        return features

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(FISH_COORDINATE_FEATURES).rows(named=True)

        data = []
        for row in tqdm(rows, desc="Vision Language Model"):
            if any(value is None for value in row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")
                continue

            name = row["name"]
            image_path = self.input_dir / name
            output_path = self.output_dir / (name + ".json")

            if not image_path.exists():
                continue

            try:
                features = self._get_features(name, image_path, output_path)
                if features is None:
                    continue
                data.append(features)

            except Exception as e:
                print(
                    f"Error feature generation with vision language model {name}: {e}"
                )
                continue

        if not data:
            return df.clear()
        result = df.drop(VLM_FEATURE_COLUMNS, strict=False).join(
            pl.DataFrame(data), on="name", how="left"
        )
        return result
