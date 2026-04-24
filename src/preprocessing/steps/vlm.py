import json
import time
from pathlib import Path

import polars as pl
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

from src.config import Config
from src.preprocessing.steps.utils import FISH_COORDINATE_FEATURES


class VLM:
    def __init__(self) -> None:
        schema = {
            "type": "OBJECT",
            "properties": {
                "background_depth": {
                    "type": "STRING",
                    "enum": ["far", "close"],
                    "description": "Far: horizon/scenery. Close: ground/mat/structure.",
                },
                "has_other_objects": {
                    "type": "BOOLEAN",
                    "description": "True if any non-fish/non-net objects like lures, gear, or buckets are present.",
                },
                "is_in_fishnet": {
                    "type": "BOOLEAN",
                    "description": "True if the fish is resting on or inside a net.",
                },
                "fish_placement": {
                    "type": "STRING",
                    "enum": [
                        "held_by_person",
                        "grass_greenery",
                        "beach_sand",
                        "rocks",
                        "board_mat_ruler",
                        "hanging_structure",
                        "water_surface_level",
                    ],
                    "description": "The primary surface or context where the fish is positioned.",
                },
                "fish_orientation": {
                    "type": "STRING",
                    "enum": [
                        "top_to_bottom",
                        "bottom_to_top",
                        "left_to_right",
                        "right_to_left",
                    ],
                    "description": "The direction the fish's head is pointing relative to the image frame.",
                },
                "lighting_condition": {
                    "type": "STRING",
                    "enum": [
                        "bright_daylight",
                        "overcast",
                        "low_light",
                        "artificial_flash",
                    ],
                },
            },
        }

        self.model_config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema
        )
        with open(".env.json", "r") as f:
            api_key = json.load(f)["GEMINI_API_KEY"]

        self.client = genai.Client(api_key=api_key)

    def extract_fish_dataset_metadata(self, image: Image.Image) -> dict:

        time.sleep(5)
        response = self.client.models.generate_content(
            model="gemma-4-31b-it",
            contents=[
                "Analyze the image. Determine the fish's orientation and placement. "
                "The fish is always fully visible. Be precise about the placement category.",
                image,
            ],
            config = self.model_config
        )

        return json.loads(response.text) # type: ignore

class VLMStep:
    def __init__(self, config: Config, rotated: bool = False):
        self.config = config
        self.input_dir = (
            config.dataset.output_dir / "rotated"
            if rotated
            else config.dataset.input_dir
        )
        self.output_dir = config.dataset.output_dir / "cache" / "vlm"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vlm = VLM()

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(self._process_images).drop_nulls()

    def _get_features(self, name: str, image_path: Path, output_path: Path) -> dict | None:
        if output_path.exists():
            with open(output_path, "r") as f:
                return json.load(f)

        image = Image.open(image_path)
        features = self.vlm.extract_fish_dataset_metadata(image)

        features["name"] = name
        with open(output_path, "w") as f:
            json.dump(features, f)

        return features

    def _process_images(self, df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(FISH_COORDINATE_FEATURES).rows(named=True)  # type: ignore

        data = []
        for row in tqdm(rows, desc="Vision Language Model"):
            if not all(row.values()):
                print(f"Skipping {row['name']} due to missing Head/Tail coordinates.")

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

        return df.join(pl.DataFrame(data), on="name", how="left") if data else df
