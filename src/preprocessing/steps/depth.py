import os
import sys

import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .base import PipelineStep


class DepthStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.input_dir = os.path.join(config["paths"]["output"], "rotated")
        self.output_dir = os.path.join(config["paths"]["output"], "depth")
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = config["params"]["device"]

        # V2 repo path
        v2_path = config["models"].get("depth_v2_repo")
        if v2_path and v2_path not in sys.path:
            sys.path.append(v2_path)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].to_list()
        missing = [
            name
            for name in names
            if not os.path.exists(os.path.join(self.output_dir, name + ".npy"))
        ]

        model = None
        if missing:
            print(f"Processing Depth Model: v2 (Missing {len(missing)}/{len(names)})")
            model = self._load_model()
            if model is None:
                print(
                    "Skipping generation due to load failure. Will process existing only."
                )
        else:
            print(f"Depth Model v2: All {len(names)} files exist. Skipping model load.")

        results = []

        for name in tqdm(names, desc="Depth V2"):
            image_path = os.path.join(self.input_dir, name)
            output_path = os.path.join(self.output_dir, name + ".npy")

            # Try to load existing depth map
            if os.path.exists(output_path):
                try:
                    depth = np.load(output_path)
                    entry = self._extract_metrics(df, name, depth)
                    results.append(entry)
                    continue
                except Exception as e:
                    print(f"Error reading {output_path}: {e}")
                    if model is None:
                        continue

            if not os.path.exists(image_path):
                continue

            if model is None:
                continue

            # Run inference
            try:
                image = cv2.imread(image_path)
                h, w = image.shape[:2]

                depth = model.infer_image(image)

                if depth.shape != (h, w):
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

                np.save(output_path, depth)

                entry = self._extract_metrics(df, name, depth)
                results.append(entry)

            except Exception as e:
                print(f"Error executing depth inference on {name}: {e}")

        # Cleanup
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge results into dataframe
        if results:
            results_df = pl.DataFrame(results)
            df = df.join(results_df, on="name", how="left")

        return df

    def _load_model(self):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2

            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
            }

            encoder = "vitl"
            model = DepthAnythingV2(**model_configs[encoder])

            checkpoint_path = os.path.join(
                "checkpoints", f"depth_anything_v2_{encoder}.pth"
            )
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            else:
                print(
                    f"Warning: DepthAnythingV2 weights not found at {checkpoint_path}"
                )
                self._download_weights(encoder, checkpoint_path)
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(
                        torch.load(checkpoint_path, map_location="cpu")
                    )

            model.to(self.device)
            model.eval()
            return model

        except Exception as e:
            print(f"Could not load DepthAnythingV2: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _download_weights(self, encoder: str, path: str):
        print(f"Downloading DepthAnythingV2 {encoder} weights to {path}...")
        url_map = {
            "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
            "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
            "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
        }

        if encoder not in url_map:
            print(f"Unknown encoder '{encoder}' for auto-download.")
            return

        try:
            torch.hub.download_url_to_file(url_map[encoder], path)
        except Exception as e:
            print(f"Failed to download weights: {e}")

    def _extract_metrics(self, df: pl.DataFrame, name: str, depth: np.ndarray) -> dict:
        row = df.filter(pl.col("name") == name)
        metrics = {"name": name}

        if row.height == 0:
            return metrics

        data = row.to_dicts()[0]

        def get_center_depth(prefix: str):
            x1_key = f"{prefix}_x1"
            if x1_key not in data:
                return None

            cx = int((data[f"{prefix}_x1"] + data[f"{prefix}_x2"]) / 2)
            cy = int((data[f"{prefix}_y1"] + data[f"{prefix}_y2"]) / 2)

            if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                return float(depth[cy, cx])
            return None

        metrics["fish_center_depth"] = get_center_depth("Fish")
        metrics["head_center_depth"] = get_center_depth("Head")
        metrics["tail_center_depth"] = get_center_depth("Tail")

        return metrics
