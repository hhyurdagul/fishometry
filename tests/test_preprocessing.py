import json
import math
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from src.artifacts import build_signature, write_manifest

from src.preprocessing.steps.blackout import BlackoutStep
from src.preprocessing.steps.depth import DepthStep
from src.preprocessing.steps.feature import FeatureStep
from src.preprocessing.steps.rotate import RotateStep
from src.preprocessing.steps.segment import SegmentStep
from src.preprocessing.steps.utils import get_center_coord
from src.preprocessing.steps.vlm import MODEL_NAME, VLM_FEATURE_COLUMNS, VLMStep
from src.preprocessing.steps.yolo import YoloStep
from src.preprocessing.run import run_pipeline as run_preprocessing_pipeline


def _dataset_config(root: Path, rotate: bool = True, fish_type: bool = False):
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(
            output_dir=root / "processed",
            input_dir=root / "raw",
            rotate=rotate,
            fish_type_available=fish_type,
        )
    )


def _coords_row(name: str) -> dict:
    # Head on the right, tail on the left, fish spanning the middle.
    return {
        "name": name,
        "Head_x1": 70,
        "Head_x2": 90,
        "Head_y1": 45,
        "Head_y2": 55,
        "Fish_x1": 20,
        "Fish_x2": 90,
        "Fish_y1": 40,
        "Fish_y2": 60,
        "Tail_x1": 10,
        "Tail_x2": 30,
        "Tail_y1": 45,
        "Tail_y2": 55,
    }


class UtilsTests(unittest.TestCase):
    def test_get_center_coord(self) -> None:
        data = {"Head_x1": 10, "Head_x2": 20, "Head_y1": 30, "Head_y2": 50}
        self.assertEqual(get_center_coord(data, "Head"), (15, 40))


class RotateTests(unittest.TestCase):
    def test_largest_rotated_rect_at_zero_angle_is_full(self) -> None:
        step = RotateStep.__new__(RotateStep)
        x, y, w, h = step._largest_rotated_rect(100, 100, math.radians(0))
        self.assertAlmostEqual(w, 100, delta=1)
        self.assertAlmostEqual(h, 100, delta=1)

    def test_process_writes_rotated_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir(parents=True)
            config = _dataset_config(root, rotate=True)
            step = RotateStep(config)

            name = "fish.png"
            cv2.imwrite(
                str(root / "raw" / name),
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            )
            df = pl.DataFrame([_coords_row(name)])
            out = step.process(df)

            self.assertTrue((root / "processed" / "rotated" / name).exists())
            self.assertEqual(out["name"].to_list(), [name])

    def test_process_noop_when_rotate_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir(parents=True)
            config = _dataset_config(root, rotate=False)
            step = RotateStep(config)
            df = pl.DataFrame([_coords_row("fish.png")])
            out = step.process(df)
            self.assertEqual(out.height, 1)
            self.assertFalse((root / "processed" / "rotated" / "fish.png").exists())


class BlackoutTests(unittest.TestCase):
    def _setup(self, root: Path, name: str) -> None:
        config = _dataset_config(root, rotate=True)
        (config.dataset.output_dir / "rotated").mkdir(parents=True)
        (config.dataset.output_dir / "segment").mkdir(parents=True)
        cv2.imwrite(
            str(config.dataset.output_dir / "rotated" / name),
            np.full((60, 80, 3), 128, dtype=np.uint8),
        )
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[20:40, 30:50] = 1
        np.save(config.dataset.output_dir / "segment" / (name + ".npy"), mask)

    def test_process_writes_blackout_and_keeps_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "fish.png"
            self._setup(root, name)
            config = _dataset_config(root, rotate=True)
            df = pl.DataFrame({"name": [name], "length": [12.0]})

            out = BlackoutStep(config).process(df)

            self.assertTrue((root / "processed" / "blackout" / name).exists())
            self.assertEqual(out["name"].to_list(), [name])

    def test_rerun_over_cache_still_returns_names(self) -> None:
        # Regression guard: the cache-hit branch must still append to valid_names,
        # otherwise a second run returns an empty frame.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "fish.png"
            self._setup(root, name)
            config = _dataset_config(root, rotate=True)
            df = pl.DataFrame({"name": [name], "length": [12.0]})

            BlackoutStep(config).process(df)  # populate cache
            out = BlackoutStep(config).process(df)  # cache hit

            self.assertEqual(out["name"].to_list(), [name])

    def test_missing_mask_drops_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _dataset_config(root, rotate=True)
            (config.dataset.output_dir / "rotated").mkdir(parents=True)
            (config.dataset.output_dir / "segment").mkdir(parents=True)
            cv2.imwrite(
                str(config.dataset.output_dir / "rotated" / "fish.png"),
                np.full((60, 80, 3), 128, dtype=np.uint8),
            )
            df = pl.DataFrame({"name": ["fish.png"], "length": [12.0]})
            out = BlackoutStep(config).process(df)
            self.assertEqual(out.height, 0)


class FeatureTests(unittest.TestCase):
    def _base_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["a", "b"],
                "Fish_w": [50.0, 40.0],
                "Fish_h": [25.0, 20.0],
                "Image_w": [100.0, 100.0],
                "Image_h": [100.0, 100.0],
            }
        )

    def test_geometric_features(self) -> None:
        config = _dataset_config(Path("."), fish_type=False)
        out = FeatureStep(config).process(self._base_df())
        row = out.filter(pl.col("name") == "a").to_dicts()[0]
        self.assertAlmostEqual(row["relative_w"], 0.5)
        self.assertAlmostEqual(row["relative_h"], 0.25)
        self.assertAlmostEqual(row["relative_area"], 0.125)
        self.assertAlmostEqual(row["fish_aspect"], 2.0)
        self.assertAlmostEqual(row["fish_area"], math.sqrt(50.0 * 25.0))

    def test_vlm_categorical_encoding(self) -> None:
        config = _dataset_config(Path("."), fish_type=False)
        df = self._base_df().with_columns(
            background_depth=pl.Series(["far", "close"]),
            has_other_objects=pl.Series([True, False]),
            is_in_fishnet=pl.Series([False, True]),
        )
        out = FeatureStep(config).process(df)
        self.assertEqual(out["background_depth"].to_list(), [1, 0])
        self.assertEqual(out["has_other_objects"].to_list(), [1, 0])
        self.assertEqual(out["is_in_fishnet"].to_list(), [0, 1])

    def test_fish_type_one_hot(self) -> None:
        config = _dataset_config(Path("."), fish_type=True)
        df = self._base_df().with_columns(fish_type=pl.Series(["Cod", "Perch"]))
        out = FeatureStep(config).process(df)
        self.assertIn("fish_type_Cod", out.columns)
        self.assertIn("fish_type_Perch", out.columns)
        # Original column retained for later per-type grouping.
        self.assertIn("fish_type", out.columns)


class DepthMetricTests(unittest.TestCase):
    def test_robust_depth_is_patch_median(self) -> None:
        step = DepthStep.__new__(DepthStep)
        depth = np.zeros((50, 50), dtype=np.float32)
        depth[24:27, 24:27] = 10.0
        # Center patch (size 9) around (25, 25) is dominated by zeros -> median 0.
        self.assertEqual(step._get_robust_depth(depth, 25, 25, 9), 0.0)

    def test_extract_metrics_gradient(self) -> None:
        step = DepthStep.__new__(DepthStep)
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:, :50] = 5.0  # tail side
        depth[:, 50:] = 20.0  # head side
        metrics = step._extract_metrics(_coords_row("x") | {"name": "x"}, depth)
        self.assertEqual(metrics["head_depth"], 20.0)
        self.assertEqual(metrics["tail_depth"], 5.0)
        self.assertEqual(metrics["depth_gradient_raw"], 15.0)
        self.assertEqual(metrics["depth_gradient_abs"], 15.0)


class SegmentFeatureTests(unittest.TestCase):
    def test_geometric_features_of_filled_rectangle(self) -> None:
        step = SegmentStep.__new__(SegmentStep)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 1  # 40 x 50 rectangle
        feats = step._extract_geometric_features("x", mask)
        self.assertGreater(feats["mask_area"], 0)
        self.assertAlmostEqual(feats["solidity"], 1.0, places=2)

    def test_empty_mask_returns_empty(self) -> None:
        step = SegmentStep.__new__(SegmentStep)
        feats = step._extract_geometric_features("x", np.zeros((10, 10), np.uint8))
        self.assertEqual(feats, {})


class CachedReadTests(unittest.TestCase):
    def test_yolo_returns_matching_cached_json(self) -> None:
        step = YoloStep.__new__(YoloStep)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            checkpoint = root / "yolo.pt"
            image.write_bytes(b"image")
            checkpoint.write_bytes(b"checkpoint")
            step.config = types.SimpleNamespace(
                model_path=types.SimpleNamespace(yolo=checkpoint),
                params=types.SimpleNamespace(yolo_classes=["Fish"]),
            )
            step.cache_variant = "initial"
            out = root / "a.json"
            payload = {"name": "a", "Fish_w": 10}
            out.write_text(json.dumps(payload), encoding="utf-8")
            signature = build_signature(
                inputs={"image": image, "checkpoint": checkpoint},
                parameters={
                    "step": "yolo",
                    "version": 1,
                    "variant": "initial",
                    "confidence": step.CONFIDENCE,
                    "classes": ["Fish"],
                },
            )
            write_manifest(out, signature)
            result = step._get_yolo_data("a", image, out)
            self.assertEqual(result, payload)

    def test_vlm_returns_matching_cached_json(self) -> None:
        step = VLMStep.__new__(VLMStep)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"image")
            out = root / "a.json"
            payload = {"name": "a", "background_depth": "far"}
            out.write_text(json.dumps(payload), encoding="utf-8")
            signature = build_signature(
                inputs={"image": image},
                parameters={
                    "step": "vlm",
                    "version": 1,
                    "model": MODEL_NAME,
                    "feature_columns": VLM_FEATURE_COLUMNS,
                },
            )
            write_manifest(out, signature)
            result = step._get_features("a", image, out)
            self.assertEqual(result, payload)


class PreprocessingReportTests(unittest.TestCase):
    def test_pipeline_reports_exact_stage_attrition(self) -> None:
        class DropOne:
            def process(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.filter(pl.col("name") != "drop.png")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / "split.csv"
            output_dir = root / "processed"
            output_csv = root / "processed.csv"
            pl.DataFrame(
                {
                    "name": ["keep.png", "drop.png", "test.png"],
                    "length": [10.0, 11.0, 12.0],
                    "is_train": [True, False, False],
                    "is_val": [False, True, False],
                    "is_test": [False, False, True],
                    "unrelated_nullable": [None, "value", None],
                }
            ).write_csv(split_path)
            config = types.SimpleNamespace(
                dataset=types.SimpleNamespace(
                    name="unit",
                    split_csv_path=split_path,
                    output_dir=output_dir,
                    output_csv_path=output_csv,
                    fish_type_available=False,
                )
            )
            with mock.patch(
                "src.preprocessing.run._pipeline_steps",
                return_value=[DropOne()],
            ):
                run_preprocessing_pipeline(config)

            result = pl.read_csv(output_csv)
            report = json.loads(
                (output_dir / "preprocessing_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["name"].to_list(), ["keep.png", "test.png"])
            self.assertIn("unrelated_nullable", result.columns)
            self.assertEqual(report["stages"][0]["dropped_names"], ["drop.png"])
            self.assertEqual(report["stages"][0]["dropped_count"], 1)


if __name__ == "__main__":
    unittest.main()
