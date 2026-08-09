"""Behavioral tests for the current five-column relative-depth contract."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from src.preprocessing.steps.depth import DepthModel, DepthStep


def _coordinates(name: str = "fish.png") -> dict[str, int | str]:
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


def _config(root: Path):
    checkpoint = root / "depth.pth"
    checkpoint.write_bytes(b"test checkpoint identity")
    return types.SimpleNamespace(
        dataset=types.SimpleNamespace(
            output_dir=root / "processed",
            input_dir=root / "raw",
            rotate=False,
        ),
        model_path=types.SimpleNamespace(depth=checkpoint),
    )


class DepthModelTests(unittest.TestCase):
    def test_missing_checkpoint_fails_when_download_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.pth"
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                DepthModel(path, download=False)

    def test_depth_output_is_resized_to_source_shape(self) -> None:
        model = DepthModel.__new__(DepthModel)
        model.model_initialized = True
        model.model = types.SimpleNamespace(
            infer_image=lambda _image: np.ones((3, 4), dtype=np.float32)
        )
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        result = model.get_depth_map(image)
        self.assertEqual(result.shape, (12, 16))


class DepthFeatureTests(unittest.TestCase):
    def test_robust_depth_uses_patch_median_at_image_edge(self) -> None:
        step = DepthStep.__new__(DepthStep)
        depth = np.arange(25, dtype=np.float32).reshape(5, 5)
        expected = float(np.median(depth[:3, :3]))
        self.assertEqual(step._get_robust_depth(depth, 0, 0, 5), expected)

    def test_extract_metrics_emits_canonical_columns(self) -> None:
        step = DepthStep.__new__(DepthStep)
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:, :50] = 5.0
        depth[:, 50:] = 20.0
        result = step._extract_metrics(_coordinates(), depth)
        self.assertEqual(
            set(result),
            {
                "name",
                "head_depth",
                "body_depth",
                "tail_depth",
                "depth_gradient_raw",
                "depth_gradient_abs",
            },
        )
        self.assertEqual(result["head_depth"], 20.0)
        self.assertEqual(result["tail_depth"], 5.0)
        self.assertEqual(result["depth_gradient_raw"], 15.0)
        self.assertEqual(result["depth_gradient_abs"], 15.0)


class DepthStepCacheTests(unittest.TestCase):
    def test_cache_reuses_exact_source_and_invalidates_changed_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            image_path = raw / "fish.png"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
            )
            config = _config(root)
            step = DepthStep(config)
            calls = []

            def infer(image: np.ndarray) -> np.ndarray:
                calls.append(image.copy())
                return np.full(image.shape[:2], len(calls), dtype=np.float32)

            step.depth_model.get_depth_map = infer
            output = config.dataset.output_dir / "depth" / "fish.png.npy"

            first = step._get_depth_map(image_path, output)
            second = step._get_depth_map(image_path, output)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(first, second)

            self.assertTrue(
                cv2.imwrite(
                    str(image_path), np.full((100, 100, 3), 255, dtype=np.uint8)
                )
            )
            third = step._get_depth_map(image_path, output)
            self.assertEqual(len(calls), 2)
            self.assertTrue((third == 2).all())

    def test_process_drops_only_rows_without_depth_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            config = _config(root)
            step = DepthStep(config)
            step.depth_model.get_depth_map = lambda image: np.ones(
                image.shape[:2], dtype=np.float32
            )
            self.assertTrue(
                cv2.imwrite(
                    str(root / "raw" / "present.png"),
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )
            )
            rows = [_coordinates("present.png"), _coordinates("missing.png")]
            frame = pl.DataFrame(rows).with_columns(
                unrelated_nullable=pl.Series([None, "kept"])
            )
            result = step.process(frame)
            self.assertEqual(result["name"].to_list(), ["present.png"])
            self.assertIn("unrelated_nullable", result.columns)


if __name__ == "__main__":
    unittest.main()
