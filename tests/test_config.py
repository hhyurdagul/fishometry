import json
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from src.config import (
    Config,
    DatasetConfig,
    ParamConfig,
    get_config,
    get_valid_configs,
)


def _config_payload(name: str) -> dict:
    return {
        "dataset": {
            "name": name,
            "rotate": True,
            "fish_type_available": False,
            "feature_sets": ["coords"],
            "depth": [True, False],
        },
        "model_path": {
            "yolo": "checkpoints/yolo.pt",
            "sam": "checkpoints/sam.pth",
            "depth": "checkpoints/depth.pth",
        },
        "params": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "yolo_classes": ["Head", "Tail", "Fish"],
        },
    }


class ParamConfigTests(unittest.TestCase):
    def test_valid_ratios_accepted(self) -> None:
        params = ParamConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        self.assertEqual(params.train_ratio, 0.7)

    def test_ratio_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "between 0.0 and 1.0"):
            ParamConfig(train_ratio=1.5, val_ratio=0.1, test_ratio=0.1)

    def test_ratio_sum_over_one_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sum to less than"):
            ParamConfig(train_ratio=0.8, val_ratio=0.3, test_ratio=0.3)

    def test_ratio_sum_equal_to_one_allowed(self) -> None:
        params = ParamConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        self.assertAlmostEqual(
            params.train_ratio + params.val_ratio + params.test_ratio, 1.0
        )


class ConfigOnDiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        Path("configs").mkdir()
        Path("checkpoints").mkdir()

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _make_dataset(self, name: str) -> None:
        raw = Path("data") / name / "raw"
        raw.mkdir(parents=True)
        pl.DataFrame({"name": [], "length": []}).write_csv(
            Path("data") / name / "raw.csv"
        )

    def _write_config(self, name: str, payload: dict | None = None) -> None:
        payload = payload if payload is not None else _config_payload(name)
        (Path("configs") / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_dataset_validator_rejects_missing_directory(self) -> None:
        with self.assertRaisesRegex(ValidationError, "does not exist"):
            DatasetConfig(name="ghost")

    def test_get_config_loads_valid_config(self) -> None:
        self._make_dataset("example")
        self._write_config("example")
        config = get_config("example")
        self.assertIsInstance(config, Config)
        self.assertEqual(config.dataset.name, "example")
        self.assertEqual(config.dataset.output_dir, Path("data/example/processed"))

    def test_get_config_missing_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            get_config("nope")

    def test_get_valid_configs_filters_invalid(self) -> None:
        self._make_dataset("good")
        self._write_config("good")
        # Invalid: dataset directory does not exist.
        self._write_config("bad", _config_payload("bad"))
        valid = get_valid_configs()
        self.assertIn("good", valid)
        self.assertNotIn("bad", valid)


if __name__ == "__main__":
    unittest.main()
