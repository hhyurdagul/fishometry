import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from src.config import get_config
from src.create_data.run import _validate_source_split, run_pipeline


class CreateDataAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_cwd = Path.cwd()
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        os.chdir(self.root)

        Path("configs").mkdir()
        self.source_dir = Path("data/example")
        self.source_raw_dir = self.source_dir / "raw"
        self.source_raw_dir.mkdir(parents=True)
        pl.DataFrame({"name": [], "length": []}).write_csv(self.source_dir / "raw.csv")

        self._write_config("example")
        self.target_config_path = self._write_config("example-zoom", indent=4)
        self.config = get_config("example")

    def tearDown(self) -> None:
        os.chdir(self._original_cwd)
        self._temporary_directory.cleanup()

    def _write_config(self, name: str, indent: int | None = None) -> Path:
        config = {
            "dataset": {
                "name": name,
                "rotate": True,
                "fish_type_available": False,
                "feature_sets": ["eye", "coords"],
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
                "yolo_classes": ["Head", "Tail", "Eye", "Fish"],
            },
        }
        path = Path("configs") / f"{name}.json"
        path.write_text(json.dumps(config, indent=indent), encoding="utf-8")
        return path

    def _write_image(self, name: str, offset: int = 0) -> None:
        image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        image = (image + offset).astype(np.uint8)
        self.assertTrue(cv2.imwrite(str(self.source_raw_dir / name), image))

    def _write_split(self) -> None:
        pl.DataFrame(
            {
                "name": ["one.png", "two.png"],
                "length": [10.5, 12.0],
                "is_train": [True, False],
                "is_val": [False, False],
                "is_test": [False, True],
            }
        ).write_csv(self.source_dir / "split.csv")

    def test_missing_source_split_fails_before_writing(self) -> None:
        config_before = self.target_config_path.read_bytes()

        with self.assertRaisesRegex(FileNotFoundError, "existing source split"):
            run_pipeline(self.config, augment=True)

        self.assertFalse(Path("data/example-zoom").exists())
        self.assertEqual(config_before, self.target_config_path.read_bytes())

    def test_missing_target_config_fails_before_writing(self) -> None:
        self._write_split()
        self._write_image("one.png")
        self._write_image("two.png", offset=7)
        self.target_config_path.unlink()

        with self.assertRaisesRegex(ValueError, "Config `example-zoom` does not exist"):
            run_pipeline(self.config, augment=True)

        self.assertFalse(Path("data/example-zoom").exists())

    def test_non_augmentation_mode_only_replaces_the_source_split(self) -> None:
        raw_path = self.source_dir / "raw.csv"
        pl.DataFrame(
            {"name": ["one.png", "two.png"], "length": [10.5, 12.0]}
        ).write_csv(raw_path)
        raw_before = raw_path.read_bytes()

        run_pipeline(self.config, augment=False)

        split_df = pl.read_csv(self.source_dir / "split.csv")
        self.assertEqual(2, split_df.height)
        self.assertEqual(
            ["name", "length", "is_train", "is_val", "is_test"],
            split_df.columns,
        )
        self.assertTrue(
            split_df.select(
                pl.sum_horizontal(
                    pl.col("is_train"), pl.col("is_val"), pl.col("is_test")
                )
                .eq(1)
                .all()
            ).item()
        )
        self.assertEqual(raw_before, raw_path.read_bytes())
        self.assertFalse(Path("data/example-zoom").exists())

    def test_source_split_target_config_and_unrelated_files_are_preserved(self) -> None:
        self._write_split()
        self._write_image("one.png")
        self._write_image("two.png", offset=7)
        source_split_before = (self.source_dir / "split.csv").read_bytes()
        config_before = self.target_config_path.read_bytes()
        legacy_path = Path("data/example-zoom/raw/legacy.png")
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"legacy")

        run_pipeline(self.config, augment=True)

        self.assertEqual(
            source_split_before, (self.source_dir / "split.csv").read_bytes()
        )
        self.assertEqual(config_before, self.target_config_path.read_bytes())
        self.assertEqual(b"legacy", legacy_path.read_bytes())

    def test_each_readable_image_produces_a_metadata_preserving_triplet(self) -> None:
        self._write_split()
        self._write_image("one.png")
        self._write_image("two.png", offset=7)

        run_pipeline(self.config, augment=True)

        target_dir = Path("data/example-zoom")
        raw_df = pl.read_csv(target_dir / "raw.csv")
        split_df = pl.read_csv(target_dir / "split.csv")
        self.assertTrue(raw_df.equals(split_df))
        self.assertEqual(6, split_df.height)

        source_rows = pl.read_csv(self.source_dir / "split.csv").to_dicts()
        target_rows = split_df.to_dicts()
        for source_row in source_rows:
            source_name = Path(source_row["name"])
            prefix = f"{source_name.stem}-"
            triplet = [
                row
                for row in target_rows
                if row["name"] == source_row["name"] or row["name"].startswith(prefix)
            ]
            self.assertEqual(3, len(triplet))
            self.assertEqual(
                {source_row["name"], "zin", "zout"},
                {
                    source_row["name"]
                    if row["name"] == source_row["name"]
                    else row["name"].split("-")[-2]
                    for row in triplet
                },
            )
            for row in triplet:
                for column in ("length", "is_train", "is_val", "is_test"):
                    self.assertEqual(source_row[column], row[column])

            original = cv2.imread(str(target_dir / "raw" / source_row["name"]))
            zoom_in_name = next(
                row["name"] for row in triplet if "-zin-" in row["name"]
            )
            zoom_out_name = next(
                row["name"] for row in triplet if "-zout-" in row["name"]
            )
            zoom_in = cv2.imread(str(target_dir / "raw" / zoom_in_name))
            zoom_out = cv2.imread(str(target_dir / "raw" / zoom_out_name))
            self.assertEqual((12, 16), original.shape[:2])
            self.assertEqual(original.shape[:2], zoom_in.shape[:2])
            self.assertLess(zoom_out.shape[0], original.shape[0])
            self.assertLess(zoom_out.shape[1], original.shape[1])

    def test_rerun_is_deterministic(self) -> None:
        self._write_split()
        self._write_image("one.png")
        self._write_image("two.png", offset=7)

        run_pipeline(self.config, augment=True)
        target_dir = Path("data/example-zoom")
        first_manifests = {
            name: (target_dir / name).read_bytes() for name in ("raw.csv", "split.csv")
        }
        first_images = {
            path.name: path.read_bytes() for path in (target_dir / "raw").iterdir()
        }

        run_pipeline(self.config, augment=True)

        self.assertEqual(
            first_manifests,
            {
                name: (target_dir / name).read_bytes()
                for name in ("raw.csv", "split.csv")
            },
        )
        self.assertEqual(
            first_images,
            {path.name: path.read_bytes() for path in (target_dir / "raw").iterdir()},
        )

    def test_split_validation_rejects_duplicates_and_multiple_assignments(self) -> None:
        duplicate_names = pl.DataFrame(
            {
                "name": ["one.png", "one.png"],
                "length": [10.5, 10.5],
                "is_train": [True, True],
                "is_val": [False, False],
                "is_test": [False, False],
            }
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _validate_source_split(duplicate_names, self.config)

        multiple_assignments = duplicate_names.with_columns(
            pl.Series("name", ["one.png", "two.png"]),
            pl.lit(True).alias("is_val"),
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_source_split(multiple_assignments, self.config)

    def test_unsafe_source_name_fails_before_writing(self) -> None:
        pl.DataFrame(
            {
                "name": ["../escape.png"],
                "length": [10.5],
                "is_train": [True],
                "is_val": [False],
                "is_test": [False],
            }
        ).write_csv(self.source_dir / "split.csv")

        with self.assertRaisesRegex(ValueError, "normalized relative path"):
            run_pipeline(self.config, augment=True)

        self.assertFalse(Path("data/example-zoom").exists())

    def test_generated_name_collision_fails_before_writing(self) -> None:
        pl.DataFrame(
            {
                "name": ["fish.png", "fish-zin-39.png"],
                "length": [10.5, 12.0],
                "is_train": [True, False],
                "is_val": [False, False],
                "is_test": [False, True],
            }
        ).write_csv(self.source_dir / "split.csv")

        with self.assertRaisesRegex(ValueError, "duplicate image name"):
            run_pipeline(self.config, augment=True)

        self.assertFalse(Path("data/example-zoom").exists())


if __name__ == "__main__":
    unittest.main()
