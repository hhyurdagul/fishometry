"""Behavioral tests for safe visualization artifact loading."""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from src.visualization.data_loading import get_datasets, load_dataset_metadata
from src.visualization.image_processing import get_image_paths, process_images


class VisualizationLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_cwd = Path.cwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        os.chdir(self.root)

    def tearDown(self) -> None:
        os.chdir(self.original_cwd)
        self.temporary_directory.cleanup()

    def test_dataset_listing_is_sorted_and_requires_current_artifacts(self) -> None:
        (Path("data") / "z-empty").mkdir(parents=True)
        for name in ("z-valid", "a-valid"):
            directory = Path("data") / name
            directory.mkdir(parents=True)
            pl.DataFrame({"name": ["fish.png"]}).write_csv(directory / "processed.csv")

        self.assertEqual(get_datasets(), ["a-valid", "z-valid"])

    def test_nested_image_name_resolves_without_flattening(self) -> None:
        dataset = Path("data/example")
        image_path = dataset / "raw/folder/fish.png"
        image_path.parent.mkdir(parents=True)
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((4, 5, 3), dtype=np.uint8))
        )

        raw_path, _, _, _ = get_image_paths("example", "folder/fish.png")
        self.assertEqual(Path(raw_path), image_path.resolve())

    def test_path_traversal_is_rejected(self) -> None:
        (Path("data/example")).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Invalid image name"):
            get_image_paths("example", "../secret.png")
        with self.assertRaisesRegex(ValueError, "Invalid dataset"):
            load_dataset_metadata("../example")

    def test_corrupt_images_do_not_crash_the_view(self) -> None:
        raw = Path("data/example/raw/fish.png")
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"not an image")

        image_raw, image_rotated, image_depth, image_blackout = process_images(
            "example", "fish.png", None
        )
        self.assertEqual(image_raw.shape, (200, 200, 3))
        self.assertIsNone(image_rotated)
        self.assertIsNone(image_depth)
        self.assertIsNone(image_blackout)


if __name__ == "__main__":
    unittest.main()
