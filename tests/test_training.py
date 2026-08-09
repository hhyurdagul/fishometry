import json
import os
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from src.artifacts import content_manifest
from src.training.artifacts import checkpoint_stem
import src.training.models.embedding as embedding
import src.training.run as training_run
from src.training.models.cnn import FishModel, build_image_dataset
from src.training.models.regression import (
    MLPRegressor,
    build_xgboost_model,
    run_model_pipeline,
)
from src.training.run import (
    MIN_PER_TYPE_TRAIN_ROWS,
    run_per_fish_task,
    seed_everything,
    validate_training_frame,
)


def _multi_scale_features(n: int, seed: int = 0) -> np.ndarray:
    """Features whose columns span very different magnitudes (like coords vs depth)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 4)) * np.array([1.0, 1.0, 300.0, 300.0])).astype(np.float32)


class SeedingTests(unittest.TestCase):
    def test_seed_everything_is_deterministic(self) -> None:
        seed_everything(123)
        first = (torch.rand(5).tolist(), np.random.rand(5).tolist())
        seed_everything(123)
        second = (torch.rand(5).tolist(), np.random.rand(5).tolist())
        self.assertEqual(first, second)

    def test_seed_sets_cudnn_deterministic(self) -> None:
        seed_everything(7)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)


class MLPScalerTests(unittest.TestCase):
    def test_fit_scales_and_predicts(self) -> None:
        x = _multi_scale_features(40, seed=1)
        y = (np.random.default_rng(2).random(40) * 80).astype(np.float32)
        x_val = _multi_scale_features(10, seed=3)
        y_val = (np.random.default_rng(4).random(10) * 80).astype(np.float32)

        model = MLPRegressor(input_dim=4, epochs=2, batch_size=8)
        model.fit(x, y, x_val, y_val)

        # Scaler is fit on training data only.
        self.assertTrue(hasattr(model.scaler, "mean_"))
        np.testing.assert_allclose(model.scaler.mean_, x.mean(axis=0), rtol=1e-4)
        preds = model.predict(x_val)
        self.assertEqual(preds.shape, (10,))

    def test_predict_is_deterministic_under_seed(self) -> None:
        x = _multi_scale_features(30, seed=5)
        y = (np.random.default_rng(6).random(30) * 50).astype(np.float32)

        def train_once() -> np.ndarray:
            seed_everything(99)
            m = MLPRegressor(input_dim=4, epochs=3, batch_size=8)
            m.fit(x, y, x, y)
            return m.predict(x)

        np.testing.assert_array_equal(train_once(), train_once())

    def test_save_persists_scaler_stats(self) -> None:
        x = _multi_scale_features(20, seed=7)
        y = (np.random.default_rng(8).random(20) * 40).astype(np.float32)
        model = MLPRegressor(input_dim=4, epochs=1, batch_size=8)
        model.fit(x, y, x, y)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mlp.pth")
            model.save(path)
            checkpoint = torch.load(path, weights_only=False)
        self.assertIn("scaler_mean", checkpoint)
        self.assertIn("scaler_scale", checkpoint)
        np.testing.assert_allclose(checkpoint["scaler_mean"], model.scaler.mean_)


class XGBoostConfigTests(unittest.TestCase):
    def test_default_max_depth_is_tuned_down(self) -> None:
        regressor = build_xgboost_model().named_steps["regressor"]
        self.assertEqual(regressor.max_depth, 4)


class CNNAuxScalerTests(unittest.TestCase):
    def _write_images(self, image_dir: Path, names: list[str]) -> None:
        image_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            cv2.imwrite(str(image_dir / name), np.zeros((8, 8, 3), dtype=np.uint8))

    def test_aux_features_are_transformed_by_scaler(self) -> None:
        names = ["a.png", "b.png", "c.png"]
        raw_aux = np.array([[0.0], [100.0], [200.0]], dtype=np.float32)
        df = pl.DataFrame(
            {"name": names, "length": [10.0, 20.0, 30.0], "aux": raw_aux.ravel()}
        )
        scaler = StandardScaler().fit(raw_aux)

        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "blackout"
            self._write_images(image_dir, names)
            dataset = build_image_dataset(
                df, image_dir, [pl.col("aux")], transform=None, aux_scaler=scaler
            )

            stored_aux = np.array([sample[1] for sample in dataset.samples])
        np.testing.assert_allclose(stored_aux, scaler.transform(raw_aux), rtol=1e-5)


class PerFishGuardTests(unittest.TestCase):
    def test_small_fish_types_are_skipped(self) -> None:
        # "Big" has >= 10 train rows, "Tiny" has fewer.
        big = MIN_PER_TYPE_TRAIN_ROWS + 2
        tiny = MIN_PER_TYPE_TRAIN_ROWS - 2
        names = [f"b{i}" for i in range(big)] + [f"t{i}" for i in range(tiny)]
        df = pl.DataFrame(
            {
                "name": names,
                "fish_type": ["Big"] * big + ["Tiny"] * tiny,
                "is_train": [True] * (big + tiny),
            }
        )

        seen: list[str] = []

        def fake_task(data, config, feature_set, depth, per_type):
            seen.append(data["fish_type"][0])
            return data.select("name").with_columns(pl.lit(1.0).alias("pred"))

        pred_df = df.select("name")
        run_per_fish_task(
            fake_task,
            df,
            config=None,
            feature_set="coords",
            depth=False,
            pred_df=pred_df,
        )

        self.assertIn("Big", seen)
        self.assertNotIn("Tiny", seen)

    def test_per_type_checkpoint_names_include_species(self) -> None:
        salmon = pl.DataFrame({"fish_type": ["Atlantic Salmon"]})
        perch = pl.DataFrame({"fish_type": ["Perch"]})
        self.assertEqual(
            checkpoint_stem("linear_coords_per_type", salmon, True),
            "linear_coords_per_type__atlantic-salmon",
        )
        self.assertEqual(
            checkpoint_stem("linear_coords_per_type", perch, True),
            "linear_coords_per_type__perch",
        )


class TrainingFrameValidationTests(unittest.TestCase):
    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["train", "val", "test"],
                "length": [10.0, 11.0, 12.0],
                "is_train": [True, False, False],
                "is_val": [False, True, False],
                "is_test": [False, False, True],
                "relative_w": [0.5, 0.6, 0.7],
                "relative_h": [0.2, 0.3, 0.4],
                "relative_area": [0.1, 0.2, 0.3],
                "fish_aspect": [2.5, 2.0, 1.75],
                "fish_area": [5.0, 6.0, 7.0],
                "unused_nullable": [None, "kept", None],
            }
        )

    def _config(self):
        return types.SimpleNamespace(
            dataset=types.SimpleNamespace(
                fish_type_available=False,
                feature_sets=["coords"],
                depth=[False],
            )
        )

    def test_unselected_nullable_column_does_not_remove_rows(self) -> None:
        validate_training_frame(self._frame(), self._config())

    def test_selected_nullable_column_fails_before_training(self) -> None:
        frame = self._frame().with_columns(
            pl.when(pl.col("name") == "val")
            .then(None)
            .otherwise(pl.col("relative_w"))
            .alias("relative_w")
        )
        with self.assertRaisesRegex(ValueError, "relative_w"):
            validate_training_frame(frame, self._config())


class CNNWeightContractTests(unittest.TestCase):
    def test_pretrained_weight_failure_is_not_silently_randomized(self) -> None:
        with mock.patch(
            "src.training.models.cnn.models.resnet18",
            side_effect=OSError("offline"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "ImageNet ResNet-18 weights are required"
            ):
                FishModel()


class EmbeddingCacheTests(unittest.TestCase):
    def _fake_config(self, output_dir: Path):
        return types.SimpleNamespace(
            dataset=types.SimpleNamespace(output_dir=output_dir)
        )

    def test_cache_reused_when_names_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rotated").mkdir()
            model_dir = root / "checkpoints"
            model_dir.mkdir()

            names = ["x.png", "y.png"]
            for name in names:
                (root / "rotated" / name).write_bytes(name.encode())
            cached = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(model_dir / "efficientnet_b3_rotated_embeddings.npy", cached)
            manifest = {
                "schema": 1,
                "backbone": "efficientnet_b3",
                "weights": str(embedding.models.EfficientNet_B3_Weights.DEFAULT),
                "images": content_manifest(root / "rotated", names),
            }
            (model_dir / "efficientnet_b3_rotated_embeddings_names.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            df = pl.DataFrame({"name": names})
            # Guard: building would raise, proving the cache short-circuits.
            original = embedding._build_efficientnet_b3
            embedding._build_efficientnet_b3 = lambda: (_ for _ in ()).throw(
                AssertionError("should not rebuild")
            )
            try:
                result = embedding._load_or_create_embeddings(
                    df, self._fake_config(root), model_dir
                )
            finally:
                embedding._build_efficientnet_b3 = original

            np.testing.assert_array_equal(result, cached)

    def test_cache_invalidated_when_names_differ(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rotated").mkdir()
            model_dir = root / "checkpoints"
            model_dir.mkdir()

            np.save(
                model_dir / "efficientnet_b3_rotated_embeddings.npy",
                np.zeros((2, 2), dtype=np.float32),
            )
            (model_dir / "efficientnet_b3_rotated_embeddings_names.json").write_text(
                '["old1.png", "old2.png"]', encoding="utf-8"
            )

            df = pl.DataFrame({"name": ["new1.png", "new2.png"]})
            for name in df["name"].to_list():
                (root / "rotated" / name).write_bytes(name.encode())
            original = embedding._build_efficientnet_b3
            embedding._build_efficientnet_b3 = lambda: (_ for _ in ()).throw(
                RuntimeError("rebuild attempted")
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "rebuild attempted"):
                    embedding._load_or_create_embeddings(
                        df, self._fake_config(root), model_dir
                    )
            finally:
                embedding._build_efficientnet_b3 = original

    def test_cache_invalidated_when_image_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "rotated"
            image_dir.mkdir()
            model_dir = root / "checkpoints"
            model_dir.mkdir()
            names = ["x.png"]
            image = image_dir / names[0]
            image.write_bytes(b"old")
            np.save(
                model_dir / "efficientnet_b3_rotated_embeddings.npy",
                np.zeros((1, 2), dtype=np.float32),
            )
            manifest = {
                "schema": 1,
                "backbone": "efficientnet_b3",
                "weights": str(embedding.models.EfficientNet_B3_Weights.DEFAULT),
                "images": content_manifest(image_dir, names),
            }
            (model_dir / "efficientnet_b3_rotated_embeddings_names.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            image.write_bytes(b"new-content")
            original = embedding._build_efficientnet_b3
            embedding._build_efficientnet_b3 = lambda: (_ for _ in ()).throw(
                RuntimeError("rebuild attempted")
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "rebuild attempted"):
                    embedding._load_or_create_embeddings(
                        pl.DataFrame({"name": names}),
                        self._fake_config(root),
                        model_dir,
                    )
            finally:
                embedding._build_efficientnet_b3 = original


class PredictionAlignmentTests(unittest.TestCase):
    def test_tabular_predictions_align_with_names(self) -> None:
        seed_everything(11)
        rng = np.random.default_rng(11)
        n = 24
        df = pl.DataFrame(
            {
                "name": [f"img{i}" for i in range(n)],
                "length": (rng.random(n) * 50).astype(np.float32),
                "relative_w": rng.random(n),
                "relative_h": rng.random(n),
                "relative_area": rng.random(n),
                "fish_aspect": rng.random(n),
                "fish_area": rng.random(n),
                "is_train": [i % 4 != 0 for i in range(n)],
                "is_val": [i % 4 == 0 for i in range(n)],
            }
        )
        config = types.SimpleNamespace(dataset=types.SimpleNamespace(name="unit-test"))

        with tempfile.TemporaryDirectory() as tmp:
            import os

            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                pred = run_model_pipeline("linear", df, config, "coords", depth=False)
            finally:
                os.chdir(cwd)

        self.assertEqual(pred["name"].to_list(), df["name"].to_list())
        self.assertEqual(pred.height, n)


class AtomicTrainingPublicationTests(unittest.TestCase):
    def _config(self, root: Path):
        dataset_dir = root / "data" / "unit"
        dataset_dir.mkdir(parents=True)
        processed_path = dataset_dir / "processed.csv"
        pl.DataFrame(
            {
                "name": ["train", "val", "test"],
                "length": [10.0, 11.0, 12.0],
                "is_train": [True, False, False],
                "is_val": [False, True, False],
                "is_test": [False, False, True],
                "relative_w": [0.5, 0.6, 0.7],
                "relative_h": [0.2, 0.3, 0.4],
                "relative_area": [0.1, 0.2, 0.3],
                "fish_aspect": [2.5, 2.0, 1.75],
                "fish_area": [5.0, 6.0, 7.0],
            }
        ).write_csv(processed_path)
        dataset = types.SimpleNamespace(
            name="unit",
            dataset_dir=dataset_dir,
            output_csv_path=processed_path,
            fish_type_available=False,
            feature_sets=["coords"],
            depth=[False],
        )
        config = types.SimpleNamespace(dataset=dataset)
        config.model_dump = lambda mode: {"dataset": {"name": "unit"}}
        return config

    @staticmethod
    def _task(label: str):
        def run(
            df,
            config,
            feature_set,
            depth,
            per_type,
            checkpoint_dir,
        ):
            (checkpoint_dir / f"{label}.joblib").write_text(label, encoding="utf-8")
            return df.select("name").with_columns(pl.lit(10.0).alias(label))

        return run

    def test_success_publishes_manifest_predictions_and_reports_together(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(training_run, "get_config", return_value=config),
                    mock.patch.object(
                        training_run,
                        "train_linear_model",
                        self._task("linear"),
                    ),
                    mock.patch.object(
                        training_run,
                        "train_xgboost_model",
                        self._task("xgboost"),
                    ),
                    mock.patch.object(
                        training_run, "train_mlp_model", self._task("mlp")
                    ),
                    mock.patch.object(
                        training_run, "train_cnn_model", self._task("cnn")
                    ),
                ):
                    training_run.main(dataset_name="unit")
            finally:
                os.chdir(original_cwd)

            current = json.loads(
                (root / "checkpoints/unit/current.json").read_text(encoding="utf-8")
            )
            run_id = current["run_id"]
            self.assertTrue(
                (root / f"checkpoints/unit/runs/{run_id}/manifest.json").is_file()
            )
            self.assertTrue((root / "data/unit/predictions.csv").is_file())
            for split in ("train", "val", "test"):
                self.assertTrue((root / f"reports/unit/{split}.csv").is_file())

    def test_failed_run_does_not_replace_current_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._config(root)
            canonical = root / "data/unit/predictions.csv"
            canonical.write_bytes(b"previous-run")

            def fail(*args, **kwargs):
                raise RuntimeError("training failed")

            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(training_run, "get_config", return_value=config),
                    mock.patch.object(training_run, "train_linear_model", fail),
                ):
                    with self.assertRaisesRegex(RuntimeError, "training failed"):
                        training_run.main(dataset_name="unit")
            finally:
                os.chdir(original_cwd)

            self.assertEqual(canonical.read_bytes(), b"previous-run")
            self.assertFalse((root / "checkpoints/unit/current.json").exists())


if __name__ == "__main__":
    unittest.main()
