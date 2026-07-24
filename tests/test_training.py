import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler

import src.training.models.embedding as embedding
from src.training.models.cnn import build_image_dataset
from src.training.models.regression import (
    MLPRegressor,
    build_xgboost_model,
    run_model_pipeline,
)
from src.training.run import (
    MIN_PER_TYPE_TRAIN_ROWS,
    run_per_fish_task,
    seed_everything,
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
        np.testing.assert_allclose(
            stored_aux, scaler.transform(raw_aux), rtol=1e-5
        )


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
        run_per_fish_task(fake_task, df, config=None, feature_set="coords",
                          depth=False, pred_df=pred_df)

        self.assertIn("Big", seen)
        self.assertNotIn("Tiny", seen)


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
            cached = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            np.save(model_dir / "efficientnet_b3_rotated_embeddings.npy", cached)
            (model_dir / "efficientnet_b3_rotated_embeddings_names.json").write_text(
                '["x.png", "y.png"]', encoding="utf-8"
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
        config = types.SimpleNamespace(
            dataset=types.SimpleNamespace(name="unit-test")
        )

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


if __name__ == "__main__":
    unittest.main()
