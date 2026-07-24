import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from src.training.data_loader import get_feature_names_and_desc
from src.training.models.baseline import train_baseline
from src.training.models.embedding import PerSpeciesRegressor
from src.training.models.regression import run_model_pipeline
from src.training.run import seed_everything


class FeatureSpecTests(unittest.TestCase):
    def test_feature_desc_strings(self) -> None:
        _, desc = get_feature_names_and_desc("linear", "coords", False, False)
        self.assertEqual(desc, "linear_coords")
        _, desc = get_feature_names_and_desc("mlp", "coords", True, True)
        self.assertEqual(desc, "mlp_coords_depth_per_type")
        _, desc = get_feature_names_and_desc("xgboost", "eye", True, False)
        self.assertEqual(desc, "xgboost_eye_depth")

    def test_coords_exprs_select_expected_columns(self) -> None:
        feats, _ = get_feature_names_and_desc("linear", "coords", depth=True)
        df = pl.DataFrame(
            {
                "relative_w": [1.0], "relative_h": [1.0], "relative_area": [1.0],
                "fish_aspect": [1.0], "fish_area": [1.0],
                "head_depth": [1.0], "body_depth": [1.0], "tail_depth": [1.0],
                "depth_gradient_raw": [1.0], "depth_gradient_abs": [1.0],
                "fish_type_Cod": [1], "length": [10.0],
            }
        )
        selected = df.select(feats).columns
        self.assertIn("relative_w", selected)
        self.assertIn("head_depth", selected)
        self.assertIn("fish_type_Cod", selected)
        self.assertNotIn("length", selected)


class BaselineTests(unittest.TestCase):
    def test_global_mean_baseline(self) -> None:
        config = types.SimpleNamespace(
            dataset=types.SimpleNamespace(fish_type_available=False)
        )
        df = pl.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "length": [10.0, 20.0, 100.0, 200.0],
                "is_train": [True, True, False, False],
            }
        )
        pred = train_baseline(df, config)
        self.assertEqual(pred.columns, ["name", "mean_regression"])
        self.assertTrue((pred["mean_regression"] == 15.0).all())

    def test_per_fish_type_baseline(self) -> None:
        config = types.SimpleNamespace(
            dataset=types.SimpleNamespace(fish_type_available=True)
        )
        df = pl.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "fish_type": ["Cod", "Cod", "Perch", "Perch"],
                "length": [10.0, 30.0, 100.0, 300.0],
                "is_train": [True, True, True, True],
            }
        )
        pred = train_baseline(df, config)
        by_name = dict(zip(pred["name"].to_list(), pred["mean_regression"].to_list()))
        self.assertEqual(by_name["a"], 20.0)   # Cod mean
        self.assertEqual(by_name["c"], 200.0)  # Perch mean


class PerSpeciesRegressorTests(unittest.TestCase):
    def test_large_species_gets_own_model_small_falls_back(self) -> None:
        rng = np.random.default_rng(0)
        # Species 0: 10 rows, constant target 10. Species 1: 3 rows, target 999.
        big_feat = rng.random((10, 1))
        small_feat = rng.random((3, 1))
        x = np.vstack(
            [
                np.hstack([big_feat, np.zeros((10, 1))]),
                np.hstack([small_feat, np.ones((3, 1))]),
            ]
        )
        y = np.concatenate([np.full(10, 10.0), np.full(3, 999.0)])

        model = PerSpeciesRegressor(LinearRegression()).fit(x, y)
        # Only species 0 (>= 8 rows) has a dedicated model.
        self.assertIn(0, model.models_)
        self.assertNotIn(1, model.models_)

        preds = model.predict(x)
        # Species 0 rows are predicted close to their own constant target.
        np.testing.assert_allclose(preds[:10], 10.0, atol=1e-6)


class RegressionPipelineTests(unittest.TestCase):
    def _make_df(self, n: int = 24) -> pl.DataFrame:
        rng = np.random.default_rng(3)
        return pl.DataFrame(
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

    def _run(self, model_name: str) -> pl.DataFrame:
        seed_everything(5)
        config = types.SimpleNamespace(
            dataset=types.SimpleNamespace(name="unit-test")
        )
        df = self._make_df()
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                return run_model_pipeline(model_name, df, config, "coords", depth=False)
            finally:
                os.chdir(cwd)

    def test_xgboost_predictions_align(self) -> None:
        pred = self._run("xgboost")
        self.assertEqual(pred.height, 24)
        self.assertIn("xgboost_coords", pred.columns)
        self.assertEqual(pred["name"].null_count(), 0)

    def test_mlp_predictions_align(self) -> None:
        pred = self._run("mlp")
        self.assertEqual(pred.height, 24)
        self.assertIn("mlp_coords", pred.columns)


if __name__ == "__main__":
    unittest.main()
