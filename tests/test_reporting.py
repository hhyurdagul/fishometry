import unittest

import numpy as np
import polars as pl

from src.reporting.metrics import compute_metrics, model_columns
from src.reporting.report import (
    _severity,
    build_document,
    build_fragment,
    build_report_data,
)


class MetricsTests(unittest.TestCase):
    def test_compute_metrics_known_values(self) -> None:
        gt = np.array([10.0, 20.0])
        pred = np.array([12.0, 18.0])
        m = compute_metrics(gt, pred)
        self.assertAlmostEqual(m["mae"], 2.0)
        self.assertAlmostEqual(m["rmse"], 2.0)
        self.assertAlmostEqual(m["mape"], 15.0)   # mean(20%, 10%)
        self.assertAlmostEqual(m["r2"], 0.84)     # 1 - 8/50
        self.assertEqual(m["n"], 2)

    def test_compute_metrics_empty_is_none(self) -> None:
        self.assertIsNone(compute_metrics(np.array([]), np.array([])))

    def test_model_columns_excludes_identifiers(self) -> None:
        cols = ["name", "length", "is_train", "is_val", "is_test", "fish_type",
                "linear_coords", "cnn_coords"]
        self.assertEqual(model_columns(cols), ["linear_coords", "cnn_coords"])


class SeverityTests(unittest.TestCase):
    def test_thresholds(self) -> None:
        self.assertEqual(_severity(2.0), "good")
        self.assertEqual(_severity(10.0), "warn")
        self.assertEqual(_severity(30.0), "crit")
        self.assertEqual(_severity(float("nan")), "warn")


class ReportDataTests(unittest.TestCase):
    def _df(self) -> pl.DataFrame:
        # good_model tracks length closely; bad_model is off by a lot.
        length = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        return pl.DataFrame(
            {
                "name": [f"f{i}" for i in range(6)],
                "length": length,
                "is_train": [True, True, True, True, False, False],
                "is_val": [False] * 6,
                "is_test": [False, False, False, False, True, True],
                "good_model": [x + 1 for x in length],
                "bad_model": [x + 25 for x in length],
            }
        )

    def test_ranking_and_headline(self) -> None:
        data = build_report_data(self._df())
        self.assertEqual(data["headline_split"], "test")
        self.assertEqual(data["n_models"], 2)
        self.assertEqual(data["best"]["model"], "good_model")
        # Entries are sorted best-first.
        self.assertEqual(data["entries"][0]["model"], "good_model")

    def test_worst_predictions_present(self) -> None:
        data = build_report_data(self._df())
        bad = next(e for e in data["entries"] if e["model"] == "bad_model")
        self.assertTrue(bad["worst"])
        self.assertGreater(bad["worst"][0]["abs_error"], 0)

    def test_falls_back_to_all_split_without_test(self) -> None:
        df = self._df().drop("is_test")
        data = build_report_data(df)
        self.assertEqual(data["headline_split"], "all")


class DocumentTests(unittest.TestCase):
    def _df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["a", "b", "c", "d"],
                "length": [10.0, 20.0, 30.0, 40.0],
                "is_train": [True, True, False, False],
                "is_val": [False, False, False, False],
                "is_test": [False, False, True, True],
                "m1": [11.0, 21.0, 31.0, 41.0],
            }
        )

    def test_document_is_self_contained(self) -> None:
        doc = build_document(self._df(), "unit")
        self.assertIn("<!doctype html>", doc)
        self.assertIn("Leaderboard", doc)
        self.assertNotIn("http://", doc)
        self.assertNotIn("https://", doc)

    def test_fragment_has_no_document_wrapper(self) -> None:
        frag = build_fragment(self._df(), "unit")
        self.assertTrue(frag.lstrip().startswith("<style>"))
        self.assertNotIn("<!doctype", frag)
        self.assertNotIn("<body", frag)


if __name__ == "__main__":
    unittest.main()
