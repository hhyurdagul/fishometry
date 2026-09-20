"""Check the context handoff without running image models or remote calls."""

import unittest
from types import SimpleNamespace
from pathlib import Path

import polars as pl

from src.context_features import (
    VLM_BOOLEAN_COLUMNS,
    VLM_CATEGORICAL_VALUES,
    VLM_FEATURE_COLUMNS,
    VLM_SCHEMA,
)
from src.preprocessing.steps.feature import FeatureStep
from src.training.data_loader import get_feature_names_and_desc


class ContextFeatureTests(unittest.TestCase):
    def test_all_context_fields_reach_training_as_numbers(self):
        # Exercise every allowed category, including those absent in smaller splits.
        size = max(map(len, VLM_CATEGORICAL_VALUES.values()))
        data = {
            name: [values[i % len(values)] for i in range(size)]
            for name, values in VLM_CATEGORICAL_VALUES.items()
        }
        data.update(
            {name: [i % 2 == 0 for i in range(size)] for name in VLM_BOOLEAN_COLUMNS}
        )
        data["num_fish"] = list(range(1, size + 1))
        data.update(
            {
                name: [10.0] * size
                for name in (
                    "Fish_w",
                    "Fish_h",
                    "Image_w",
                    "Image_h",
                    "length",
                    "mask_area",
                    "mask_perimeter",
                    "major_axis",
                    "minor_axis",
                    "solidity",
                )
            }
        )
        config = SimpleNamespace(
            dataset=SimpleNamespace(
                rotate=False,
                input_dir=Path("."),
                fish_type_available=False,
            )
        )
        encoded = FeatureStep(config).process(pl.DataFrame(data))
        selectors, _ = get_feature_names_and_desc(feature_set="features")
        selected = encoded.select(selectors)
        self.assertTrue(all(dtype.is_numeric() for dtype in selected.dtypes))
        self.assertNotIn("length", selected.columns)
        for name in VLM_BOOLEAN_COLUMNS + ["num_fish"]:
            self.assertIn(name, selected.columns)
        for name, values in VLM_CATEGORICAL_VALUES.items():
            for value in values:
                self.assertIn(f"{name}_{value}", selected.columns)
        self.assertEqual(selected["num_fish"].to_list(), data["num_fish"])

    def test_schema_requires_context_without_predicting_targets(self):
        self.assertEqual(len(VLM_FEATURE_COLUMNS), 27)
        self.assertEqual(set(VLM_SCHEMA["required"]), set(VLM_SCHEMA["properties"]))
        self.assertTrue({"name", "length", "fish_type"}.isdisjoint(VLM_FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
