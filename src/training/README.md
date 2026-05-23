# Training Pipeline

This package trains length-prediction models from `processed.csv` and writes a wide prediction table for analysis.

## Entrypoint

```bash
uv run python -m src.training.run --dataset-name data-inside
uv run python -m src.training.run --dataset-name data-inside-zoom
uv run python -m src.training.run --dataset-name data-outside
```

## Inputs

- `data/<dataset>/processed.csv`
- `data/<dataset>/processed/blackout/` for CNN models
- dataset-specific feature settings from `configs/<dataset>.json`

## Outputs

The main output is:

```text
data/<dataset>/predictions.csv
```

This file keeps identifying columns such as `name`, `length`, split flags, and optionally `fish_type`. Model predictions are added as extra columns.

## Models and Feature Sets

The orchestrator trains:

- baseline mean-length prediction,
- linear regression,
- XGBoost,
- MLP,
- CNN with blackout images and tabular features.

For outside data, an additional embedding/ridge-style model is included.

The configured `feature_sets` and `depth` values decide which feature combinations are trained. Prediction column names follow:

```text
<model>_<feature-set>[_depth][_per_type]
```

For example, `cnn_coords_depth_per_type` is a CNN model using coordinate features plus depth, trained separately per fish type.

## Per-Fish-Type Training

When `fish_type_available` is `true`, the training pipeline also trains per-fish-type variants. This is mainly used for `data-outside`, where fish species differ and a grouped baseline or per-type model can be more informative than a single global model.
