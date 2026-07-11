# Training Pipeline

The training package turns a dataset's processed feature table and processed images into a wide prediction table. A run executes the complete experiment matrix defined by the dataset configuration; the command does not provide switches for selecting an individual model or resuming an earlier run.

See [models/README.md](models/README.md) for architecture, optimization, validation, image-transform, and checkpoint details.

## Run From the Repository Root

Install the project environment first, then pass an existing configuration name explicitly:

```bash
uv sync

uv run python -m src.training.run --dataset-name data-inside
uv run python -m src.training.run --dataset-name data-inside-zoom
uv run python -m src.training.run --dataset-name data-outside
```

Paths are resolved relative to the repository root. Although the option is represented as nullable by the CLI, omitting `--dataset-name` does not select a useful default.

CUDA is used automatically by the PyTorch models when available. CPU execution is supported but the full CNN matrix, especially the per-fish-type outside run, can take a long time.

## Prerequisites

The selected dataset must already have completed split creation and preprocessing. A training run requires:

- `configs/<dataset-name>.json` with `dataset.feature_sets`, `dataset.depth`, and `dataset.fish_type_available` set correctly;
- `data/<dataset-name>/raw/` and `data/<dataset-name>/raw.csv`, because configuration validation still checks the raw dataset even though training reads processed artifacts;
- `data/<dataset-name>/processed.csv` with unique image names, target lengths, split flags, and every column selected by the configured experiments;
- one blackout image under `data/<dataset-name>/processed/blackout/` for every retained table row, because every configured run currently includes CNN training;
- rotated images under `data/data-outside/processed/rotated/` for the outside-only embedding/Ridge experiments.

The first CNN run attempts to load pretrained ResNet-18 weights. If they are unavailable, the CNN silently falls back to random ResNet-18 initialization. The outside embedding model requires pretrained EfficientNet-B3 weights when its embedding cache does not yet exist; those weights must already be in the local Torch cache or be downloadable.

## Run Sequence

The orchestrator performs these operations in order:

1. Load the selected configuration and `processed.csv`.
2. Drop every row containing a null in any column, including columns not used by a particular experiment.
3. Create the training-only mean baseline.
4. Train global linear, XGBoost, MLP, and CNN models for every configured feature-set/depth pair.
5. When `fish_type_available` is true, repeat those four model families separately for every retained fish type.
6. Only when the CLI value is exactly `data-outside`, train the outside EfficientNet/Ridge variants for every configured feature-set/depth pair.
7. Join every prediction column by `name` and replace `data/<dataset-name>/predictions.csv`.

Model checkpoints are written as each fit completes, but the prediction table is written only after the entire matrix succeeds. A failed run can therefore leave newly replaced checkpoints beside an older prediction table.

## Experiment Matrix

Let `F` be the number of configured feature sets, `D` the number of configured depth values, and `S` the number of retained fish types.

| Experiment | Fits performed | Prediction columns |
| --- | ---: | ---: |
| Mean baseline | 1 calculation | 1 |
| Four global model families | `4 x F x D` | `4 x F x D` |
| Per-type core models, when enabled | `4 x F x D x S` | `4 x F x D` |
| Outside EfficientNet/Ridge | `F x D` | `F x D` |

Per-type fits for different fish types are concatenated into one prediction column for each model/feature/depth combination. They therefore increase the number of fits, not the number of output columns by `S`.

The checked-in configurations expand as follows:

| Dataset | Feature sets | Depth settings | Per-type core pass | Outside embedding pass | Prediction columns |
| --- | --- | --- | --- | --- | ---: |
| `data-inside` | `eye`, `coords` | with and without depth | No | No | 17 |
| `data-inside-zoom` | `eye`, `coords` | with and without depth | No | No | 17 |
| `data-outside` | `coords`, `features` | with and without depth | Yes | Yes | 37 |

The counts include the mean baseline but exclude identifier, target, and split columns. For the outside dataset, the 16 per-type output columns represent `16 x S` separate fits.

## Feature Sets

Every tabular selection first includes all columns whose names start with `fish_type_`. Controlled datasets have no such columns; the outside dataset uses them as one-hot fish-type indicators.

| Feature set | Selected columns |
| --- | --- |
| `eye` | Eye width and height plus detected fish width and height |
| `coords` | Relative fish width, relative fish height, relative area, fish aspect ratio, and fish area |
| `features` | All `coords` values; mask area, perimeter, major axis, minor axis, and solidity; background depth; other-object and fishnet flags; and every one-hot placement, orientation, and lighting-condition column |

When the depth value is `true`, five values are appended to the chosen set: head, body, and tail depth plus raw and absolute depth-gradient values. This flag adds tabular features; it does not select a different image backbone or depth-estimation checkpoint.

The feature-set strings are not validated against the supported names. An unknown value selects no base group, leaving only any `fish_type_` columns, and can consequently fail or train an unintended model. The current configurations use only the sets in the table.

## How Splits Are Used

All models fit target values only from rows where `is_train` is true. Their use of the validation split differs:

| Model | Training split | Validation split | Test split during fitting |
| --- | --- | --- | --- |
| Mean baseline | Computes the mean | Not used | Not used |
| Linear regression | Fits coefficients | Not used | Not used |
| XGBoost | Fits trees | Early-stopping evaluation set | Not used |
| MLP | Fits weights | Selects the best epoch across 100 epochs | Not used |
| CNN | Fits image backbone and head | Selects the best epoch and controls early stopping | Not used |
| EfficientNet/Ridge | Fits Ridge models and chooses regularization by internal cross-validation | Not used | Not used |

For a per-type core pass, the dataframe is first restricted to one fish type and the same split rules are then applied. Small fish-type groups must still contain usable training and validation rows for models that require validation.

The test split remains held out from fitting and model selection. Predictions are nevertheless generated for train, validation, and test rows so downstream analysis can choose the desired split explicitly.

## Outside EfficientNet/Ridge Experiments

The additional outside model combines three inputs:

- a frozen EfficientNet-B3 embedding from each rotated image;
- the selected tabular feature set, optionally including depth;
- fish type, used both through the one-hot table columns and as a routing code.

A median-imputation, standardization, and Ridge cross-validation pipeline is first fitted globally on all training rows. A separate clone is also fitted for each fish type with at least eight retained training examples. Predictions for an eligible type use its specialized model; smaller or unseen groups use the global fallback. This specialization is internal, so these columns do **not** carry the `_per_type` suffix.

The validation and test targets are not used. Embeddings are computed for all retained rows without using their targets, then reused across the four outside feature/depth variants.

## Prediction Table Contract

The output is a wide CSV at:

```text
data/<dataset-name>/predictions.csv
```

It contains one row for every complete row retained after `processed.csv` is globally null-filtered, not necessarily every row originally emitted by preprocessing. The leading columns are:

- `name`, `length`, `is_train`, `is_val`, and `is_test`;
- `fish_type` immediately after `name` when fish types are enabled.

Every model is asked to predict every retained row. Global models produce values for all splits; per-type models produce values for all splits within their type. Image names are the join key and must be unique and consistent across the table and image directories.

Prediction columns use:

```text
<model>_<feature-set>[_depth][_per_type]
```

Examples include `linear_eye`, `xgboost_features_depth`, and `cnn_coords_depth_per_type`. The baseline is always `mean_regression`; on fish-type datasets it is a training-set mean per type rather than one global mean. Outside embedding columns begin with `efficientnet_ridge_`.

Baseline, CNN, and embedding/Ridge predictions are rounded to two decimal places before output. Linear, XGBoost, and MLP predictions retain their returned floating-point precision.

Because the table includes fitted-on rows, evaluation code must filter by `is_test` for a held-out estimate or by `is_val` for validation analysis. Metrics over the unfiltered table mix all three roles.

## Checkpoints and Embedding Cache

Completed fits are saved under `checkpoints/<dataset-name>/`:

- linear, XGBoost, and EfficientNet/Ridge models use `.joblib`;
- MLP and CNN models use `.pth` state dictionaries;
- the outside rotated-image embeddings use `efficientnet_b3_rotated_embeddings.npy`.

Checkpoint names match prediction column names. Existing files with the same name are replaced. The training command never loads a saved model and has no resume or skip-existing behavior; checkpoints are artifacts for later inspection or custom inference, not inputs to the orchestrated run.

All fish-type-specific fits for one experiment write to the same `_per_type` checkpoint path. Each type overwrites the preceding type, so the final file contains only whichever type was processed last even though `predictions.csv` correctly combines predictions from every type. The iteration order is not a stable model registry. Do not treat a `_per_type` checkpoint as the complete set of species models.

The EfficientNet embedding cache is reused solely because the `.npy` file exists. It does not store image names and is not checked against the current row count, order, image content, or preprocessing state. Delete that cache before rerunning after adding, removing, reordering, replacing, or reprocessing outside rows. A stale cache can either cause an array-shape failure or silently pair an embedding with the wrong row.

## Reproducibility and Failure Conditions

- XGBoost sets `random_state=42`. The MLP and CNN do not set Python, NumPy, Torch, data-loader, or deterministic-kernel seeds, so their predictions and best epochs can vary between runs.
- CNN initialization can change from pretrained to random depending on whether pretrained weights are available. The fallback is automatic.
- Existing checkpoints and `predictions.csv` are replaced without confirmation. Preserve experiment artifacts outside their generated paths when a historical run must remain immutable.
- Empty training splits fail every learned model. Empty validation splits fail the MLP and can make XGBoost unusable; the CNN falls back to training loss only when its validation image dataset is empty.
- The CNN image loader skips missing blackout files, but output construction still assumes one prediction per retained table row. In practice every retained row must have a readable blackout image or the run will fail rather than emit a partial CNN column.
- The embedding model requires every retained outside row to have a readable rotated image. It does not skip missing files.
- Tabular core models do not impute or scale their inputs, and the orchestrator's global null removal is their only missing-value handling. The outside Ridge pipeline includes imputation and scaling, though normally no null remains by that point.
- A duplicate `name`, overlapping split flags, or missing required feature columns is not repaired by training and can produce invalid joins, leakage, or an exception.

When a run fails, fix or regenerate the upstream processed artifacts first. Then remove any stale outside embedding cache when row identity or order changed and rerun the complete command.
