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

The CNN requires torchvision's pretrained ResNet-18 weights and fails explicitly if they cannot be loaded; it never substitutes random initialization under the same experiment name. EfficientNet-B3 and pinned DINOv2 backbones likewise require cached or downloadable pretrained weights when a matching embedding cache does not exist.

## Run Sequence

The orchestrator performs these operations in order:

1. Load the selected configuration and `processed.csv`.
2. Validate unique names, boolean exclusive split membership, non-empty partitions, and non-null identifiers plus configured feature columns. Unselected nullable columns remain in the table.
3. Create an immutable run directory and the training-only mean baseline.
4. Train global linear, XGBoost, MLP, and CNN models for every configured feature-set/depth pair.
5. When `fish_type_available` is true, repeat those four model families separately for every fish type with at least ten training rows.
6. Only when the CLI value is exactly `data-outside`, train both EfficientNet/Ridge and DINOv2/Ridge variants for every configured feature-set/depth pair.
7. Write versioned predictions, per-split metrics, checkpoint metadata, and content hashes into the run manifest.
8. Atomically replace canonical `predictions.csv` and report CSVs, then publish `checkpoints/<dataset-name>/current.json`.

A failed run can leave an unreferenced incomplete run directory, but it does not overwrite the canonical prediction table, reports, or current-run pointer.

## Experiment Matrix

Let `F` be the number of configured feature sets, `D` the number of configured depth values, and `S` the number of retained fish types.

| Experiment | Fits performed | Prediction columns |
| --- | ---: | ---: |
| Mean baseline | 1 calculation | 1 |
| Four global model families | `4 x F x D` | `4 x F x D` |
| Per-type core models, when enabled | `4 x F x D x S` | `4 x F x D` |
| Outside EfficientNet/Ridge and DINOv2/Ridge | `2 x F x D` | `2 x F x D` |

Per-type fits for different fish types are concatenated into one prediction column for each model/feature/depth combination. They therefore increase the number of fits, not the number of output columns by `S`.

The checked-in configurations expand as follows:

| Dataset | Feature sets | Depth settings | Per-type core pass | Outside embedding passes | Prediction columns |
| --- | --- | --- | --- | --- | ---: |
| `data-inside` | `eye`, `coords` | with and without depth | No | No | 17 |
| `data-inside-zoom` | `eye`, `coords` | with and without depth | No | No | 17 |
| `data-outside` | `coords`, `features` | with and without depth | Yes | EfficientNet and DINOv2 | 41 |

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

## Outside Embedding/Ridge Experiments

The additional outside model combines three inputs:

- a frozen EfficientNet-B3 embedding from each rotated image;
- the selected tabular feature set, optionally including depth;
- fish type, used both through the one-hot table columns and as a routing code.

A median-imputation, standardization, and Ridge cross-validation pipeline is first fitted globally on all training rows. A separate clone is also fitted for each fish type with at least eight retained training examples. Predictions for an eligible type use its specialized model; smaller or unseen groups use the global fallback. This specialization is internal, so these columns do **not** carry the `_per_type` suffix.

The validation and test targets are not used. Embeddings are computed for all retained rows without using their targets, then reused across the four outside feature/depth variants.

DINOv2 uses a pinned upstream commit and multiple content-validated rotated/blackout image views. Its derived tabular features obey the same feature-set and depth gates as the reported experiment name.

## Prediction Table Contract

The output is a wide CSV at:

```text
data/<dataset-name>/predictions.csv
```

It contains one row for every validated preprocessing row used by the configured experiments. The leading columns are:

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

## Checkpoints, Runs, and Embedding Cache

Each invocation stages artifacts below:

```text
checkpoints/<dataset-name>/runs/<run-id>/
data/<dataset-name>/runs/<run-id>/predictions.csv
reports/<dataset-name>/runs/<run-id>/{train,val,test}.csv
```

Linear, XGBoost, EfficientNet/Ridge, and DINOv2/Ridge use `.joblib`; MLP and CNN use `.pth`. Every checkpoint includes the feature ordering and experiment metadata needed to identify its config, depth flag, species scope, image source, backbone, preprocessing constants, and upstream DINO revision where applicable.

Per-type checkpoint names append a normalized fish-type identifier after the prediction-column name, so one species cannot overwrite another. Embedding caches live under `checkpoints/<dataset-name>/cache/`; their manifests include ordered image names, image content hashes, backbone/weight or repository revision, view, and framing.

The run manifest records hashes for the processed input, predictions, reports, and every checkpoint. `current.json` is written only after canonical outputs are published. Training never resumes from checkpoints or skips an experiment automatically.

## Reproducibility and Failure Conditions

- Python, NumPy, and Torch are seeded; deterministic cuDNN behavior is enabled.
- Missing pretrained CNN weights fail the experiment instead of changing its initialization.
- Empty or invalid split partitions, duplicate names, overlapping flags, missing configured features, or nulls in selected inputs fail before training.
- Every retained row must have a readable blackout image; outside embedding models additionally require the configured image views.
- Core tabular models require complete selected inputs. The outside Ridge pipelines retain their own median imputation and standardization.
- Canonical predictions, reports, and the current-run pointer remain on the previous complete run if any fit or publication preparation fails.
