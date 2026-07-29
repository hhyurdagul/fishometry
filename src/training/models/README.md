# Training Models

This package implements the estimators used by the [training orchestrator](../README.md). The normal entry point supplies a complete-case processed dataframe, a dataset configuration, one feature-set name, one depth setting, and a flag used to name per-fish-type results.

Each training function returns predictions keyed by image `name`. The orchestrator joins those results into the dataset's wide `predictions.csv`; the model functions do not calculate evaluation metrics.

## Shared Tabular Feature Specification

Linear regression, XGBoost, MLP, CNN auxiliary inputs, and EfficientNet/Ridge all use the same configured selector:

1. Include every `fish_type_` one-hot column that exists.
2. Add the requested base feature group.
3. Append the five depth columns when depth is enabled.
4. Build the prediction/checkpoint name from the model, feature set, optional `_depth`, and optional `_per_type` suffix.

The supported groups are:

| Group | Inputs |
| --- | --- |
| `eye` | `Eye_w`, `Eye_h`, `Fish_w`, `Fish_h` |
| `coords` | `relative_w`, `relative_h`, `relative_area`, `fish_aspect`, `fish_area` |
| `features` | All `coords` inputs; `mask_area`, `mask_perimeter`, `major_axis`, `minor_axis`, `solidity`; `background_depth`, `has_other_objects`, `is_in_fishnet`; and all columns beginning with `fish_placement_`, `fish_orientation_`, and `lightning_condition_` |
| Depth addition | `head_depth`, `body_depth`, `tail_depth`, `depth_gradient_raw`, `depth_gradient_abs` |

An unrecognized group does not raise a feature-specific error. It selects only matching fish-type one-hot columns, if any, which may create a zero-column matrix or an unintended fish-type-only experiment.

The `per_type` argument does not itself split data. The orchestrator filters to one fish type before calling a core model, and the argument only adds the `_per_type` suffix. Direct callers must perform that filtering themselves.

## Mean Baseline

The baseline uses only rows marked as training data.

- Without fish types, it predicts the global training-set mean length for every row.
- With fish types, it computes a separate training-set mean for each type and joins that value to all rows of the type. There is no additional global baseline in this case.
- Predictions are rounded to two decimal places and named `mean_regression`.
- No checkpoint is written.

A fish type with no retained training row receives a null baseline after the join.

## Linear Regression

The linear model is scikit-learn ordinary least squares with its default intercept behavior.

- It fits selected, unscaled feature values from `is_train` rows.
- It does not use validation rows, regularization, imputation, or feature scaling.
- It predicts all rows in the dataframe.
- The serialized scikit-learn pipeline is saved as `<experiment-name>.joblib`.

The pipeline contains only the regressor; the feature names and ordering are not stored as an explicit preprocessing schema.

## XGBoost Regression

The tree model uses `XGBRegressor` with:

- at most 100 estimators;
- maximum tree depth 16;
- learning rate 0.1;
- `random_state=42`;
- all available CPU worker threads;
- early stopping after 20 validation rounds without improvement.

Training features are taken from `is_train` rows. The complete `is_val` matrix and targets are supplied as the evaluation set. Inputs are not imputed or scaled. The fitted scikit-learn/XGBoost pipeline is saved as `<experiment-name>.joblib`, then used to predict all rows.

The validation split must be present and compatible with the training feature matrix. The test split is not supplied to fitting or early stopping.

## Tabular MLP

The MLP is a fully connected regression network:

```text
input -> Linear(32) -> ReLU -> Linear(8) -> ReLU -> Linear(1)
```

Its fixed training settings are:

- 100 epochs;
- batch size 16;
- Adam optimizer with learning rate `1e-3`;
- mean squared error loss;
- shuffled training batches;
- CUDA when available, otherwise CPU.

Input features are converted directly to `float32` without imputation or scaling. Validation loss is calculated after every epoch, and an independent copy of the lowest-validation-loss state is restored after all 100 epochs. There is no patience-based early stop for the MLP.

The validation dataset must contain at least one row; otherwise validation-loss averaging divides by zero. No random seed is set, so initialization, batch order, and potentially device kernels make repeated fits nondeterministic.

The `.pth` checkpoint stores only:

- `model_state_dict`;
- `input_dim`.

It does not contain optimizer state, epoch, best validation loss, feature ordering, training settings, or a ready-to-call model object.

## Blackout-Image CNN

The CNN combines a fish-only blackout image with the same selected auxiliary tabular features used by the regression models.

### Architecture

The image backbone is ResNet-18. Loading first requests torchvision's default pretrained weights; any exception causes a silent fallback to an untrained ResNet-18. The classification layer is replaced with an identity mapping, producing 512 image features. All backbone parameters remain trainable.

The image features are concatenated with the auxiliary vector and passed through:

```text
Linear(512 + auxiliary width, 256) -> ReLU -> Linear(256, 1)
```

The orchestrated experiments always provide a configured feature set. A lower-level image-only mode exists when the feature set is `None`; its optional depth branch expects three center-depth fields and is not part of the standard experiment matrix.

### Image Transform

Each blackout image is:

1. opened and converted to RGB;
2. resized directly to `224 x 224` pixels, without preserving aspect ratio;
3. converted to a tensor;
4. normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

There are no random crops, flips, color transforms, or other training-time augmentations. Auxiliary values are converted to `float32` and are not standardized.

### Optimization and Validation

The fixed settings are 100 maximum epochs, batch size 16, Adam at `1e-4`, and mean squared error loss. Training batches are shuffled. The model runs on CUDA when available and CPU otherwise.

The lowest validation-loss state is retained, and training stops after five consecutive epochs without improvement. If no validation image is available, training loss is used as the selection signal instead. At least one training blackout image is mandatory.

Image records whose blackout file is absent are skipped while datasets are assembled. This does not create a safe partial-output path: the final prediction dataframe pairs predictions with the full input name column, so any skipped prediction image normally causes a length mismatch and aborts the run. Every retained row should therefore have a readable blackout image.

Predictions are rounded to two decimal places. The `.pth` checkpoint stores `model_state_dict` and `aux_size`; it omits optimizer state, epoch, feature ordering, transforms, and whether pretrained initialization succeeded.

## Outside EfficientNet/Ridge Model

This model runs only in the standard `data-outside` orchestration and requires the original `fish_type` column plus a rotated image for every retained row.

### Frozen Image Embeddings

EfficientNet-B3 is loaded with torchvision's default pretrained weights. Its classifier is replaced with an identity mapping, and the network is held in evaluation mode under `no_grad`; it is never fine-tuned. Each rotated image is transformed with the preprocessing recipe bundled with those pretrained weights, then mapped to a 1,536-value embedding. Extraction uses batches of eight, no shuffled order, no worker processes, and CUDA when available.

Unlike the CNN, failure to obtain pretrained weights is not caught. If no embedding cache exists, the default weights must be locally cached or downloadable.

### Combined Regressor

For each configured feature/depth pair, the selected tabular matrix is concatenated with the cached image embeddings. A numeric fish-type code is appended for routing, but removed before each Ridge model is fitted. Fish type can still be present in the regressor inputs through the selected `fish_type_` one-hot columns.

The base regression pipeline contains:

1. median imputation;
2. standard scaling;
3. `RidgeCV` over 80 logarithmically spaced alpha values from `1e-2` through `1e10`.

One global pipeline is fitted on every training row. A cloned pipeline is also fitted for each fish type with at least eight retained training rows. During prediction, eligible types use their specialized pipeline and all other types use the global fallback.

This internal type routing happens even though the orchestrator passes `per_type=False`, so output names are `efficientnet_ridge_<feature-set>[_depth]`, without `_per_type`. Validation flags are ignored; RidgeCV chooses alpha using internal cross-validation over training rows. Test targets are never used. Predictions for every split are rounded to two decimals.

### Cache and Artifact

The first variant writes all embeddings, in dataframe row order, to:

```text
checkpoints/<dataset-name>/efficientnet_b3_rotated_embeddings.npy
```

Later variants and later runs load that array whenever the file exists. The cache has no image-name manifest, signature, row-count validation, or preprocessing fingerprint. Delete it whenever rows or rotated images change. Reordering rows without changing their count is particularly dangerous because it can silently assign the wrong embedding to each target.

Each Ridge artifact is a `.joblib` dictionary containing the fitted routed model, feature-set/depth/per-type values, fish-type lookup, and image-backbone/source labels. It does not include the image embeddings or pretrained backbone weights.

## Outside Multi-View DINOv2/Ridge Model

This model runs only in the standard `data-outside` orchestration and has the same shape as the EfficientNet/Ridge model — frozen image features concatenated with tabular geometry, fitted per fish type with a ridge. It differs in three ways that were measured to matter on the outdoor data.

### Self-supervised backbone

The image features come from DINOv2 (`dinov2_vitb14` and `dinov2_vitl14`, loaded from `facebookresearch/dinov2` through `torch.hub`) rather than an ImageNet classifier. Held-out comparisons on this dataset put DINOv2 features well ahead of EfficientNet-B3, ConvNeXt-Base, EfficientNetV2-M and Swin-V2-B features, and the ImageNet backbones were not distinguishable from one another.

### Multiple views per image

Each image is embedded nine times, under different resolutions and framings, and the results are concatenated:

| Backbone | Source | Size | Framing |
| --- | --- | ---: | --- |
| `dinov2_vitb14` | `rotated` | 224 | squash |
| `dinov2_vitb14` | `rotated` | 336 | squash |
| `dinov2_vitb14` | `rotated` | 448 | crop |
| `dinov2_vitb14` | `rotated` | 518 | squash |
| `dinov2_vitb14` | `blackout` | 224 | squash |
| `dinov2_vitl14` | `rotated` | 224 | squash |
| `dinov2_vitl14` | `rotated` | 336 | squash |
| `dinov2_vitl14` | `rotated` | 448 | squash |
| `dinov2_vitl14` | `rotated` | 448 | crop |

`squash` resizes the whole frame to a square and accepts the aspect distortion; `crop` resizes the short side and takes a centre crop. The two framings retain different amounts of surrounding scene, which is where the monocular scale cues sit. The single `blackout` view sees the isolated fish with no context and contributes a different error pattern. All sizes are multiples of the 14-pixel DINOv2 patch.

### Derived geometry features

Beyond the configured feature set, the model derives further inputs from columns already present in `processed.csv`, so no preprocessing rerun is required:

- Scale-referenced spans: straight-line head-to-tail pixel distance and fish box diagonal, each divided by the image diagonal; mask area and major axis relative to the image diagonal.
- Dimensionless body proportions: head and tail box size relative to the fish diagonal, head-to-tail box ratio, head and tail aspect ratios, mask fill ratio, perimeter-squared-over-area compactness, major-over-minor elongation.
- Frame placement: fish centre coordinates as image fractions, image aspect ratio and diagonal.
- Depth contrasts: body-minus-background relative depth, absolute head-minus-tail relative depth.
- Log transforms of the size-valued columns above.

### Shrunk per-species ridge

The regression pipeline is the same median-impute, standard-scale, `RidgeCV` stack, over 120 alphas from `1e-3` to `1e8`. A global pipeline is fitted on all training rows, and a separate pipeline is fitted for each fish type with at least eight training rows. Unlike the EfficientNet/Ridge model, an eligible type's prediction is not replaced outright: it is a `0.85 / 0.15` blend of the species pipeline and the global pipeline. The shrinkage stabilises types whose training count is small relative to the feature width.

Validation flags are ignored; RidgeCV selects alpha by internal cross-validation over training rows, and test targets are never used. Output names are `dino_ridge_<feature-set>[_depth]`, without `_per_type`, because the type routing is internal.

### Cache and artifact

Each view writes its own cache under `checkpoints/<dataset-name>/`:

```text
dino_<backbone>_<source>_<size>_<framing>.npy
dino_<backbone>_<source>_<size>_<framing>_names.json
```

The companion `_names.json` records the exact ordered image-name list the array was built for. The cache is reused only when that list matches the current dataframe, so a reordered or filtered `processed.csv` triggers re-extraction instead of silently binding embeddings to the wrong rows. Extraction needs network access on first run to fetch the DINOv2 weights.

The `.joblib` artifact holds the fitted blended model, feature-set/depth/per-type values, fish-type lookup, backbone label, and the view list. It does not include embeddings or backbone weights.

## Checkpoint Summary

All artifacts are written under `checkpoints/<dataset-name>/`.

| Model | Extension | Saved content | Loaded by orchestrator on rerun |
| --- | --- | --- | --- |
| Mean baseline | None | No checkpoint | No |
| Linear | `.joblib` | Fitted scikit-learn pipeline | No |
| XGBoost | `.joblib` | Fitted XGBoost pipeline | No |
| MLP | `.pth` | State dictionary and input width | No |
| CNN | `.pth` | State dictionary and auxiliary width | No |
| EfficientNet/Ridge | `.joblib` | Routed Ridge model and metadata | No |
| EfficientNet embeddings | `.npy` | Ordered embedding matrix | Yes, based only on file existence |
| DINOv2/Ridge | `.joblib` | Blended Ridge model and metadata | No |
| DINOv2 embeddings | `.npy` + `.json` | Ordered embedding matrix per view, with its name manifest | Yes, only when the manifest matches the current rows |

Artifacts with the same experiment name are overwritten without confirmation. There is no checkpoint loading, resume, or skip-existing path for learned models.

For per-fish-type core experiments, every fish type uses the same `_per_type` filename. The checkpoints overwrite one another as types are processed, leaving only the last fitted type on disk. The corresponding prediction column is still correct because predictions are collected in memory before the overwrite. Recovering all type-specific models requires changing the artifact strategy; the current checkpoint directory is not a complete per-type model registry.

## Prediction and Evaluation Contract

Every estimator predicts the dataframe it receives, not only test rows. The orchestrator supplies all complete retained rows to global models and all complete rows of one type to each per-type model. Split flags are preserved in the final table so evaluation can isolate validation or held-out test observations.

The model package does not enforce:

- one and only one active split flag per row;
- unique image names;
- nonempty train and validation subsets;
- consistency between selected columns and a saved checkpoint;
- consistency between image rows and the embedding cache.

Those are upstream data and orchestration requirements. Violating them can cause exceptions, duplicated joins, leakage, or predictions associated with the wrong image.

## Reproducibility Notes

Ordinary least squares and the Ridge configuration are deterministic for fixed ordered inputs. XGBoost explicitly sets a random seed. The MLP and CNN set no random seeds, and CUDA kernels are not forced into deterministic operation. CNN runs may also differ fundamentally when pretrained ResNet weights are available in one environment but not another. Saved prediction tables should therefore be treated as experiment artifacts and retained alongside the exact processed data and configuration used to produce them.
