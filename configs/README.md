# Configs

This folder contains JSON configuration files for the datasets and pipeline settings.

## Files

- `data-inside.json`: controlled laboratory dataset.
- `data-inside-zoom.json`: zoom-augmented controlled dataset.
- `data-outside.json`: arbitrary outdoor dataset with fish types.
- `data-outside-custom.json`: custom outside-data variant.

## Schema

Each config has three main sections:

- `dataset`: dataset behavior and available metadata.
- `model_path`: checkpoint paths for YOLO, SAM, and DepthAnythingV2.
- `params`: split ratios and YOLO class labels.

Important `dataset` fields:

- `name`: dataset folder under `data/`.
- `rotate`: controls whether preprocessing rotates images before downstream feature extraction.
- `fish_type_available`: enables fish-type-aware splitting, one-hot encoding, and per-type training.
- `feature_sets`: feature groups used during training.
- `depth`: list of depth-feature toggles to train with.

## Rotation Setting

`rotate` is the main switch for rotated versus non-rotated preprocessing.

- `true`: run YOLO on raw images, rotate images using head/tail coordinates, rerun YOLO on rotated images, then run depth, segmentation, blackout, and features on rotated images.
- `false`: skip image rotation and run downstream preprocessing on raw images.

VLM metadata extraction is currently skipped for rotated preprocessing and runs only when `rotate` is `false`.

## Dataset Differences

`data-inside` uses controlled lab images and has no fish-type column. Its YOLO class list includes `Eye`, so training can use the `eye` feature set.

`data-inside-zoom` follows the same controlled-data setup as `data-inside`, but images are zoom-augmented copies of the inside dataset.

`data-outside` has `fish_type_available: true`, so split creation is stratified by fish type and training also creates per-fish-type prediction columns.
