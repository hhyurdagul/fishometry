# Fishometry

Fishometry is my master's thesis codebase for predicting fish length from images without using a physical reference object in the image.

The project compares fish photographed under controlled laboratory conditions with fish photographed in arbitrary outdoor conditions. It builds dataset splits, optionally creates a zoom-augmented controlled dataset, extracts computer-vision features, trains prediction models, and provides a Streamlit app for inspecting results.

## Documentation Map

- [configs](configs/README.md): dataset configuration files and the meaning of each config field
- [src](src/README.md): overview of the Python source package
- [create_data](src/create_data/README.md): split creation and zoom augmentation pipeline
- [create_data steps](src/create_data/steps/README.md): individual split and augmentation components
- [preprocessing](src/preprocessing/README.md): YOLO, rotation, depth, segmentation, blackout, VLM, and feature pipeline
- [preprocessing steps](src/preprocessing/steps/README.md): details for each preprocessing component
- [training](src/training/README.md): model orchestration and prediction outputs
- [training models](src/training/models/README.md): model implementations
- [visualization](src/visualization/README.md): Streamlit analysis app
- [visualization views](src/visualization/views/README.md): app pages and result views

## Dataset Flow

The project uses three dataset variants:

- `data-inside`: controlled laboratory fish images.
- `data-outside`: arbitrary outdoor fish images with multiple fish types.
- `data-inside-zoom`: zoom-augmented data derived from `data-inside`.

The intended flow is:

1. Create splits for `data-inside` and `data-outside`.
2. Create `data-inside-zoom` from the already split `data-inside` metadata so the augmented examples keep the same train/validation/test assignment as their source image.
3. Run preprocessing from each dataset config.
4. Train prediction models and write `predictions.csv`.
5. Inspect metadata, images, errors, and model comparisons in the Streamlit app.

## Main Commands

Install dependencies with `uv`, then run commands from the repository root.

```bash
uv run python -m src.create_data.run --dataset-name data-inside
uv run python -m src.create_data.run --dataset-name data-outside
uv run python -m src.create_data.run --dataset-name data-inside --augment

uv run python -m src.preprocessing.run --dataset-name data-inside
uv run python -m src.preprocessing.run --dataset-name data-inside-zoom
uv run python -m src.preprocessing.run --dataset-name data-outside

uv run python -m src.training.run --dataset-name data-inside
uv run python -m src.training.run --dataset-name data-inside-zoom
uv run python -m src.training.run --dataset-name data-outside

uv run streamlit run src/visualization/app.py
```

## Important Generated Files

Each dataset lives under `data/<dataset-name>/`.

- `raw/`: original images.
- `raw.csv`: original metadata, including fish length and optionally fish type.
- `split.csv`: metadata with `is_train`, `is_val`, and `is_test`.
- `processed.csv`: preprocessing output with extracted features.
- `processed/rotated/`: rotated images when rotation is enabled.
- `processed/depth/`: cached depth maps.
- `processed/segment/`: cached segmentation masks.
- `processed/blackout/`: fish-only images on a black canvas for CNN input.
- `processed/cache/`: cached YOLO and VLM results.
- `predictions.csv`: final wide prediction table.

Large runtime artifacts such as `data/`, `checkpoints/`, and `third_party/` are intentionally not part of the normal source documentation.

## Rotation Behavior

Rotation is controlled by `dataset.rotate` in each JSON config.

- When `rotate` is `true`, preprocessing runs initial YOLO on raw images, rotates the image using head and tail detections, reruns YOLO on the rotated image, and then runs depth, segmentation, blackout image creation, and feature extraction on the rotated image.
- When `rotate` is `false`, the rotation step is skipped and downstream preprocessing uses the original raw image.
- VLM metadata extraction currently runs only when rotation is disabled.

This makes the config file the source of truth for whether preprocessing happens on rotated or non-rotated images.
