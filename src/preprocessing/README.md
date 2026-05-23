# Preprocessing Pipeline

This package turns split metadata and raw images into processed features, cached computer-vision artifacts, and CNN-ready images.

## Entrypoint

```bash
uv run python -m src.preprocessing.run --dataset-name data-inside
uv run python -m src.preprocessing.run --dataset-name data-inside-zoom
uv run python -m src.preprocessing.run --dataset-name data-outside
```

## Pipeline Order

`src.preprocessing.run` executes these steps:

1. Initial YOLO detection on raw images.
2. Rotation using detected head and tail coordinates.
3. YOLO detection again on the rotated image, or raw image if rotation is disabled.
4. Depth estimation.
5. SAM segmentation.
6. Blackout image creation.
7. VLM metadata extraction when rotation is disabled.
8. Feature engineering.

## Inputs

- `data/<dataset>/split.csv`
- `data/<dataset>/raw/`
- model checkpoints configured in `configs/<dataset>.json`

## Outputs

- `data/<dataset>/processed.csv`
- `data/<dataset>/processed/rotated/`
- `data/<dataset>/processed/depth/`
- `data/<dataset>/processed/segment/`
- `data/<dataset>/processed/blackout/`
- `data/<dataset>/processed/cache/`

## Rotation

The config field `dataset.rotate` controls which image version downstream steps use.

- `true`: rotate images and run later steps on `processed/rotated/`.
- `false`: skip rotation and run later steps on `raw/`.

The first YOLO pass always uses raw images because it provides the coordinates needed for rotation.
