# Preprocessing Steps

This folder contains the individual preprocessing components.

## YOLO Detection

`yolo.py` loads the configured YOLO checkpoint and detects configured classes such as `Head`, `Tail`, `Eye`, and `Fish`.

The initial YOLO step runs on raw images. The second YOLO step runs on rotated images when `dataset.rotate` is `true`, otherwise it runs on raw images.

Cached detections are written under `processed/cache/`.

## Rotation

`rotate.py` uses head and tail centers to align the fish along the x-axis. The rotated image is written to `processed/rotated/`.

If `dataset.rotate` is `false`, this step returns the dataframe unchanged.

## Depth

`depth.py` uses DepthAnythingV2 to create a depth map and extracts robust median depth values around head, body, and tail coordinates.

Depth maps are cached as `.npy` files under `processed/depth/`.

## Segmentation

`segment.py` uses Segment Anything with head and tail points as prompts. It saves masks under `processed/segment/` and extracts geometric mask features such as area, perimeter, axes, and solidity.

## Blackout Images

`blackout.py` applies the segmentation mask to keep only the fish, centers it on a black canvas, and writes the result to `processed/blackout/`. These images are used by CNN-style training.

## VLM Metadata

`vlm.py` uses Gemini to extract scene metadata such as placement, orientation, lighting, background depth, and whether other objects or a fishnet are visible.

This step currently returns no new features when `dataset.rotate` is `true`.

## Feature Engineering

`feature.py` creates relative geometry features, fish-type dummy variables when available, and encoded VLM features when those columns exist.
