# Preprocessing Pipeline

The preprocessing package converts split metadata and source images into a complete feature table plus reusable computer-vision artifacts. It runs detection, optional alignment, relative-depth estimation, segmentation, isolated-image creation, original-image context enrichment, and feature engineering in a fixed order.

## Command

Run from the repository root:

```bash
uv run python -m src.preprocessing.run --dataset-name data-inside
uv run python -m src.preprocessing.run --dataset-name data-inside-zoom
uv run python -m src.preprocessing.run --dataset-name data-outside
```

`--dataset-name` is required and selects `configs/<dataset-name>.json`.

## Prerequisites

The selected config and `data/<dataset>/split.csv` must be valid. Every run requires the configured YOLO and Segment Anything checkpoints plus the raw images named by the split. A configured `true` depth experiment additionally requires the Depth Anything V2 checkpoint and initialized submodule. A configured `features` experiment constructs the context stage and therefore requires `.env.json` containing `GEMINI_API_KEY`.

GPU execution is used automatically when available. CPU execution is possible but depth, segmentation, and neural inference can be slow.

## Input Contract

`split.csv` must contain:

- `name`
- `length`
- `is_train`
- `is_val`
- `is_test`
- `fish_type` when the config enables fish types

Required input columns cannot contain nulls. Image names must be unique normalized paths relative to the raw image folder, and every row must belong to exactly one split. Unrelated nullable columns are retained.

## Pipeline Order

| Order | Stage | Image source | Main result | Row behavior |
| --- | --- | --- | --- | --- |
| 1 | Initial detection | Raw image | Head, tail, fish, and optional eye boxes used for alignment | Rows without complete usable detections are removed |
| 2 | Rotation | Raw image | Tail-to-head alignment and cropped rotated image | Skipped when rotation is disabled |
| 3 | Final detection | Rotated image, or raw image when rotation is disabled | Image dimensions and final body-part boxes | Rows without complete usable detections are removed |
| 4 | Relative depth, when any configured depth flag is true | Same image as final detection | Full depth array and sampled head/body/tail values | Failed or missing results become null and are removed |
| 5 | Segmentation | Same image as final detection | Binary fish mask and contour geometry | Failed or missing results become null and are removed |
| 6 | Blackout image | Image plus segmentation mask | Isolated fish centered on a 224 by 224 canvas | Failed writes remove the row |
| 7 | Original-image context, when `features` is configured | Raw image or valid cache | Scene placement, orientation, lighting, and object indicators | Missing required context removes the row |
| 8 | Feature engineering | Enriched metadata | Relative geometry, fish-type dummies, and encoded context | Missing required columns or invalid categories abort the run |

After all selected stages finish, `processed.csv` is replaced atomically and `processed/preprocessing_report.json` records each stage's input count, output count, and dropped names.

## Rotation Modes

### Rotation enabled

All current configs enable rotation.

1. Initial detections are made on raw images.
2. The center-to-center tail/head direction is aligned with the horizontal axis.
3. The expanded rotated image is cropped to remove introduced black borders.
4. A new detection pass discards the original coordinate system and measures the rotated image.
5. Depth, segmentation, blackout creation, and geometric features use rotated images.

Original-image context is deliberately separate from geometric alignment. Existing context cache entries are joined because placement, lighting, surrounding objects, and fishnet presence describe the original scene. New remote context calls are skipped while rotation is enabled.

### Rotation disabled

- Rotation returns the dataframe unchanged.
- Both detector stages use raw images and the same `yolo_initial` cache, so the second stage reloads the first result.
- Depth, segmentation, and blackout creation use raw images.
- Missing original-image context can be requested remotely and cached.

## Outputs

| Location | Format | Contents |
| --- | --- | --- |
| `processed/cache/yolo_initial/` | One JSON file per image | Raw-image detection boxes and dimensions |
| `processed/cache/yolo_rotated/` | One JSON file per image | Rotated-image detection boxes and dimensions |
| `processed/cache/vlm/` | One JSON file per image | Original-image scene context |
| `processed/rotated/` | Original image extension | Aligned and cropped image |
| `processed/depth/` | NumPy array named `<image-name>.npy` | Full relative-depth map |
| `processed/segment/` | NumPy array named `<image-name>.npy` | Binary segmentation mask |
| `processed/blackout/` | Original image extension | Isolated fish on a fixed 224 by 224 canvas |
| `processed.csv` | CSV | Final labels, split flags, detections, depth, mask geometry, context, and engineered features |
| `processed/preprocessing_report.json` | JSON | Per-stage row attrition for the most recent successful run |

The final table does not contain paths to generated artifacts. Image name is the key used to resolve every artifact.

## Final Feature Families

### Detection geometry

For each configured label, the table records integer box boundaries, width, and height. It also records image dimensions from the final detection pass.

### Relative geometry

- Fish width divided by image width.
- Fish height divided by image height.
- Fish bounding-box area divided by image area.
- Fish width-to-height aspect ratio.
- Square root of fish bounding-box area, stored as `fish_area`.

### Relative depth

- Median relative depth near the head center.
- Median relative depth near the fish-box center.
- Median relative depth near the tail center.
- Raw and absolute depth-gradient fields.

Depth is monocular relative depth, not a calibrated physical distance.

### Segmentation geometry

- Largest external contour area.
- Closed-contour perimeter.
- Major and minor axes from an ellipse fit when the contour has enough points.
- Solidity, calculated as contour area divided by convex-hull area.

### Original-image context

- Background depth category.
- Whether other objects are visible.
- Whether the fish is in or on a fishnet.
- Fish placement category.
- Fish orientation category.
- Lighting category.

Boolean and categorical values are converted into numeric or dummy columns when the expected source columns are available.

### Fish type

Fish-aware datasets receive one dummy column per observed type while retaining the original `fish_type` column.

## Row Attrition

Preprocessing records complete cases for the stages required by the configured experiment matrix. Rows can disappear because:

- A required split value is null or split membership is invalid, which aborts before processing.
- The raw or rotated image is missing or unreadable.
- The detector finds no object, duplicate class IDs, or an incomplete required class set.
- Rotation, depth inference, depth sampling, segmentation, blackout writing, or required context extraction fails.

Unrelated nullable columns do not remove rows. The final `processed.csv` commonly contains fewer records than `split.csv`; the structured preprocessing report makes the exact stage and names observable. Training validates only identifiers, split fields, and columns selected by configured experiments.

## Cache and Rerun Behavior

Every detection JSON, rotated image, depth array, segmentation mask, blackout image, and context response has a sidecar manifest. The signature includes source-image content and all relevant checkpoint identities, stage parameters, class ordering, prompts, remote model identifiers, and implementation-version values. A missing or mismatched manifest invalidates the cached artifact.

Embedding caches apply the same principle to the ordered image-name and content manifest plus backbone/revision metadata. There is no force or clean mode, and old unreferenced artifacts are not deleted automatically.

## Failure Behavior

Per-image failures are logged and represented as row attrition; `preprocessing_report.json` identifies the stage and dropped names. Required split-contract violations fail before a model is constructed. The final CSV and report are published only after the selected stage list succeeds.

Original-image context remains deliberately separate from aligned geometry. With rotation enabled, valid cached context can be reused, while uncached remote calls remain disabled. This preserves scene information from the source photograph without describing the rotated crop as a new scene.

## Downstream Contract

Training expects the configured feature columns to exist in `processed.csv` and expects corresponding blackout images for CNN experiments. The outdoor embedding experiment also expects rotated images. The `features` training family requires the context-derived columns present in the current outdoor processed artifacts.

See [steps/README.md](steps/README.md) for the exact mechanics of each stage and [training/README.md](../training/README.md) for how the generated columns are selected.
