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

The selected config must be valid, and `data/<dataset>/split.csv` must already exist. Preprocessing constructs every stage before processing the first row, so a run also requires:

- The configured YOLO checkpoint.
- The configured Segment Anything ViT-L checkpoint.
- The configured Depth Anything V2 ViT-L checkpoint.
- The initialized `third_party/Depth-Anything-V2` submodule.
- `.env.json` containing `GEMINI_API_KEY`, even when rotation prevents new context requests.
- Raw images whose names match the split metadata.

GPU execution is used automatically when available. CPU execution is possible but depth, segmentation, and neural inference can be slow.

## Input Contract

`split.csv` must contain:

- `name`
- `length`
- `is_train`
- `is_val`
- `is_test`
- `fish_type` when the config enables fish types

Input rows containing any null are removed immediately. Image names are expected to be unique and relative to the raw image folder.

## Pipeline Order

| Order | Stage | Image source | Main result | Row behavior |
| --- | --- | --- | --- | --- |
| 1 | Initial detection | Raw image | Head, tail, fish, and optional eye boxes used for alignment | Rows without complete usable detections are removed |
| 2 | Rotation | Raw image | Tail-to-head alignment and cropped rotated image | Skipped when rotation is disabled |
| 3 | Final detection | Rotated image, or raw image when rotation is disabled | Image dimensions and final body-part boxes | Rows without complete usable detections are removed |
| 4 | Relative depth | Same image as final detection | Full depth array and sampled head/body/tail values | Failed or missing results become null and are removed |
| 5 | Segmentation | Same image as final detection | Binary fish mask and contour geometry | Failed or missing results become null and are removed |
| 6 | Blackout image | Image plus segmentation mask | Isolated fish centered on a 224 by 224 canvas | Failures are logged but do not directly remove metadata rows |
| 7 | Original-image context | Raw image or existing cache | Scene placement, orientation, lighting, and object indicators | Cache is reused before rotation policy is checked |
| 8 | Feature engineering | Enriched metadata | Relative geometry, fish-type dummies, and encoded context | Missing required columns or invalid categories can abort the run |

After all stages finish, the current dataframe replaces `data/<dataset>/processed.csv`.

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

Preprocessing is a complete-case pipeline rather than a one-to-one conversion of the split table. Rows can disappear because:

- The split row already contains a null.
- The raw or rotated image is missing.
- The detector finds no object, duplicate class IDs, or an incomplete required class set.
- Rotation cannot produce a readable output.
- Depth inference or sampling fails.
- Segmentation fails or yields unusable geometry.
- Only some context rows are cached and the joined context columns are null for the rest.

The final `processed.csv` therefore commonly contains fewer records than `split.csv`. Training applies another full-row null removal before fitting.

## Cache and Rerun Behavior

Every intermediate is keyed only by image filename. If an expected output exists, preprocessing normally reuses it without checking:

- Source image contents or modification time.
- Checkpoint identity.
- Detector class order.
- Rotation setting.
- Context prompt or remote model version.
- Preprocessing implementation version.

There is no force, clean, or cache-manifest option. A failed run can leave a mix of new and old intermediates while preserving the previous final `processed.csv`. A later successful run can also leave orphaned artifacts for rows no longer present.

Remove only the affected generated directories when deliberately changing inputs or models. Preserve the dataset split unless a new experimental partition is intended.

## Current Known Limitations

These behaviors are documented but intentionally not changed by the documentation update:

- The detector result dimensions are assigned to `Image_w` and `Image_h` in reverse order. Relative width and relative height therefore use swapped denominators; the relative area denominator remains equivalent.
- The depth extraction assignment overwrites head depth with tail depth and stores tail depth as both gradient fields rather than a head-to-tail difference.
- Missing depth weights do not currently reach the intended automatic download because the warning path references the model path before it is assigned. Provide the checkpoint explicitly.
- Context generation emits `lighting_condition`, while feature encoding currently searches for `lightning_condition`. Newly generated lighting categories are therefore not converted to dummy columns by that stage; older cached artifacts may already contain legacy dummy columns.
- Blackout failures do not remove rows immediately. A later CNN run can fail if `processed.csv` references an image that was not written.
- A partial original-image context cache can cause uncached rows to be removed after the join. When no context entries are available, no context columns are added.

Cached original-image context being reused for rotated geometry is intended behavior, not a limitation.

## Downstream Contract

Training expects the configured feature columns to exist in `processed.csv` and expects corresponding blackout images for CNN experiments. The outdoor embedding experiment also expects rotated images. The `features` training family requires the context-derived columns present in the current outdoor processed artifacts.

See [steps/README.md](steps/README.md) for the exact mechanics of each stage and [training/README.md](../training/README.md) for how the generated columns are selected.
