# Dataset Configurations

JSON files in this directory define dataset identity, preprocessing model locations, split ratios, detector labels, and the experiment combinations used during training. Commands select a config by filename stem; the config's dataset name then determines all data locations.

Run config-driven commands from the repository root because paths are relative to the current working directory.

## Current Configs

| Config | Dataset | Rotation | Fish types | Feature sets | Depth variants |
| --- | --- | --- | --- | --- | --- |
| `data-inside.json` | Controlled laboratory data | Enabled | No | `eye`, `coords` | With and without depth |
| `data-inside-zoom.json` | Zoom-derived controlled data | Enabled | No | `eye`, `coords` | With and without depth |
| `data-outside.json` | Multi-species outdoor data | Enabled | Yes | `coords`, `features` | With and without depth |

`data-inside-zoom.json` is predefined and remains the source of truth for preprocessing and training the generated zoom dataset. The augmentation workflow creates its data artifacts but does not generate or overwrite this config.

## Config Shape

```json
{
  "dataset": {
    "name": "data-inside",
    "rotate": true,
    "fish_type_available": false,
    "feature_sets": ["eye", "coords"],
    "depth": [true, false]
  },
  "model_path": {
    "yolo": "checkpoints/yolo11m-inside_fish_model.pt",
    "sam": "checkpoints/sam.pth",
    "depth": "checkpoints/depth_anything_v2_vitl.pth"
  },
  "params": {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "yolo_classes": ["Head", "Tail", "Eye", "Fish"]
  }
}
```

## `dataset`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | Required | Folder name under `data/` and namespace for generated artifacts and training checkpoints |
| `rotate` | `true` | Whether fish are aligned horizontally before the second detection and all downstream image stages |
| `fish_type_available` | `false` | Enables per-type splitting, fish-type dummy features, grouped baseline predictions, and per-type experiment variants |
| `feature_sets` | `["coords", "scaled"]` | Feature-family names iterated by every configured training model; current configs override the unused `scaled` default |
| `depth` | `[true, false]` | Whether training runs each feature family with and without the five depth columns |

`feature_sets` and `depth` configure experiments only. They do not turn preprocessing stages on or off.

Supported feature-family names in the current training implementation are:

- `eye`: eye width and height plus fish bounding-box width and height.
- `coords`: relative fish width, relative fish height, relative box area, fish aspect ratio, and square-root box area.
- `features`: coordinate features plus segmentation geometry, original-image scene context, and encoded context categories.

Fish-type dummy columns are automatically included in all feature families when fish types are available.

## `model_path`

| Field | Consumer | Expected model |
| --- | --- | --- |
| `yolo` | Both detection passes | A detector whose class IDs match `yolo_classes` order |
| `sam` | Segmentation | Segment Anything ViT-L checkpoint |
| `depth` | Relative-depth estimation | Depth Anything V2 ViT-L checkpoint |

Config loading does not verify these files. The relevant preprocessing step raises an error when its checkpoint is missing or incompatible.

## `params`

| Field | Default | Runtime behavior |
| --- | --- | --- |
| `train_ratio` | `0.7` | Training count is `floor(group_size * train_ratio)` |
| `val_ratio` | `0.15` | Validation count is `round(group_size * val_ratio)` |
| `test_ratio` | `0.15` | Validated as part of the ratio total, but not used directly; test receives every row remaining after train and validation |
| `yolo_classes` | `Head`, `Tail`, `Fish` | Ordered detector label mapping; the final entry is the fallback label for any class ID not mapped earlier |

Every ratio must be strictly greater than zero and strictly less than one. Their sum may not exceed one. Because test receives the remainder, a sum below one makes the effective test share larger than `test_ratio`.

When fish types are enabled, floor/round/remainder calculations are performed independently for every type. Small groups can consequently have an empty validation or test partition.

## Detector Class Ordering

Detector class IDs are interpreted positionally. With:

```json
["Head", "Tail", "Eye", "Fish"]
```

class IDs 0, 1, and 2 map to `Head`, `Tail`, and `Eye`; all other IDs map to the final `Fish` label. The configured order must match the checkpoint's training labels. The preprocessing pipeline expects one usable detection for each required body part and rejects a prediction containing duplicate raw class IDs.

## Dataset Validation

Loading any config requires these paths to exist already:

```text
data/<dataset-name>/
|-- raw/
`-- raw.csv
```

The raw metadata table must contain:

- `name`: unique image filename.
- `length`: numeric target value.
- `fish_type`: required when `fish_type_available` is `true`.

Config validation checks only the dataset directory, raw directory, and raw table. It does not validate table columns, image correspondence, unique names, split files, processed files, checkpoints, or API credentials.

The helper that lists valid configs silently excludes a config when parsing or dataset validation fails. Direct selection reports the validation error.

Augmentation is the one bootstrap exception: it validates the predefined zoom config without requiring the target dataset to exist, creates the target raw artifacts, and leaves the config unchanged. Ordinary preprocessing and training config loading then uses the normal dataset-presence validation.

## Derived Paths

For dataset name `<dataset>`, the shared config layer resolves:

| Purpose | Path |
| --- | --- |
| Raw images | `data/<dataset>/raw/` |
| Raw metadata | `data/<dataset>/raw.csv` |
| Split metadata | `data/<dataset>/split.csv` |
| Processed artifacts | `data/<dataset>/processed/` |
| Processed feature table | `data/<dataset>/processed.csv` |
| Dataset root | `data/<dataset>/` |

Training adds `checkpoints/<dataset>/` and `data/<dataset>/predictions.csv` outside the computed config properties.

## Rotation and Original-Image Context

With rotation enabled:

1. Initial detection runs on the raw image.
2. The fish is rotated using head and tail centers.
3. Detection runs again on the rotated image.
4. Depth, segmentation, blackout creation, and geometric features use the rotated image.
5. Previously cached scene context from the original image is still joined when present; new remote context requests are skipped.

This cache behavior is intentional: scene placement, lighting, surrounding objects, and fishnet presence describe the original photograph even when geometry is measured after rotation.

With rotation disabled, both detector passes use the raw-image cache, downstream image stages use raw images, and uncached original-image context may be requested.

## Adding a Dataset Config

1. Create the dataset directory, raw image folder, and raw metadata table first.
2. Copy the closest existing JSON config.
3. Set a unique dataset name that matches the folder.
4. Match detector labels to the checkpoint class order.
5. Choose only feature families whose preprocessing columns will exist.
6. Decide whether fish-type-aware splitting and per-type training are valid for the metadata.
7. Confirm the three ratios and checkpoint locations.
8. Run split creation before preprocessing, then train only after `processed.csv` is complete.

Do not point two config files at the same dataset directory unless overwriting the same split, processed table, predictions, and checkpoints is intentional.
