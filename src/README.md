# Source Package

`src` contains the config-driven Fishometry pipelines. The packages are manually sequenced and communicate through dataset artifacts rather than an in-memory end-to-end orchestrator.

## Package Map

| Package | Responsibility | Primary input | Primary output |
| --- | --- | --- | --- |
| `config.py` | Parse configs, validate base dataset presence, and derive shared paths | `configs/<name>.json` | Validated config object |
| `create_data/` | Create source splits and derive the zoom-controlled dataset | Raw metadata and images, or an existing source split for augmentation | `split.csv` and optional zoom raw data |
| `preprocessing/` | Detect, align, estimate depth, segment, isolate fish, attach scene context, and engineer features | Split metadata, images, checkpoints, and context cache | `processed.csv` and reusable image artifacts |
| `training/` | Run feature/depth/model experiments and save predictions and checkpoints | Processed features and prepared images | `predictions.csv` and model artifacts |
| `visualization/` | Inspect artifacts and calculate interactive result metrics | Processed metadata, image artifacts, and predictions | Streamlit views; no persistent analysis output |

## Handoff Contracts

### Raw data to create-data

- `raw.csv` requires unique `name` and numeric `length` values.
- Fish-aware datasets also require `fish_type`.
- Every metadata name must identify an image under `raw/`.
- Null rows are removed before splitting.

### Create-data to preprocessing

- `split.csv` retains input metadata.
- It adds boolean `is_train`, `is_val`, and `is_test` columns.
- Exactly one split flag should be true per row.
- Zoom-derived rows inherit the source image's persisted split.

### Preprocessing to training

- `processed.csv` retains names, labels, fish types when applicable, and split flags.
- It adds detector coordinates, relative depth, segmentation geometry, optional original-image context, and engineered features.
- Blackout images provide the convolutional model input.
- Rotated images provide the outdoor embedding model input.
- Rows that cannot complete required stages can be absent from the processed table.

### Training to visualization

- `predictions.csv` is a wide table keyed by image name.
- Identifier columns include the target, split flags, and optional fish type.
- Every other column represents one experiment's predicted length.
- Predictions cover all surviving splits; visualization performs the split filtering and calculates metrics.

## Shared Runtime Rules

- Run module commands from the repository root because data, config, checkpoint, and third-party paths are relative.
- Generated artifacts are overwritten or reused in place; there is no run registry or global transaction.
- Image names are join keys throughout the project and are assumed to be unique.
- Caches are keyed by filename rather than input hashes or config versions.
- Data, checkpoints, and runtime caches are ignored by Git and are not portable with the source alone.
- The pipeline does not automatically remove stale or orphaned outputs.

## Execution Order

```text
config + raw data
        |
        v
source split creation
        |
        +----> optional zoom derivation from persisted controlled split
        |
        v
preprocessing and feature creation
        |
        v
training and prediction generation
        |
        v
interactive visualization and evaluation
```

See the package READMEs for commands, exact schemas, caches, failure behavior, and model details. See the repository [knowledge base](../KNOWLEDGE_BASE.md) for an implementation-independent account of the full research pipeline.
