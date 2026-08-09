# Fishometry

Fishometry is a master's thesis project for estimating fish length from a single image without requiring a ruler or another physical reference object in the frame. The project compares controlled laboratory photographs with heterogeneous outdoor photographs and evaluates how geometry, relative depth, segmentation, scene context, species information, and image features affect length prediction.

The repository contains four manually sequenced pipelines:

1. Create stable train, validation, and test assignments.
2. Derive the zoom-augmented controlled dataset from the persisted controlled split.
3. Preprocess images into detections, image artifacts, and engineered features.
4. Train an experiment matrix and inspect its predictions in Streamlit.

There is no all-in-one orchestrator. Each stage consumes artifacts produced by the preceding stage.

## Documentation

- [Pipeline knowledge base](KNOWLEDGE_BASE.md): implementation-independent explanation of the research workflow, outputs, experiments, and interpretation
- [Configuration reference](configs/README.md): config schema, current datasets, paths, validation, and experiment switches
- [Source overview](src/README.md): package boundaries and handoff contracts
- [Create data](src/create_data/README.md): split creation and zoom dataset workflow
- [Create data steps](src/create_data/steps/README.md): split and augmentation mechanics
- [Preprocessing](src/preprocessing/README.md): end-to-end computer-vision and feature pipeline
- [Preprocessing steps](src/preprocessing/steps/README.md): behavior and artifacts of every preprocessing stage
- [Training](src/training/README.md): experiment orchestration and prediction-table contract
- [Training models](src/training/models/README.md): model architectures, features, fitting, and checkpoints
- [Visualization](src/visualization/README.md): Streamlit application requirements and operation
- [Visualization views](src/visualization/views/README.md): filters, metrics, charts, and image views

The READMEs under `third_party/` belong to the vendored project and are not Fishometry documentation.

## Dataset Variants

| Dataset | Purpose | Raw metadata | Split behavior | Training additions |
| --- | --- | --- | --- | --- |
| `data-inside` | Controlled laboratory images | `name`, `length` | One global split | Eye and coordinate feature experiments |
| `data-inside-zoom` | Original, zoom-in, and zoom-out versions of controlled images | Source metadata plus inherited split flags | Each three-image family keeps the persisted `data-inside` assignment | Same experiment families as controlled data |
| `data-outside` | Outdoor images from multiple fish types | `fish_type`, `name`, `length` | Split independently within each fish type | Global, per-fish-type, context-rich, and embedding experiments |

Dataset and runtime artifacts are intentionally ignored by Git. Do not assume another checkout contains the local images, generated tables, caches, or checkpoints present on a development machine.

## Prerequisites

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The Depth Anything V2 Git submodule
- YOLO, Segment Anything, and Depth Anything checkpoints referenced by the selected config
- A CUDA-capable GPU is strongly recommended for preprocessing and neural training, although CPU execution is supported by most stages
- `.env.json` with a `GEMINI_API_KEY` entry when a configured `features` experiment requires original-image context

From the repository root:

```bash
git submodule update --init --recursive
uv sync
```

Place model checkpoints at the locations declared in the dataset configs. Configuration loading verifies the dataset folder, raw image folder, and raw metadata table, but individual preprocessing stages validate their own model files.

## Required Dataset Layout

Before loading a source dataset config, create:

```text
data/<dataset>/
|-- raw/
|   `-- <image files>
`-- raw.csv
```

`raw.csv` requires one row per image:

- `name`: image filename relative to `raw/`; names are expected to be unique.
- `length`: numeric ground-truth fish length, using one consistent unit within the dataset.
- `fish_type`: required only when the config enables fish-type-aware behavior.

Required metadata values are validated before splitting. Names must be safe, unique relative paths; every referenced image must exist, decode successfully, and have unique file content.

## End-to-End Workflow

Run commands from the repository root because configs and data paths are relative to it.

### 1. Create source splits

```bash
uv run python -m src.create_data.run --dataset-name data-inside
uv run python -m src.create_data.run --dataset-name data-outside
```

Each command replaces that dataset's `split.csv`. Re-running split creation can change assignments, so treat the persisted split as an experiment input once downstream work begins.

### 2. Create the zoom-derived dataset

```bash
uv run python -m src.create_data.run --dataset-name data-inside --augment
```

Augmentation requires `data/data-inside/split.csv` and the predefined `data-inside-zoom` config. It does not split the controlled data again. Every readable source image yields an original copy, one zoom-in image, and one zoom-out image with identical labels and split flags.

### 3. Preprocess each dataset

```bash
uv run python -m src.preprocessing.run --dataset-name data-inside
uv run python -m src.preprocessing.run --dataset-name data-inside-zoom
uv run python -m src.preprocessing.run --dataset-name data-outside
```

Preprocessing runs only the expensive optional stages required by the experiment contract: depth is skipped when every configured depth flag is false, and original-image context is skipped when `features` is absent. Detection, rotation, segmentation, blackout generation, and final feature engineering remain required by the core model matrix.

### 4. Train experiments

```bash
uv run python -m src.training.run --dataset-name data-inside
uv run python -m src.training.run --dataset-name data-inside-zoom
uv run python -m src.training.run --dataset-name data-outside
```

Training writes checkpoints, predictions, metrics, and a content manifest into an immutable run directory. Canonical predictions and reports, followed by `checkpoints/<dataset>/current.json`, are replaced only after the full matrix succeeds.

### 5. Explore results

```bash
uv run python -m streamlit run src/visualization/app.py
```

The app defaults to metrics over all available splits. Select `test` explicitly when reporting held-out performance.

## Artifact Flow

Each dataset is stored under `data/<dataset>/`.

| Artifact | Producer | Contents or purpose |
| --- | --- | --- |
| `raw/` | Dataset preparation or augmentation | Original input images |
| `raw.csv` | Dataset preparation or augmentation | Ground-truth metadata; augmented metadata also carries inherited split flags |
| `split.csv` | Create-data pipeline | Metadata plus mutually exclusive `is_train`, `is_val`, and `is_test` flags |
| `processed/cache/` | Preprocessing | Reusable YOLO and original-image context responses |
| `processed/rotated/` | Preprocessing | Horizontally aligned fish images when rotation is enabled |
| `processed/depth/` | Preprocessing | Cached relative-depth arrays |
| `processed/segment/` | Preprocessing | Cached binary segmentation masks |
| `processed/blackout/` | Preprocessing | Isolated fish images centered on a 224 by 224 black canvas |
| `processed.csv` | Preprocessing | Labels, split flags, detections, requested depth/context values, mask geometry, and engineered features |
| `processed/preprocessing_report.json` | Preprocessing | Per-stage input, output, and dropped-name attrition |
| `checkpoints/<dataset>/runs/<run-id>/` | Training | Versioned model checkpoints and a content-hashed run manifest |
| `checkpoints/<dataset>/cache/` | Training | Content-validated image embedding caches |
| `predictions.csv` | Training | Atomically published identifiers, labels, split flags, and one prediction column per completed experiment |

Processed and prediction row counts can be smaller than split row counts. Detection, missing-image, depth, segmentation, context, or null-feature failures remove records at several handoffs.

## Cache and Rerun Policy

Every reusable preprocessing and embedding artifact has a sidecar or companion manifest covering ordered image identities and content hashes plus relevant checkpoint, parameter, prompt, model-revision, and feature-schema values. A mismatch recomputes the artifact instead of silently reusing it.

There is no force or clean mode. Orphaned artifacts from older inputs are reported or left in place rather than pruned automatically; immutable training run directories preserve complete historical fits. Keep the persisted split stable when comparing experiments.

## Result Interpretation

- Training rows fit model parameters.
- Validation rows guide selected tree and neural models.
- Test rows are held out from fitting and model selection.
- Predictions are generated for all surviving rows, which lets the app display any split.
- Training writes MAE, MAPE, RMSE, and R² for train, validation, and test; the visualization layer recalculates metrics for interactive subsets.
- Outdoor results can be compared globally, by fish type, and across global versus per-type estimators.

See [the knowledge base](KNOWLEDGE_BASE.md) for the research rationale and a code-independent description of every stage.
