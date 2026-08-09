# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Fishometry is an MSc thesis project that estimates fish length from a single image with no ruler or
reference object in frame. Three dataset variants are compared: `data-inside` (controlled lab photos),
`data-inside-zoom` (zoom-in/zoom-out derivatives of the controlled images), and `data-outside`
(multi-species outdoor photos). Work is split into four **manually sequenced** pipelines — there is no
all-in-one orchestrator, and each stage consumes files written by the previous one.

## Setup

```bash
git submodule update --init --recursive   # third_party/Depth-Anything-V2
uv sync
```

Also required before preprocessing: `.env.json` at the repo root containing `GEMINI_API_KEY` (imported
unconditionally, even when rotation prevents new context calls), and the YOLO / SAM / Depth-Anything
checkpoints at the paths named in `configs/<dataset>.json`. `data/`, `checkpoints/`, and `reports/` are
gitignored — a fresh checkout has source only.

## Commands

Run everything from the repository root; all paths resolve relative to CWD.

```bash
# 1. splits (rewrites data/<dataset>/split.csv)
uv run python -m src.create_data.run --dataset-name data-inside
uv run python -m src.create_data.run --dataset-name data-outside

# 2. zoom dataset, derived from the persisted data-inside split
uv run python -m src.create_data.run --dataset-name data-inside --augment

# 3. preprocessing (always runs the full stage list)
uv run python -m src.preprocessing.run --dataset-name data-inside

# 4. training (runs the whole experiment matrix; no model selection flag)
uv run python -m src.training.run --dataset-name data-outside

# 5. inspect results
uv run python -m streamlit run src/visualization/app.py
```

Tests are stdlib `unittest` — `pytest` is **not** a dependency and is not installed in `.venv`:

```bash
uv run python -m unittest discover -s tests            # all
uv run python -m unittest tests.test_training          # one module
uv run python -m unittest tests.test_training.PerFishGuardTests.test_small_fish_types_are_skipped
```

Thesis build (see `Thesis/BUILD.md`), driven entirely off `data/*/predictions.csv`:

```bash
uv run python Thesis/scripts/make_metrics.py && \
uv run python Thesis/scripts/make_figures.py && \
uv run python Thesis/scripts/make_appendices.py && \
uv run python Thesis/scripts/build.py && \
uv run python Thesis/scripts/check_numbers.py
```

## Architecture

Stages communicate through files under `data/<dataset>/`, joined on the `name` column:

`raw.csv` + `raw/` → **create_data** → `split.csv` → **preprocessing** → `processed.csv` (+ `processed/`
image artifacts) → **training** → `predictions.csv` + `checkpoints/<dataset>/` + `reports/<dataset>/` →
**visualization** (read-only Streamlit).

- `src/config.py` is the only config layer. A pydantic `Config` is loaded from `configs/<name>.json` and
  derives every dataset path as a computed field. Loading validates that `data/<name>/raw/` and `raw.csv`
  exist — nothing else (not checkpoints, not columns, not split files).
- Every stage is a class with `process(df) -> df` (`src/preprocessing/steps/*.py`) or
  `process(df) -> (df, config)` (`src/create_data/steps/base.py`). `run.py` builds the step list and
  threads one Polars dataframe through it. Rows that fail a stage are dropped, so row counts shrink
  down the chain.
- Training expands `dataset.feature_sets × dataset.depth` over four model families (linear, XGBoost, MLP,
  blackout-image CNN), plus a per-fish-type pass when `fish_type_available`, plus the EfficientNet/Ridge
  and DINOv2/Ridge embedding models for outdoor data. Prediction and checkpoint names share one scheme:
  `<model>_<feature-set>[_depth][_per_type]`.
- The visualization app never reads a config: it enumerates `data/` subdirectories and computes metrics
  itself from `predictions.csv`.

## Non-obvious behavior

- **`name` is the global join key** across every table and image directory; duplicates silently corrupt
  joins. Nothing enforces uniqueness after `create_data`.
- **`src/training/run.py` calls `drop_nulls()` on the whole `processed.csv`**, so a null in a column no
  configured experiment uses still removes the row from every experiment.
- **The embedding models are gated on the literal string `"data-outside"`** in `src/training/run.py`.
  A new outdoor dataset under a different name silently skips them.
- **Per-type checkpoints overwrite each other.** All fish types of one experiment write the same
  `*_per_type.pth`/`.joblib`, so only the last-fitted type survives on disk. The prediction column is
  still correct (predictions are combined in memory).
- **Cache invalidation is inconsistent.** Preprocessing caches (`processed/cache/`, `rotated/`, `depth/`,
  `segment/`, `blackout/`) and the EfficientNet `.npy` embedding cache are keyed by filename or mere
  existence — they go stale silently after image, checkpoint, or config changes. Only the DINOv2 caches
  carry a `_names.json` manifest and re-extract on mismatch. Delete the EfficientNet cache whenever
  outdoor rows are added, removed, or reordered.
- **There is no force or clean mode anywhere.** Deliberate recomputation means deleting the specific
  artifacts by hand; orphaned files from older runs are never pruned.
- **Re-running split creation reshuffles assignments**, invalidating comparisons against existing
  `processed.csv`/`predictions.csv`. Treat a persisted `split.csv` as an experiment input.
- Training replaces checkpoints as each fit completes but writes `predictions.csv` only after the whole
  matrix succeeds, so a failed run leaves new checkpoints beside a stale prediction table.

## Documentation

These are the authoritative deep references; keep them in sync when changing behavior.

| Doc | Covers |
| --- | --- |
| `KNOWLEDGE_BASE.md` | Implementation-independent account of the research pipeline, experiments, and interpretation |
| `README.md` | End-to-end workflow, dataset variants, artifact flow |
| `configs/README.md` | Config schema, feature-set names, detector class ordering, split ratio math |
| `src/README.md` | Package boundaries and handoff contracts |
| `src/{create_data,preprocessing,training,visualization}/README.md` (+ their `steps/`, `models/`, `views/`) | Per-stage mechanics, schemas, failure modes |
| `Thesis/BUILD.md` | Thesis generation and the manual Word steps |

READMEs under `third_party/` belong to the vendored project, not to Fishometry.

