# Create Data Pipeline

The create-data pipeline prepares the metadata that every later stage uses. It
has two independent modes:

1. assign source observations to training, validation, and test sets; or
2. derive a zoom-augmented dataset from an already persisted source split.

Augmentation never creates a new split. This is an important lineage rule: an
original image and both images derived from it always remain in the same split.

## Run From the Repository Root

```bash
uv run python -m src.create_data.run --dataset-name data-inside
uv run python -m src.create_data.run --dataset-name data-outside
uv run python -m src.create_data.run --dataset-name data-inside --augment
```

The commands use paths relative to the repository root. The selected dataset
configuration and the source dataset directory must already be present.

## Source Dataset Contract

A source dataset contains:

```text
data/<dataset-name>/
|-- raw/
|   `-- <image files>
`-- raw.csv
```

`raw.csv` must contain one row per image. The common metadata fields are:

| Column | Meaning |
| --- | --- |
| `name` | Image name relative to the dataset's `raw/` directory. |
| `length` | Ground-truth fish length used as the prediction target. |
| `fish_type` | Fish category; required when `fish_type_available` is enabled. |

Extra metadata columns are carried through the split and augmentation outputs.
Image names must be non-empty normalized relative paths. Absolute paths, parent
directory traversal, and paths that resolve outside `raw/` are rejected before
augmentation writes destination artifacts.

## Split Mode

Run split mode without `--augment`:

```bash
uv run python -m src.create_data.run --dataset-name <dataset-name>
```

The pipeline reads `raw.csv`, removes every row containing a null value, shuffles
the remaining observations, and adds three boolean columns:

- `is_train`
- `is_val`
- `is_test`

For a group of `n` rows, the training count is the floor of
`n * train_ratio`, the validation count is `round(n * val_ratio)`, and all
remaining rows become test rows. Consequently, the configured `test_ratio` is
validated as part of the ratio set but the test count is the remainder rather
than a separately rounded count.

When `fish_type_available` is enabled, the calculation is performed separately
for every fish type and the groups are concatenated afterward. This preserves
representation by fish type, subject to the rounding limits of small groups.

The result is written to:

```text
data/<dataset-name>/split.csv
```

Running split mode again replaces that file with a newly generated assignment.
Any downstream preprocessing or model comparison that must share an assignment
should therefore use the persisted file rather than rerunning split mode.

## Augmentation Mode

Run augmentation only after the source split has been created:

```bash
uv run python -m src.create_data.run \
  --dataset-name <source-dataset> \
  --augment
```

The destination name is always `<source-dataset>-zoom`. A matching configuration
must already exist in `configs/`, and its declared dataset name must match that
destination exactly. The pipeline reads and validates both prerequisites before
creating or changing any destination artifact. It never creates, rewrites, or
normalizes the configuration file.

### Source Split Validation

The persisted source `split.csv` must satisfy all of these conditions:

- `name`, `length`, `is_train`, `is_val`, and `is_test` are present;
- `fish_type` is present when the source configuration enables fish types;
- required values are not null;
- image names are unique;
- every source and generated image path remains inside its dataset's `raw/`
  directory;
- all three split columns are boolean; and
- each row has exactly one split flag set to `true`.

Failure stops augmentation before the destination dataset is written. In
particular, the command does not fall back to splitting `raw.csv` when the source
split is missing.

The complete set of planned original, zoom-in, and zoom-out names must also be
unique. A generated name that collides with another source or generated name
stops the run before the destination directory is created.

### Images Produced

Every source image that can be decoded produces exactly three encoded files:

| Variant | Transformation | Dimensions |
| --- | --- | --- |
| Original | Decode and re-encode without geometric augmentation. | Same as source. |
| Zoom in | Keep a centered crop, then resize it to the source size. | Same as source. |
| Zoom out | Resize the complete image to smaller dimensions. | Smaller than source. |

Zoom magnitudes are drawn from 0.20 through 0.50. A zoom-in therefore retains
roughly 50% through 80% of each original dimension before resizing. A zoom-out
retains roughly 50% through 80% of each dimension in the output itself. Integer
rounding is applied to pixel dimensions, with a one-pixel minimum.

Generated names add `-zin-<percent>` or `-zout-<percent>` before the file
extension. A new local random generator is initialized with seed `42` on every
run, making magnitudes, suffixes, manifests, and image names stable when the
source manifest order is unchanged.

Missing files and files that cannot be decoded are reported and skipped. They do
not contribute an original or augmented metadata row.

### Metadata and Split Inheritance

All metadata on the source row is copied to each member of its three-image
group. This includes `length`, optional `fish_type`, extra metadata, and the
three split flags. The transformation therefore increases the observations
within each split without moving information between splits.

The destination contains:

```text
data/<source-dataset>-zoom/
|-- raw/
|   |-- <re-encoded originals>
|   |-- <zoom-in images>
|   `-- <zoom-out images>
|-- raw.csv
`-- split.csv
```

`raw.csv` and `split.csv` contain matching rows, columns, and assignments for all
successfully decoded source images.

## Preservation and Reruns

Augmentation treats the source split and destination configuration as immutable.
It overwrites the deterministic image paths and the two destination manifests,
but it does not delete any other file in the destination. Files produced by an
older naming scheme or a changed source split can therefore remain on disk even
when they are no longer referenced by the current manifests. Remove such stale
artifacts manually only after confirming they are no longer needed.

## Pipeline Handoff

Preprocessing consumes the persisted split manifest and raw images. For a zoom
dataset, run preprocessing with the destination dataset name after augmentation
has completed. Training then consumes preprocessing outputs while retaining the
same train, validation, and test assignment established here.

## Common Failures

| Failure | Resolution |
| --- | --- |
| Source split does not exist | Run split mode for the source dataset first. |
| Zoom configuration does not exist | Add and review the predefined destination configuration before augmentation. |
| Required column is absent or null | Repair the source metadata or regenerate the source split from valid raw metadata. |
| Split flags are strings or overlap | Store boolean flags and assign exactly one split per row. |
| Image is skipped | Confirm its relative name, file presence, format, and readability. |
| Old images remain after a rerun | Compare the destination directory with the current manifests and remove unreferenced files deliberately. |

Implementation details for the individual stages are documented in
[`steps/README.md`](steps/README.md).
