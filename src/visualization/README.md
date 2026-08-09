# Visualization App

The visualization package is a read-only Streamlit application for inspecting a
dataset after preprocessing and comparing the wide prediction table produced by
training. It does not run preprocessing or training, modify artifacts, or read a
dataset configuration. What appears in the interface is determined entirely by
the directories and tables already present under `data/`.

## Launch

Install the project dependencies and start the application from the repository
root:

```bash
uv sync
uv run python -m streamlit run src/visualization/app.py
```

Running from the repository root is required because all data paths are resolved
relative to the current working directory. Streamlit prints the local browser URL
after it starts.

## Dataset Discovery

The dataset selector lists every immediate subdirectory of `data/`. Discovery
does not check for a matching configuration or validate the directory contents,
so a partially generated dataset can appear in the selector even when some views
cannot use it.

For the selected dataset, the app tries to load metadata in this order:

1. `data/<dataset>/processed.csv`
2. `data/<dataset>/processed/processed_train.csv`
3. `data/<dataset>/processed/processed_val.csv`
4. `data/<dataset>/processed/processed_test.csv`

The combined `processed.csv` takes precedence. When it is absent or unreadable,
the available split-specific files are combined and receive a `split` column.
Each metadata table must contain `name` for the app to enumerate images. A
metadata warning in the sidebar does not prevent prediction-only views from
working.

Predictions are loaded independently from:

```text
data/<dataset>/predictions.csv
```

When a prediction table exists, Data Explorer uses its `name` column as the image
list and offers split and fish-type filters. Without predictions, it falls back
to the names in processed metadata. Raw image files alone are not scanned to
build the selector; at least one of these tables must provide image names.

## Prediction Table Contract

The prediction table is a wide table with one row per image. It must contain:

- `name`: image filename and row identifier.
- `length`: ground-truth fish length.
- one or more numeric model columns containing predicted lengths.

It can also contain `fish_type`, `is_train`, `is_val`, and `is_test`. These six
identifier and grouping fields are excluded when discovering models; every other
column is treated as a model prediction column. Keep unrelated metadata out of
this table, because it would otherwise be interpreted as predictions.

Split choices are shown only for boolean split columns that contain at least one
`true` row. Fish-type controls and the two fish-type modes appear only when
`fish_type` contains at least one non-null value. Null ground-truth or prediction
values are removed before metrics are calculated, separately for each model, so
sample counts can differ between models.

## Image Artifact Resolution

For a selected image, the app looks for the original in the following order:

1. `data/<dataset>/raw/<name>`
2. `data/<dataset>/splits/<name>`
3. `data/<dataset>/<name>`

The latter two locations are compatibility fallbacks. If no original is found,
a black placeholder is displayed.

The remaining image products are resolved beneath `processed/`:

- `rotated/<name>` for the normalized image. When processed metadata contains
  fish, head, or tail detections, the display adds a labeled fish rectangle, a
  red head-center point, and a green tail-center point.
- `blackout/<name>` for the fish-only image, with common JPEG/PNG extension
  fallbacks.
- `depth/<depth-model>/<stem>.npy` for a model-specific depth map.
- `depth/<stem>.npy` as the root-level depth fallback.

Immediate subdirectories of `processed/depth/` populate the sidebar Depth Model
selector. The selected model is checked first; if its array is unavailable, the
app tries the root-level depth layout. A dataset that only uses root-level arrays
will not show a depth-model selector, but those arrays can still be displayed.
Depth arrays are normalized independently to an 8-bit range and rendered with a
magma color map. Consequently, colors show within-image relative depth and are
not a shared numeric scale for comparing different images.

Missing rotated, depth, and blackout products are reported in their display
positions and do not stop the rest of the view.

## Modes

### Data Explorer

Available whenever a dataset directory is discovered. It requires image names
from either predictions or processed metadata. Prediction-backed datasets can be
filtered by split and fish type. The selected sample is shown as a two-row,
two-column grid containing raw, rotated/annotated, depth, and blackout images.
An expanded JSON panel shows all non-null metadata for the image and, when
available, a nested map of every model prediction.

### Prediction Visualization

Requires a valid prediction table. This is an aggregate view, not a single-image
viewer. It selects one model and optionally filters rows by split and fish type,
then displays:

- sample count, MAE, R2, and MAPE;
- an interactive predicted-versus-actual scatter plot, colored by each sample's
  percentage error and overlaid with the perfect-prediction diagonal;
- actual and predicted values as paired lines, sorted by descending percentage
  error or ascending actual length;
- absolute-error and percentage-error histograms;
- mean percentage error across 3 to 10 equal-width actual-length bins, together
  with each bin's range, count, mean, and standard deviation.

### Error Analysis

Requires a valid prediction table. It applies the same model, split, and
fish-type filters and reports aggregate metrics. A deep-dive table ranks samples
by descending percentage or absolute error. Selecting a ranked sample shows its
ground truth, prediction, error, and a three-column raw, rotated/annotated, and
depth image layout. Processed metadata enriches annotations but is optional.
Blackout images are not shown in this mode.

### Correlation Analysis

Available whenever processed metadata exists. It lets the user select numeric geometry, depth, and species-statistic fields, then reports correlations with length, a feature/target heatmap, scatter plots with regression lines, and descriptive statistics. At least ten complete rows are required for the selected columns.

### Model Comparison

Requires a valid prediction table. It evaluates every discovered model over the
selected split and fish-type subset. The leaderboard is ordered by increasing
MAPE and includes MAPE, MAE, R2, and sample count. A bar chart can visualize any
of the three metrics. Because missing predictions are dropped per model, use the
sample-count column when deciding whether metric rows are directly comparable.

### Fish Type Comparison

Shown only when non-null fish types are present. It compares one selected model
across fish types for the chosen split. The table includes an `All` aggregate and
one row per type; the bar chart includes the individual types and can display
MAPE, MAE, or R2.

### Model x Fish Type Heatmap

Shown only when non-null fish types are present. It calculates a metric for every
selected model and fish-type pair. All models are selected initially. The view
offers MAPE, MAE, and R2 heatmaps with values written into each cell, tooltips
containing all metrics and sample counts, and a pivoted metric table below.

## Split Defaults and Metric Interpretation

Every split selector defaults to `all`. In that state, charts and metrics use all
rows in the prediction table, normally combining training, validation, and test
samples. The app does not default to held-out evaluation. Select `test` explicitly
when reporting final generalization performance, and select `val` when examining
validation behavior.

All metrics are recalculated over the currently selected model, split, and
fish-type rows:

- **MAE** is the mean absolute difference between predicted and ground-truth
  length, in the same unit as `length`.
- **MAPE** is the mean absolute error divided by ground-truth length and expressed
  as a percentage. Ground-truth lengths must be nonzero for this metric to be
  meaningful.
- **R2** is `1 - residual sum of squares / total sum of squares`. It can be
  negative when predictions are worse than using the subset mean. The app reports
  `0` when every selected ground-truth length is identical.
- **Samples** is the number of non-null ground-truth/prediction pairs included.

The error-analysis table also uses signed residual internally as prediction minus
ground truth, absolute error as its magnitude, and per-sample percentage error as
absolute error divided by ground truth.

## Cache Refresh

Dataset discovery is evaluated on each rerun. Processed and prediction CSV caches include file size and nanosecond modification time, so atomically published replacements are loaded automatically. A manual Streamlit cache clear remains useful only for external files whose own loader does not yet carry a version key.

## Correlation View

`Correlation Analysis` is a standard sidebar mode and receives the selected dataset's processed metadata. The two fish-type comparison modes remain conditional on non-null `fish_type` values.

## Troubleshooting

- **No datasets found:** start the app from the repository root and confirm that `data/` contains a dataset with `processed.csv` or `predictions.csv`.
- **Metadata warning:** generate a readable `processed.csv` containing a `name` column.
- **No prediction CSV or no model columns:** run training and verify the wide
  prediction-table contract above.
- **No images in Data Explorer:** verify that `name` exists in predictions or
  processed metadata and that the current filters retain at least one row.
- **No depth selector:** model-specific depth subdirectories were not found. A
  compatible root-level depth array can still render without the selector.

See [views/README.md](views/README.md) for the component-level behavior of each
screen.
