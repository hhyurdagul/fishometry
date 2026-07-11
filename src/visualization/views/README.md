# Visualization Views

This package contains the page renderers used by the Fishometry Streamlit app.
The application owns dataset, depth-model, and mode selection; each renderer owns
the controls and displays within one analysis screen. All screens are read-only.

## Shared Prediction Behavior

Prediction analyses consume the wide `predictions.csv` table. `name`,
`fish_type`, `length`, `is_train`, `is_val`, and `is_test` are treated as
identifier fields; every other column is treated as a model. Selected model
columns are normalized into model/sample rows with these derived fields:

- ground-truth length;
- predicted length;
- signed residual, calculated as prediction minus ground truth;
- absolute error;
- absolute percentage error.

Rows with a null ground truth or null prediction are dropped. Metrics are then
calculated on the remaining rows: MAE is mean absolute error, MAPE is mean
absolute percentage error, and R2 compares residual variation with variation in
the selected ground truths. R2 is set to zero when the selected ground truths
have no variation. Ground-truth values must be nonzero for percentage error to be
well-defined.

Split controls list `all` first, followed by each of train, validation, and test
whose boolean flag contains at least one true row. Therefore every view defaults
to all available rows, not the held-out test set. Fish-type lists contain the
sorted, unique, non-null values from the prediction table. An empty fish-type
multiselect means all types.

## Data Explorer

The explorer is the sample-level artifact browser.

### Inputs and filters

If predictions are available, their names define the selectable samples and the
view offers split and fish-type filters. If they are absent, the view uses image
names from processed metadata and does not show these filters. Processed metadata
is optional when predictions supply names, but without either source the view has
nothing to display.

### Displays

The selected sample is rendered in a stable two-by-two layout:

1. raw image;
2. rotated image with available fish, head, and tail annotations;
3. selected or fallback depth map;
4. blackout image.

The fish annotation is a labeled rectangle. Head and tail bounding boxes are
reduced to red and green center points. The depth map is min-max normalized per
sample and colorized with the magma palette. Missing raw images become a black
placeholder, while missing rotated, depth, or blackout products produce an
informational message in that grid position.

The expanded metadata JSON contains non-null fields from the processed row. When
the prediction table has a matching name, it also contains a `Predictions` object
mapping every discovered model to its value for that sample.

## Prediction Visualization

Despite its name, this renderer is the aggregate single-model dashboard. It does
not select or display an individual image.

The controls choose one model, one split, and optionally one or more fish types.
It presents four summary values: sample count, MAE, R2, and MAPE. The selected
filters apply consistently to all subsequent displays.

### Predicted versus actual

An interactive scatter plot places actual length on the horizontal axis and
predicted length on the vertical axis. Points are colored by the sample's
absolute percentage error and expose name, fish type, actual value, prediction,
and percentage error in the tooltip. A dashed green diagonal marks perfect
predictions.

### Sorted comparison

Actual and predicted lengths are drawn as two point-marked lines over a generated
sample index. The default ordering is descending percentage error, which places
the worst relative errors first. The alternative is ascending actual length.

### Error distributions

Side-by-side histograms show absolute error and absolute percentage error, each
using at most 30 bins.

### Error by length range

A slider selects 3 through 10 bins, defaulting to 5. Samples are divided into
equal-width bins based on actual length. A bar chart shows mean percentage error
for each observed range, and a table reports range, sample count, mean percentage
error, and its standard deviation.

## Error Analysis

The error-analysis renderer combines ranked prediction errors with image
artifacts. It requires predictions; processed metadata is optional.

One model, split, and optional fish-type subset drive the MAE, MAPE, R2, and
sample-count summary. The deep-dive table includes name, fish type, ground truth,
prediction, absolute error, and percentage error. It can be ordered by descending
percentage error, the default, or descending absolute error.

The image selector preserves that error ranking. For the selected sample, an
information panel reports ground truth, prediction, absolute error, and percentage
error. The image row then displays raw, rotated/annotated, and depth products in
three columns. The selected sidebar depth model is used when possible, followed
by the root-level depth fallback. Metadata supplies detection annotations when it
has a matching image row. This screen intentionally omits the blackout product.

## Model Comparison

The model-comparison renderer calculates metrics independently for every model
column after applying the selected split and optional fish-type filters.

The leaderboard is sorted by increasing MAPE and reports model, MAPE, MAE, R2,
and sample count. A metric selector switches the accompanying model bar chart
between MAPE, MAE, and R2; its tooltips retain all metrics and the row count.
Because null predictions are removed per model, the sample count must be checked
before comparing scores across incomplete columns.

## Fish Type Comparison

This view is routed only when the prediction table has at least one non-null
`fish_type`. It selects one model and one split, then calculates an `All` result
and a separate result for every available fish type. The metrics table contains
MAPE, MAE, R2, and sample count. The bar chart excludes the aggregate row and
compares individual types using the selected metric.

Here, fish type is the dataset's species/category label. The view compares
performance by that label; it does not infer species from the image or retrain a
species-specific model.

## Model x Fish Type Heatmap

This view also requires non-null fish types. The controls choose a split and a
set of models; the split defaults to `all`, and every model is selected by
default. Metrics are calculated for each selected model/fish-type pair.

A selector switches among MAPE, MAE, and R2. The heatmap places fish types on the
horizontal axis and models on the vertical axis, writes the selected metric into
each cell, and exposes all metrics plus sample count in tooltips. A pivoted table
of the selected metric follows the chart. Heatmap height expands with the number
of selected models.

## Correlation Analysis

The correlation renderer is implemented and exported from the views package, but
the application entry point neither imports it nor defines a sidebar route for
it. It is not an active mode in the standard Streamlit interface.

If routed in the future, it expects processed metadata with `length` and searches
for available bounding-box, scaled-dimension, head/tail, depth, and
species-statistic features. It drops rows containing nulls in the chosen analysis
columns and requires at least ten complete samples. Its displays comprise:

- each selected feature's correlation with length;
- a labeled feature/target correlation matrix;
- an interactive feature-versus-length scatter plot with a regression line and
  optional fish-type coloring;
- mean, standard deviation, minimum, maximum, and target correlation for each
  selected feature.

Exporting a renderer only makes it importable; it does not make it selectable in
Streamlit. The currently routed views are Data Explorer, Prediction
Visualization, Error Analysis, Model Comparison, and, when fish types exist,
Fish Type Comparison and Model x Fish Type Heatmap.

## Image and Depth Lookup

Both Data Explorer and Error Analysis use the shared image loader. Raw images are
checked in `raw/`, then the compatibility `splits/` location, then the dataset
root. Rotated and blackout images are read from their respective processed
directories.

Depth discovery and depth lookup are deliberately separate:

- subdirectories directly below `processed/depth/` become selectable depth-model
  names in the application sidebar;
- a selected model resolves arrays below its subdirectory;
- a missing model-specific array falls back to an array directly below
  `processed/depth/`;
- root-level arrays can be displayed even when there are no depth-model
  subdirectories and therefore no selector.

Model-specific and root-level arrays are expected as NumPy `.npy` data associated
with the image filename. Each loaded array is normalized independently for
display, so its colors must not be interpreted as calibrated cross-image depth.

## Cache Behavior

The loaders cache dataset names, metadata, predictions, depth-model directories,
and per-image prediction maps. When upstream steps create or replace artifacts
during an app session, clear Streamlit's cache from the application menu and
rerun, or restart the Streamlit process. Merely changing a widget or refreshing
the browser may reuse cached results.
