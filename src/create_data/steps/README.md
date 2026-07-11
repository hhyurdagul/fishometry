# Create Data Steps

The create-data stages share a small dataframe-in, dataframe-out contract. The
runner decides which stage is valid for the selected mode: ordinary creation
uses only splitting, while augmentation reads an existing split and uses only
augmentation. The two stages are intentionally not chained for an augmentation
run.

## Base Step Contract

`base.py` defines the common pipeline-step interface. A step receives the
current dataset configuration at construction and its `process` operation
returns:

1. the transformed Polars dataframe; and
2. the configuration that owns the returned artifacts.

For a split, the returned configuration is the source configuration. For
augmentation, it is the predefined zoom-dataset configuration.

## Split Step

`split.py` assigns every input row to one of three experiment partitions.

### Input

The runner supplies the source `raw.csv` after dropping rows that contain nulls.
The dataframe retains every remaining source metadata column.

### Operation

For each frame being split, the step:

1. calculates `int(row_count * train_ratio)` training rows;
2. calculates `round(row_count * val_ratio)` validation rows;
3. shuffles the rows;
4. initializes all split flags to `false`;
5. marks the first calculated range as training;
6. marks the next calculated range as validation;
7. marks every remaining row as test; and
8. shuffles the concatenated result again.

The three output flags are `is_train`, `is_val`, and `is_test`. The test size is
the remainder after training and validation, so it absorbs rounding differences.

If the dataset configuration enables fish types, each distinct `fish_type`
subset is split independently. The independently assigned subsets are then
combined. This is categorical stratification by separate calculation rather
than a guarantee that every small category receives at least one row in every
partition.

### Output and Rerun Behavior

The runner writes the returned dataframe to the source dataset's `split.csv`.
Running the command again replaces the previous manifest. Consumers that need a
fixed experimental partition should preserve and reuse the existing manifest.

## Augmentation Step

`augment.py` materializes the zoom dataset. It is constructed with both the
source configuration and the already loaded destination configuration.

### Preconditions Owned by the Runner

Before the step starts, the runner has already:

- confirmed that the source `split.csv` exists;
- loaded that persisted manifest instead of invoking the split step;
- checked required metadata, unique image names, boolean flags, and exactly one
  active split per row;
- rejected unsafe source paths and any collision among planned original and
  generated names;
- required a predefined `<source>-zoom` configuration; and
- confirmed that the configuration declares the expected destination name.

These checks happen before the augmentation step creates the destination raw
image directory.

### Image Transformations

For each readable source image, the step writes:

1. **Original:** the decoded pixels re-encoded under the source image name.
2. **Zoom in:** a centered crop produced from a random 0.20-0.50 magnitude,
   resized back to the source width and height.
3. **Zoom out:** the complete image resized to the smaller width and height
   implied by a separate random 0.20-0.50 magnitude.

Pixel dimensions use integer truncation and never fall below one pixel. Zoom-in
names contain `-zin-<integer-percent>` and zoom-out names contain
`-zout-<integer-percent>` before the original extension.

The step owns a local random generator seeded with `42`. It does not modify the
process-wide random state, and each new augmentation run starts from the same
state. An unchanged source manifest order therefore produces the same output
names on every run.

If a source path is missing or its image cannot be decoded, the step logs the
condition and skips the complete three-image group. An image-write failure is a
hard error rather than a silently retained metadata row.

### Metadata and Artifacts

Each of the three rows is a copy of its source row with only `name` changed for
the augmented variants. Length, optional fish type, additional columns, and all
split flags remain identical within the group.

The step writes the returned dataframe to the destination `raw.csv`; the runner
writes that same dataframe to destination `split.csv`. It returns the destination
configuration so later orchestration resolves paths against the derived dataset.

The step does not write any configuration and does not modify the source split.
It creates parent directories as needed and overwrites the deterministic files
it owns, but performs no directory cleanup. Unreferenced legacy artifacts are
left in place for deliberate review or removal.

## Zoom Helper

The reusable zoom helper accepts an image, a direction, and a magnitude. It
clamps magnitudes into the safe interval from zero to 0.90. `in` performs the
center-crop-and-resize operation; `out` performs dimension reduction. An
unrecognized direction returns the input unchanged.

See the parent [`README.md`](../README.md) for commands, dataset layouts,
validation rules, and downstream handoff.
