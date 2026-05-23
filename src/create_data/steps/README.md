# Create Data Steps

This folder contains reusable components for dataset creation.

## `base.py`

Defines the `PipelineStep` interface used by create-data steps. Each step receives a Polars dataframe and returns an updated dataframe and config.

## `split.py`

`SplitStep` shuffles metadata and adds boolean split columns:

- `is_train`
- `is_val`
- `is_test`

When fish types are available, the step splits each fish type separately before concatenating the result.

## `augment.py`

`AugmentStep` creates a zoom-augmented dataset from the source dataset.

For each source image it writes:

- the original image copy,
- one zoom-in image,
- one zoom-out image.

The generated rows copy the source metadata, so length labels and split flags are preserved for the augmented examples.
