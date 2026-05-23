# Create Data Pipeline

This package creates dataset splits and optionally builds the zoom-augmented inside dataset.

## Entrypoint

```bash
uv run python -m src.create_data.run --dataset-name data-inside
uv run python -m src.create_data.run --dataset-name data-outside
uv run python -m src.create_data.run --dataset-name data-inside --augment
```

## Inputs

The pipeline reads the selected config from `configs/<dataset-name>.json`, then reads:

- `data/<dataset-name>/raw.csv`
- `data/<dataset-name>/raw/`

## Outputs

- `data/<dataset-name>/split.csv`: metadata with `is_train`, `is_val`, and `is_test`.
- When `--augment` is used on `data-inside`, the pipeline creates a `data-inside-zoom` style dataset with copied original images plus zoom-in and zoom-out variants.

## Split Logic

The split ratios come from the config `params` section. If `fish_type_available` is enabled, splitting is done separately for each fish type so each type contributes train, validation, and test rows.

For the zoom dataset, augmented rows keep the metadata of the source row, including its split flags. This lets `data-inside-zoom` use the same conceptual split as `data-inside`.
