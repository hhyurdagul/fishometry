"""Data creation pipeline runner."""

import json

import polars as pl
import typer
from pydantic import TypeAdapter

from src.config import (
    CONFIG_ROOT,
    Config,
    DatasetConfig,
    ModelConfig,
    ParamConfig,
    get_config,
)
from src.create_data.steps.split import SplitStep
from src.create_data.steps.augment import AugmentStep, resolve_image_path

app = typer.Typer(add_completion=False, help="Run the data creation pipeline.")

SPLIT_COLUMNS = ("is_train", "is_val", "is_test")


def _validate_source_split(df: pl.DataFrame, config: Config) -> None:
    required_columns = {"name", "length", *SPLIT_COLUMNS}
    if config.dataset.fish_type_available:
        required_columns.add("fish_type")

    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "Source split is missing required columns: " + ", ".join(missing_columns)
        )

    null_columns = sorted(
        column for column in required_columns if df[column].null_count() > 0
    )
    if null_columns:
        raise ValueError(
            "Source split contains null required values in: " + ", ".join(null_columns)
        )

    if df["name"].n_unique() != df.height:
        raise ValueError("Source split image names must be unique")

    for name in df["name"].to_list():
        resolve_image_path(config.dataset.input_dir, name)

    non_boolean_columns = [
        column for column in SPLIT_COLUMNS if df.schema[column] != pl.Boolean
    ]
    if non_boolean_columns:
        raise ValueError(
            "Source split flags must be boolean columns: "
            + ", ".join(non_boolean_columns)
        )

    split_count = pl.sum_horizontal(
        [pl.col(column).cast(pl.Int8) for column in SPLIT_COLUMNS]
    )
    invalid_rows = df.filter(split_count != 1)
    if invalid_rows.height:
        raise ValueError(
            "Every source row must belong to exactly one of train, validation, or test"
        )


def _load_augmentation_config(source_config: Config) -> Config:
    """Load a target config without requiring its data directory to exist yet."""
    target_name = f"{source_config.dataset.name}-zoom"
    config_path = (CONFIG_ROOT / target_name).with_suffix(".json")
    if not config_path.exists():
        raise ValueError(f"Config `{target_name}` does not exist")

    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("dataset"), dict):
        raise ValueError(f"Config `{target_name}` has an invalid dataset section")

    dataset_values = {}
    for field_name, field_info in DatasetConfig.model_fields.items():
        if field_name in payload["dataset"]:
            adapter = TypeAdapter(field_info.annotation)
            dataset_values[field_name] = adapter.validate_python(
                payload["dataset"][field_name]
            )
        elif field_info.is_required():
            raise ValueError(
                f"Config `{target_name}` is missing dataset field `{field_name}`"
            )
        else:
            dataset_values[field_name] = field_info.get_default(
                call_default_factory=True
            )

    dataset_config = DatasetConfig.model_construct(**dataset_values)
    if dataset_config.name != target_name:
        raise ValueError(
            f"Config `{target_name}` must define dataset name `{target_name}`"
        )

    try:
        model_config = ModelConfig.model_validate(payload.get("model_path"))
        param_config = ParamConfig.model_validate(payload.get("params"))
    except Exception as error:
        raise ValueError(f"Config `{target_name}` is invalid: {error}") from error

    return Config.model_construct(
        dataset=dataset_config,
        model_path=model_config,
        params=param_config,
    )


def run_pipeline(config: Config, augment: bool) -> None:
    print(f"Starting data creation pipeline for {config.dataset.name}...")

    if augment:
        if not config.dataset.split_csv_path.exists():
            raise FileNotFoundError(
                "Augmentation requires an existing source split at "
                f"{config.dataset.split_csv_path}"
            )

        df = pl.read_csv(config.dataset.split_csv_path)
        _validate_source_split(df, config)
        target_config = _load_augmentation_config(config)

        print("Running AugmentStep...")
        df, target_config = AugmentStep(config, target_config).process(df)
        df.write_csv(target_config.dataset.split_csv_path)
        print(f"Data creation pipeline for {target_config.dataset.name} finished.")
        print(f"Saved processed data to {target_config.dataset.split_csv_path}")
        return

    df = pl.read_csv(config.dataset.input_csv_path).drop_nulls()
    step = SplitStep(config)
    print(f"Running {step.__class__.__name__}...")
    df, config = step.process(df)

    df.write_csv(config.dataset.split_csv_path)
    print(f"Data creation pipeline for {config.dataset.name} finished.")
    print(f"Saved processed data to {config.dataset.split_csv_path}")


@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Dataset name"),
    augment: bool = typer.Option(False, help="Augment data"),
):
    config = get_config(dataset_name)

    run_pipeline(config, augment)


if __name__ == "__main__":
    app()
