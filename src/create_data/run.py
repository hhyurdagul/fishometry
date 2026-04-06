"""
Data creation pipeline runner

Usage:
    python -m src.preprocessing.run data-inside
    python -m src.preprocessing.run data-inside --augment
    python -m src.preprocessing.run data-outside
"""

import os
import polars as pl
import typer
from src.config import get_config
from src.create_data.steps.split import SplitStep
from src.create_data.steps.augment import AugmentStep

app = typer.Typer(add_completion=False, help="Run the data creation pipeline.")


def run_pipeline(config, augment):
    print(f"Starting data creation pipeline for {config.name}...")
    config.output_dir.mkdir(exist_ok=True)

    # Initialize data
    df = pl.read_csv(config.input_csv_path).drop_nulls()
    steps = [
        SplitStep(config)
    ]

    if augment:
        steps.append(AugmentStep(config))

    # Run pipeline
    for step in steps:
        print(f"Running {step.__class__.__name__}...")
        df = step.process(df)

    # Save final result
    df.write_csv(config.output_csv_path)
    print(f"Data creation pipeline for {config.name} finished.")
    print(f"Saved processed data to {config.output_csv_path}")

@app.command()
def main(
    dataset_name: str = typer.Option(..., help="Dataset name"),
    augment: bool = typer.Option(False, help="Augment data"),
):
    config = get_config(dataset_name)

    run_pipeline(config, augment)


if __name__ == "__main__":
    app()
