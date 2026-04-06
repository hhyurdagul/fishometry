"""
Dataset split utility.

This module enforces the dataset layout used by this project. Raw data must live
under `data/<dataset_name>/raw.csv`, and split files are always written to
`data/<dataset_name>/splits/`. If the dataset is not located in the `data`
folder with those exact names and folders, this script will not work.

Usage:
    python -m src.create_data.split --dataset data-inside
    python -m src.create_data.split --dataset data-outside
"""
from src.config import get_valid_configs

from pathlib import Path
from typing import Annotated 

import polars as pl
import typer

app = typer.Typer(add_completion=False, help="Split dataset into train/val/test.")

DATA_DIR = Path("data")
RAW_FILENAME = "raw.csv"
SPLITS_DIRNAME = "splits"
STRATIFY_DATASETS = {"data-outside"}


def save_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Save train, validation, and test splits under the enforced output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.write_csv(output_dir / "train.csv")
    val_df.write_csv(output_dir / "val.csv")
    test_df.write_csv(output_dir / "test.csv")

    print(f"Saved splits to {output_dir}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")


def compute_split_indices(
    size: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    """Compute split boundaries for train and validation sets."""
    train_end = int(size * train_ratio)
    val_end = int(size * (train_ratio + val_ratio))
    return train_end, val_end


def split_frame(
    df: pl.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Shuffle and split one dataframe into train, validation, and test frames."""
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    train_end, val_end = compute_split_indices(len(shuffled), train_ratio, val_ratio)

    train_df = shuffled.slice(0, train_end)
    val_df = shuffled.slice(train_end, val_end - train_end)
    test_df = shuffled.slice(val_end, len(shuffled) - val_end)
    return train_df, val_df, test_df


def split_dataset(
    dataset: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> None:
    """
    Split a dataset stored under the enforced project structure.

    Raw data must exist at `data/<dataset>/raw.csv`. Output is always written to
    `data/<dataset>/splits/`. Any dataset outside that structure is unsupported
    and will fail.
    """
    if not 0.0 < train_ratio < 1.0:
        raise typer.BadParameter("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise typer.BadParameter("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise typer.BadParameter("train_ratio + val_ratio must be less than 1.")

    dataset_dir = DATA_DIR / dataset
    raw_path = dataset_dir / RAW_FILENAME
    output_dir = dataset_dir / SPLITS_DIRNAME

    if not raw_path.exists():
        raise typer.BadParameter(
            f"Expected raw dataset at {raw_path}. "
            "Raw data must be located under data/<dataset>/raw.csv."
        )

    df = pl.read_csv(raw_path)

    if dataset in STRATIFY_DATASETS:
        if "fish_type" not in df.columns:
            raise typer.BadParameter(
                f"Dataset '{dataset}' requires a 'fish_type' column for stratified splitting."
            )

        train_parts = []
        val_parts = []
        test_parts = []

        for fish_type in df["fish_type"].unique().sort():
            group_df = df.filter(pl.col("fish_type") == fish_type)
            train_df, val_df, test_df = split_frame(
                group_df,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            train_parts.append(train_df)
            val_parts.append(val_df)
            test_parts.append(test_df)

        train_df = pl.concat(train_parts).sample(fraction=1.0, shuffle=True, seed=seed)
        val_df = pl.concat(val_parts).sample(fraction=1.0, shuffle=True, seed=seed)
        test_df = pl.concat(test_parts).sample(fraction=1.0, shuffle=True, seed=seed)
        print(f"Processing {dataset} with stratified splitting by fish_type...")
    else:
        train_df, val_df, test_df = split_frame(
            df,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        print(f"Processing {dataset} with random splitting...")

    save_splits(train_df, val_df, test_df, output_dir)


@app.command()
def main(
    dataset: Annotated[str, typer.Option(help="Dataset name under the data folder")] = "data-inside",
    seed: int = typer.Option(42, help="Random seed"),
    train_ratio: float = typer.Option(0.7, help="Training set ratio"),
    val_ratio: float = typer.Option(0.15, help="Validation set ratio"),
) -> None:
    """Split one dataset using the enforced `data/<dataset>/raw.csv` structure."""
    if dataset not in get_valid_configs():
        raise ValueError(f"Dataset `{dataset}` does not exist")
    split_dataset(
        dataset=dataset,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


if __name__ == "__main__":
    app()
