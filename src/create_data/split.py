"""
Dataset Split Utility

Splits raw.csv into train/val/test splits.

Usage:
    python -m src.create_data.split --input data/data-inside/raw.csv --output data/data-inside/splits
    python -m src.create_data.split --input data/data-outside/raw.csv --output data/data-outside/splits
"""

import os
import argparse
import polars as pl


def split_and_save(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_dir: str,
):
    """
    Save train/val/test DataFrames to CSV files.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save split CSVs
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.write_csv(train_path)
    val_df.write_csv(val_path)
    test_df.write_csv(test_path)

    print(f"Saved splits to {output_dir}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")


def split_data_inside(
    data_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Split data-inside dataset into train/val/test splits using random sampling.

    Args:
        data_path: Path to input CSV file
        output_dir: Directory to save split CSVs
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set (default 0.7)
        val_ratio: Fraction for validation set (default 0.15)
    """
    print(f"Processing split for {data_path} (data-inside)...")
    try:
        df = pl.read_csv(data_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return

    # Shuffle the dataset
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)

    split_and_save(train_df, val_df, test_df, output_dir)


def split_data_outside(
    data_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Split data-outside dataset into train/val/test splits with stratification by fish_type.

    Ensures each fish_type has the same ratio in train/val/test splits.

    Args:
        data_path: Path to input CSV file
        output_dir: Directory to save split CSVs
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set (default 0.7)
        val_ratio: Fraction for validation set (default 0.15)
    """
    print(
        f"Processing split for {data_path} (data-outside, stratified by fish_type)..."
    )
    try:
        df = pl.read_csv(data_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return

    if "fish_type" not in df.columns:
        print("Error: 'fish_type' column not found in dataset")
        return

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Split each fish_type group separately to maintain ratios
    for fish_type in df["fish_type"].unique().sort():
        group_df = df.filter(pl.col("fish_type") == fish_type)
        group_df = group_df.sample(fraction=1.0, shuffle=True, seed=seed)

        n = len(group_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dfs.append(group_df.slice(0, train_end))
        val_dfs.append(group_df.slice(train_end, val_end - train_end))
        test_dfs.append(group_df.slice(val_end, n - val_end))

    # Concatenate all groups
    train_df = pl.concat(train_dfs)
    val_df = pl.concat(val_dfs)
    test_df = pl.concat(test_dfs)

    # Shuffle each split to mix fish types
    train_df = train_df.sample(fraction=1.0, shuffle=True, seed=seed)
    val_df = val_df.sample(fraction=1.0, shuffle=True, seed=seed)
    test_df = test_df.sample(fraction=1.0, shuffle=True, seed=seed)

    split_and_save(train_df, val_df, test_df, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV (raw.csv)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for splits"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation set ratio"
    )
    args = parser.parse_args()

    if "data-outside" in args.input:
        split_data_outside(
            args.input,
            args.output,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        split_data_inside(
            args.input,
            args.output,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )


if __name__ == "__main__":
    main()
