"""
Dataset Split Utility

Splits raw.csv into train/val/test splits.

Usage:
    python -m src.data.split --input data/data-inside/raw.csv --output data/data-inside/splits
    python -m src.data.split --input data/data-outside/raw.csv --output data/data-outside/splits
"""

import argparse
import os
import polars as pl


def split_and_save(
    data_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Split a dataset CSV into train/val/test splits.

    Args:
        data_path: Path to input CSV file
        output_dir: Directory to save split CSVs
        seed: Random seed for reproducibility
        train_ratio: Fraction for training set (default 0.7)
        val_ratio: Fraction for validation set (default 0.15)
    """
    print(f"Processing split for {data_path}...")
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

    split_and_save(
        args.input,
        args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
