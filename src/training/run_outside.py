"""
Training Orchestrator for Outside Data

Runs training pipelines specifically for data-outside dataset.

Usage:
    python -m src.training.run_outside --all         # Run all pipelines
    python -m src.training.run_outside --pipeline 1  # Run specific pipeline
"""

import argparse
import sys
import subprocess

DATASET = "data-outside"


def run_pipeline(pipeline_id, cnn=False):
    """Run a specific pipeline configuration."""
    cnn_flag = ["--cnn", "--cnn-epochs", "100"] if cnn else []
    cnn_desc = " +CNN" if cnn else ""

    # Pipeline 0: Baseline
    if pipeline_id == 0:
        print(f"\n>>> Pipeline 0: Baseline <<<")
        cmd = [sys.executable, "-m", "src.training.baseline", "--dataset", DATASET]
        subprocess.run(cmd, check=True)

    # Pipeline 1: Fish coordinates as features (Linear, XGBoost, MLP)
    elif pipeline_id == 1:
        print(f"\n>>> Pipeline 1: Fish coordinates as features <<<")
        for module in [
            "src.training.linear",
            "src.training.xgboost",
            "src.training.mlp",
        ]:
            cmd = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                DATASET,
                "--feature-set",
                "coords",
            ]
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 2: Scaled fish coordinates as features (Linear, XGBoost, MLP)
    elif pipeline_id == 2:
        print(f"\n>>> Pipeline 2: Scaled fish coordinates as features <<<")
        for module in [
            "src.training.linear",
            "src.training.xgboost",
            "src.training.mlp",
        ]:
            cmd = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                DATASET,
                "--feature-set",
                "scaled",
            ]
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 3: Fish coordinates + fish depth as features (Linear, XGBoost, MLP)
    elif pipeline_id == 3:
        print(f"\n>>> Pipeline 3: Fish coordinates + fish depth as features <<<")
        for module in [
            "src.training.linear",
            "src.training.xgboost",
            "src.training.mlp",
        ]:
            cmd = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                DATASET,
                "--feature-set",
                "coords",
                "--depth",
            ]
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 4: Scaled fish coordinates + fish depth as features (Linear, XGBoost, MLP)
    elif pipeline_id == 4:
        print(f"\n>>> Pipeline 4: Scaled fish coordinates + fish depth as features <<<")
        for module in [
            "src.training.linear",
            "src.training.xgboost",
            "src.training.mlp",
        ]:
            cmd = [
                sys.executable,
                "-m",
                module,
                "--dataset",
                DATASET,
                "--feature-set",
                "scaled",
                "--depth",
            ]
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 5: CNN with scaled fish coordinates + fish depth as features
    elif pipeline_id == 5:
        print(f"\n>>> Pipeline 5: CNN scaled fish coordinates + fish depth as features <<<")
        cmd = [
            sys.executable,
            "-m",
            "src.training.cnn",
            "--dataset",
            DATASET,
            "--feature-set",
            "scaled",
            "--depth",
            "--epochs",
            "100",
        ]
        subprocess.run(cmd, check=True)

    # Pipeline 6: Per-fish-type modeling using fish coordinates as features
    elif pipeline_id == 6:
        print(f"\n>>> Pipeline 6: Per-fish-type modeling using fish coordinates as features{cnn_desc} <<<")
        cmd = (
            [
                sys.executable,
                "-m",
                "src.training.per_fishtype",
                "--dataset",
                DATASET,
                "--feature-set",
                "coords",
                "--epochs",
                "200",
            ]
            + cnn_flag
        )
        subprocess.run(cmd, check=True)

    # Pipeline 7: Per-fish-type using scaled fish coordinates as features
    elif pipeline_id == 7:
        print(f"\n>>> Pipeline 7: Per-fish-type using scaled fish coordinates as features{cnn_desc} <<<")
        cmd = (
            [
                sys.executable,
                "-m",
                "src.training.per_fishtype",
                "--dataset",
                DATASET,
                "--feature-set",
                "scaled",
                "--epochs",
                "200",
            ]
            + cnn_flag
        )
        subprocess.run(cmd, check=True)

    # Pipeline 8: Per-fish-type using fish coordinates + fish depth as features
    elif pipeline_id == 8:
        print(
            f"\n>>> Pipeline 8: Per-fish-type using fish coordinates + fish depth as features{cnn_desc} <<<"
        )
        cmd = (
            [
                sys.executable,
                "-m",
                "src.training.per_fishtype",
                "--dataset",
                DATASET,
                "--feature-set",
                "coords",
                "--depth",
                "--epochs",
                "200",
            ]
            + cnn_flag
        )
        subprocess.run(cmd, check=True)

    # Pipeline 9: Per-fish-type using scaled fish coordinates + fish depth as features
    elif pipeline_id == 9:
        print(
            f"\n>>> Pipeline 9: Per-fish-type using scaled fish coordinates + fish depth as features{cnn_desc} <<<"
        )
        cmd = (
            [
                sys.executable,
                "-m",
                "src.training.per_fishtype",
                "--dataset",
                DATASET,
                "--feature-set",
                "scaled",
                "--depth",
                "--epochs",
                "200",
            ]
            + cnn_flag
        )
        subprocess.run(cmd, check=True)

    else:
        print(f"Unknown pipeline: {pipeline_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Training orchestrator for outside data"
    )
    parser.add_argument(
        "--pipeline",
        type=int,
        choices=list(range(10)),
        help="Specific pipeline to run (0-9)",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Include CNN training for per-fishtype pipelines (6-9)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipelines",
    )
    args = parser.parse_args()

    if args.pipeline is not None:
        run_pipeline(args.pipeline, cnn=args.cnn)
    elif args.all:
        for pid in range(10):
            run_pipeline(pid, cnn=args.cnn)
    else:
        # Default: run common pipelines
        print("Running default pipelines for data-outside...")
        run_pipeline(0, cnn=args.cnn)  # Baseline
        run_pipeline(1, cnn=args.cnn)  # coords
        run_pipeline(3, cnn=args.cnn)  # scaled
        run_pipeline(5, cnn=args.cnn)  # scaled + depth
        run_pipeline(8, cnn=args.cnn)  # per-fishtype scaled


if __name__ == "__main__":
    main()
