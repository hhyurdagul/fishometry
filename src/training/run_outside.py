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


def run_pipeline(pipeline_id, stats=False):
    """Run a specific pipeline configuration."""
    stats_flag = ["--stats"] if stats else []
    stats_desc = " with stats" if stats else ""

    # Pipeline 0: Baseline
    if pipeline_id == 0:
        print(f"\n>>> Pipeline 0: Baseline <<<")
        cmd = [sys.executable, "-m", "src.training.baseline", "--dataset", DATASET]
        subprocess.run(cmd, check=True)

    # Pipeline 1: coords features (Linear, XGBoost, MLP)
    elif pipeline_id == 1:
        print(f"\n>>> Pipeline 1: coords features{stats_desc} <<<")
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
            ] + stats_flag
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 2: eye features (Linear, XGBoost, MLP)
    elif pipeline_id == 2:
        print(f"\n>>> Pipeline 2: eye features{stats_desc} <<<")
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
                "eye",
            ] + stats_flag
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 3: scaled features (Linear, XGBoost, MLP)
    elif pipeline_id == 3:
        print(f"\n>>> Pipeline 3: scaled features{stats_desc} <<<")
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
            ] + stats_flag
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 4: coords + depth features (Linear, XGBoost, MLP)
    elif pipeline_id == 4:
        print(f"\n>>> Pipeline 4: coords + depth{stats_desc} <<<")
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
            ] + stats_flag
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 5: scaled + depth features (Linear, XGBoost, MLP)
    elif pipeline_id == 5:
        print(f"\n>>> Pipeline 5: scaled + depth{stats_desc} <<<")
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
            ] + stats_flag
            if module == "src.training.mlp":
                cmd.extend(["--epochs", "200"])
            subprocess.run(cmd, check=True)

    # Pipeline 6: CNN with scaled + depth
    elif pipeline_id == 6:
        print(f"\n>>> Pipeline 6: CNN scaled + depth{stats_desc} <<<")
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
        ] + stats_flag
        subprocess.run(cmd, check=True)

    # Pipeline 7: Per-fish-type with coords
    elif pipeline_id == 7:
        print(f"\n>>> Pipeline 7: Per-fish-type coords{stats_desc} <<<")
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            DATASET,
            "--feature-set",
            "coords",
            "--epochs",
            "200",
        ] + stats_flag
        subprocess.run(cmd, check=True)

    # Pipeline 8: Per-fish-type with scaled
    elif pipeline_id == 8:
        print(f"\n>>> Pipeline 8: Per-fish-type scaled{stats_desc} <<<")
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            DATASET,
            "--feature-set",
            "scaled",
            "--epochs",
            "200",
        ] + stats_flag
        subprocess.run(cmd, check=True)

    # Pipeline 9: Per-fish-type with coords + depth
    elif pipeline_id == 9:
        print(f"\n>>> Pipeline 9: Per-fish-type coords + depth{stats_desc} <<<")
        cmd = [
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
        ] + stats_flag
        subprocess.run(cmd, check=True)

    # Pipeline 10: Per-fish-type with scaled + depth
    elif pipeline_id == 10:
        print(f"\n>>> Pipeline 10: Per-fish-type scaled + depth{stats_desc} <<<")
        cmd = [
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
        ] + stats_flag
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
        choices=list(range(11)),
        help="Specific pipeline to run (0-10)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Include species stats features",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipelines",
    )
    args = parser.parse_args()

    if args.pipeline is not None:
        run_pipeline(args.pipeline, stats=args.stats)
    elif args.all:
        for pid in range(11):
            run_pipeline(pid, stats=args.stats)
    else:
        # Default: run common pipelines
        print("Running default pipelines for data-outside...")
        run_pipeline(0, stats=args.stats)  # Baseline
        run_pipeline(1, stats=args.stats)  # coords
        run_pipeline(3, stats=args.stats)  # scaled
        run_pipeline(5, stats=args.stats)  # scaled + depth
        run_pipeline(8, stats=args.stats)  # per-fishtype scaled


if __name__ == "__main__":
    main()
