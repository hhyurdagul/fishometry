"""
Training Orchestrator

Runs preprocessing and training pipelines for different configurations.

Usage:
    python -m src.training.run --pipeline 1 --dataset data-inside
    python -m src.training.run  # Runs default tasks
"""

import argparse
import sys
import subprocess


def run_specific_pipeline(dataset_name, pipeline_id, all_splits=True):
    """Run a specific pipeline configuration."""
    print(f"\n>>> Running Pipeline {pipeline_id} for {dataset_name} <<<")

    # Mapping config names
    if dataset_name == "data-inside":
        config_path = "configs/config_inside.yaml"
    elif dataset_name == "data-inside-zoom":
        config_path = "configs/config_inside_zoom.yaml"
    elif dataset_name == "data-outside":
        config_path = "configs/config_outside.yaml"
    else:
        config_path = (
            f"configs/config_{dataset_name.replace('data-', '').replace('-', '_')}.yaml"
        )

    splits = ["train", "val", "test"] if all_splits else ["train"]

    # Run Baseline
    if pipeline_id == 0:
        cmd = [sys.executable, "-m", "src.training.baseline", "--dataset", dataset_name]
        subprocess.run(cmd, check=True)

    # Train Models (Linear Regression, XGBoost, MLP)
    if pipeline_id == 1:
        # Linear Regression
        cmd = [
            sys.executable,
            "-m",
            "src.training.regression",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
        ]
        subprocess.run(cmd, check=True)
        # XGBoost
        cmd = [
            sys.executable,
            "-m",
            "src.training.xgboost",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
        ]
        subprocess.run(cmd, check=True)
        # MLP
        cmd = [
            sys.executable,
            "-m",
            "src.training.mlp",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 2:
        # Linear Regression
        cmd = [
            sys.executable,
            "-m",
            "src.training.regression",
            "--dataset",
            dataset_name,
            "--feature-set",
            "eye",
        ]
        subprocess.run(cmd, check=True)
        # XGBoost
        cmd = [
            sys.executable,
            "-m",
            "src.training.xgboost",
            "--dataset",
            dataset_name,
            "--feature-set",
            "eye",
        ]
        subprocess.run(cmd, check=True)
        # MLP
        cmd = [
            sys.executable,
            "-m",
            "src.training.mlp",
            "--dataset",
            dataset_name,
            "--feature-set",
            "eye",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 3:
        # Linear Regression
        cmd = [
            sys.executable,
            "-m",
            "src.training.regression",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
        ]
        subprocess.run(cmd, check=True)
        # XGBoost
        cmd = [
            sys.executable,
            "-m",
            "src.training.xgboost",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
        ]
        subprocess.run(cmd, check=True)
        # MLP
        cmd = [
            sys.executable,
            "-m",
            "src.training.mlp",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 4:
        # Linear Regression with depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.regression",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--depth",
        ]
        subprocess.run(cmd, check=True)
        # XGBoost with depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.xgboost",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--depth",
        ]
        subprocess.run(cmd, check=True)
        # MLP with depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.mlp",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--depth",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 5:
        # Linear Regression with scaled + depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.regression",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--depth",
        ]
        subprocess.run(cmd, check=True)
        # XGBoost with scaled + depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.xgboost",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--depth",
        ]
        subprocess.run(cmd, check=True)
        # MLP with scaled + depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.mlp",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--depth",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 6:
        # CNN with scaled + depth
        cmd = [
            sys.executable,
            "-m",
            "src.training.cnn",
            "--dataset",
            dataset_name,
            "--epochs",
            "100",
            "--feature-set",
            "scaled",
            "--depth",
        ]
        subprocess.run(cmd, check=True)

    # Per fish models, only for data-outside
    if pipeline_id == 7:
        # MLP
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 8:
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 9:
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            dataset_name,
            "--feature-set",
            "coords",
            "--depth",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)

    if pipeline_id == 10:
        cmd = [
            sys.executable,
            "-m",
            "src.training.per_fishtype",
            "--dataset",
            dataset_name,
            "--feature-set",
            "scaled",
            "--depth",
            "--epochs",
            "200",
        ]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Training orchestrator")
    parser.add_argument(
        "--pipeline",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Specific pipeline to run",
    )
    parser.add_argument("--dataset", type=str, help="Specific dataset")
    args = parser.parse_args()

    tasks = []

    # 0: Baseline
    # 1: Regression with coord features
    # 2: Regression with eye features
    # 3: Regression with scaled features
    # 4: Regression with depth+coord features
    # 5: Regression with depth+scaled features
    # 6: CNN with depth+scaled features and blackout images
    if args.dataset and args.pipeline:
        tasks.append((args.dataset, args.pipeline))
    else:
        # tasks.append(("data-inside", 0))
        # tasks.append(("data-inside", 1))
        # tasks.append(("data-inside", 2))
        # tasks.append(("data-inside", 6))
        #
        # tasks.append(("data-inside-zoom", 0))
        # tasks.append(("data-inside-zoom", 1))
        # tasks.append(("data-inside-zoom", 2))
        # tasks.append(("data-inside-zoom", 4))
        # tasks.append(("data-inside-zoom", 6))

        tasks.append(("data-outside", 0))
        tasks.append(("data-outside", 1))
        # tasks.append(("data-outside", 3))
        # tasks.append(("data-outside", 4))
        # tasks.append(("data-outside", 5))
        # tasks.append(("data-outside", 6))
        # tasks.append(("data-outside", 7))
        # tasks.append(("data-outside", 8))
        # tasks.append(("data-outside", 9))
        # tasks.append(("data-outside", 10))

    for ds, pid in tasks:
        run_specific_pipeline(ds, pid)


if __name__ == "__main__":
    main()
