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
from src.utils.io import load_config


def get_steps_for_pipeline(config, pipeline_id):
    """Get preprocessing steps for a given pipeline configuration.

    Imports are lazy to avoid requiring heavy dependencies (ultralytics, etc.)
    when only running training without preprocessing.
    """
    # Lazy imports - only needed if preprocessing is actually run
    from src.preprocessing.steps.yolo import YoloStep
    from src.preprocessing.steps.rotate import RotateStep
    from src.preprocessing.steps.depth import DepthStep
    from src.preprocessing.steps.segment import SegmentStep
    from src.preprocessing.steps.blackout import BlackoutStep

    # Pipeline 1: Yolo -> Rotate -> Yolo -> Regression (Coords)
    # Pipeline 2: Yolo -> Rotate -> Yolo -> Depth -> Regression (Coords+Depth)
    # Pipeline 3: Yolo -> Rotate -> Yolo -> Segment -> Blackout -> CNN

    # Common prefix
    steps = [
        YoloStep(config, stage="initial"),
        RotateStep(config),
        YoloStep(config, stage="rotated"),
    ]

    if pipeline_id == 1:
        # Just need coords, so we are done with preprocessing
        pass

    elif pipeline_id == 2:
        steps.append(DepthStep(config))

    elif pipeline_id == 3:
        # Segment -> Blackout
        steps.append(SegmentStep(config))
        steps.append(BlackoutStep(config))

    return steps


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

    config = load_config(config_path)
    steps = get_steps_for_pipeline(config, pipeline_id)

    splits = ["train", "val", "test"] if all_splits else ["train"]

    # Run Pipeline Steps (Preprocessing)
    for split in splits:
        in_path = f"data/{dataset_name}/splits/{split}.csv"
        out_path = f"data/{dataset_name}/processed/processed_{split}.csv"

        # Preprocessing can be uncommented if needed
        # print(f" Processing split: {split}")
        # run_pipeline(config, in_path, out_path, steps=steps)

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

    if args.dataset and args.pipeline:
        tasks.append((args.dataset, args.pipeline))
    else:
        tasks.append(("data-inside", 0))
        tasks.append(("data-inside-zoom", 0))
        tasks.append(("data-outside", 0))

    for ds, pid in tasks:
        run_specific_pipeline(ds, pid)


if __name__ == "__main__":
    main()
