import argparse
import os
import sys
import subprocess
from src.utils.io import load_config
from src.pipeline import run_pipeline
from src.steps.yolo_step import YoloStep
from src.steps.rotate_step import RotateStep
from src.steps.depth_step import DepthStep
from src.steps.segment_step import SegmentStep
from src.steps.blackout_step import BlackoutStep
from src.steps.visualize_step import VisualizeStep

def get_steps_for_pipeline(config, pipeline_id):
    # Pipeline 1: Yolo -> Rotate -> Yolo -> Regression (Coords)
    # Pipeline 2: Yolo -> Rotate -> Yolo -> Depth -> Regression (Coords+Depth)
    # Pipeline 3: Yolo -> Rotate -> Yolo -> Segment -> Blackout -> CNN
    
    # Common prefix
    steps = [
        YoloStep(config, stage="initial"),
        RotateStep(config),
        YoloStep(config, stage="rotated")
    ]
    
    if pipeline_id == 1:
        # Just need coords, so we are done with preprocessing? 
        # Actually pipeline runner updates the CSV with coords.
        # So running up to Yolo(rotated) is enough for P1 features.
        pass
        
    elif pipeline_id == 2:
        steps.append(DepthStep(config))
        
    elif pipeline_id == 3:
        # Segment -> Blackout
        steps.append(SegmentStep(config))
        steps.append(BlackoutStep(config))
        
    return steps

def run_specific_pipeline(dataset_name, pipeline_id, all_splits=True):
    print(f"\n>>> Running Pipeline {pipeline_id} for {dataset_name} <<<")
    
    config_path = f"configs/config_{dataset_name.replace('data-','').replace('-','_')}.yaml"
    # Mapping config names: data-inside -> config_inside.yaml, data-inside-zoom -> config_inside_zoom.yaml
    # Helper:
    if dataset_name == "data-inside":
        config_path = "configs/config_inside.yaml"
    elif dataset_name == "data-inside-zoom":
        config_path = "configs/config_inside_zoom.yaml"
    elif dataset_name == "data-outside":
        config_path = "configs/config_outside.yaml"
        
    config = load_config(config_path)
    steps = get_steps_for_pipeline(config, pipeline_id)
    
    splits = ["train", "val", "test"] if all_splits else ["train"]
    
    # Run Pipeline Steps (Preprocessing)
    for split in splits:
        in_path = f"data/{dataset_name}/splits/{split}.csv"
        out_path = f"data/{dataset_name}/processed/processed_{split}.csv"
        
        # We need to ensure we don't overwrite previous pipeline results if they share steps?
        # Actually, P1, P2, P3 share initial steps. 
        # If we run P3, it does everything P1 does (Yolo, Rotate, Yolo) + Segment/Blackout.
        # If we run P2, it adds Depth.
        # Ideally we should just run the superset of steps needed for all requested pipelines?
        # But user asked for specific pipelines.
        # For efficiency, if I run P3, I get P1 features too.
        # Re-running might be redundant but safer for isolation. 
        
        # print(f" Processing split: {split}")
        # run_pipeline(config, in_path, out_path, steps=steps)

    # Run Baseline
    if pipeline_id == 0:
        cmd = [sys.executable, "-m", "src.train_baseline", "--dataset", dataset_name]
        subprocess.run(cmd, check=True)
        
    # Train Model
    if pipeline_id == 1:
        # cmd = [sys.executable, "-m", "src.train_regression", "--dataset", dataset_name, "--feature-set", "coords"]
        # subprocess.run(cmd, check=True)
        # Append MLP
        cmd = [sys.executable, "-m", "src.train_mlp", "--dataset", dataset_name, "--feature-set", "coords", "--epochs", "200"]
        subprocess.run(cmd, check=True)
    
    if pipeline_id == 2:
        # cmd = [sys.executable, "-m", "src.train_regression", "--dataset", dataset_name, "--feature-set", "eye"]
        # subprocess.run(cmd, check=True)
        # Append MLP
        cmd = [sys.executable, "-m", "src.train_mlp", "--dataset", dataset_name, "--feature-set", "eye", "--epochs", "200"]
        subprocess.run(cmd, check=True)
    
    # Train Model
    if pipeline_id == 3:
        # cmd = [sys.executable, "-m", "src.train_regression", "--dataset", dataset_name, "--feature-set", "scaled"]
        # subprocess.run(cmd, check=True)
        # Append MLP
        cmd = [sys.executable, "-m", "src.train_mlp", "--dataset", dataset_name, "--feature-set", "scaled", "--epochs", "200"]
        subprocess.run(cmd, check=True)
        
    if pipeline_id == 4:
        # cmd = [sys.executable, "-m", "src.train_regression", "--dataset", dataset_name, "--feature-set", "coords", "--depth-model", "v2"]
        # subprocess.run(cmd, check=True)
        # Append MLP
        cmd = [sys.executable, "-m", "src.train_mlp", "--dataset", dataset_name, "--feature-set", "coords", "--depth-model", "v2", "--epochs", "200"]
        subprocess.run(cmd, check=True)

    if pipeline_id == 5:
        # cmd = [sys.executable, "-m", "src.train_regression", "--dataset", dataset_name, "--feature-set", "scaled", "--depth-model", "v2"]
        # subprocess.run(cmd, check=True)
        # Append MLP
        cmd = [sys.executable, "-m", "src.train_mlp", "--dataset", dataset_name, "--feature-set", "scaled", "--depth-model", "v2", "--epochs", "200"]
        subprocess.run(cmd, check=True)
    
    if pipeline_id == 6:
        cmd = [sys.executable, "-m", "src.train_cnn", "--dataset", dataset_name, "--epochs", "100", "--feature-set", "scaled", "--depth-model", "v2"]
        subprocess.run(cmd, check=True)

    # Run Analysis
    cmd = [sys.executable, "-m", "src.analyze_results", "--dataset", dataset_name]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=int, choices=[1, 2, 3, 4, 5, 6], help="Specific pipeline to run")
    parser.add_argument("--dataset", type=str, help="Specific dataset")
    args = parser.parse_args()

    # Rule:
    # Data-inside: Pipeline 1 and 3
    # Data-inside-zoom: Pipeline 1, 2 and 3
    
    tasks = []
    
    if args.dataset and args.pipeline:
        tasks.append((args.dataset, args.pipeline))
    else:
        tasks.append(("data-inside", 0))
        # tasks.append(("data-inside", 1))
        # tasks.append(("data-inside", 2))
        # tasks.append(("data-inside", 6))

        tasks.append(("data-inside-zoom", 0)) 
        # tasks.append(("data-inside-zoom", 1))
        # tasks.append(("data-inside-zoom", 2))
        # tasks.append(("data-inside-zoom", 4))
        # tasks.append(("data-inside-zoom", 6))
        
        tasks.append(("data-outside", 0))
        # tasks.append(("data-outside", 1))
        # tasks.append(("data-outside", 3))
        # tasks.append(("data-outside", 4))
        # tasks.append(("data-outside", 5))
        # tasks.append(("data-outside", 6))
        
    for ds, pid in tasks:
        run_specific_pipeline(ds, pid)

if __name__ == "__main__":
    main()
