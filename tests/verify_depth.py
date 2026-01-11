
import os
import sys
import polars as pl
from src.utils.io import load_config
from src.steps.yolo_step import YoloStep
from src.steps.depth_step import DepthStep

# Setup basics
config = load_config("configs/config_inside.yaml")
input_dir = os.path.join(config["paths"]["output"], "rotated")

# Pick one image
image_name = sorted(os.listdir(input_dir))[0]
print(f"Testing on image: {image_name}")

# Create dummy DF
df = pl.DataFrame({"name": [image_name]})

# Initialize step
yolo_step = YoloStep(config)
depth_step = DepthStep(config)

# Run process
result_df = depth_step.process(yolo_step.process(df))

print("\nResult Dataframe:")
print(result_df)

# Check folders
output_dir = depth_step.output_dir
print(f"\nChecking outputs in {output_dir}:")
for model in ["v3", "v2", "pro"]:
    model_dir = os.path.join(output_dir, f"depth-{model}")
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"  {model}: {len(files)} files found. {files}")
    else:
        print(f"  {model}: Folder missing!")
