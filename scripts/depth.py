import os
import torch
import numpy as np
import pandas as pd
from depth_arch import depth_pro
from depth_arch.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

tqdm.pandas()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load model and preprocessing transform
pro_depth_model, transform = depth_pro.create_model_and_transforms()
pro_depth_model.to(DEVICE)
pro_depth_model.eval()

depth_anything_model = DepthAnythingV2(
    encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]
)
depth_anything_model.load_state_dict(
    torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location="cpu")
)
depth_anything_model = depth_anything_model.to(DEVICE).eval()


root_image_dir = "data/yolo_output/rotated_images"


def get_depth_pro_data(row):
    image_path = os.path.join(root_image_dir, row["name"])
    if not os.path.isfile(image_path):
        return

    # Load and preprocess an image.
    # It reads the image (height x width x 3)
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(DEVICE)

    # Run inference.
    prediction = pro_depth_model.infer(image, f_px=f_px)
    depth = prediction["depth"].cpu().detach().numpy()  # Depth in [m].
    focallength_px = prediction["focallength_px"].cpu().detach().numpy()

    try:
        out_data = row.to_dict()
        out_data.update(
            {
                "focal_length": focallength_px,
                "depth_min": np.min(depth),
                "depth_max": np.max(depth),
                "depth_head_center": depth[int(row["head_y_center"]), int(row["head_x_center"])],
                "depth_body_center": depth[int(row["fish_y_center"]), int(row["fish_x_center"])],
                "depth_tail_center": depth[int(row["tail_y_center"]), int(row["tail_x_center"])],
            }
        )
        return out_data
    except Exception:
        return None

if __name__ == "__main__":
    fish_df = pd.read_csv("data/raw_fish_data.csv")
    df = pd.read_csv("data/yolo_output/image_features_after_rotation.csv")
    depth_data = pd.DataFrame.from_records(df.progress_apply(get_depth_pro_data, axis=1).dropna()).dropna()
    new_data = pd.merge(fish_df, depth_data, on='name')
    new_data.to_csv("data/fish_data_after_depth.csv", index=False)