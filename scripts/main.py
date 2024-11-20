import os
import cv2
import torch
import numpy as np
import pandas as pd
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm

tqdm.pandas()

IMAGE_ROOT = "./data/images"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
# model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
state_dict = torch.load(f'checkpoints/latest.pth', map_location='cpu')['model']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

def get_depth_data(row):
    image_path = os.path.join(IMAGE_ROOT, row["name"])
    if not os.path.isfile(image_path):
        return
    
    # It reads the image (height x width x 3)
    raw_img = cv2.imread(image_path)
    # HxW raw depth map in numpy
    depth = model.infer_image(raw_img)
    
    out_data = {
        "name": row["name"],
        "Depth_Head": depth[row["FishHeadYCordinate"], row["FishHeadXCordinate"]],
        "Depth_Body": depth[row["FishCenterYCordinate"], row["FishCenterXCordinate"]],
        "Depth_Tail": depth[row["FishTailYCordinate"], row["FishTailXCordinate"]],
        "Depth_Max": np.max(depth),
        "Depth_Min": np.min(depth),
    }
    return out_data
    

def test():
    pd.read_csv("./data/Balik.csv").head(1).progress_apply(get_depth_data, axis=1)

def main():
    df = pd.read_csv("./data/Balik.csv")
    depth_data = pd.DataFrame.from_records(df.progress_apply(get_depth_data, axis=1))
    new_data = pd.merge(df, depth_data, on='name')
    new_data.to_csv("./data/BalikOutDepthAnythingV2-2.csv", index=False)

main()