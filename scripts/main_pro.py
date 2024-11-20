import os
import torch
import numpy as np
import pandas as pd
import depth_pro
from tqdm import tqdm

tqdm.pandas()

IMAGE_ROOT = "./data/fine_images"
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.to(DEVICE)
model.eval()

def get_fish_width_and_height(fish_x, fish_center_x, fish_y, fish_center_y):
    if fish_center_x > fish_x:
        width = (fish_center_x - fish_x) * 2
        left_x = fish_x
        right_x = left_x + width\

    else:
        width = (fish_x - fish_center_x) * 2
        right_x = fish_x
        left_x = right_x - width

    if fish_center_y > fish_y:
        height = (fish_center_y - fish_y) * 2
        top_y = fish_y
        bottom_y = top_y + height

    else:
        height = (fish_y - fish_center_y) * 2
        bottom_y = fish_y
        top_y = bottom_y - height

    return width, height

def get_fish_width_and_height_ratio(img, fish_x, fish_center_x, fish_y, fish_center_y):
    width, height = get_fish_width_and_height(fish_x, fish_center_x, fish_y, fish_center_y)
    return width / img.shape[1], height / img.shape[0]

def get_depth_data(row):
    image_path = os.path.join(IMAGE_ROOT, row["name"])
    if not os.path.isfile(image_path):
        return

    # Load and preprocess an image. 
    # It reads the image (height x width x 3)
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(DEVICE)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].cpu().detach().numpy()  # Depth in [m].
    focallength_px = prediction["focallength_px"].cpu().detach().numpy()

    fish_width, fish_height = get_fish_width_and_height(row["fishx"], row["FishCenterXCordinate"], row["fishy"], row["FishCenterYCordinate"])
    fish_width_ratio = fish_width / image.shape[1]
    fish_height_ratio = fish_height / image.shape[0]

    out_data = {
        "name": row["name"],
        "ImageWidth": image.shape[1],
        "ImageHeight": image.shape[0],
        "Depth_Head": depth[row["FishHeadYCordinate"], row["FishHeadXCordinate"]],
        "Depth_Body": depth[row["FishCenterYCordinate"], row["FishCenterXCordinate"]],
        "Depth_Tail": depth[row["FishTailYCordinate"], row["FishTailXCordinate"]],
        "Depth_Max": np.max(depth),
        "Depth_Min": np.min(depth),
        "FishWidth": fish_width,
        "FishHeight": fish_height,
        "FishWidthRatio": fish_width_ratio,
        "FishHeightRatio": fish_height_ratio,
        "FocalLength": focallength_px
    }
    return out_data

def get_depth_data_new(row):
    image_path = os.path.join(IMAGE_ROOT, row["name"])
    if not os.path.isfile(image_path):
        return

    # Load and preprocess an image. 
    # It reads the image (height x width x 3)
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image).to(DEVICE)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].cpu().detach().numpy()  # Depth in [m].
    focallength_px = prediction["focallength_px"].cpu().detach().numpy()

    fish_center_x = int((row["fish_x2"] + row["fish_x1"]) / 2)
    fish_center_y = int((row["fish_y2"] + row["fish_y1"]) / 2)

    # fish_width, fish_height = get_fish_width_and_height(row["fishx"], row["FishCenterXCordinate"], row["fishy"], row["FishCenterYCordinate"])
    # fish_width_ratio = fish_width / image.shape[1]
    # fish_height_ratio = fish_height / image.shape[0]

    out_data = {
        "name": row["name"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
        "depth_center": depth[fish_center_y, fish_center_x],
        "depth_max": np.max(depth),
        "depth_min": np.min(depth),
        "fish_width": row["fish_x2"] - row["fish_x1"],
        "fish_height": row["fish_y2"] - row["fish_y1"],
        "fish_width_ratio": (row["fish_x2"] - row["fish_x1"]) / row["image_width"],
        "fish_height_ratio": (row["fish_y2"] - row["fish_y1"]) / row["image_height"],
        "FocalLength": focallength_px
    }
    return out_data

def test():
    pd.read_csv("./data/Balik.csv").head(1).progress_apply(get_depth_data_new, axis=1)

def main():
    df = pd.read_csv("./data/Balik.csv")
    depth_data = pd.DataFrame.from_records(df.progress_apply(get_depth_data_new, axis=1))
    new_data = pd.merge(df, depth_data, on='name')
    new_data.to_csv("./data/BalikOutDepthProNew.csv", index=False)

main()