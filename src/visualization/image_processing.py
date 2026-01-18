"""Image path resolution and processing utilities."""

import os
import cv2
import numpy as np


def get_image_paths(dataset, image_name, depth_model=None):
    """Resolve paths for raw, rotated, depth, and blackout images."""
    # Try finding image in raw or splits
    found_path = None
    possible_paths = [
        f"data/{dataset}/raw/{image_name}",
        f"data/{dataset}/splits/{image_name}",
        f"data/{dataset}/{image_name}",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            found_path = p
            break

    rot_path = f"data/{dataset}/processed/rotated/{image_name}"

    # Depth extension check
    depth_path = None
    if depth_model:
        base_depth_dir = f"data/{dataset}/processed/depth/{depth_model}"
        p = f"{base_depth_dir}/{image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
        if not os.path.exists(p):
            p = f"{base_depth_dir}/{image_name}.npy"

        if os.path.exists(p):
            depth_path = p

    # Fallback to root depth folder if not found or no model selected
    if not depth_path:
        p = f"data/{dataset}/processed/depth/{image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy')}"
        if not os.path.exists(p):
            p = f"data/{dataset}/processed/depth/{image_name}.npy"
        if os.path.exists(p):
            depth_path = p

    # Blackout image path
    blackout_path = f"data/{dataset}/processed/blackout/{image_name}"
    if not os.path.exists(blackout_path):
        # Try with different extensions
        for ext in [".jpg", ".jpeg", ".png"]:
            alt_path = f"data/{dataset}/processed/blackout/{image_name.replace('.jpg', ext).replace('.jpeg', ext)}"
            if os.path.exists(alt_path):
                blackout_path = alt_path
                break

    return found_path, rot_path, depth_path, blackout_path


def process_images(dataset, image_name, data_row, depth_model=None):
    """Load and process all image variants for display."""
    raw_path, rot_path, depth_path, blackout_path = get_image_paths(
        dataset, image_name, depth_model
    )

    # Raw image
    img_raw = None
    if raw_path and os.path.exists(raw_path):
        img_raw = cv2.imread(raw_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    else:
        img_raw = np.zeros((200, 200, 3), dtype=np.uint8)  # Placeholder

    # Rotated image with annotations
    img_rot = None
    if os.path.exists(rot_path):
        img_rot = cv2.imread(rot_path)
        img_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)

        if data_row:
            # Fish BBox
            if "Fish_x1" in data_row and data_row["Fish_x1"] is not None:
                pt1 = (int(data_row["Fish_x1"]), int(data_row["Fish_y1"]))
                pt2 = (int(data_row["Fish_x2"]), int(data_row["Fish_y2"]))
                cv2.rectangle(img_rot, pt1, pt2, (0, 120, 255), 2)
                cv2.putText(
                    img_rot,
                    "Fish",
                    pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 120, 255),
                    2,
                )

            # Head center
            if "Head_x1" in data_row and data_row["Head_x1"] is not None:
                cx = int((data_row["Head_x1"] + data_row["Head_x2"]) / 2)
                cy = int((data_row["Head_y1"] + data_row["Head_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (255, 0, 0), -1)

            # Tail center
            if "Tail_x1" in data_row and data_row["Tail_x1"] is not None:
                cx = int((data_row["Tail_x1"] + data_row["Tail_x2"]) / 2)
                cy = int((data_row["Tail_y1"] + data_row["Tail_y2"]) / 2)
                cv2.circle(img_rot, (cx, cy), 5, (0, 255, 0), -1)

    # Depth map
    img_depth = None
    if depth_path and os.path.exists(depth_path):
        depth = np.load(depth_path)
        norm_depth = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        img_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2RGB)

    # Blackout image
    img_blackout = None
    if blackout_path and os.path.exists(blackout_path):
        img_blackout = cv2.imread(blackout_path)
        img_blackout = cv2.cvtColor(img_blackout, cv2.COLOR_BGR2RGB)

    return img_raw, img_rot, img_depth, img_blackout
