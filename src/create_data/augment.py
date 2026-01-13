"""
Data Augmentation Utilities

Provides zoom augmentation and dataset creation with zoomed images.

Usage:
    python -m src.create_data.augment --source data-inside --dest data-inside-zoom
"""

import os
import random
import argparse

import cv2
import polars as pl


def augment_zoom(image, zoom_type: str = "in", magnitude: float = 0.1):
    """
    Apply zoom augmentation to an image.

    Args:
        image: Input image (numpy array)
        zoom_type: "in" for zoom in, "out" for zoom out
        magnitude: Zoom magnitude (0.0-0.9)

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]

    # Validation
    magnitude = max(0.0, min(0.9, magnitude))  # Clamp to safe range
    ratio = 1.0 - magnitude

    if zoom_type == "in":
        # Zoom in: Crop center region of size (h*ratio, w*ratio) and resize to (h, w)
        # Ratio 0.8 means we keep 80% of the image (0.2 magnitude zoom)
        nh, nw = int(h * ratio), int(w * ratio)

        # Top left corner
        y = (h - nh) // 2
        x = (w - nw) // 2

        cropped = image[y : y + nh, x : x + nw]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    elif zoom_type == "out":
        # Zoom out: Simply resize image to (h*ratio, w*ratio)
        # Ratio 0.8 means result is 80% of original size (0.2 magnitude zoom)
        nh, nw = int(h * ratio), int(w * ratio)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized

    return image


def process_zoom_dataset(
    source_name: str = "data-inside", dest_name: str = "data-inside-zoom"
):
    """
    Create a zoomed dataset from an existing dataset.

    For each image in the source dataset:
    - Copies the original image
    - Creates one zoom-in variant
    - Creates one zoom-out variant

    Args:
        source_name: Source dataset name (e.g., "data-inside")
        dest_name: Destination dataset name (e.g., "data-inside-zoom")
    """
    print(f"Augmenting {source_name} -> {dest_name}")

    splits = ["train", "val", "test"]
    base_dir = f"data/{source_name}"
    dest_dir = f"data/{dest_name}"

    os.makedirs(os.path.join(dest_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "splits"), exist_ok=True)

    # We also want to create a raw.csv for the zoomed dataset
    all_rows = []

    for split in splits:
        split_path = os.path.join(base_dir, "splits", f"{split}.csv")
        if not os.path.exists(split_path):
            print(f"Split {split} not found at {split_path}, skipping.")
            continue

        print(f"Processing split: {split}")
        df = pl.read_csv(split_path)

        new_rows = []

        for row in df.to_dicts():
            name = row["name"]
            base_name, ext = os.path.splitext(name)
            # Copy other columns (like length)
            base_row = {k: v for k, v in row.items() if k != "name"}

            src_img_path = os.path.join(base_dir, "raw", name)
            if not os.path.exists(src_img_path):
                print(f"Image {name} not found, skipping.")
                continue

            img = cv2.imread(src_img_path)
            if img is None:
                print(f"Failed to read {src_img_path}")
                continue

            # 1. Original
            dest_img_path = os.path.join(dest_dir, "raw", name)
            cv2.imwrite(dest_img_path, img)
            new_rows.append({"name": name, **base_row})

            # 2. Zoom In
            # Generate exactly one zoom-in
            # Magnitude 0.2 to 0.5 (Mild to moderate)
            rin = random.uniform(0.2, 0.5)
            img_in = augment_zoom(img, "in", rin)

            # Naming: name-zin-{magnitude}
            suffix = int(rin * 100)
            name_in = f"{base_name}-zin-{suffix}{ext}"

            cv2.imwrite(os.path.join(dest_dir, "raw", name_in), img_in)
            new_rows.append({"name": name_in, **base_row})

            # 3. Zoom Out
            # Generate exactly one zoom-out
            # Magnitude 0.2 to 0.5
            rout = random.uniform(0.2, 0.5)
            img_out = augment_zoom(img, "out", rout)

            suffix = int(rout * 100)
            name_out = f"{base_name}-zout-{suffix}{ext}"

            cv2.imwrite(os.path.join(dest_dir, "raw", name_out), img_out)
            new_rows.append({"name": name_out, **base_row})

        # Save split csv
        split_df = pl.DataFrame(new_rows)
        split_df.write_csv(os.path.join(dest_dir, "splits", f"{split}.csv"))

        all_rows.extend(new_rows)

    # Save global raw.csv
    pl.DataFrame(all_rows).write_csv(os.path.join(dest_dir, "raw.csv"))
    print(f"Finished creating {dest_name}")


def main():
    parser = argparse.ArgumentParser(description="Create zoom-augmented dataset")
    parser.add_argument(
        "--source", type=str, default="data-inside", help="Source dataset name"
    )
    parser.add_argument(
        "--dest", type=str, default="data-inside-zoom", help="Destination dataset name"
    )
    args = parser.parse_args()

    process_zoom_dataset(args.source, args.dest)


if __name__ == "__main__":
    main()
