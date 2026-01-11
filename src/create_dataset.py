import os
import cv2
import polars as pl
import random
from src.utils.zoom import augment_zoom

def split_and_save(data_path, output_dir, seed=42):
    print(f"Processing split for {data_path}...")
    try:
        df = pl.read_csv(data_path)
    except Exception as e:
        print(f"Error reading {data_path}: {e}")
        return

    # Shuffle the dataset
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.write_csv(train_path)
    val_df.write_csv(val_path)
    test_df.write_csv(test_path)
    
    print(f"Saved splits to {output_dir}")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

def process_zoom_dataset(source_name="data-inside", dest_name="data-inside-zoom"):
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
            # Magnitude 0.1 to 0.3 ( Mild )
            rin = random.uniform(0.1, 0.3)
            img_in = augment_zoom(img, "in", rin)
                 
            # Naming: name-zin-{magnitude}
            suffix = int(rin * 100)
            name_in = f"{base_name}-zin-{suffix}{ext}"
                 
            cv2.imwrite(os.path.join(dest_dir, "raw", name_in), img_in)
            new_rows.append({"name": name_in, **base_row})

            # 3. Zoom Out
            # Generate exactly one zoom-out
            # Magnitude 0.1 to 0.3
            rout = random.uniform(0.1, 0.3)
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
    # 1. Split base datasets
    print("--- Splitting Base Datasets ---")
    datasets = [
        # ("data/data-inside/raw.csv", "data/data-inside/splits"),
        ("data/data-outside/raw.csv", "data/data-outside/splits")
    ]
    
    for input_csv, output_dir in datasets:
        if os.path.exists(input_csv):
            split_and_save(input_csv, output_dir)
        else:
            print(f"Warning: {input_csv} not found. Skipping.")
            
    echo_space = lambda: print("\n")
    echo_space()

    # 2. Create Zoomed Dataset (depends on splits of data-inside)
    print("--- Creating Zoomed Dataset ---")
    # process_zoom_dataset("data-inside", "data-inside-zoom")

if __name__ == "__main__":
    main()
