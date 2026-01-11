import cv2
import os
import sys
import random

# Add src to path
sys.path.append(os.getcwd())

from src.utils.zoom import augment_zoom

def verify_on_real_data():
    input_dir = "data/data-inside/raw"
    output_dir = "tests/real_data_check"
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick a few sample images
    sample_images = ["1.jpeg", "10.jpeg", "100.jpeg"]
    
    print(f"Generating visual verification samples in {output_dir}")
    
    for img_name in sample_images:
        path = os.path.join(input_dir, img_name)
        if not os.path.exists(path):
            print(f"Skipping {path} (not found)")
            continue
            
        print(f"Processing {img_name}...")
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to read {path}")
            continue
            
        # Save original copy
        cv2.imwrite(os.path.join(output_dir, f"orig_{img_name}"), img)
        
        # Zoom In (using typical ranges from our code)
        # Magnitude 0.2 (was Ratio 0.8)
        rin = 0.2
        img_in = augment_zoom(img, "in", rin)
        cv2.imwrite(os.path.join(output_dir, f"zoom_in_{rin}_{img_name}"), img_in)
        
        # Zoom Out (using typical ranges)
        # Magnitude 0.2
        rout = 0.2
        img_out = augment_zoom(img, "out", rout)
        cv2.imwrite(os.path.join(output_dir, f"zoom_out_{rout}_{img_name}"), img_out)
        
    print("Done.")

if __name__ == "__main__":
    verify_on_real_data()
