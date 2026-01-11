import os
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm
from src.steps.base_step import PipelineStep

class VisualizeStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.input_dir = os.path.join(config["paths"]["output"], "rotated")
        self.depth_dir = os.path.join(config["paths"]["output"], "depth")
        self.segment_dir = os.path.join(config["paths"]["output"], "segment")
        self.output_dir = os.path.join(config["paths"]["output"], "visualization")
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        names = df["name"].to_list()
        
        for name in tqdm(names, desc="Visualizing"):
            image_path = os.path.join(self.input_dir, name)
            output_path = os.path.join(self.output_dir, name)
            
            if os.path.exists(output_path):
                continue

            if not os.path.exists(image_path):
                continue
                
            image = cv2.imread(image_path)
            
            # Load depth
            depth_path = os.path.join(self.depth_dir, name + ".npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                # Normalize depth for visualization
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                
                # Resize if needed (should match)
                if depth_vis.shape[:2] != image.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))
                    
                # Concatenate
                image = np.hstack((image, depth_vis))
            
            # Load segmentation
            seg_path = os.path.join(self.segment_dir, name + ".npy")
            if os.path.exists(seg_path):
                mask = np.load(seg_path)
                # Overlay mask
                # Mask is boolean or 0/1
                mask_vis = np.zeros_like(image)
                # We need to match the concatenated width if we concatenated depth
                # But mask corresponds to original image.
                
                # Let's visualize: Original with Mask | Depth
                
                # Reload original to overlay mask
                orig = cv2.imread(image_path)
                
                # Create colored mask
                colored_mask = np.zeros_like(orig)
                colored_mask[mask > 0] = [0, 255, 0] # Green
                
                masked_img = cv2.addWeighted(orig, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box and center if available
                row = df.filter(pl.col("name") == name)
                if row.height > 0:
                    data = row.to_dicts()[0]
                    # Fish bbox
                    if "Fish_x1" in data:
                        cv2.rectangle(masked_img, 
                                      (int(data["Fish_x1"]), int(data["Fish_y1"])), 
                                      (int(data["Fish_x2"]), int(data["Fish_y2"])), 
                                      (0, 0, 255), 2)
                        # Draw center point
                        cx = int((data["Fish_x1"] + data["Fish_x2"]) / 2)
                        cy = int((data["Fish_y1"] + data["Fish_y2"]) / 2)
                        cv2.circle(masked_img, (cx, cy), 5, (0, 255, 255), -1)
                
                # Re-stack
                if os.path.exists(depth_path):
                     image = np.hstack((masked_img, depth_vis))
                else:
                     image = masked_img

            cv2.imwrite(os.path.join(self.output_dir, name), image)
            
        return df
