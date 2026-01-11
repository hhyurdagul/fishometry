import os
import torch
import polars as pl
import json
from tqdm import tqdm
from ultralytics import YOLO
from src.steps.base_step import PipelineStep
from src.utils.io import load_image, get_raw_name

class YoloStep(PipelineStep):
    def __init__(self, config, stage="initial"):
        super().__init__(config)
        self.stage = stage
        self.model_path = config["models"]["yolo"]
        self.conf = config["params"]["yolo_conf"]
        self.input_dir = config["paths"]["raw"] if stage == "initial" else config["paths"]["output"] + "/rotated"

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        model = YOLO(self.model_path)
        
        coord_data = []
        names = df["name"].to_list()
        if "limit" in self.config["params"] and self.config["params"]["limit"]:
            names = names[:self.config["params"]["limit"]]
        
        # Prepare cache dir
        cache_dir = os.path.join(self.config["paths"]["output"], "cache", f"yolo_{self.stage}")
        os.makedirs(cache_dir, exist_ok=True)
        
        for name in tqdm(names, desc=f"YOLO {self.stage}"):
            image_path = os.path.join(self.input_dir, name)
            if not os.path.exists(image_path):
                continue
                
            cache_path = os.path.join(cache_dir, f"{name}.json")
            
            # Check cache
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r") as f:
                        prediction = json.load(f)
                    if prediction: # non-empty cache
                        coord_data.append({"name": name, **prediction})
                    continue # Skip inference
                except Exception:
                    pass # corrupted cache, re-run
            
            prediction = self.get_prediction(model, image_path)
            
            # Cache result (save even if None/Empty to avoid re-running?)
            # If prediction is None, we can save "null" or empty dict?
            # Current logic: coord_data only appends if prediction exists.
            # So if we save empty, next time we load empty and don't append. Correct.
            
            with open(cache_path, "w") as f:
                json.dump(prediction, f)
                
            if prediction:
                coord_data.append({"name": name, **prediction})
        
        if not coord_data:
            print(f"Warning: No VALID detections found in YoloStep {self.stage}.")
            # If we return df as is, subsequent steps might fail if they expect coords. 
            # But better than crashing here.
            # Ideally filter df to empty? Or just return.
            return df

        new_df = pl.DataFrame(coord_data)
        
        # Drop existing coord columns if any to avoid collision
        exclude_cols = new_df.filter(pl.col("name").is_in(new_df["name"])).columns
        exclude_cols = [c for c in exclude_cols if c != "name"]
        
        # We only want to join rows that exist in new_df (successful detections)
        # The inner join does that.
        
        # To avoid "duplicate column" error if columns exist in df:
        cols_to_drop = [c for c in new_df.columns if c != "name" and c in df.columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

        df = df.join(new_df, on="name", how="inner")

        return df

    def only_one_head_tail_and_fish_exist(self, cls: torch.Tensor) -> bool:
        # For data-inside, we might have class 2 (Eye).
        # We want 1 Head (0), 1 Tail (1), and 1 Fish.
        # If class 2 is Eye, we should ignore it when counting "Fish".
        
        has_head = (cls == 0).sum() == 1
        has_tail = (cls == 1).sum() == 1
        
        if not (has_head and has_tail):
            return False
            
        # Check for Fish
        # Filter out Head(0), Tail(1)
        others = cls[(cls != 0) & (cls != 1)]
        
        if "inside_fish_model" in self.model_path:
            # Ignore class 2 (Eye)
            others = others[others != 2]
            
        return others.nelement() == 1

    def get_prediction(self, model, image_path) -> dict | None:
        results = model.predict(image_path, conf=self.conf, verbose=False)
        if not results:
            return None
        
        prediction = results[0]
        cls = prediction.boxes.cls
        
        if not self.only_one_head_tail_and_fish_exist(cls):
            return None

        if "inside_fish_model" in self.model_path:
            name_map = {0: "Head", 1: "Tail", 2: "Eye"}
        else:
            name_map = {0: "Head", 1: "Tail"}
        
        # Get original image dimensions
        img_h, img_w = prediction.orig_shape
        
        data = {
            "img_w": img_w,
            "img_h": img_h
        }
        
        for box in prediction.boxes:
            c = box.cls.item()
            label = name_map.get(c, "Fish")
            data.update(self.get_xxyywh(label, box))
        return data

    def get_xxyywh(self, type: str, box):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        _, _, w, h = box.xywh[0].int().tolist()
        return {
            f"{type}_x1": x1,
            f"{type}_x2": x2,
            f"{type}_y1": y1,
            f"{type}_y2": y2,
            f"{type}_w": w,
            f"{type}_h": h,
        }
