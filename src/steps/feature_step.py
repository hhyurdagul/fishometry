import polars as pl
import numpy as np
import os
import json
from src.steps.base_step import PipelineStep

class FeatureStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.train_path = config["paths"].get("train_split")
        self.stats = None
        self.max_dim = None
        self.fish_types = None
        
        # Load or calculate stats from training set
        if self.train_path and os.path.exists(self.train_path):
            self._calculate_stats(self.train_path)
        else:
            print("Warning: Train path not found or not provided in FeatureStep. Stats might be missing.")

    def _calculate_stats(self, train_path):
        df_train = pl.read_csv(train_path)
        
        # Max dimensions for scaling
        # Assuming img_w and img_h are present. If not, we can't scale properly yet?
        # Expectation: YoloStep has run and populated img_w/img_h.
        # But wait, we are calculating stats from the RAW split or PROCESSED split?
        # The split file probably doesn't have img_w/img_h yet if it's the raw split file?
        # Actually, the split file is just name, length, fish_type.
        # The PROCESSED training file will have img_w/img_h.
        # But we are running the pipeline TO create the processed file.
        # This creates a circular dependency if we need processed train to process train.
        
        # Solution: 
        # For scaling factor (max_dim), we can iterate over all raw images in the training list once.
        # Or, just use a large constant? No, user wanted ratio.
        # Better: iterate over the raw images for the training set to find max dims.
        # For fish stats (mean/median length), we can use the split file directly as it has 'length' and 'fish_type'.
        
        # 1. Fish Stats
        if "fish_type" in df_train.columns:
            # Group by fish_type
            stats_df = df_train.group_by("fish_type").agg([
                pl.col("length").mean().alias("mean_length"),
                pl.col("length").median().alias("median_length"),
                pl.col("length").min().alias("min_length"),
                pl.col("length").max().alias("max_length"),
                pl.col("length").std().alias("std_length")
            ])
            self.stats = {row["fish_type"]: {"mean": row["mean_length"], "median": row["median_length"],
                                             "min": row["min_length"], "max": row["max_length"], "std": row["std_length"]} 
                         for row in stats_df.to_dicts()}
            self.fish_types = sorted(list(self.stats.keys())) # For deterministic one-hot encoding
        else:
            # Global stats
            self.stats = {
                "global": {
                    "mean": df_train["length"].mean(),
                    "median": df_train["length"].median(),
                    "min": df_train["length"].min(),
                    "max": df_train["length"].max(),
                    "std": df_train["length"].std()
                }
            }
            self.fish_types = []

        # 2. Max Dimensions (for scaling)
        # We need to look up image sizes.
        # Since reading all images is slow, maybe we can do this continuously?
        # Or just find the max over the current batch and update? 
        # But 'scaled' implies a global scaler.
        # User said: "ratio between biggest image and current image sizes".
        # If we just do it per image (current / max_possible), we need max_possible.
        # Let's try to get it from the raw images directory for the names in train.
        
        # Actually, we can just defer this scaling calculation? 
        # No, FeatureStep needs to output it.
        # Let's assume we can scan the directory or maybe just use 5120xSomething as a known max?
        # The "identify" command earlier showed 5120 width.
        # To be robust, let's scan the training set images.
        
        # Optimization: We only do this once.
        raw_dir = self.config["paths"]["raw"]
        max_w, max_h = 0, 0
        
        # We only need to check if we don't have a cached value?
        # For now, let's just loop. It might be a few thousand images.
        # Checking file headers is fast. But we don't want to depend on 'identify'.
        # We can use cv2 or PIL.
        
        # Wait, if we are in 'train' split pipeline, 'df' passed to process() IS the training data (partially processed).
        # We can use that?
        # But for 'val' split, we need 'train' max.
        
        # Compromise: We calculate max_dim on the fly from the current DF if it's training, 
        # and save it? Or finding max of *entire* dataset is better?
        # User said "ratio between biggest image and current image sizes".
        # Usually this means max in dataset.
        
        # Let's iterate names in train_df and check sizes?
        # That's too slow if many images.
        # Alternative: The user might have meant just use the max of the CURRENT image? No "ratio between biggest image".
        
        # Let's try to find max_dim lazily or just pre-calculate.
        # I'll implement a helper that scans the train images quickly.
        
        pass # Will be done in _get_max_dims

    def _get_max_dims(self, names, raw_dir):
        # This might be slow. Is there a better way?
        # Maybe we can just store the max dim in a config or cache?
        # For now, let's assume standard maxes or scan.
        # To avoid being too slow, let's limit the scan or use a known max if possible.
        # But precise scaling requires precise max.
        
        # Let's use `identify` if available (Linux) for speed?
        # Or just PIL Image.open(path).size (lazy load).
        from PIL import Image
        
        max_w, max_h = 0, 0
        # Check first 100 to guess? No, unsafe.
        # Let's checks all.
        
        cache_path = os.path.join(self.config["paths"]["output"], "cache", "max_dims.json")
        if os.path.exists(cache_path):
             with open(cache_path, "r") as f:
                 d = json.load(f)
                 return d["w"], d["h"]

        print("Calculating max image dimensions from training set...")
        for name in names:
            p = os.path.join(raw_dir, name)
            if os.path.exists(p):
                try:
                    with Image.open(p) as img:
                        w, h = img.size
                        max_w = max(max_w, w)
                        max_h = max(max_h, h)
                except:
                    pass
        
        # Cache it
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"w": max_w, "h": max_h}, f)
            
        return max_w, max_h

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. Ensure fish_type exists (if not, add null/empty)
        if "fish_type" not in df.columns:
            # Try to join from original data if possible?
            # The input df usually comes from the raw csv which includes fish_type.
            # If not, we assume single species or check config?
            # For data-inside, raw.csv has no fish_type.
            pass

        # 2. Get Max Dims
        if self.max_dim is None:
            # We need the list of training names to find max dims.
            if self.train_path:
                train_names = pl.read_csv(self.train_path)["name"].to_list()
                self.max_dim = max(self._get_max_dims(train_names, self.config["paths"]["raw"]))
            else:
                 # Fallback: use max of current dataframe?
                 # Or just 1.0 (no scaling)
                 self.max_dim = 1.0
        
        # 3. Add Scaled Features
        # Requires img_w, img_h in df
        if "img_w" in df.columns and "img_h" in df.columns:
            # fish_w_scaled = fish_w * (max_dim / max(img_w, img_h))
            # ratio = max_dim / max(img_w, img_h)
            
            # Use map_rows or vector ops
            # Vector ops are hard with row-wise max(w, h).
            # pl.max_horizontal(["img_w", "img_h"])
            
            img_max = pl.max_horizontal(["img_w", "img_h"])
            scale_factor = self.max_dim / img_max
            
            df = df.with_columns([
                (pl.col("Fish_w") * scale_factor).alias("Fish_w_scaled"),
                (pl.col("Fish_h") * scale_factor).alias("Fish_h_scaled")
            ])
        
        # 4. Fish Type Encoding & Stats
        # If fish_type column exists
        if "fish_type" in df.columns:
            # One-hot encoding
            # We use self.fish_types to determine columns.
            for ft in self.fish_types:
                # col name: fish_type_Salmon
                df = df.with_columns(
                    (pl.col("fish_type") == ft).cast(pl.Int8).alias(f"fish_type_{ft}")
                )
            
            # Stats
            # Map mean/median based on fish_type
            # We can use a join or map_dict?
            # map_dict is easier.
            
            mean_map = {k: v["mean"] for k, v in self.stats.items()}
            median_map = {k: v["median"] for k, v in self.stats.items()}
            min_map = {k: v["min"] for k, v in self.stats.items()}
            max_map = {k: v["max"] for k, v in self.stats.items()}
            std_map = {k: v["std"] for k, v in self.stats.items()}
            
            # Handle unknown types? Fill with global mean?
            # For now assume known.
            
            df = df.with_columns([
                pl.col("fish_type").replace(mean_map).cast(pl.Float64).alias("species_mean_length"),
                pl.col("fish_type").replace(median_map).cast(pl.Float64).alias("species_median_length"),
                pl.col("fish_type").replace(min_map).cast(pl.Float64).alias("species_min_length"),
                pl.col("fish_type").replace(max_map).cast(pl.Float64).alias("species_max_length"),
                pl.col("fish_type").replace(std_map).cast(pl.Float64).alias("species_std_length")
            ])
            
        else:
            # No fish_type column (data-inside)
            # Use global stats
            if self.stats and "global" in self.stats:
                g_mean = self.stats["global"]["mean"]
                g_median = self.stats["global"]["median"]
                g_min = self.stats["global"]["min"]
                g_max = self.stats["global"]["max"]
                g_std = self.stats["global"]["std"]
                
                df = df.with_columns([
                    pl.lit(g_mean).alias("species_mean_length"),
                    pl.lit(g_median).alias("species_median_length"),
                    pl.lit(g_min).alias("species_min_length"),
                    pl.lit(g_max).alias("species_max_length"),
                    pl.lit(g_std).alias("species_std_length")
                ])
                
        return df
