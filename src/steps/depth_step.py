import os
import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
import sys
from src.steps.base_step import PipelineStep

os.environ["DA3_LOG_LEVEL"] = "ERROR"

class DepthStep(PipelineStep):
    def __init__(self, config):
        super().__init__(config)
        self.input_dir = os.path.join(config["paths"]["output"], "rotated")
        self.output_dir = os.path.join(config["paths"]["output"], "depth")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.repo_path = config["models"]["depth_repo"]
        src_path = os.path.join(self.repo_path, "src")
        if src_path not in sys.path:
            sys.path.append(src_path)
            
        self.device = config["params"]["device"]

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        # models_to_run = ["v3", "v2", "pro"]
        models_to_run = ["v2"]
        
        # We'll collect depth columns for each model
        # We need to initialize the new columns in df if they don't exist, 
        # or we join them at the end. Joining at the end is cleaner.
        
        # Dictionary to store results for each model: {model_name: [{name: ..., fish_center_depth: ...}, ...]}
        all_results = {}

        for model_key in models_to_run:
            # Check if we need to run inference
            model_out_dir = os.path.join(self.output_dir, f"depth-{model_key}")
            os.makedirs(model_out_dir, exist_ok=True)
            
            names = df["name"].to_list()
            missing = []
            for name in names:
                # We also need to check input existence? Assuming input exists if in DF.
                # But strict check:
                if not os.path.exists(os.path.join(model_out_dir, name + ".npy")):
                    missing.append(name)
            
            model = None
            if len(missing) > 0:
                print(f"Processing Depth Model: {model_key} (Missing {len(missing)}/{len(names)})")
                model = self._load_model(model_key)
                if model is None:
                    print(f"Skipping generation for {model_key} due to load failure. Will process existing only.")
            else:
                 print(f"Depth Model {model_key}: All {len(names)} files exist. Skipping model load.")

            # Collection results
            model_results = []
            
            # If we have missing files but no model, we can't generate them.
            # We will just skip them in the loop.
            
            for name in tqdm(names, desc=f"Depth {model_key.upper()}"):
                image_path = os.path.join(self.input_dir, name)
                output_name = name + ".npy"
                output_path = os.path.join(model_out_dir, output_name)
                
                # Check exist
                if os.path.exists(output_path):
                    try:
                        depth = np.load(output_path)
                        intrinsics = {} 
                        # Note: We rely on V3 being run at least once for intrinsics if we want them?
                        # Or if we want to save intrinsics we should have saved them.
                        # For now, we just proceed.
                        entry = self._extract_metrics(df, name, depth, model_key, intrinsics)
                        model_results.append(entry)
                        continue
                    except Exception as e:
                         print(f"Error reading {output_path}: {e}")
                         # If readout fails, try to re-infer if model available
                         if model is None:
                             continue

                if not os.path.exists(image_path):
                    continue
                
                if model is None:
                    # Missing file and no model
                    continue

                # Inference
                try:
                    depth, intrinsics = self._run_inference(model, model_key, image_path)
                    
                    # Resize if needed
                    image = cv2.imread(image_path)
                    h, w = image.shape[:2]
                    
                    if depth.shape != (h, w):
                        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                    # Save
                    np.save(output_path, depth)
                    
                    # Extract metrics
                    entry = self._extract_metrics(df, name, depth, model_key, intrinsics)
                    model_results.append(entry)
                    
                except Exception as e:
                    print(f"Error executing {model_key} on {name}: {e}")
                    # traceback.print_exc()

            all_results[model_key] = model_results
            
            if model is not None:
                del model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        # Merge all results into initial dataframe
        for model_key, results in all_results.items():
            if not results:
                continue
            
            model_df = pl.DataFrame(results)
            # Join on name
            df = df.join(model_df, on="name", how="left")

        return df

    def _load_model(self, model_key):
        if model_key == "v3":
            try:
                from depth_anything_3.api import DepthAnything3
                model = DepthAnything3(model_name="da3nested-giant-large")
                model.to(self.device)
                return model
            except ImportError:
                print("Could not import DepthAnything3")
                return None
        
        elif model_key == "v2":
            try:
                # V2 import logic
                v2_path = self.config["models"].get("depth_v2_repo")
                if v2_path and v2_path not in sys.path:
                    sys.path.append(v2_path)
                
                from depth_anything_v2.dpt import DepthAnythingV2
                
                # V2 needs specific config
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                # Using vitl as a good default or we can make it configurable
                encoder = 'vitl'
                model = DepthAnythingV2(**model_configs[encoder])
                
                # Check for weights file - assuming user might need to download or we use a default location
                # For now let's hope it loads or checks for weights. 
                # Actually V2 usually requires load_state_dict explicitly.
                # Since we don't have explicit weights path in config yet, let's assume a default location or try to find it.
                # If we don't have weights, V2 initialization typically fails or produces random noise. 
                # We will check if 'checkpoints' dir has them.
                checkpoint_path = os.path.join("checkpoints", f"depth_anything_v2_{encoder}.pth")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                else:
                    print(f"Warning: DepthAnythingV2 weights not found at {checkpoint_path}. Attempting to download or run without weights (not recommended).")
                    # In a real scenario, we might want to auto-download here.
                    self._download_v2_weights(encoder, checkpoint_path)
                    if os.path.exists(checkpoint_path):
                         model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                
                model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"Could not load DepthAnythingV2: {e}")
                import traceback
                traceback.print_exc()
                return None

        elif model_key == "pro":
            try:
                pro_path = self.config["models"].get("depth_pro_repo")
                if pro_path and pro_path not in sys.path:
                    sys.path.append(os.path.join(pro_path, "src"))
                    sys.path.append(pro_path)

                import depth_pro
                from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
                
                # Update checkpoint path to absolute or relative from cwd
                checkpoint_path = os.path.join(pro_path, "checkpoints", "depth_pro.pt")
                
                # Update config
                cfg = DEFAULT_MONODEPTH_CONFIG_DICT
                cfg.checkpoint_uri = checkpoint_path
                
                # Depth Pro load
                model, transform = depth_pro.create_model_and_transforms(config=cfg, device=self.device)
                model.eval()
                # Return tuple to keep transform handy or attach it to model
                model.transform = transform 
                return model
            except Exception as e:
                print(f"Could not load Depth Pro: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return None

    def _download_v2_weights(self, encoder, path):
        print(f"Downloading DepthAnythingV2 {encoder} weights to {path}...")
        url_map = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        if encoder not in url_map:
            print("Unknown encoder for auto-download.")
            return

        url = url_map[encoder]
        try:
            import torch 
            torch.hub.download_url_to_file(url, path)
        except Exception as e:
            print(f"Failed to download weights: {e}")

    def _run_inference(self, model, model_key, image_path):
        import torch
        intrinsics = {}
        
        if model_key == "v3":
            # inference returns a Prediction object (list)
             pred = model.inference(
                image=[image_path],
                export_dir=None,
                export_format="mini_npz"
            )
             depth = pred.depth[0]
             if hasattr(pred, "intrinsics") and pred.intrinsics is not None:
                 K = pred.intrinsics[0]
                 intrinsics = {
                    "focal_length_x": K[0, 0],
                    "focal_length_y": K[1, 1],
                    "principal_point_x": K[0, 2],
                    "principal_point_y": K[1, 2],
                    "skew": K[0, 1]
                }
             return depth, intrinsics

        elif model_key == "v2":
            # V2 infer_image returns numpy array
            image = cv2.imread(image_path)
            depth = model.infer_image(image) # (H, W)
            return depth, intrinsics # No intrinsics from V2 usually

        elif model_key == "pro":
            # Depth Pro inference
            import depth_pro
            image, _, f_px = depth_pro.load_rgb(image_path)
            image = self._apply_transform(model.transform, image)
            
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"] # torch tensor usually
            if isinstance(depth, torch.Tensor):
                depth = depth.squeeze().cpu().numpy()
            
            # Depth Pro estimates focal length usually
            if f_px is not None:
                 intrinsics["focal_length_estimate"] = float(f_px)
                 
            return depth, intrinsics
            
        return np.zeros((10,10)), {}

    def _apply_transform(self, transform, image):
        # Helper for Depth Pro transform handling if needed
        # depth_pro load_rgb returns a Tensor or PIL image? 
        # Actually depth_pro.load_rgb returns (image: torch.Tensor, original_image: PIL.Image, f_px: float)
        # But wait, create_model_and_transforms returns a transform that expects specific input.
        # Let's check typical usage. 
        # usage: image, _, f_px = depth_pro.load_rgb(path)
        #        image = transform(image)
        return transform(image)

    def _extract_metrics(self, df, name, depth, model_key, intrinsics):
        row = df.filter(pl.col("name") == name)
        metrics = {"name": name}
        
        if row.height > 0:
            data = row.to_dicts()[0]
            
            def get_center_depth(prefix):
                if f"{prefix}_x1" not in data: return None
                cx = int((data[f"{prefix}_x1"] + data[f"{prefix}_x2"]) / 2)
                cy = int((data[f"{prefix}_y1"] + data[f"{prefix}_y2"]) / 2)
                
                if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                    return depth[cy, cx]
                return None

            metrics[f"fish_center_depth_{model_key}"] = get_center_depth("Fish")
            metrics[f"head_center_depth_{model_key}"] = get_center_depth("Head")
            metrics[f"tail_center_depth_{model_key}"] = get_center_depth("Tail")
            
            # Save intrinsics with model suffix if it's V3 (or others if we want)
            # Logic requested: "when there is a head_depth in data from now on there should be 3 depth value for each model"
            # We already did that above.
            
            # If intrinsics present, save them. 
            # Note: overwriting intrinsics from different models might be confusing if they differ. 
            # User request: "Data manipulations and all saved depth images should stay as well."
            # V3 was providing intrinsics. Let's keep V3 intrinsics as the "main" ones or namespaced.
            # Assuming V3 is the 'best' for intrinsics for now, or we namespace them.
            for k, v in intrinsics.items():
                metrics[f"{k}_{model_key}"] = v
                
        return metrics
