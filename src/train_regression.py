import argparse
import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(split_path, feature_sets=["coords"], depth_model=None):
    df = pl.read_csv(split_path)
    
    required = []
    
    # Base Features
    for fs in feature_sets:
        if fs == "coords":
            required.extend(["Fish_w", "Fish_h", "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"])
        elif fs == "eye":
            required.extend(["Eye_w", "Eye_h", "Fish_w", "Fish_h"])
        elif fs == "scaled":
            required.extend(["Fish_w_scaled", "Fish_h_scaled"])
            
    # Depth Features
    if depth_model:
        models = ["v3", "v2", "pro"] if depth_model == "all" else [depth_model]
        for m in models:
            required.append(f"head_center_depth_{m}")
            required.append(f"tail_center_depth_{m}")

    # Auxiliary Features (Stats & Fish Type)
    # Auto-detect if they exist
    aux_cols = [c for c in df.columns if c.startswith("fish_type_") or c.startswith("species_")]
    required.extend(aux_cols)
    
    # Drop rows missing features
    # Check if columns exist
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
         # Filter out missing auxiliary columns from requirement if they prefer?
         # But for base features, we must fail.
         # Actually, for reliability, let's fail if base features missing, but warn/skip for aux?
         # "scaled" is a base feature now.
         
         # Simplify: Check base features.
         base_req = [c for c in required if not (c.startswith("fish_type_"))]
         missing_base = [c for c in base_req if c not in df.columns]
         if missing_base:
             raise ValueError(f"Missing required base columns: {missing_base}")
         
         # For aux, if missing, we just don't include them?
         # But we added them to `required` because we found them in `df.columns`! 
         # Wait, logic above: `aux_cols` comes from `df.columns`. So they exist.
         # So `missing_cols` should only contain missing BASE features.
         pass

    df = df.drop_nulls(subset=required)
    
    X = df.select(required).to_numpy()
    y = df.select("length").to_numpy().ravel()
    names = df.select("name").to_series().to_list()
    
    return X, y, names

def train_and_eval(train_path, val_path, test_path, feature_sets, dataset_name, depth_model=None):
    feature_desc = "_".join(sorted(feature_sets))
    if depth_model:
        feature_desc += f"_{depth_model}"
        
    print(f"Training Regression Model (Features: {feature_desc})")
    
    try:
        X_train, y_train, names_train = load_data(train_path, feature_sets, depth_model)
        X_val, y_val, names_val = load_data(val_path, feature_sets, depth_model)
        X_test, y_test, names_test = load_data(test_path, feature_sets, depth_model)
    except ValueError as e:
        print(f"Skipping training due to data issue: {e}")
        return
    
    print(f"Train size: {len(X_train)}")
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Eval & Save Predictions
    
    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    model_dir = os.path.join("checkpoints", dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for split_name, X, y, names in [("train", X_train, y_train, names_train), ("val", X_val, y_val, names_val), ("test", X_test, y_test, names_test)]:
        if len(X) == 0:
            print(f"{split_name}: No samples")
            continue
            
        pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, pred)
        print(f"{split_name.capitalize()} MAPE: {mape:.4f}")
        
        # Save output
        res_df = pl.DataFrame({
            "name": names,
            "gt_length": np.round(y, 1),
            "pred_length": np.round(pred, 1)
        })
        
        filename = f"{feature_desc}_{split_name}.csv"
        res_df.write_csv(os.path.join(pred_dir, filename))
        
    # Save Model
    save_name = f"regression_{feature_desc}.joblib"
    joblib.dump(model, os.path.join(model_dir, save_name))
    print(f"Model saved to {os.path.join(model_dir, save_name)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset name e.g. data-inside")
    parser.add_argument("--feature-set", type=str, nargs='+', choices=["coords", "scaled", "eye"], required=True)
    parser.add_argument("--depth-model", type=str, choices=["v3", "v2", "pro", "all"], default=None, help="Specific depth model or all")
    args = parser.parse_args()
    
    base_dir = f"data/{args.dataset}/processed"
    train_path = f"{base_dir}/processed_train.csv"
    val_path = f"{base_dir}/processed_val.csv"
    test_path = f"{base_dir}/processed_test.csv"
    
    train_and_eval(train_path, val_path, test_path, args.feature_set, args.dataset, args.depth_model)

if __name__ == "__main__":
    main()
