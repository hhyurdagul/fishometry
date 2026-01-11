import os
import glob
import argparse
import polars as pl
import altair as alt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

def analyze_predictions(dataset_name):
    print(f"Analyzing results (Altair) for {dataset_name}...")
    
    pred_dir = f"data/{dataset_name}/predictions"
    if not os.path.exists(pred_dir):
        print(f"No predictions found at {pred_dir}")
        return

    output_dir = f"data/{dataset_name}/processed/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(pred_dir, "*.csv"))
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        model_name, split = basename.replace(".csv", "").rsplit("_", 1)
        
        print(f"Processing {model_name} - {split}...")
        
        try:
            df = pl.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
            
        if df.height == 0:
            print("Empty file.")
            continue
            
        # Calculate metrics
        y_true = df["gt_length"].to_numpy()
        y_pred = df["pred_length"].to_numpy()
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Add Error and APE column for visualization
        df = df.with_columns([
            (pl.col("pred_length") - pl.col("gt_length")).alias("error"),
            ((pl.col("pred_length") - pl.col("gt_length")).abs() / pl.col("gt_length")).alias("ape")
        ])
        df = df.with_columns(pl.lit(f"{model_name}-{split}").alias("model"))
        
        title = f"{model_name} ({split}) - MAPE: {mape:.3f}, R2: {r2:.3f}"
        
        # Altair Charts
        # 1. Scatter: GT vs Pred
        base = alt.Chart(df.to_pandas()).encode(
            tooltip=['name', 'gt_length', 'pred_length', 'error', alt.Tooltip('ape', format='.2%')]
        ).properties(
            title=title,
            width=800
        )
        
        # ... (Scatter Code kept similar, implicitly re-using base)
        
        # Scatter: GT vs Pred
        scatter = base.mark_circle(size=60).encode(
            x=alt.X('gt_length', title='Ground Truth Length'),
            y=alt.Y('pred_length', title='Predicted Length'),
            color=alt.value('steelblue')
        ).interactive()
        
        line = base.mark_line(color='red', strokeDash=[5,5]).encode(
            x='gt_length',
            y='gt_length'
        )
        
        chart1 = scatter + line
        
        # 2. Prediction vs GT per Image
        c_gt = base.mark_circle(color='green', opacity=0.7).encode(
            x=alt.X('name', axis=alt.Axis(labelAngle=-90), sort='x'),
            y='gt_length'
        )
        c_pred = base.mark_circle(color='blue', opacity=0.7).encode(
            x=alt.X('name', axis=alt.Axis(labelAngle=-90), sort='x'),
            y='pred_length'
        )
        rule = base.mark_rule(color='gray').encode(
            x=alt.X('name', sort='x'),
            y='gt_length',
            y2='pred_length'
        )
        
        chart2 = (c_gt + c_pred + rule).properties(height=300, title="Predictions vs GT per Image")
        
        # 3. APE Bar Chart
        chart3 = base.mark_bar().encode(
            x=alt.X('name', axis=alt.Axis(labelAngle=-90), sort='x'),
            y=alt.Y('ape', title='Absolute Percentage Error', axis=alt.Axis(format='%')),
            color=alt.condition(
                alt.datum.error > 0,
                alt.value("orange"),  # Overestimation
                alt.value("purple")   # Underestimation
            )
        ).properties(height=200, title="APE per Image (Orange=Over, Purple=Under)")
        
        # Combine
        final_chart = chart1 & chart2 & chart3
        
        save_path = os.path.join(output_dir, f"{model_name}_{split}_interactive.html")
        final_chart.save(save_path)
        print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    analyze_predictions(args.dataset)

if __name__ == "__main__":
    main()
