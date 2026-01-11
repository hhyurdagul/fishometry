import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error

def calculate_length(row):
    # Intrinsics
    fx = row["focal_length_x"]
    fy = row["focal_length_y"]
    cx = row["principal_point_x"]
    cy = row["principal_point_y"]
    
    # Head
    u_h = (row["Head_x1"] + row["Head_x2"]) / 2
    v_h = (row["Head_y1"] + row["Head_y2"]) / 2
    d_h = row["head_center_depth"]
    
    # Tail
    u_t = (row["Tail_x1"] + row["Tail_x2"]) / 2
    v_t = (row["Tail_y1"] + row["Tail_y2"]) / 2
    d_t = row["tail_center_depth"]
    
    if None in [fx, fy, cx, cy, d_h, d_t]:
        return None
        
    # Back-project to 3D
    def back_project(u, v, d):
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
        return np.array([X, Y, Z])
        
    P_h = back_project(u_h, v_h, d_h)
    P_t = back_project(u_t, v_t, d_t)
    
    # Euclidean distance
    length = np.linalg.norm(P_h - P_t)
    return length

df = pl.read_csv("../data/data-inside/raw.csv").select("name", "length").join(
    pl.read_csv("../data/data-inside/processed.csv"),
    on="name",
    how="inner",
)

# Calculate estimated length
# Polars apply is a bit tricky with multiple columns, convert to pandas or use struct
# For simplicity, let's use map_rows or iterate (slow but fine for small data)
# Or better, use expressions if possible, but back_project is custom.
# Let's use map_elements on a struct of all needed cols.

# needed_cols = [
#     "focal_length_x", "focal_length_y", "principal_point_x", "principal_point_y",
#     "Head_x1", "Head_x2", "Head_y1", "Head_y2", "head_center_depth",
#     "Tail_x1", "Tail_x2", "Tail_y1", "Tail_y2", "tail_center_depth"
# ]

needed_cols = [
    "Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"
]

# Filter out rows with nulls in needed cols
df_clean = df.drop_nulls(subset=needed_cols)

estimated_lengths = []
for row in df_clean.to_dicts():
    l = calculate_length(row)
    estimated_lengths.append(l)

df_clean = df_clean.with_columns(pl.Series("estimated_length", estimated_lengths))

print("Correlation between estimated and actual length:")
print(df_clean.select(pl.corr("length", "estimated_length")))

# Regression using estimated length as feature
X = df_clean.select("estimated_length").to_numpy()
y = df_clean.select("length").to_numpy()

n_samples = len(X)
cv = min(5, n_samples) if n_samples > 1 else 2

if n_samples < 2:
    print("Not enough samples for regression.")
else:
    # LeaveOneOut if very small? Or just standard CV with smaller folds.
    # cross_val_predict needs at least 2 folds.
    if cv < 2:
        print("Need at least 2 samples for cross validation.")
    else:
        try:
            pred = cross_val_predict(LinearRegression(fit_intercept=False), X, y, cv=cv)
            print(f"MAPE: {mean_absolute_percentage_error(y, pred)}")
        except ValueError as e:
            print(f"Regression failed: {e}")

print(df_clean.select("name", "length", "estimated_length").head())
print(f"MAPE: {mean_absolute_percentage_error(df_clean.select('length').to_numpy(), df_clean.select('estimated_length').to_numpy())}")


df_clean = df.drop_nulls(subset=["Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2"])
X = df_clean.select("Fish_x1", "Fish_x2", "Fish_y1", "Fish_y2").to_numpy()
y = df_clean.select("length").to_numpy()

pred = cross_val_predict(LinearRegression(), X, y, cv=cv)
print(f"MAPE: {mean_absolute_percentage_error(y, pred)}")


df = pl.read_csv("../data/data-inside/processed.csv")
print(df.select("length").describe())