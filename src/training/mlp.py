"""
MLP (Multi-Layer Perceptron) Training

Trains a simple feedforward neural network.

Usage:
    python -m src.training.mlp --dataset data-inside --feature-set coords --epochs 200
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from src.training.data_loader import load_data, get_feature_description


class TabularDataset(Dataset):
    """Dataset for tabular data."""

    def __init__(self, X, y, names):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.names = names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.names[idx]


class FishMLP(nn.Module):
    """Simple MLP for fish length prediction."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    dataset_name,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    feature_sets=["coords"],
    depth_model=None,
    include_stats=False,
):
    feature_desc = get_feature_description(feature_sets, depth_model, include_stats)

    print(f"Training MLP on {dataset_name} with features: {feature_desc}...")

    base_dir = f"data/{dataset_name}/processed"

    # Load Data
    try:
        X_train, y_train, names_train = load_data(
            f"{base_dir}/processed_train.csv", feature_sets, depth_model, include_stats
        )
        X_val, y_val, names_val = load_data(
            f"{base_dir}/processed_val.csv", feature_sets, depth_model, include_stats
        )
        X_test, y_test, names_test = load_data(
            f"{base_dir}/processed_test.csv", feature_sets, depth_model, include_stats
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    if len(X_val) > 0:
        X_val = scaler.transform(X_val)
    if len(X_test) > 0:
        X_test = scaler.transform(X_test)

    train_set = TabularDataset(X_train, y_train, names_train)
    val_set = TabularDataset(X_val, y_val, names_val)
    test_set = TabularDataset(X_test, y_test, names_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_set, batch_size=batch_size) if len(val_set) > 0 else None
    )
    test_loader = (
        DataLoader(test_set, batch_size=batch_size) if len(test_set) > 0 else None
    )

    input_dim = X_train.shape[1]
    model = FishMLP(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mape = float("inf")
    save_path = f"checkpoints/{dataset_name}/mlp_model_{feature_desc}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).flatten()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # Validation
        val_mape = 0
        if val_loader:
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch).flatten()
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(y_batch.numpy())
            val_mape = mean_absolute_percentage_error(val_targets, val_preds)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_set):.4f} - Val MAPE: {val_mape:.4f}"
            )

        if val_mape < best_mape:
            best_mape = val_mape
            torch.save(model.state_dict(), save_path)

    # Evaluation
    model.load_state_dict(torch.load(save_path))
    model.eval()

    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        if not loader:
            continue

        preds = []
        targets = []
        batch_names = []

        with torch.no_grad():
            for X_batch, y_batch, n_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).flatten()
                preds.extend(outputs.cpu().numpy())
                targets.extend(y_batch.numpy())
                batch_names.extend(n_batch)

        mape = mean_absolute_percentage_error(targets, preds)
        print(f"{split_name.capitalize()} MAPE: {mape:.4f}")

        df = pl.DataFrame(
            {
                "name": batch_names,
                "gt_length": np.round(targets, 1),
                "pred_length": np.round(preds, 1),
            }
        )
        df.write_csv(os.path.join(pred_dir, f"mlp_{feature_desc}_{split_name}.csv"))


def main():
    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--feature-set",
        type=str,
        nargs="+",
        choices=["coords", "scaled", "eye"],
        default=["coords"],
    )
    parser.add_argument("--depth", action="store_true", help="Use depth v2 features")
    parser.add_argument(
        "--stats", action="store_true", help="Include species stats features"
    )
    args = parser.parse_args()

    train_mlp(
        args.dataset,
        epochs=args.epochs,
        feature_sets=args.feature_set,
        depth_model="depth" if args.depth else None,
        include_stats=args.stats,
    )


if __name__ == "__main__":
    main()
