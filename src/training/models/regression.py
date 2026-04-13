"""
Regression model training helpers.

Provides a shared dataframe/config-based interface for linear regression,
XGBoost, and MLP models.
"""

import os

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor

from src.config import Config
from src.training.data_loader import get_feature_names_and_desc


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): # type: ignore
        return self.X[idx], self.y[idx]


class FishMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


class MLPRegressor:
    def __init__(self, input_dim: int, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FishMLP(input_dim).to(self.device)

    def fit(self, X, y, X_val, y_val):
        dataset = TabularDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TabularDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state_dict = None

        for _ in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch).flatten()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)


            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self.model(X_batch).flatten()
                    val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)

            current_val_loss = val_loss / len(val_dataset)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        return self

    def predict(self, X):
        dataset = TabularDataset(X, np.zeros(len(X), dtype=np.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _y_batch in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch).flatten()
                predictions.extend(outputs.cpu().numpy())

        return np.asarray(predictions)

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
            },
            path,
        )


def build_linear_model() -> Pipeline:
    return Pipeline([("regressor", LinearRegression())])


def build_xgboost_model(
    n_estimators: int = 100,
    max_depth: int = 16,
    learning_rate: float = 0.1,
) -> Pipeline:
    return Pipeline([
        
        ("regressor", XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
        ))
    ])

def build_mlp_model(
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> MLPRegressor:
    return MLPRegressor(input_dim, epochs=epochs, batch_size=batch_size, lr=lr)


MODELS = {
    "linear": build_linear_model,
    "xgboost": build_xgboost_model,
    "mlp": build_mlp_model,
}


def run_model_pipeline(
    model_name: str,
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool = False,
) -> pl.DataFrame:
    features, feature_desc = get_feature_names_and_desc(model_name, feature_set, depth)

    X_train = df.filter(pl.col("is_train")).select(features).to_numpy()
    y_train = df.filter(pl.col("is_train")).select("length").to_numpy().ravel()

    X_val = df.filter(pl.col("is_val")).select(features).to_numpy()
    y_val = df.filter(pl.col("is_val")).select("length").to_numpy().ravel()

    print(f"Training {feature_desc} model on {config.dataset.name}...")
    
    if model_name == "linear":
        model = build_linear_model()
        model.fit(X_train, y_train)
    elif model_name == "xgboost":
        model = build_xgboost_model()
        model.fit(X_train, y_train, regressor__eval_set=[(X_val, y_val)], regressor__verbose=False)
    else:
        model = build_mlp_model(X_train.shape[1])
        model.fit(X_train, y_train, X_val, y_val)

    model_dir = os.path.join("checkpoints", config.dataset.name)
    os.makedirs(model_dir, exist_ok=True)

    if isinstance(model, MLPRegressor):
        model.save(os.path.join(model_dir, feature_desc + ".pth"))
    else:
        joblib.dump(model, os.path.join(model_dir, feature_desc + ".joblib"))
    
    pred = pl.DataFrame({
        "name": df["name"].to_numpy(), 
        feature_desc: model.predict(df.select(features).to_numpy())
    })

    return pred




def train_linear_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool = False,
) -> pl.DataFrame:
    return run_model_pipeline("linear", df, config, feature_set, depth)


def train_xgboost_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool = False,
) -> pl.DataFrame:
    return run_model_pipeline("xgboost", df, config, feature_set, depth)

def train_mlp_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str,
    depth: bool = False,
) -> pl.DataFrame:
    return run_model_pipeline("mlp", df, config, feature_set, depth)
