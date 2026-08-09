"""
CNN model training helpers.

Provides a shared dataframe/config-based interface for image regression with
optional auxiliary tabular features.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.config import Config
from src.training.artifacts import (
    atomic_torch_save,
    checkpoint_directory,
    checkpoint_stem,
)
from src.training.data_loader import get_feature_names_and_desc

IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_cnn_feature_spec(
    feature_set: str | None,
    depth: bool = False,
    per_type: bool = False,
) -> tuple[list[pl.Expr], str]:
    if feature_set is None:
        feature_desc = "cnn"
        feature_exprs: list[pl.Expr] = [pl.selectors.starts_with("fish_type_")]
        if depth:
            feature_exprs.append(
                pl.col(
                    [
                        "head_depth",
                        "body_depth",
                        "tail_depth",
                        "depth_gradient_raw",
                        "depth_gradient_abs",
                    ]
                )
            )
            feature_desc = f"{feature_desc}_depth"
        if per_type:
            feature_desc = f"{feature_desc}_per_type"
        return feature_exprs, feature_desc

    return get_feature_names_and_desc("cnn", feature_set, depth, per_type)


def select_aux_features(df: pl.DataFrame, feature_exprs: list[pl.Expr]) -> np.ndarray:
    if not feature_exprs:
        return np.empty((len(df), 0), dtype=np.float32)

    aux_df = df.select(feature_exprs)
    if aux_df.width == 0:
        return np.empty((len(df), 0), dtype=np.float32)

    return aux_df.to_numpy().astype(np.float32, copy=False)


class FishModel(nn.Module):
    """ResNet-based model with optional auxiliary features."""

    def __init__(self, aux_size: int = 0):
        super().__init__()

        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception as error:
            raise RuntimeError(
                "ImageNet ResNet-18 weights are required for the CNN experiment"
            ) from error

        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512 + aux_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, aux=None):
        features = self.backbone(x)
        if aux is not None and aux.numel() > 0:
            features = torch.cat([features, aux], dim=1)
        return self.fc(features)


class FishImageDataset(Dataset):
    """Dataset for blackout fish images and optional auxiliary features."""

    def __init__(
        self,
        names: list[str],
        targets: np.ndarray,
        image_dir: Path,
        aux_features: np.ndarray | None = None,
        transform: transforms.Compose | None = None,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.samples: list[tuple[Path, np.ndarray, float, str]] = []

        if aux_features is None:
            aux_features = np.empty((len(names), 0), dtype=np.float32)
        elif aux_features.ndim == 1:
            aux_features = aux_features.reshape(-1, 1)

        for idx, name in enumerate(names):
            image_path = self.image_dir / name
            if not image_path.exists():
                continue

            aux = np.asarray(aux_features[idx], dtype=np.float32)
            target = float(targets[idx])
            self.samples.append((image_path, aux, target, name))

        self.names = [sample[3] for sample in self.samples]
        self.aux_size = aux_features.shape[1] if aux_features.ndim == 2 else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, aux, target, name = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            torch.from_numpy(aux),
            torch.tensor(target, dtype=torch.float32),
            name,
        )


class CNNRegressor:
    def __init__(
        self,
        aux_size: int,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-4,
    ):
        self.aux_size = aux_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FishModel(aux_size=aux_size).to(self.device)
        self.aux_scaler: StandardScaler | None = None

    def _build_loader(
        self, dataset: FishImageDataset, shuffle: bool = False
    ) -> DataLoader:
        batch_size = (
            min(self.batch_size, len(dataset)) if len(dataset) > 0 else self.batch_size
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def fit(self, train_dataset: FishImageDataset, val_dataset: FishImageDataset):
        if len(train_dataset) == 0:
            raise ValueError("No training images available for CNN training.")

        train_loader = self._build_loader(train_dataset, shuffle=True)
        val_loader = self._build_loader(val_dataset) if len(val_dataset) > 0 else None

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

        early_stopping = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for images, aux, targets, _names in train_loader:
                images = images.to(self.device)
                aux = aux.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images, aux).flatten()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            current_val_loss = train_loss / len(train_dataset)
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, aux, targets, _names in val_loader:
                        images = images.to(self.device)
                        aux = aux.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(images, aux).flatten()
                        val_loss += criterion(outputs, targets).item() * images.size(0)

                current_val_loss = val_loss / len(val_dataset)

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss / len(train_dataset):.4f} - "
                f"Val Loss: {current_val_loss:.4f}"
            )

            if current_val_loss < best_val_loss:
                early_stopping = 0
                best_val_loss = current_val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

            if early_stopping >= EARLY_STOPPING_PATIENCE:
                break

            early_stopping += 1

        self.model.load_state_dict(best_state_dict)
        return self

    def predict(self, dataset: FishImageDataset) -> np.ndarray:
        if len(dataset) == 0:
            return np.asarray([], dtype=np.float32)

        loader = self._build_loader(dataset)

        self.model.eval()
        predictions: list[float] = []
        with torch.no_grad():
            for images, aux, _targets, _names in loader:
                images = images.to(self.device)
                aux = aux.to(self.device)

                outputs = self.model(images, aux).flatten()
                predictions.extend(outputs.cpu().numpy())

        return np.asarray(predictions, dtype=np.float32)

    def save(self, path: Path, metadata: dict | None = None) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "aux_size": self.aux_size,
            "metadata": metadata or {},
        }
        if self.aux_scaler is not None:
            checkpoint["aux_scaler_mean"] = self.aux_scaler.mean_
            checkpoint["aux_scaler_scale"] = self.aux_scaler.scale_
        atomic_torch_save(checkpoint, path)


def build_cnn_model(
    aux_size: int,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> CNNRegressor:
    return CNNRegressor(aux_size, epochs=epochs, batch_size=batch_size, lr=lr)


def build_image_dataset(
    df: pl.DataFrame,
    image_dir: Path,
    feature_exprs: list[pl.Expr],
    transform: transforms.Compose | None = None,
    aux_scaler: StandardScaler | None = None,
) -> FishImageDataset:
    names = df["name"].to_list()
    targets = df["length"].to_numpy().ravel().astype(np.float32)
    aux_features = select_aux_features(df, feature_exprs)
    if aux_scaler is not None and aux_features.shape[1] > 0:
        aux_features = aux_scaler.transform(aux_features).astype(np.float32, copy=False)
    return FishImageDataset(names, targets, image_dir, aux_features, transform)


def run_cnn_pipeline(
    df: pl.DataFrame,
    config: Config,
    feature_set: str | None,
    depth: bool = False,
    per_type: bool = False,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    feature_exprs, feature_desc = get_cnn_feature_spec(feature_set, depth, per_type)
    image_dir = config.dataset.output_dir / "blackout"
    if not image_dir.exists():
        raise FileNotFoundError(f"Blackout image directory not found: {image_dir}")

    transform = build_image_transform()
    train_df = df.filter(pl.col("is_train"))
    val_df = df.filter(pl.col("is_val"))

    aux_scaler: StandardScaler | None = None
    train_aux = select_aux_features(train_df, feature_exprs)
    if train_aux.shape[1] > 0:
        aux_scaler = StandardScaler().fit(train_aux)

    train_dataset = build_image_dataset(
        train_df, image_dir, feature_exprs, transform, aux_scaler
    )
    val_dataset = build_image_dataset(
        val_df, image_dir, feature_exprs, transform, aux_scaler
    )
    pred_dataset = build_image_dataset(
        df, image_dir, feature_exprs, transform, aux_scaler
    )
    if pred_dataset.names != df["name"].to_list():
        raise ValueError("Every training row must have a readable blackout image")

    print(f"Training {feature_desc} model on {config.dataset.name}...")
    model = build_cnn_model(
        aux_size=train_dataset.aux_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    model.aux_scaler = aux_scaler
    model.fit(train_dataset, val_dataset)
    predictions = model.predict(pred_dataset)

    model_dir = checkpoint_directory(config, checkpoint_dir)
    stem = checkpoint_stem(feature_desc, df, per_type)
    model.save(
        model_dir / f"{stem}.pth",
        {
            "model": "cnn",
            "dataset": config.dataset.name,
            "feature_set": feature_set,
            "depth": depth,
            "per_type": per_type,
            "feature_names": df.select(feature_exprs).columns,
            "image_size": IMAGE_SIZE,
            "image_source": "blackout",
            "backbone": str(models.ResNet18_Weights.DEFAULT),
            "imagenet_mean": IMAGENET_MEAN,
            "imagenet_std": IMAGENET_STD,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
        },
    )

    return pl.DataFrame(
        {
            "name": pred_dataset.names,
            feature_desc: np.round(predictions, 2),
        }
    )


def train_cnn_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str | None,
    depth: bool = False,
    per_type: bool = False,
    checkpoint_dir: Path | None = None,
) -> pl.DataFrame:
    return run_cnn_pipeline(
        df,
        config,
        feature_set=feature_set,
        depth=depth,
        per_type=per_type,
        epochs=100,
        batch_size=16,
        lr=1e-4,
        checkpoint_dir=checkpoint_dir,
    )
