
"""
Image-embedding regression helpers.

This module adds an outside-data model that follows the same dataframe/config
interface as the existing training helpers. It uses frozen torchvision image
features from the rotated images plus tabular geometry features, then fits a
per-species Ridge regressor.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from PIL import Image
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from src.config import Config
from src.training.data_loader import get_feature_names_and_desc


class PerSpeciesRegressor(BaseEstimator, RegressorMixin):
    """Fit one model per fish type, with a global fallback."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, x, y):
        species_codes = x[:, -1].astype(np.int64)
        base_x = x[:, :-1]
        self.global_model_ = clone(self.base_estimator).fit(base_x, y)
        self.models_ = {}
        for species_code in np.unique(species_codes):
            mask = species_codes == species_code
            if int(mask.sum()) >= 8:
                self.models_[int(species_code)] = clone(self.base_estimator).fit(base_x[mask], y[mask])
        return self

    def predict(self, x):
        species_codes = x[:, -1].astype(np.int64)
        base_x = x[:, :-1]
        predictions = self.global_model_.predict(base_x)
        for species_code, model in self.models_.items():
            mask = species_codes == species_code
            if np.any(mask):
                predictions[mask] = model.predict(base_x[mask])
        return predictions


class ImageNameDataset(Dataset):
    def __init__(self, image_dir: Path, names: list[str], transform):
        self.image_dir = image_dir
        self.names = names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_dir / self.names[idx]) as image:
            return self.transform(image.convert("RGB"))


def _build_efficientnet_b3():
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)
    model.classifier = torch.nn.Identity()
    return model, weights.transforms()


def _load_or_create_embeddings(df: pl.DataFrame, config: Config, model_dir: Path) -> np.ndarray:
    image_dir = config.dataset.output_dir / "rotated"
    if not image_dir.exists():
        raise FileNotFoundError(f"Rotated image directory not found: {image_dir}")

    names = df["name"].to_list()
    cache_path = model_dir / "efficientnet_b3_rotated_embeddings.npy"
    names_path = model_dir / "efficientnet_b3_rotated_embeddings_names.json"

    # Only reuse the cache when it was built for this exact ordered name list,
    # otherwise embeddings would bind to the wrong rows after processed.csv changes.
    if cache_path.exists() and names_path.exists():
        with names_path.open("r", encoding="utf-8") as f:
            cached_names = json.load(f)
        if cached_names == names:
            return np.load(cache_path)

    model, transform = _build_efficientnet_b3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(
        ImageNameDataset(image_dir, names, transform),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            outputs = model(images.to(device)).flatten(1)
            embeddings.append(outputs.cpu().numpy())

    result = np.vstack(embeddings).astype(np.float32)
    np.save(cache_path, result)
    with names_path.open("w", encoding="utf-8") as f:
        json.dump(names, f)
    return result


def train_efficientnet_ridge_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str = "",
    depth: bool = False,
    per_type: bool = False,
) -> pl.DataFrame:
    if "fish_type" not in df.columns:
        raise ValueError("efficientnet_ridge requires fish_type in the dataframe.")

    feature_set = feature_set or "features"
    feature_exprs, feature_desc = get_feature_names_and_desc(
        "efficientnet_ridge",
        feature_set,
        depth,
        per_type,
    )

    model_dir = Path("checkpoints") / config.dataset.name
    model_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings = _load_or_create_embeddings(df, config, model_dir)
    species_lookup = {name: idx for idx, name in enumerate(sorted(df["fish_type"].unique().to_list()))}
    species_codes = np.asarray([species_lookup[name] for name in df["fish_type"].to_list()], dtype=np.float32)

    tabular = df.select(feature_exprs).to_numpy().astype(np.float32)
    x = np.concatenate([tabular, image_embeddings, species_codes.reshape(-1, 1)], axis=1)
    y = df["length"].to_numpy().astype(np.float32)
    train_mask = df["is_train"].to_numpy().astype(bool)

    base_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=np.logspace(-2, 10, 80))),
        ]
    )
    model = PerSpeciesRegressor(base_model)

    print(f"Training {feature_desc} model on {config.dataset.name}...")
    model.fit(x[train_mask], y[train_mask])
    predictions = model.predict(x)

    model_path = model_dir / f"{feature_desc}.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_set": feature_set,
            "depth": depth,
            "per_type": per_type,
            "species_lookup": species_lookup,
            "image_backbone": "efficientnet_b3",
            "image_source": "rotated",
        },
        model_path,
    )

    return pl.DataFrame(
        {
            "name": df["name"].to_numpy(),
            feature_desc: np.round(predictions, 2),
        }
    )
