from .baseline import train_baseline
from .regression import (
    train_linear_model,
    train_mlp_model,
    train_xgboost_model,
)
from .cnn import train_cnn_model
from .dino import train_dino_ridge_model
from .embedding import train_efficientnet_ridge_model


__all__ = [
    "train_baseline",
    "train_linear_model",
    "train_xgboost_model",
    "train_mlp_model",
    "train_cnn_model",
    "train_efficientnet_ridge_model",
    "train_dino_ridge_model",
]
