"""
CNN Training

Trains a ResNet-based CNN with optional auxiliary features.

Usage:
    python -m src.training.cnn --dataset data-inside --feature-set coords --epochs 100
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import polars as pl
from PIL import Image
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

from src.training.data_loader import load_data, get_feature_description


class FishModel(nn.Module):
    """ResNet-based model with optional auxiliary features."""

    def __init__(self, aux_size=0):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Original ResNet fc is Linear(512, 1000).
        # We replace it with Identity to get 512 features.
        self.backbone.fc = nn.Identity()

        # New head
        self.fc = nn.Sequential(
            nn.Linear(512 + aux_size, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x, aux=None):
        features = self.backbone(x)
        if aux is not None:
            features = torch.cat([features, aux], dim=1)
        return self.fc(features)


class FishDataset(Dataset):
    """Dataset for fish images with auxiliary features."""

    def __init__(
        self,
        data_path,
        img_dir,
        transform=None,
        feature_sets=["coords"],
        depth_model=None,
    ):
        self.img_dir = img_dir
        self.transform = transform

        # Load data using regression logic
        try:
            self.X, self.y, self.names = load_data(data_path, feature_sets, depth_model)
        except ValueError as e:
            print(f"Warning: {e}")
            self.X, self.y, self.names = np.array([]), np.array([]), []

        # Filter existence and align
        valid_indices = []
        for i, name in enumerate(self.names):
            if os.path.exists(os.path.join(img_dir, name)):
                valid_indices.append(i)

        self.X = self.X[valid_indices] if len(self.X) > 0 else np.array([])
        self.y = self.y[valid_indices] if len(self.y) > 0 else np.array([])
        self.names = [self.names[i] for i in valid_indices]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        length = self.y[idx]
        aux = self.X[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(aux, dtype=torch.float32),
            torch.tensor(length, dtype=torch.float32),
            img_name,
        )


def train_cnn(
    dataset_name,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    feature_sets=["coords"],
    depth_model=None,
):
    feature_desc = get_feature_description(feature_sets, depth_model)

    print(f"Training CNN on {dataset_name} with features: {feature_desc}...")

    base_dir = f"data/{dataset_name}/processed"
    img_dir = os.path.join(base_dir, "blackout")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = FishDataset(
        f"{base_dir}/processed_train.csv", img_dir, transform, feature_sets, depth_model
    )
    val_set = FishDataset(
        f"{base_dir}/processed_val.csv", img_dir, transform, feature_sets, depth_model
    )
    test_set = FishDataset(
        f"{base_dir}/processed_test.csv", img_dir, transform, feature_sets, depth_model
    )

    print(f"Sizes: Train={len(train_set)} Val={len(val_set)} Test={len(test_set)}")

    if len(train_set) == 0:
        print("Error: No training data available.")
        return

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model
    aux_size = train_set.X.shape[1] if len(train_set.X) > 0 else 0
    print(f"Auxiliary Feature Size: {aux_size}")

    model = FishModel(aux_size=aux_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mape = float("inf")
    save_path = f"checkpoints/{dataset_name}/cnn_model_{feature_desc}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, aux, lbls, _ in train_loader:
            imgs, aux, lbls = imgs.to(device), aux.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, aux).flatten()
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for imgs, aux, lbls, _ in val_loader:
                imgs, aux = imgs.to(device), aux.to(device)
                outputs = model(imgs, aux).flatten()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(lbls.numpy())

        val_mape = (
            mean_absolute_percentage_error(val_targets, val_preds) if val_targets else 0
        )
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_set):.4f} - Val MAPE: {val_mape:.4f}"
        )

        if val_mape < best_mape:
            best_mape = val_mape
            torch.save(model.state_dict(), save_path)
            print("  Values improved, saved model.")

    # Final Evaluation & Save
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()

    pred_dir = os.path.join("data", dataset_name, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        preds = []
        targets = []
        names = []

        if len(loader.dataset) == 0:
            continue

        with torch.no_grad():
            for imgs, aux, lbls, img_names in loader:
                imgs, aux = imgs.to(device), aux.to(device)
                outputs = model(imgs, aux).flatten()

                preds.extend(outputs.cpu().numpy())
                targets.extend(lbls.numpy())
                names.extend(img_names)

        mape = mean_absolute_percentage_error(targets, preds)
        print(f"{split_name.capitalize()} MAPE: {mape:.4f}")

        # Save CSV
        df = pl.DataFrame(
            {
                "name": names,
                "gt_length": np.round(targets, 1),
                "pred_length": np.round(preds, 1),
            }
        )
        df.write_csv(os.path.join(pred_dir, f"cnn_{feature_desc}_{split_name}.csv"))


def main():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--feature-set",
        type=str,
        nargs="+",
        choices=["coords", "scaled", "eye"],
        default=["coords"],
    )
    parser.add_argument("--depth", action="store_true", help="Use depth v2 features")
    args = parser.parse_args()

    train_cnn(
        args.dataset,
        epochs=args.epochs,
        feature_sets=args.feature_set,
        depth_model="depth" if args.depth else None,
    )


if __name__ == "__main__":
    main()
