"""Multi-view DINOv2 regression.

This is the same shape of model as `embedding.py` — frozen image features plus
tabular geometry, fitted per fish type with a ridge — but it changes three
things that turned out to matter on the outdoor data:

1. The backbone is DINOv2 (self-supervised ViT) instead of an ImageNet
   classifier. Its features carry far more usable scale/context information
   here than EfficientNet, ConvNeXt, Swin or EfficientNetV2 features do.
2. Each image is embedded under several resolutions and framings, and the
   views are concatenated. Different framings keep different amounts of the
   surrounding scene, which is where the monocular scale cues live.
3. The per-species fit is shrunk toward the global fit instead of replacing it
   outright, which stabilises the species with few training rows.

It also derives extra geometry features from the columns already present in
processed.csv, so no preprocessing rerun is needed.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from PIL import Image
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.training.data_loader import get_feature_names_and_desc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALPHAS = np.logspace(-3, 8, 120)
MIN_SPECIES_ROWS = 8
SPECIES_BLEND = 0.85

# (hub model, image source, square input size, framing). DINOv2 uses 14px
# patches, so the sizes are multiples of 14. The rotated views keep the scene
# context; the blackout view sees the isolated fish only and adds a different
# kind of error, which is why it earns a place in the concatenation.
VIEWS: tuple[tuple[str, str, int, str], ...] = (
    ("dinov2_vitb14", "rotated", 224, "squash"),
    ("dinov2_vitb14", "rotated", 336, "squash"),
    ("dinov2_vitb14", "rotated", 448, "crop"),
    ("dinov2_vitb14", "rotated", 518, "squash"),
    ("dinov2_vitb14", "blackout", 224, "squash"),
    ("dinov2_vitl14", "rotated", 224, "squash"),
    ("dinov2_vitl14", "rotated", 336, "squash"),
    ("dinov2_vitl14", "rotated", 448, "squash"),
    ("dinov2_vitl14", "rotated", 448, "crop"),
)

EPS = 1e-6

GEOMETRY_COLUMNS = [
    "g_ht_rel", "g_ht_over_fishdiag", "g_fish_diag_rel", "g_head_frac", "g_tail_frac",
    "g_head_over_tail", "g_head_aspect", "g_tail_aspect", "g_cx", "g_cy",
    "g_img_aspect", "g_img_diag", "g_fill", "g_compact", "g_elong", "g_mask_rel",
    "g_major_rel", "g_depth_contrast", "g_depth_span",
]

LOG_SOURCES = [
    "relative_w", "relative_h", "relative_area", "fish_area", "g_ht_rel",
    "g_fish_diag_rel", "g_head_frac", "g_tail_frac", "g_mask_rel", "g_major_rel",
    "mask_area", "mask_perimeter", "major_axis", "minor_axis", "g_img_diag",
]


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add scale-free geometry ratios and log sizes derived from existing columns."""
    c = pl.col
    img_diag = (c("Image_w") ** 2 + c("Image_h") ** 2).sqrt()
    head_cx = (c("Head_x1") + c("Head_x2")) / 2
    head_cy = (c("Head_y1") + c("Head_y2")) / 2
    tail_cx = (c("Tail_x1") + c("Tail_x2")) / 2
    tail_cy = (c("Tail_y1") + c("Tail_y2")) / 2
    head_tail = ((head_cx - tail_cx) ** 2 + (head_cy - tail_cy) ** 2).sqrt()
    fish_diag = (c("Fish_w") ** 2 + c("Fish_h") ** 2).sqrt()

    df = df.with_columns(
        # Straight-line head-to-tail pixel span: the direct analogue of the label.
        g_ht_rel=head_tail / img_diag,
        g_ht_over_fishdiag=head_tail / (fish_diag + EPS),
        g_fish_diag_rel=fish_diag / img_diag,
        # Body proportions, dimensionless and therefore species-descriptive.
        g_head_frac=(c("Head_w") * c("Head_h")).sqrt() / (fish_diag + EPS),
        g_tail_frac=(c("Tail_w") * c("Tail_h")).sqrt() / (fish_diag + EPS),
        g_head_over_tail=(
            (c("Head_w") * c("Head_h")) / (c("Tail_w") * c("Tail_h") + EPS)
        ).sqrt(),
        g_head_aspect=c("Head_w") / (c("Head_h") + EPS),
        g_tail_aspect=c("Tail_w") / (c("Tail_h") + EPS),
        # Placement in frame: a weak ground-plane / camera-distance cue.
        g_cx=((c("Fish_x1") + c("Fish_x2")) / 2) / c("Image_w"),
        g_cy=((c("Fish_y1") + c("Fish_y2")) / 2) / c("Image_h"),
        g_img_aspect=c("Image_w") / c("Image_h"),
        g_img_diag=img_diag,
        # Segmentation shape descriptors.
        g_fill=c("mask_area") / (c("Fish_w") * c("Fish_h") + EPS),
        g_compact=c("mask_perimeter") ** 2 / (c("mask_area") + EPS),
        g_elong=c("major_axis") / (c("minor_axis") + EPS),
        g_mask_rel=c("mask_area").sqrt() / img_diag,
        g_major_rel=c("major_axis") / img_diag,
        # Relative-depth contrasts between the fish and its surroundings.
        g_depth_contrast=c("body_depth") - c("background_depth"),
        g_depth_span=(c("head_depth") - c("tail_depth")).abs(),
    )
    return df.with_columns(
        [(pl.col(col).abs() + EPS).log().alias(f"log_{col}") for col in LOG_SOURCES]
    )


class ShrunkPerSpeciesRidge(BaseEstimator, RegressorMixin):
    """Global ridge, blended with a per-species ridge where the species has data."""

    def __init__(self, blend: float = SPECIES_BLEND, min_rows: int = MIN_SPECIES_ROWS):
        self.blend = blend
        self.min_rows = min_rows

    @staticmethod
    def _pipeline() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", RidgeCV(alphas=ALPHAS)),
        ])

    def fit(self, x, y):
        codes = x[:, -1].astype(np.int64)
        base_x = x[:, :-1]
        self.global_model_ = self._pipeline().fit(base_x, y)
        self.models_ = {}
        for code in np.unique(codes):
            mask = codes == code
            if int(mask.sum()) >= self.min_rows:
                self.models_[int(code)] = self._pipeline().fit(base_x[mask], y[mask])
        return self

    def predict(self, x):
        codes = x[:, -1].astype(np.int64)
        base_x = x[:, :-1]
        predictions = self.global_model_.predict(base_x)
        for code, model in self.models_.items():
            mask = codes == code
            if np.any(mask):
                species = model.predict(base_x[mask])
                predictions[mask] = (
                    self.blend * species + (1 - self.blend) * predictions[mask]
                )
        return predictions


class _ImageDataset(Dataset):
    def __init__(self, image_dir: Path, names: list[str], size: int, framing: str):
        self.image_dir = image_dir
        self.names = names
        self.size = size
        self.framing = framing

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_dir / self.names[idx]) as image:
            image = image.convert("RGB")
            if self.framing == "squash":
                # Keep the whole frame, accept the aspect-ratio distortion.
                image = image.resize((self.size, self.size), Image.BILINEAR)
            else:
                # Short-side resize then centre crop.
                w, h = image.size
                scale = self.size / min(w, h)
                image = image.resize((max(self.size, round(w * scale)),
                                      max(self.size, round(h * scale))), Image.BILINEAR)
                w, h = image.size
                left, top = (w - self.size) // 2, (h - self.size) // 2
                image = image.crop((left, top, left + self.size, top + self.size))

            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(
                IMAGENET_STD, np.float32
            )
            return torch.from_numpy(array).permute(2, 0, 1)


def _view_embeddings(
    names: list[str], processed_dir: Path, model_dir: Path, view: tuple[str, str, int, str]
) -> np.ndarray:
    backbone, source, size, framing = view
    image_dir = processed_dir / source
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    tag = f"{backbone}_{source}_{size}_{framing}"
    cache_path = model_dir / f"dino_{tag}.npy"
    names_path = model_dir / f"dino_{tag}_names.json"

    # Only reuse the cache when it was built for this exact ordered name list,
    # otherwise embeddings would bind to the wrong rows after processed.csv changes.
    if cache_path.exists() and names_path.exists():
        with names_path.open("r", encoding="utf-8") as f:
            if json.load(f) == names:
                return np.load(cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", backbone).to(device).eval()
    loader = DataLoader(
        _ImageDataset(image_dir, names, size, framing),
        batch_size=4 if size > 336 else 8,
        shuffle=False,
        num_workers=4,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images in loader:
            chunks.append(model(images.to(device)).flatten(1).float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = np.vstack(chunks).astype(np.float32)
    np.save(cache_path, result)
    with names_path.open("w", encoding="utf-8") as f:
        json.dump(names, f)
    return result


def train_dino_ridge_model(
    df: pl.DataFrame,
    config: Config,
    feature_set: str = "",
    depth: bool = False,
    per_type: bool = False,
) -> pl.DataFrame:
    if "fish_type" not in df.columns:
        raise ValueError("dino_ridge requires fish_type in the dataframe.")

    feature_set = feature_set or "features"
    feature_exprs, feature_desc = get_feature_names_and_desc(
        "dino_ridge", feature_set, depth, per_type
    )
    # The derived geometry is always supplied; it is cheap and never hurt in testing.
    df = add_derived_features(df)
    feature_exprs = list(feature_exprs) + [pl.col(GEOMETRY_COLUMNS + [f"log_{c}" for c in LOG_SOURCES])]

    model_dir = Path("checkpoints") / config.dataset.name
    model_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = config.dataset.output_dir

    names = df["name"].to_list()
    embeddings = [_view_embeddings(names, processed_dir, model_dir, v) for v in VIEWS]

    species_lookup = {n: i for i, n in enumerate(sorted(df["fish_type"].unique().to_list()))}
    species_codes = np.asarray(
        [species_lookup[n] for n in df["fish_type"].to_list()], dtype=np.float32
    )

    tabular = df.select(feature_exprs).to_numpy().astype(np.float32)
    x = np.concatenate([tabular, *embeddings, species_codes.reshape(-1, 1)], axis=1)
    y = df["length"].to_numpy().astype(np.float32)
    train_mask = df["is_train"].to_numpy().astype(bool)

    print(f"Training {feature_desc} model on {config.dataset.name}...")
    model = ShrunkPerSpeciesRidge().fit(x[train_mask], y[train_mask])
    predictions = model.predict(x)

    joblib.dump(
        {
            "model": model,
            "feature_set": feature_set,
            "depth": depth,
            "per_type": per_type,
            "species_lookup": species_lookup,
            "image_backbone": "dinov2_multiview",
            "views": VIEWS,
        },
        model_dir / f"{feature_desc}.joblib",
    )

    return pl.DataFrame({
        "name": df["name"].to_numpy(),
        feature_desc: np.round(predictions, 2),
    })
