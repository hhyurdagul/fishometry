"""Versioned and atomic training-artifact helpers."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any

import joblib
import polars as pl
import torch

from src.config import Config


def checkpoint_directory(config: Config, override: Path | None = None) -> Path:
    path = override or Path("checkpoints") / config.dataset.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_stem(feature_desc: str, df: pl.DataFrame, per_type: bool) -> str:
    if not per_type:
        return feature_desc
    if "fish_type" not in df.columns:
        raise ValueError("Per-type checkpoints require a fish_type column")
    fish_types = df["fish_type"].drop_nulls().unique().to_list()
    if len(fish_types) != 1:
        raise ValueError("A per-type checkpoint must contain exactly one fish type")
    slug = re.sub(r"[^a-z0-9]+", "-", str(fish_types[0]).lower()).strip("-")
    if not slug:
        raise ValueError("Fish type cannot produce an empty checkpoint name")
    return f"{feature_desc}__{slug}"


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        torch.save(payload, stream)
    os.replace(temporary_path, path)


def atomic_joblib_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        joblib.dump(payload, stream)
    os.replace(temporary_path, path)
