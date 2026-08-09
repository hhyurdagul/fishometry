"""Content-aware manifests and atomic artifact writers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

MANIFEST_SCHEMA = 1


def file_sha256(path: Path) -> str:
    """Hash current file bytes for artifact provenance."""
    digest = hashlib.sha256()
    with path.resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def build_signature(
    *,
    inputs: Mapping[str, Path],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe every input that changes the meaning of one artifact."""
    return {
        "schema": MANIFEST_SCHEMA,
        "inputs": {name: input_record(path) for name, path in sorted(inputs.items())},
        "parameters": json.loads(json.dumps(parameters, sort_keys=True, default=str)),
    }


def manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_name(f"{artifact_path.name}.meta.json")


def cache_matches(artifact_path: Path, signature: Mapping[str, Any]) -> bool:
    metadata_path = manifest_path(artifact_path)
    if not artifact_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            return json.load(stream) == signature
    except (OSError, ValueError, TypeError):
        return False


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary_path, path)


def write_manifest(artifact_path: Path, signature: Mapping[str, Any]) -> None:
    atomic_write_json(manifest_path(artifact_path), signature)


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        np.save(stream, array)
    os.replace(temporary_path, path)


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"Failed to write image {path}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def content_manifest(root: Path, names: Sequence[str]) -> list[dict[str, str]]:
    """Bind an ordered dataframe name list to exact source image bytes."""
    return [{"name": name, "sha256": file_sha256(root / name)} for name in names]
