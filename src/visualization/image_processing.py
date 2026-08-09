"""Safe image path resolution and display preparation."""

from pathlib import Path, PurePosixPath

import cv2
import numpy as np


def _safe_dataset_dir(dataset: str) -> Path:
    root = Path("data").resolve()
    candidate = (root / dataset).resolve()
    if candidate.parent != root or not candidate.is_dir():
        raise ValueError(f"Invalid dataset directory: {dataset}")
    return candidate


def _safe_image_name(image_name: str) -> Path:
    relative = PurePosixPath(image_name)
    if (
        not image_name
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
    ):
        raise ValueError(f"Invalid image name: {image_name}")
    return Path(*relative.parts)


def get_image_paths(dataset: str, image_name: str):
    """Resolve raw and processed image variants without path traversal."""
    dataset_dir = _safe_dataset_dir(dataset)
    relative = _safe_image_name(image_name)

    raw_path = next(
        (
            path
            for path in (
                dataset_dir / "raw" / relative,
                dataset_dir / "splits" / relative,
                dataset_dir / relative,
            )
            if path.is_file()
        ),
        None,
    )
    rotated = dataset_dir / "processed" / "rotated" / relative
    depth = dataset_dir / "processed" / "depth" / Path(f"{relative}.npy")
    blackout = dataset_dir / "processed" / "blackout" / relative
    if not blackout.is_file():
        for suffix in (".jpg", ".jpeg", ".png"):
            alternative = blackout.with_suffix(suffix)
            if alternative.is_file():
                blackout = alternative
                break

    return (
        str(raw_path) if raw_path else None,
        str(rotated),
        str(depth) if depth.is_file() else None,
        str(blackout),
    )


def _load_rgb(path: str | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    image = cv2.imread(path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def process_images(dataset: str, image_name: str, data_row: dict | None):
    """Load readable image variants and annotate rotated detections."""
    raw_path, rotated_path, depth_path, blackout_path = get_image_paths(
        dataset, image_name
    )

    image_raw = _load_rgb(raw_path)
    if image_raw is None:
        image_raw = np.zeros((200, 200, 3), dtype=np.uint8)

    image_rotated = _load_rgb(rotated_path)
    if image_rotated is not None and data_row:
        if data_row.get("Fish_x1") is not None:
            point_1 = (
                int(data_row["Fish_x1"]),
                int(data_row["Fish_y1"]),
            )
            point_2 = (
                int(data_row["Fish_x2"]),
                int(data_row["Fish_y2"]),
            )
            cv2.rectangle(image_rotated, point_1, point_2, (0, 120, 255), 2)
            cv2.putText(
                image_rotated,
                "Fish",
                point_1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 120, 255),
                2,
            )
        if data_row.get("Head_x1") is not None:
            center = (
                int((data_row["Head_x1"] + data_row["Head_x2"]) / 2),
                int((data_row["Head_y1"] + data_row["Head_y2"]) / 2),
            )
            cv2.circle(image_rotated, center, 5, (255, 0, 0), -1)
        if data_row.get("Tail_x1") is not None:
            center = (
                int((data_row["Tail_x1"] + data_row["Tail_x2"]) / 2),
                int((data_row["Tail_y1"] + data_row["Tail_y2"]) / 2),
            )
            cv2.circle(image_rotated, center, 5, (0, 255, 0), -1)

    image_depth = None
    if depth_path:
        try:
            depth = np.load(depth_path, allow_pickle=False)
            if depth.ndim == 2 and np.isfinite(depth).all():
                normalized = cv2.normalize(
                    depth,
                    np.empty_like(depth),
                    0,
                    255,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_8U,
                )
                image_depth = cv2.cvtColor(
                    cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA),
                    cv2.COLOR_BGR2RGB,
                )
        except (OSError, ValueError):
            image_depth = None

    return (
        image_raw,
        image_rotated,
        image_depth,
        _load_rgb(blackout_path),
    )
