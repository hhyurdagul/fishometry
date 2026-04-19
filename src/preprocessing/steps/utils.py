FISH_COORDINATE_FEATURES = [
    "name",
    "Head_x1",
    "Head_x2",
    "Head_y1",
    "Head_y2",
    "Tail_x1",
    "Tail_x2",
    "Tail_y1",
    "Tail_y2",
]


def get_center_coord(data: dict, prefix: str) -> tuple[int, int]:
    return (
        int((data[f"{prefix}_x1"] + data[f"{prefix}_x2"]) / 2),
        int((data[f"{prefix}_y1"] + data[f"{prefix}_y2"]) / 2),
    )
