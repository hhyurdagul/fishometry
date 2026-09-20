"""Context vocabulary observed in data-outside/raw.csv.

Kept in source so inference and training do not depend on local raw data.
Identifiers, fish species, and target length are not context predictions.
"""

VLM_BOOLEAN_COLUMNS = [
    "is_held_by_human",
    "human_hand_visible",
    "human_body_visible",
    "is_curled",
    "has_measure_tape",
    "has_fishnet",
    "has_fishing_rod_reel",
    "has_lure_or_hook",
    "has_bucket_or_container",
    "has_other_manmade_objects",
    "water_visible",
    "glare_or_wet_reflections",
    "is_multiple_fish",
]

VLM_CATEGORICAL_VALUES = {
    "holding_method": [
        "hands_single",
        "hands_two",
        "lap_or_body",
        "lip_gripper_tool",
        "none",
        "stringer_hang",
    ],
    "curvature_degree": ["moderate_curve", "slight_curve", "straight"],
    "fish_orientation": [
        "diagonal_down",
        "diagonal_up",
        "horizontal_left_to_right",
        "horizontal_right_to_left",
        "vertical_head_down",
        "vertical_head_up",
    ],
    "fish_view_angle": [
        "dorsal_top",
        "head_on",
        "lateral_profile",
        "three_quarter_oblique",
        "ventral_bottom",
    ],
    "fish_completeness": [
        "full_body",
        "head_truncated",
        "partially_occluded",
        "tail_truncated",
    ],
    "fish_state": ["alive_fresh", "dead", "gutted_or_filleted"],
    "measure_tape_type": ["bump_board", "measuring_tape", "none", "ruler"],
    "fishnet_type": ["keepnet", "landing_net", "none"],
    "background_category": [
        "boat_deck",
        "concrete_dock",
        "grass_vegetation",
        "indoor_surface",
        "rocks_gravel",
        "sand_beach",
        "soil_mud_dirt",
        "water_freshwater",
        "water_sea",
        "wood_board_mat",
    ],
    "environment_type": ["boat_marine", "dock_pier", "indoor", "natural_outdoor"],
    "background_depth": ["close_surface", "far_horizon", "medium_ground"],
    "lighting_condition": [
        "bright_direct_sunlight",
        "diffuse_overcast",
        "indoor_light",
        "low_light_shadow",
        "night_flash",
    ],
    "image_quality": ["mild_blur", "overexposed", "sharp_clear", "underexposed"],
}

VLM_INTEGER_COLUMNS = ["num_fish"]

VLM_FEATURE_COLUMNS = [
    "is_held_by_human",
    "holding_method",
    "human_hand_visible",
    "human_body_visible",
    "is_curled",
    "curvature_degree",
    "fish_orientation",
    "fish_view_angle",
    "fish_completeness",
    "fish_state",
    "has_measure_tape",
    "measure_tape_type",
    "has_fishnet",
    "fishnet_type",
    "has_fishing_rod_reel",
    "has_lure_or_hook",
    "has_bucket_or_container",
    "has_other_manmade_objects",
    "background_category",
    "water_visible",
    "environment_type",
    "background_depth",
    "lighting_condition",
    "glare_or_wet_reflections",
    "image_quality",
    "num_fish",
    "is_multiple_fish",
]

VLM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        **{name: {"type": "BOOLEAN"} for name in VLM_BOOLEAN_COLUMNS},
        **{
            name: {"type": "STRING", "enum": values}
            for name, values in VLM_CATEGORICAL_VALUES.items()
        },
        "num_fish": {"type": "INTEGER", "minimum": 1},
    },
    "required": VLM_FEATURE_COLUMNS,
}
