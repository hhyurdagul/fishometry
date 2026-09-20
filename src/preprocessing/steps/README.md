# Preprocessing Steps

This directory contains the ordered image and feature transformations used by the preprocessing runner. Each stage receives the current metadata table, creates or loads per-image artifacts, and either joins new columns or writes images for downstream training.

## Shared Image Selection

Final detection, depth, segmentation, and blackout creation use:

- `processed/rotated/` when the selected dataset enables rotation.
- `raw/` when rotation is disabled.

Initial detection and original-image context always refer to raw images. Image name is the join and cache key throughout.

YOLO, depth, and SAM release their loaded models when their respective stage finishes, including on failure. Reference cycles are collected before unused CUDA memory is returned, so earlier stages do not compete with segmentation for GPU capacity.

## `yolo.py`: Detection

### Model loading

- The configured checkpoint must exist when the step is constructed.
- The model is loaded lazily on the first uncached prediction.
- Inference uses a confidence threshold of `0.8`.
- CPU or GPU selection is delegated to the detector library.

### Accepted prediction

A prediction is discarded when it contains no boxes or contains more than one box with the same raw class ID. Configured labels are positional: all class IDs except the final fallback are mapped by list position, and any remaining ID receives the final label.

For every accepted box, the step stores:

- `x1`, `x2`, `y1`, and `y2` integer boundaries.
- Integer box width and height.
- The image dimensions returned by inference.

The table is joined by image name and then all null rows are dropped. This means an image must provide every column present in the joined detection schema to survive.

### Passes and caches

- Initial pass: reads raw images and writes `processed/cache/yolo_initial/<name>.json`.
- Final rotated pass: reads rotated images and writes `processed/cache/yolo_rotated/<name>.json`.
- With rotation disabled, both passes use the initial raw-image cache.

Cached JSON is accepted only when its sidecar manifest matches the image hash, checkpoint hash, class order, confidence, pass variant, and step version.

## `rotate.py`: Horizontal Alignment

Rotation requires head and tail box centers from the initial detector.

1. Compute the vector from tail center to head center.
2. Rotate around the image center until that vector is horizontal and points toward positive x.
3. Expand the canvas so the original image is not clipped by the affine transform.
4. Vertically flip cases whose rotation magnitude exceeds 90 degrees.
5. Estimate and crop the largest axis-aligned rectangle that excludes rotation borders.
6. Write the result with the original filename under `processed/rotated/`.

Existing output images are reused only when their sidecar manifests match the source image, detection coordinates, rotation parameters, and step version. Coordinate transformation is deliberately not propagated: the final detector measures body parts again in the rotated coordinate system.

When rotation is disabled, the stage returns the table unchanged.

## `depth.py`: Relative Depth

### Model

- Uses the vendored Depth Anything V2 implementation with a ViT-L encoder.
- Loads the configured weights lazily on the first uncached image.
- Chooses CUDA when available, otherwise CPU.
- Resizes the inferred array to the input image dimensions when needed.

The output is relative monocular depth. Values are useful for within-image and learned comparisons but are not calibrated physical distances.

### Cache and sampled values

The full floating-point map is stored at `processed/depth/<name>.npy`. Its manifest covers the source image, depth checkpoint, model settings, and step version; any mismatch reruns inference.

The step calculates box centers for head, fish body, and tail. It takes the median of a clipped 9 by 9 patch around each center to reduce single-pixel noise. Joined columns are:

- `head_depth`
- `body_depth`
- `tail_depth`
- `depth_gradient_raw`, equal to head depth minus tail depth
- `depth_gradient_abs`, the absolute head-to-tail difference

Any image or metric error is logged, and null removal over these five required outputs excludes that row. Missing weights trigger the configured download path before model construction.

## `segment.py`: Fish Mask and Shape

### Mask generation

- Loads Segment Anything ViT-L lazily from the configured checkpoint.
- Uses the final head and tail centers as two positive point prompts.
- Requests one mask rather than multiple candidates.
- Runs the image encoder and prompt/mask decoder on CUDA when available; expands and thresholds the low-resolution mask on CPU at the original image dimensions. Point coordinates and geometric feature units remain unchanged.
- Clears the per-image embedding after every attempt. A CUDA out-of-memory error retries the same image once on CPU after releasing its CUDA state; the next uncached image attempts CUDA again. CPU retries can be substantially slower and still require enough system RAM.
- Saves the binary result at `processed/segment/<name>.npy`.

Existing masks are reused only when the source image, SAM checkpoint, prompt coordinates, and step version match the sidecar manifest.

After an earlier CUDA failure, rerun the normal preprocessing command using the existing split and caches. Successful masks remain reusable; images without a valid cached mask are retried. Only an unsuccessful CPU retry follows the normal per-image error/drop behavior.

### Geometric features

The mask is converted to an 8-bit binary image. External contours are found, and only the largest contour is treated as the fish. The stage extracts:

| Column | Meaning |
| --- | --- |
| `mask_area` | Largest contour area in pixels |
| `mask_perimeter` | Closed contour length in pixels |
| `major_axis` | Larger fitted-ellipse axis, or zero for fewer than five contour points |
| `minor_axis` | Smaller fitted-ellipse axis, or zero for fewer than five contour points |
| `solidity` | Contour area divided by convex-hull area |

No contour produces no feature row, which causes that image to be removed during the null-drop handoff.

## `blackout.py`: Isolated Fish Image

Blackout creation consumes the selected image and its saved segmentation mask.

1. Resize the mask with nearest-neighbor interpolation if it does not match the image.
2. Set every background pixel to black.
3. Find the mask bounds and crop tightly around the fish.
4. Scale the crop so its longest constrained dimension fits a 224 by 224 canvas while preserving aspect ratio.
5. Center it on a black RGB canvas.
6. Write it under `processed/blackout/` with the original filename and extension.

Existing output images are reused. Missing images, missing masks, empty masks, and per-image exceptions are skipped. This step does not join columns or remove metadata rows, so downstream training must still find a blackout image for every selected CNN record.

## `vlm.py`: Original-Image Context

The context stage describes the unmodified scene rather than the rotated fish geometry.

### Structured fields

The 27 context fields and allowed categories match the observed vocabulary in
`data/data-outside/raw.csv`, declared in `src/context_features.py`. Raw annotations
only define the vocabulary; `VLMStep` still predicts values from the image.
`name`, `fish_type`, and the target `length` are excluded from the response schema.
Every context field is required, and partial/occluded fish are supported.

| Field | Values |
| --- | --- |
| `holding_method` | `hands_single`, `hands_two`, `lap_or_body`, `lip_gripper_tool`, `none`, `stringer_hang` |
| `curvature_degree` | `moderate_curve`, `slight_curve`, `straight` |
| `fish_orientation` | `diagonal_down`, `diagonal_up`, `horizontal_left_to_right`, `horizontal_right_to_left`, `vertical_head_down`, `vertical_head_up` |
| `fish_view_angle` | `dorsal_top`, `head_on`, `lateral_profile`, `three_quarter_oblique`, `ventral_bottom` |
| `fish_completeness` | `full_body`, `head_truncated`, `partially_occluded`, `tail_truncated` |
| `fish_state` | `alive_fresh`, `dead`, `gutted_or_filleted` |
| `measure_tape_type` | `bump_board`, `measuring_tape`, `none`, `ruler` |
| `fishnet_type` | `keepnet`, `landing_net`, `none` |
| `background_category` | `boat_deck`, `concrete_dock`, `grass_vegetation`, `indoor_surface`, `rocks_gravel`, `sand_beach`, `soil_mud_dirt`, `water_freshwater`, `water_sea`, `wood_board_mat` |
| `environment_type` | `boat_marine`, `dock_pier`, `indoor`, `natural_outdoor` |
| `background_depth` | `close_surface`, `far_horizon`, `medium_ground` |
| `lighting_condition` | `bright_direct_sunlight`, `diffuse_overcast`, `indoor_light`, `low_light_shadow`, `night_flash` |
| `image_quality` | `mild_blur`, `overexposed`, `sharp_clear`, `underexposed` |
| `num_fish` | Positive integer count of visible fish |

Boolean fields: `is_held_by_human`, `human_hand_visible`, `human_body_visible`, `is_curled`, `has_measure_tape`, `has_fishnet`, `has_fishing_rod_reel`, `has_lure_or_hook`, `has_bucket_or_container`, `has_other_manmade_objects`, `water_visible`, `glare_or_wet_reflections`, `is_multiple_fish`.

The stage reads `GEMINI_API_KEY` from `.env.json` during construction. Each uncached request is rate-spaced by five seconds and must return JSON conforming to the declared schema.

### Cache and image source

The cache path is `processed/cache/vlm/<name>.json`. Its sidecar manifest covers
the source image, full response schema, prompt, remote model identifier, and step
version. The new schema invalidates old six-field responses. Matching responses
are reused; otherwise the original raw image is submitted, regardless of rotation.
Rows without complete context values do not survive this stage.

## `feature.py`: Engineered Features

This final stage does not write image artifacts. It transforms the enriched table.

### Relative geometry

| Column | Calculation |
| --- | --- |
| `relative_w` | Fish box width / image width |
| `relative_h` | Fish box height / image height |
| `relative_area` | Fish box area / image area |
| `fish_aspect` | Fish box width / fish box height |
| `fish_area` | Square root of fish box area |

Detector output stores width and height in their matching columns, so all relative-geometry denominators use the intended image axis.

### Fish-type encoding

When fish types are enabled, the stage creates `fish_type_<value>` dummy columns and then restores the original categorical `fish_type` column for grouping and display.

### Context encoding

- All boolean context fields become integers; `num_fish` remains an integer.
- All categorical fields become dummy columns named `<field>_<value>`, including
  the three background-depth levels (`close_surface`, `medium_ground`, `far_horizon`).
- Only context columns present in the input are encoded. Fish species encoding is separate.
- Training's `features` bundle selects all encoded context fields from the same shared vocabulary.

## `utils.py`: Coordinate Contract

The shared coordinate selection requires image name and all four box boundaries for `Head`, `Fish`, and `Tail`. Center coordinates are integer averages of each box's horizontal and vertical bounds. Depth and segmentation depend on this complete schema.

## Failure and Reuse Summary

- Per-image errors are logged and become explicit row attrition in the preprocessing report.
- Each step drops nulls only from the columns it requires or produces.
- Missing global prerequisites, malformed input contracts, missing columns, or feature-encoding errors abort the pipeline before final publication.
- Artifacts are reused only when content-aware manifests match; unreferenced historical artifacts are reported but not pruned automatically.

See the parent [preprocessing README](../README.md) for the full run contract and downstream handoff.

## Temporary context-stage override

The preprocessing runner currently skips `VLMStep` and passes
`encode_vlm_features=False` to `FeatureStep`. This preserves raw context columns
without interpreting their categories as VLM output, and removes the runner's
Gemini credential requirement. `FeatureStep` still defaults to encoding VLM
features for other callers. Restore both extraction and encoding together.
