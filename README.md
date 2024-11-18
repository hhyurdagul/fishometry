# Fish Depth

## Table of Contents

- [Yolo Image Detection](#yolo-image-detection)
- [Depth Estimation](#depth-estimation)
- [Length Prediction](#length-prediction)


## Preparation

- Inputs
    - data/images
- Outputs
    - data/yolo_input/raw_images_for_yolo_to_train
- Code
    - scripts/select_images_for_yolo_gradio.py
    - scripts/preprocess_yolo_input.py

## Yolo Image Detection

### Train

- Inputs
    - data/yolo_input/raw_images_for_yolo_to_train
- Outputs
    - models/yolov11m-v0.py
- Code
    - scripts/yolo.py

1. From raw photos pick different types of fish and copy to new folder.
2. Create train set by reordering the class numbers in label files.
3. Use this data to train a YOLOv11m model.
4. Save the model.

### Inference

- Inputs
    - data/yolo_input/raw_images_for_yolo_to_predict
    - data/raw_fish_data.csv
- Outputs
    - data/yolo_output/image_features_before_rotation.csv
    - data/yolo_output/rotated_images
    - data/yolo_output/image_features_after_rotation.csv
    - data/depth/depth_input_images
    - data/fish_data_after_yolo.csv

1. On the non-labeled photos, run inference on the YOLOv11m model.
    - Get the detected features and save to data/yolo_output/image_features_before_rotation.csv
2. Use the rotation code to rotate images using head and tail coordinates.
    - Rotate images from data/yolo_input/raw_images_for_yolo_to_predict and save to data/yolo_output/rotated_images
3. Re-Run inference on the rotated images using the YOLOv11m model.
    - Get the detected features and save to data/yolo_output/image_features_after_rotation.csv
    - Features
        - image_width, image_height
        - head_x1, head_y1, head_x2, head_y2
        - tail_x1, tail_y1, tail_x2, tail_y2
        - fish_x1, fish_y1, fish_x2, fish_y2
        - head_center_x, head_center_y, tail_center_x, tail_center_y, fish_center_x, fish_center_y
        - fish_width, fish_height, fish_width_scaled, fish_height_scaled
4. Copy the contents for depth estimation.
    - Copy data/yolo_output/image_features_after_rotation.csv to data/fish_data_after_yolo.csv
    - Copy data/yolo_output/rotated_images to data/depth/depth_input_images

## Depth Estimation

- Inputs
    - data/fish_data_after_yolo.csv
    - data/depth/depth_input_images
- Outputs
    - data/depth/depth_output_images
    - data/fish_data_after_depth.csv

1. Run depth estimation on the raw data.
    - Save the contents to data/depth/depth_output
2. Get features from detected depth of the images.
    - Merge it with data/fish_data_after_yolo.csv and save to data/fish_data_after_depth.csv
    - Features
        - Depth_head, Depth_tail, Depth_fish


## Length Prediction

- Inputs
    - data/fish_data_after_depth.csv
- Outputs
    - data/fish_data_after_length.csv

1. Run length prediction on the raw data.
    - Read the data from data/fish_data_after_depth.csv
    - Create Linear Regression model and predict the length
    - Save the contents to data/fish_data_after_length.csv
