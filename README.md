# Fish Depth

## Yolo Image Detection

### Train

- Inputs
    - data/yolo_input/raw_images_for_yolo_to_train
- Outputs
    - models/yolov11m-v0.py

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
    - data/depth/depth_input
    - data/fish_data_after_yolo.csv

1. On the non-labeled photos, run inference on the YOLOv11m model.
    - Get the detected features and save to data/yolo_output/image_features_before_rotation.csv
2. Use the rotation code to rotate images using head and tail coordinates.
    - Rotate images from data/yolo_input/raw_images_for_yolo_to_predict and save to data/yolo_output/rotated_images
3. Re-Run inference on the rotated images using the YOLOv11m model.
    - Get the detected features and save to data/yolo_output/image_features_after_rotation.csv
    - Copy the contents to data/depth/depth_input
    - Merge features with raw_fish_data.csv and save to fish_data_after_yolo.csv


## Depth Estimation

Input: data/raw_fish_data.csv, data/depth/depth_input

1. Run depth estimation on the raw data.
    - Save the contents to data/depth/depth_output
2. Get features from detected depth of the images.
    - Features
        - Depth_Head, Depth_Tail, Depth_Center
        - 
    - Merge it with raw_data and save to fish_output.csv