from ultralytics import YOLO, checks, hub
from glob import glob
import pandas as pd

model = YOLO("checkpoints/yolov11m-fish_model.pt")

def train_yolo():
    checks()

    hub.login('1572b4afca7a8f8edf3315d1bc1ea4b48404bf9000')

    model = YOLO('https://hub.ultralytics.com/models/qReiuPj4S03FfAxTxPjo')
    results = model.train()
    return results

def image_have_head_and_tail(classes):
    have_head = True if 0 in classes else False
    have_tail = True if 1 in classes else False
    return have_head and have_tail

# - Features
#     - name
#     - image_width, image_height
#     - head_x1, head_y1, head_x2, head_y2
#     - tail_x1, tail_y1, tail_x2, tail_y2
#     - fish_x1, fish_y1, fish_x2, fish_y2
#     - head_center_x, head_center_y, tail_center_x, tail_center_y, fish_center_x, fish_center_y
#     - head_width, head_height, head_width_scaled, head_height_scaled
#     - tail_width, tail_height, tail_width_scaled, tail_height_scaled
#     - fish_width, fish_height, fish_width_scaled, fish_height_scaled

def get_features(result):
    boxes = result.boxes
    classes = boxes.cls
    if not image_have_head_and_tail(classes):
        return None

    image_name = result.path.split("/")[-1]
    image_height, image_width = result.orig_shape
    data = {
        "name": image_name,
        "image_height": image_height,
        "image_width": image_width,
    }
    for idx, class_ in enumerate(classes):
        xyxy = boxes.xyxy[idx].detach().tolist()
        xywh = boxes.xywh[idx].detach().tolist()
        if class_ == 0:
            data["head_x1"] = xyxy[0]
            data["head_y1"] = xyxy[1]
            data["head_x2"] = xyxy[2]
            data["head_y2"] = xyxy[3]
            data["head_x_center"] = xywh[0]
            data["head_y_center"] = xywh[1]
            data["head_width"] = xywh[2]
            data["head_height"] = xywh[3]
            data["head_width_scaled"] = xywh[2] / image_width
            data["head_height_scaled"] = xywh[3] / image_height
        elif class_ == 1:
            data["tail_x1"] = xyxy[0]
            data["tail_y1"] = xyxy[1]
            data["tail_x2"] = xyxy[2]
            data["tail_y2"] = xyxy[3]
            data["tail_x_center"] = xywh[0]
            data["tail_y_center"] = xywh[1]
            data["tail_width"] = xywh[2]
            data["tail_height"] = xywh[3]
            data["tail_width_scaled"] = xywh[2] / image_width
            data["tail_height_scaled"] = xywh[3] / image_height
        else:
            data["fish_x1"] = xyxy[0]
            data["fish_y1"] = xyxy[1]
            data["fish_x2"] = xyxy[2]
            data["fish_y2"] = xyxy[3]
            data["fish_x_center"] = xywh[0]
            data["fish_y_center"] = xywh[1]
            data["fish_width"] = xywh[2]
            data["fish_height"] = xywh[3]
            data["fish_width_scaled"] = xywh[2] / image_width
            data["fish_height_scaled"] = xywh[3] / image_height

    return data


def get_result_before_rotation():
    # Run inference
    images = glob("data/yolo_input/raw_images_for_yolo_to_predict/*.jpg")
    batch_size = len(images) // 4
    results = []
    for i in range(0, len(images), batch_size):
        results.extend(model(images[i:i+batch_size]) )

    data = []
    for res in results:
        features = get_features(res)
        if features:
            data.append(features)

    df = pd.DataFrame(data).round(2)
    df.to_csv("data/yolo_output/image_features_before_rotation.csv", index=False)

def get_result_after_rotation():
    # Run inference
    images = glob("data/yolo_output/rotated_images/*.jpg")
    batch_size = len(images) // 4
    results = []
    for i in range(0, len(images), batch_size):
        results.extend(model(images[i:i+batch_size]) )

    data = []
    for res in results:
        features = get_features(res)
        if features:
            data.append(features)

    df = pd.DataFrame(data).round(2)
    df.to_csv("data/yolo_output/image_features_after_rotation.csv", index=False)


if __name__ == '__main__':
    # train_yolo()
    # get_result_before_rotation()
    get_result_after_rotation()
