import os
import shutil
import zipfile
import yaml
import random

random.seed(0)

class_map = {
    "0": "0", # Head
    "1": "1", # Tail
    "7": "2", # Salmon
    "22": "3", # Pike
    "27": "4", # Rudd
    "30": "5", # Sea_trout
}

def is_val():
    return random.random() <= 0.1


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )

def check_dataset(path):
    from ultralytics.hub import check_dataset

    check_dataset(path, task="detect")

root_images_dir = "data/images"
root_yolo_input_dir = "data/yolo_input"
root_yolo_data_dir = root_yolo_input_dir + "/raw_images_for_yolo_to_train"


def create_folders():
    if not os.path.exists(root_yolo_input_dir):
        os.makedirs(root_yolo_input_dir)

    if not os.path.exists(root_yolo_data_dir):
        os.makedirs(root_yolo_data_dir)

    if not os.path.exists(root_yolo_data_dir + "/images/train"):
        os.makedirs(root_yolo_data_dir + "/images/train")

    if not os.path.exists(root_yolo_data_dir + "/images/val"):
        os.makedirs(root_yolo_data_dir + "/images/val")

    if not os.path.exists(root_yolo_data_dir + "/labels/train"):
        os.makedirs(root_yolo_data_dir + "/labels/train")

    if not os.path.exists(root_yolo_data_dir + "/labels/val"):
        os.makedirs(root_yolo_data_dir + "/labels/val")


def create_yaml_file():
    with open(root_images_dir + "/classes.txt", "r") as f_in, open(
        root_yolo_data_dir + "/raw_images_for_yolo_to_train.yaml", "w"
    ) as f_out:
        data = {
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "Head",
                1: "Tail",
                2: "Salmon",
                3: "Pike",
                4: "Rudd",
                5: "Sea_trout",
            }
        }
        yaml.dump(data, f_out)


def get_all_dirs():
    root_dir = root_images_dir
    return [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d != "Removed"
    ]


def get_images_and_labels(dir_):
    images = []
    labels = []
    for file in os.listdir(dir_):
        if file.endswith(".jpg"):
            images.append(os.path.join(dir_, file))
            labels.append(os.path.join(dir_, file[:-4] + ".txt"))
    return images, labels


def copy_images_and_labels(images, labels):
    for image, label in zip(images, labels):
        try:
            if is_val():
                with open(label, "r") as f, open(root_yolo_data_dir + "/labels/val/" + os.path.basename(label), "w") as f_out:
                    for line in f.readlines():
                        class_id, x, y, w, h = line.split()
                        f_out.write(f"{class_map[class_id]} {x} {y} {w} {h}\n")
                shutil.copy(image, root_yolo_data_dir + "/images/val")
            else:
                with open(label, "r") as f, open(root_yolo_data_dir + "/labels/train/" + os.path.basename(label), "w") as f_out:
                    for line in f.readlines():
                        class_id, x, y, w, h = line.split()
                        f_out.write(f"{class_map[class_id]} {x} {y} {w} {h}\n")
                shutil.copy(image, root_yolo_data_dir + "/images/train")
        except Exception as e:
            print(f"Error processing {image}: {str(e)}")
            


def main():
    create_folders()
    create_yaml_file()
    dirs = get_all_dirs()
    for dir_ in dirs:
        images, labels = get_images_and_labels(dir_)
        copy_images_and_labels(images, labels)

    with zipfile.ZipFile(root_yolo_input_dir + "/yolo_fish_data.zip", "w") as zipf:
        zipdir(root_yolo_data_dir, zipf)
    
    check_dataset(root_yolo_input_dir + "/yolo_fish_data.zip")


if __name__ == "__main__":
    main()