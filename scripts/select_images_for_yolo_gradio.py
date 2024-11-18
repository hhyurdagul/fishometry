import os
import gradio as gr
from glob import glob
import cv2

index = -1
# Get all images in all directories if the directory is not Removed
images_list = [i for i in glob("data/images/*/*.jpg") if i.split("/")[2] != "Removed"]

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return None


def get_index_label(index):
    return f"Index: {index+1}/{len(images_list)}"


def select_image(image_name):
    global index
    index = images_list.index(image_name)
    return load_image(image_name), get_index_label(index)


def increase_index(index):
    return min(index + 1, len(images_list) - 1)

def next_image():
    global index
    index = increase_index(index)
    return load_image(images_list[index]), images_list[index], get_index_label(index)

def delete_image():
    global index
    image_name = images_list[index]
    image_path = "/".join(image_name.split("/")[-2:])
    old_root_dir = "/".join(image_name.split("/")[:-2])
    new_root_dir = old_root_dir + "/Removed"
    os.rename(image_name, os.path.join(new_root_dir, image_path))
    os.rename(image_name[:-4] + ".txt", os.path.join(new_root_dir, image_path[:-4] + ".txt"))
    index = increase_index(index)
    return load_image(images_list[index]), images_list[index], get_index_label(index)

# Create image viewer and two buttons labeled as next and delete
with gr.Blocks() as demo:
    with gr.Row():
        image_output = gr.Image(label="Image")
        with gr.Column():
            index_output = gr.Textbox(label="Image Index", value=f"Index: {index}/{len(images_list)}")
            selected_image = gr.Dropdown(label="Image Name", choices=images_list)
            next_button = gr.Button("Next")
            delete_button = gr.Button("Delete")


    selected_image.change(
        select_image,
        inputs=selected_image,
        outputs=[image_output, index_output]
    )

    next_button.click(
        next_image,
        outputs=[image_output, selected_image, index_output]
    )

    delete_button.click(
        delete_image,
        outputs=[image_output, selected_image, index_output]
    )


if __name__ == "__main__":
    demo.launch()