import os
import gradio as gr
from glob import glob
import cv2

index = -1
root_image_dir = "data/images"
removed_image_dir = "data/images/Removed"
# Get all images in all directories if the directory is not Removed
images_list = [i for i in glob(f"{root_image_dir}/*/*.jpg") if i.split("/")[2] != "Removed"]

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
    image_path = images_list[index]
    image_name = image_path.split("/")[-1]

    txt_path = image_path[:-4] + ".txt"
    txt_name = image_name[:-4] + ".txt"

    sub_image_dir = image_path.split("/")[-2]

    new_image_path = os.path.join(removed_image_dir, sub_image_dir, image_name)
    new_txt_path = os.path.join(removed_image_dir, sub_image_dir, txt_name)

    os.rename(image_path, new_image_path)
    os.rename(txt_path, new_txt_path)

    print(f"Image moved from {image_path} to {new_image_path}")

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