import gradio as gr
import pandas as pd
import cv2
import os
import numpy as np

# Read the CSV file
df = pd.read_csv('data/fish_data_after_length.csv')

def draw_points_on_image(image, row):
    """Draw head, tail, and fish bounding boxes on the image"""
    # Draw head bounding box (red)
    cv2.rectangle(image, 
                 (int(row['head_x1']), int(row['head_y1'])),
                 (int(row['head_x2']), int(row['head_y2'])),
                 (0, 0, 255), 2)
    
    # Draw fish bounding box (green)
    cv2.rectangle(image, 
                 (int(row['fish_x1']), int(row['fish_y1'])),
                 (int(row['fish_x2']), int(row['fish_y2'])),
                 (0, 255, 0), 2)
    
    # Draw tail bounding box (blue)
    cv2.rectangle(image, 
                 (int(row['tail_x1']), int(row['tail_y1'])),
                 (int(row['tail_x2']), int(row['tail_y2'])),
                 (255, 0, 0), 2)
    
    return image

def filter_and_display(selected_name):
    if selected_name is None:
        return None
    
    # Get the selected fish data
    selected_row = df[df['name'] == selected_name].iloc[0]
    
    # Read the image
    image_path = os.path.join('data/yolo_output/rotated_images', selected_row['name'])
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw points on the image
    annotated_image = draw_points_on_image(image.copy(), selected_row)
    
    return annotated_image

def get_filtered_names(loss_threshold):
    if loss_threshold is None:
        return []
    filtered_df = df[df['Loss'] >= loss_threshold]
    return gr.update(choices=filtered_df['name'].tolist()), filtered_df[['name', 'Length', 'Predictions', 'Loss']]

def on_select(evt: gr.SelectData, dropdown_value):
    if evt.index[1] == 0:
        return evt.value
    return dropdown_value

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Fish Visualization App")
    
    with gr.Row():
        loss_input = gr.Number(label="Loss Threshold", value=0)
        name_dropdown = gr.Dropdown(choices=df['name'].tolist(), label="Select Fish Image")
    
    image_output = gr.Image(label="Annotated Fish Image")
    dataframe_output = gr.DataFrame(df)
    
    # Update dropdown based on loss threshold
    loss_input.change(
        fn=get_filtered_names,
        inputs=[loss_input],
        outputs=[name_dropdown, dataframe_output]
    )
    
    # Update image based on selection
    name_dropdown.change(
        fn=filter_and_display,
        inputs=[name_dropdown],
        outputs=[image_output]
    )
    
    dataframe_output.select(
        fn=on_select,
        inputs=[name_dropdown],
        outputs=[name_dropdown]
    )

if __name__ == "__main__":
    demo.launch()
