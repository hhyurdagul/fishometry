# Source Package

This folder contains the Python implementation of the Fishometry pipelines.

## Packages

- `config.py`: Pydantic config models and config loading helpers.
- `create_data/`: split creation and zoom augmentation.
- `preprocessing/`: image preprocessing and feature extraction.
- `training/`: model training and prediction CSV creation.
- `visualization/`: Streamlit app for inspecting datasets and predictions.

The project is designed around small pipeline steps that receive metadata, add outputs or generated artifacts, and pass the enriched metadata to the next step.
