# Training Models

This folder contains the model implementations used by `src.training.run`.

## Model Types

- `baseline.py`: mean-length baseline. For datasets with fish types, the baseline can use grouped means.
- `regression.py`: tabular regression models such as linear regression, XGBoost, and MLP.
- `cnn.py`: image-based model using blackout images together with selected tabular features.
- `embedding.py`: embedding-based model used for outside-data experiments.

All models return prediction dataframes keyed by image name so the orchestrator can join outputs into `predictions.csv`.

## Feature Naming

Feature column selection is coordinated through `src.training.data_loader.get_feature_names_and_desc`.

Supported feature groups include:

- `eye`: eye and fish dimensions for controlled inside data.
- `coords`: relative fish geometry features.
- `features`: richer geometric, segmentation, VLM, and encoded context features.

Depth features are appended when the `depth` flag is enabled.
