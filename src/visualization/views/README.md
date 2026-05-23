# Visualization Views

This folder contains Streamlit page components used by `src.visualization.app`.

## Views

- `explorer.py`: dataset and image explorer for raw, rotated, depth, and blackout outputs.
- `prediction_viz.py`: prediction display for individual images.
- `analysis.py`: error analysis over model predictions.
- `comparison.py`: model comparison views, including fish-type-specific comparisons.
- `correlation.py`: correlation-oriented analysis helpers.

The app chooses which views to show based on the selected dataset and whether `predictions.csv` includes fish-type information.
