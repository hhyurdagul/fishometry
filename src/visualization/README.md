# Visualization App

This package contains the Streamlit app used to inspect datasets, preprocessing outputs, and prediction results.

## Entrypoint

```bash
uv run streamlit run src/visualization/app.py
```

Run the command from the repository root so relative paths under `data/` resolve correctly.

## Expected Files

The app works best after preprocessing and training have produced:

- `data/<dataset>/processed.csv`
- `data/<dataset>/processed/rotated/`
- `data/<dataset>/processed/depth/`
- `data/<dataset>/processed/blackout/`
- `data/<dataset>/predictions.csv`

## App Modes

- Data Explorer: inspect images, metadata, detections, depth maps, and blackout images.
- Prediction Visualization: inspect predictions for selected images.
- Error Analysis: inspect residuals and error patterns.
- Model Comparison: compare model metrics.
- Fish Type Comparison: available when predictions include `fish_type`.
- Model x Fish Type Heatmap: available when predictions include `fish_type`.
