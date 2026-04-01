# Fishometry

Predict fish length from images without a physical reference object.

## Layout

- `src/create_data`: dataset split and zoom augmentation utilities
- `src/preprocessing`: image preprocessing pipeline and step implementations
- `src/training`: model trainers and orchestration entrypoints
- `src/visualization`: Streamlit analysis apps
- `scripts`: manual verification scripts that are not part of automated tests
- `tests`: lightweight automated smoke tests

## Common Commands

```bash
uv run python -m src.create_data.split --help
uv run python -m src.create_data.augment --help
uv run python -m src.preprocessing.run --config configs/config_inside.yaml --all-splits
uv run python -m src.training.run --help
uv run streamlit run src/visualization/app.py
uv run pytest -q
```

## Notes

- Large runtime artifacts live under `data/`, `checkpoints/`, and `third_party/` and are ignored by git.
- The Streamlit app is designed to run from the repository root so relative data paths resolve correctly.
