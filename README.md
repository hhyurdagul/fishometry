# How to Run

```bash
# Install dependencies
uv sync
# Run pipeline for inside data
uv run env PYTHONPATH=. python src/pipeline.py --config configs/config_inside.yaml
# Run pipeline for outside data
uv run env PYTHONPATH=. python src/pipeline.py --config configs/config_outside.yaml
```
