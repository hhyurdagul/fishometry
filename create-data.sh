#!/bin/sh

echo "Creating data..."

echo "Splitting data-inside"
uv run -m src.create_data.split --input data/data-inside/raw.csv --output data/data-inside/splits

echo "Splitting data-outside"
uv run -m src.create_data.split --input data/data-outside/raw.csv --output data/data-outside/splits

echo "Augmenting data-inside to data-inside-zoom"
uv run -m src.create_data.augment --source data-inside --dest data-inside-zoom

