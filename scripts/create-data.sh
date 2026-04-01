#!/bin/sh

echo "Creating data..."

echo "Splitting data-inside"
uv run -m src.create_data.split --dataset data-inside

echo "Splitting data-outside"
uv run -m src.create_data.split --dataset data-outside

echo "Augmenting data-inside to data-inside-zoom"
uv run -m src.create_data.augment --source data-inside --dest data-inside-zoom

