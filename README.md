**Geonni** determines where a street-level photo was taken, in the style of GeoGuessr.

### Setup

1. Create a virtual environment and then install required packages. For example using uv: 
    1. `uv venv`
    2. `source .venv/bin/activate`
    3. `uv pip install torch torchvision notebook`
2. Download and prepare the data.
    1. Download the street-location-images dataset from Kaggle (https://www.kaggle.com/datasets/killusions/street-location-images/) and place it in the `datasets` directory.
    2. Restructure the dataset for further use with `data_restructuring.ipynb`.
