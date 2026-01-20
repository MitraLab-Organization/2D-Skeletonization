# Vectorization Suite

Python pipeline to convert binary masks into vectorized graph/skeleton GeoJSONs.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run the main script with optional arguments:
```bash
python script_to_run.py \
  --input_dir ./mask \
  --output_cc_dir ./CC1 \
  --output_json_dir ./skelJSON1 \
  --debug
```

## Features
- **Skeletonization**: Converts binary masks to skeletons.
- **Graph Analysis**: Decomposes skeletons into nodes and arcs using `networkx`.
- **Optimization**: Uses `scipy.ndimage.find_objects` to efficiently process large images by cropping connected components.
- **Output**: Generates GeoJSON files suitable for web visualization.
