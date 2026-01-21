# DM2D Whole-Brain Skeletonization Pipeline

This repository contains the official Docker implementation of the DM2D pipeline for neuron skeletonization.

## 1. Quick Start: Pull the Image

The fastest way to get started is to pull the pre-built image from Docker Hub:

```bash
docker pull samikbanerjee69/dm_full_pipeline_docker_cshl:latest
```

---

## 2. Usage: Processing Whole Images

This is the primary mode for users who want to skeletonize new, large brain sections (JP2 or TIFF format).

### Command
Run the following command to process a single image. Replace `/path/to/local/input.jp2` with your actual file path.

```bash
docker run --rm -v /path/to/local/input.jp2:/input/image.jp2 -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-image /input/image.jp2
```

### Outputs
After the run completes, check your local `outputs/whole_image/` folder:

| File Type | Location | Description |
|-----------|----------|-------------|
| **Vectorized Skeleton** | `outputs/whole_image/{ImageID}_{Section}.json` | The final skeleton in GeoJSON format. |
| **High-Res Visualization** | `outputs/whole_image/visualization/{ImageID}_{Section}_full.tif` | Full-resolution overlay of skeleton on original image. |
| **Preview Image** | `outputs/whole_image/visualization/{ImageID}_{Section}_preview.jpg` | Compressed, easy-to-open preview image. |
| **Binary Mask** | `outputs/whole_image/mask/{ImageID}_{Section}.jpg` | Binary pixel mask of the skeleton. |

*(Note: If the input filename does not follow the `ID_Section.ext` convention, the output will default to `Brain_0`.)*

---

## 3. Usage: Reproducing Paper Results

To reproduce the benchmark results from the paper:

### Data
The PMD and STP datasets are **bundled inside the Docker image** at `/data/`. No download or mounting required.

### Run Reproduction Command
This command runs inference, evaluation, and generates comparison plots for **both** datasets.

```bash
docker run --rm -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest paper
```

### Outputs
Results will be organized in `outputs/`:

| Path | Contents |
|------|----------|
| `outputs/figures/` | Comparison bar plots and figures. |
| `outputs/tables/` | CSV files containing quantitative metrics (Precision, Recall, F1). |
| `outputs/pmd/dm2d/` | Raw skeleton files for the PMD dataset. |
| `outputs/pmd/dm2d_evaluation/` | Detailed evaluation logs. |

---

## 4. Advanced: Build Docker from Source

If you need to modify the code or build the image locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MitraLab-Organization/2D-Skeletonization.git
    cd 2D-Skeletonization
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t samikbanerjee69/dm_full_pipeline_docker_cshl:latest .
    ```
    *Note: This requires ~8GB RAM and may take 10-20 minutes.*

3.  **Run:**
    Use the image name in any of the commands above.

---

## 5. Manual Setup (Without Docker)

For users who prefer to run the pipeline natively:

### Prerequisites
- Python 3.9+
- Conda (recommended)
- CMake, g++
- OpenCV (with pkg-config)

### Step 1: Clone and Setup Environment
```bash
git clone https://github.com/MitraLab-Organization/2D-Skeletonization.git
cd 2D-Skeletonization
conda env create -f environment.yml
conda activate dm2d
```

### Step 2: Download Model Weights
```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1dleWViW9W3tir021gr8T4njCA8gfne67 -O Skeletonization_Suite/models
```

### Step 3: Compile C++ Binaries

**DM2D Core:**
```bash
g++ -O3 Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/ComputeGraphReconstruction.cpp -o Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/a.out

g++ -O3 Skeletonization_Suite/DM_2D_code/DiMo2d/code/paths_src/ComputePaths.cpp -o Skeletonization_Suite/DM_2D_code/DiMo2d/code/paths_src/a.out
```

**DIPHA (Persistence Computation):**
```bash
cd Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-2d-thresh
rm -rf build && mkdir build && cd build
cmake .. && make
cd ../../../../..
```

**DM++ Morse Code (for whole-image processing):**
```bash
g++ -O3 Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/src/ComputeGraphReconstruction.cpp -o Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/src/a.out $(pkg-config --cflags --libs opencv4)

g++ -O3 Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/paths_src/ComputePaths.cpp -o Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/paths_src/a.out
```

### Step 4: Run the Pipeline
```bash
cd Skeletonization_Suite
python run_dm2d_tiles.py --lkl_dir ../data/pmd/lkl --output_dir ../outputs/pmd/dm2d --ve_persistence 0 --et_persistence 64 --min_size 40
```

