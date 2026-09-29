# DM2D Whole-Brain Skeletonization Pipeline

This repository contains the official Docker implementation of the DM2D pipeline for neuron skeletonization.

## 1. Quick Start: Pull the Image

The fastest way to get started is to pull the pre-built image from Docker Hub:

```bash
docker pull samikbanerjee69/dm_full_pipeline_docker_cshl:latest
```

---

## 2. Usage: Reproducing Paper Results

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

## 3. Usage: Whole Image - Full Pipeline (`run-image`)

This is the primary mode for users who want to skeletonize new, large brain sections (JP2 or TIFF format). It runs ALBU inference (a 4-fold ResNet34 encoder-decoder) to produce a likelihood map, clips low-confidence background pixels to zero, then skeletonizes that map with DM2D the same way the paper-reproduction path does.

### Single Image
Run the following command to process a single image using the **PMD** parameter preset (default is custom, so mode must be specified or parameters provided).

```bash
docker run --rm -v /path/to/local/input.jp2:/input/image.jp2 -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-image /input/image.jp2 --mode pmd
```

### Batch Processing (`run-folder`)
To process an entire folder of original images:
```bash
docker run --rm -v /path/to/images:/input -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-folder /input --mode pmd
```

### Outputs
After the run completes, check your local `outputs/whole_image/` folder:

| File Type | Location | Description |
|-----------|----------|-------------|
| **Vectorized Skeleton** | `outputs/whole_image/{ImageID}_{Section}.json` | The final skeleton in GeoJSON format. |
| **High-Res Visualization** | `outputs/whole_image/visualization/{ImageID}_{Section}_full.tif` | Full-resolution overlay of skeleton on original image. |
| **Preview Image** | `outputs/whole_image/visualization/{ImageID}_{Section}_preview.jpg` | Compressed, easy-to-open preview image. |
| **Binary Mask** | `outputs/whole_image/mask/{ImageID}_{Section}.jpg` | Binary pixel mask of the skeleton. |

*(Note: If the input filename does not follow the `ID_Section.ext` convention and `--output` is not specified, the output will default to `Brain_0`.)*

---

## 4. Usage: Staged Processing (Likelihood and Skeleton as Separate Steps)

`run-image`/`run-folder` do the whole job in one call, but sometimes it's useful to stop halfway: inspect or reuse the likelihood map without skeletonizing it, or skeletonize a likelihood image you already have without touching ALBU or the raw image at all. Two more commands expose each half independently, and together they reproduce `run-image`'s output exactly.

### `image-to-likelihood`: ALBU Only
Runs the same ALBU inference and background-clipping step as `run-image`, then stops.

```bash
docker run --rm -v /path/to/local/input.jp2:/input/image.jp2 -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest image-to-likelihood /input/image.jp2 --mode pmd --output MyBrain_001
```

Options: `--mode`, `--norm_factor`, `--likelihood_threshold`, `--mask`, `--output`. Result: `outputs/likelihood/lkl/MyBrain_001_0.jpg`.

### `likelihood-to-skeleton`: DM2D Only
Takes a likelihood image — either from the command above, or the intermediate `lkl/*.jpg` file `run-image` writes along the way — and runs DM2D, vectorization, and visualization on it. No raw image or neural network involved.

```bash
docker run --rm -v /path/to/likelihood.jpg:/input/likelihood.jpg -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest likelihood-to-skeleton /input/likelihood.jpg --mode pmd --output MyBrain_001
```

Options: `--mode`, `--ve_persistence_threshold`, `--persistence_threshold`, `--min_size`, `--output`. Result: `outputs/skeleton/` (same file layout as `run-image`'s `whole_image/` output, described in the table above).

### Chaining
Chained together, these two commands are equivalent to running `run-image` directly:

```bash
docker run --rm ... samikbanerjee69/dm_full_pipeline_docker_cshl:latest image-to-likelihood /input.jp2 --mode pmd --output MyBrain_001
docker run --rm ... samikbanerjee69/dm_full_pipeline_docker_cshl:latest likelihood-to-skeleton /outputs/likelihood/lkl/MyBrain_001_0.jpg --mode pmd --output MyBrain_001
```

---

## 5. Common Parameters for Whole Image Processing

`run-image`/`run-folder`, `image-to-likelihood`, and `likelihood-to-skeleton` share a common parameter engine.

### Parameter Modes (Presets)
The pipeline supports three parameter modes via the `--mode` flag:

| Mode | Use Case | Parameters Used |
|---|---|---|
| **`pmd`** | For PMD-like datasets | `ve_persistence=0`, `et_persistence=64`, `min_size=40`, `norm_factor=16`, `likelihood_threshold=40`, `mask=true` |
| **`stp`** | For STP-like datasets | `ve_persistence=0`, `et_persistence=32`, `min_size=12`, `norm_factor=256`, `likelihood_threshold=40`, `mask=true` |
| **`custom`** | Fully custom parameters | Requires specifying exact thresholds (see below) |

### Custom Parameters

If you use `--mode custom` (or omit the mode flag entirely), you can specify fine-grained thresholds. Which flags are required depends on which command you're running:

| Flag | Description | Used In |
|---|---|---|
| `--persistence_threshold <N>` | (Required) ET persistence, removes spurious branches | `run-image`/`run-folder`, `likelihood-to-skeleton` |
| `--min_size <N>` | (Required) Removes disconnected skeleton components smaller than this size | `run-image`/`run-folder`, `likelihood-to-skeleton` |
| `--ve_persistence_threshold <N>` | (Optional, default 0) VE persistence | `run-image`/`run-folder`, `likelihood-to-skeleton` |
| `--norm_factor <N>` | (Required) ALBU pixel normalization divisor | `run-image`/`run-folder`, `image-to-likelihood` |
| `--likelihood_threshold <N>` | (Required) Likelihood background clip-to-zero cutoff | `run-image`/`run-folder`, `image-to-likelihood` |
| `--mask <true/false>` | (Required) Restrict ALBU processing to tissue via Otsu thresholding, instead of the full frame | `run-image`/`run-folder`, `image-to-likelihood` |
| `--output <name>` | (Optional) Custom name for the results (e.g., `MyBrain_001`) | All three (single image only) |

**Examples:**

*Custom parameters with custom output name:*
```bash
docker run --rm ... dm2d run-image /input/image.jp2 --persistence_threshold 16 --min_size 30 --norm_factor 16 --likelihood_threshold 40 --mask true --output MyBrain_001
```

*Custom parameters with VE persistence:*
```bash
docker run --rm ... dm2d run-image /input/image.jp2 --ve_persistence_threshold 5 --persistence_threshold 16 --min_size 30 --norm_factor 16 --likelihood_threshold 40 --mask true
```

*Override mask in a preset mode:*
```bash
docker run --rm ... dm2d run-image /input/image.jp2 --mode pmd --mask false
```

---

## 6. Advanced: Build Docker from Source

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

### Exploring Source Code Inside the Docker Image

If you only pulled the image and don't have the GitHub repo, you can still inspect and modify the source code:

```bash
# Open an interactive shell inside the container
docker run --rm -it samikbanerjee69/dm_full_pipeline_docker_cshl:latest bash
```

Once inside, the full source code is at `/app/`:
```
/app/
├── Skeletonization_Suite/       # Main pipeline code
│   ├── run_whole_image.py       # Whole-image entry point
│   ├── pipeline_processDetect_skel_samik_singleChanel.py
│   ├── albu_calculations_singleChanel.py
│   ├── run_dm2d_tiles.py        # Tiled evaluation entry point
│   └── models/                  # Pre-trained model weights
├── Vectorization/               # Vectorization & graph analysis
├── Utilities/                   # Evaluation & visualization scripts
├── Baselines/                   # Baseline methods (PHD, diffskel, etc.)
└── docker-entrypoint.sh         # Docker command dispatcher
```

You can browse and edit files with standard tools (`cat`, `vi`, `nano` etc.):
```bash
cat /app/Skeletonization_Suite/run_whole_image.py
```

To **copy source code out** of the container to your local machine:
```bash
# From your host machine (not inside the container)
docker create --name tmp_dm2d samikbanerjee69/dm_full_pipeline_docker_cshl:latest
docker cp tmp_dm2d:/app ./dm2d_source
docker rm tmp_dm2d
```

---

## 7. Manual Setup (Without Docker)

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

### Step 4: Run the Pipeline
```bash
cd Skeletonization_Suite
python run_dm2d_tiles.py --lkl_dir ../data/pmd/lkl --output_dir ../outputs/pmd/dm2d --ve_persistence 0 --et_persistence 64 --min_size 40
```
