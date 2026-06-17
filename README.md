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

## 3. Usage: Whole Image - Full Pipeline (`run-pipeline`)

This is the primary mode for users who want to skeletonize new, large brain sections (JP2 or TIFF format). This process runs the AI (ALBU) inference, Topological skeletonization, and vectorization.

### High-Speed GPU Processing (`gpu_parallel`)

For significantly faster inference and tensor operations, use the `gpu_parallel` Docker tag. This requires an NVIDIA GPU and the Docker `--gpus all` flag. If your images are `.jp2` files, you must also mount your Kakadu installation.

**General Template:**
```bash
docker run --rm --gpus all \
  -v /path/to/your/kakadu:/kakadu \
  -e LD_LIBRARY_PATH=/kakadu/lib/Linux-x86-64-gcc \
  -v /path/to/local/input.jp2:/input/image.jp2 \
  -v $(pwd)/outputs:/outputs \
  samikbanerjee69/dm_full_pipeline_docker_cshl:gpu_parallel \
  run-pipeline /input/image.jp2 --mode pmd
```

**Real-world Example (MD1022):**
```bash
docker run --rm --gpus all \
  -v /home/samik/v8_2_1-02038E:/kakadu \
  -e LD_LIBRARY_PATH=/kakadu/lib/Linux-x86-64-gcc \
  -v /nfs/data/main/M38/mba_converted_imaging_data/MD1022/MD1022/MD1022-F50-2025.02.06-03.16.05_MD1022_2_0100_lossy.jp2:/input/image.jp2 \
  -v $(pwd)/dm2doutputs/md1022:/outputs \
  samikbanerjee69/dm_full_pipeline_docker_cshl:gpu_parallel \
  run-pipeline /input/image.jp2 --mode pmd
```

### Standard CPU Processing (`latest`)
If you do not have a GPU, you can run the standard CPU pipeline.

```bash
docker run --rm -v /path/to/local/input.jp2:/input/image.jp2 -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-pipeline /input/image.jp2 --mode pmd
```

### Batch Processing (`run-pipeline-folder`)
To process an entire folder of original images using the GPU parallel tag:
```bash
docker run --rm --gpus all -v /path/to/your/kakadu:/kakadu -e LD_LIBRARY_PATH=/kakadu/lib/Linux-x86-64-gcc -v /path/to/images:/input -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:gpu_parallel run-pipeline-folder /input --mode pmd
```
---

## 4. Usage: Whole Image - Only DM2D (`run-dm2d`)

To exclusively run the topological processing and vectorization natively on a pre-computed probability map (LKL output), bypassing the ALBU Neural Network inference.

### Single LKL Image
```bash
docker run --rm -v /path/to/lkl.jpg:/input/lkl.jpg -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-dm2d /input/lkl.jpg --mode pmd
```

### Batch Processing (`run-dm2d-folder`)
To process an entire folder of LKL probability map images:
```bash
docker run --rm -v /path/to/images:/input -v $(pwd)/outputs:/outputs samikbanerjee69/dm_full_pipeline_docker_cshl:latest run-dm2d-folder /input --mode pmd
```

---

## 5. Common Parameters for Whole Image Processing

Both `run-pipeline` and `run-dm2d` share a common parameter engine.

### Parameter Modes (Presets)
The pipeline supports three parameter modes via the `--mode` flag:

| Mode | Use Case | Parameters Used |
|---|---|---|
| **`pmd`** | For PMD-like datasets | `ve_persistence=0`, `et_persistence=64`, `min_size=40`, `norm_factor=16`, `mask=true` |
| **`stp`** | For STP-like datasets | `ve_persistence=0`, `et_persistence=32`, `min_size=12`, `norm_factor=256`, `mask=true` |
| **`custom`** | Fully custom parameters | Requires specifying exact thresholds (see below) |

### Custom Parameters

If you use `--mode custom` (or omit the mode flag entirely), you can specify fine-grained thresholds.
Depending on your run mode (`run-pipeline` vs `run-dm2d`), different parameters are required or ignored by the backend:

| Flag | Description | Used In |
|---|---|---|
| `--persistence_threshold <N>` | (Required) Edge-Triangle persistence, removes spurious branches | Both |
| `--min_size <N>` | (Required) Removes disconnected skeleton components smaller than this size | Both |
| `--ve_persistence_threshold <N>` | (Optional, default 0) Vertex-Edge persistence | Both |
| `--norm_factor <N>` | (Required for full pipeline) ALBU pixel normalization divisor | `run-pipeline` only |
| `--mask <true/false>` | (Required for full pipeline) Enable/disable Otsu tissue masking | `run-pipeline` only |
| `--output <name>` | (Optional) Specify a custom name for the results (e.g., `MyBrain_001`) | Both (Single image only) |

**Examples:**

*Custom parameters for full pipeline with custom output name:*
```bash
docker run --rm ... dm2d run-pipeline /input/image.jp2 --persistence_threshold 16 --min_size 30 --norm_factor 16 --mask true --output MyBrain_001
```

*Custom parameters for DM2D only:*
```bash
docker run --rm ... dm2d run-dm2d /input/lkl.jpg --persistence_threshold 16 --min_size 30
```

*Override Mask in Preset Modes (pipeline only):*
```bash
docker run --rm ... dm2d run-pipeline /input/image.jp2 --mode pmd --mask false
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
