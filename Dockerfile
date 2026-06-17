    # DM2D Docker - Neuron Skeletonization Pipeline
    # 
    # Core method: DM2D skeletonization
    # Includes comparison baselines: NeuTube, VESS, PHD, diffskel
    #
    # Build:
    #   docker build -t dm2d .
    #
    # Run:
    #   docker run -v /path/to/data:/data -v /path/to/outputs:/outputs dm2d paper

    FROM mambaorg/micromamba:latest

    USER root
    WORKDIR /app

    # Enable GPU compute via nvidia-container-toolkit
    ENV NVIDIA_VISIBLE_DEVICES=all
    ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

    # ============================================================
    # System dependencies
    # ============================================================
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libgl1 \
        libglib2.0-0 \
        wget \
        unzip \
        swig \
        autoconf \
        libfftw3-dev \
        libpng-dev \
        libxml2-dev \
        libhdf5-dev \
        openmpi-bin \
        libopenmpi-dev \
        default-jre-headless \
        xvfb \
        xauth \
        && rm -rf /var/lib/apt/lists/*

    # ============================================================
    # Python environment
    # ============================================================
    COPY environment.yml /tmp/environment.yml
    RUN --mount=type=cache,target=/opt/conda/pkgs \
        --mount=type=cache,target=/root/.cache/pip \
        micromamba create -y -f /tmp/environment.yml -n wholebrain

    RUN micromamba clean -afy \
        && rm -f /tmp/environment.yml

    ENV MAMBA_ROOT_PREFIX=/opt/conda
    ENV PATH=/opt/conda/envs/wholebrain/bin:$PATH

    # ============================================================
    # Fiji (ImageJ2) for VESS and PHD
    # ============================================================
    RUN wget -q https://downloads.imagej.net/fiji/stable/fiji-stable-linux64-jdk.zip \
        && unzip -q fiji-stable-linux64-jdk.zip -d /opt \
        && rm fiji-stable-linux64-jdk.zip \
        && ln -s /opt/Fiji.app/ImageJ-linux64 /usr/local/bin/fiji

    COPY Baselines/imagej_plugins/*.jar /opt/Fiji.app/plugins/

    ENV IMAGEJ_DIR=/opt/Fiji.app
    ENV IJ_JAR=/opt/Fiji.app/jars/ij-*.jar

    # ============================================================
    # NeuTube (Build from source)
    # ============================================================
    COPY Baselines/neutube /app/Baselines/neutube

    WORKDIR /app/Baselines/neutube/neurolabi/lib
    RUN ./build.sh

    WORKDIR /app/Baselines/neutube/neurolabi
    RUN ./update_library

    WORKDIR /app/Baselines/neutube/neurolabi/cpp/lib
    RUN rm -rf build && ./build.sh

    WORKDIR /app/Baselines/neutube/neurolabi/python/module
    RUN make

    ENV NEUTUBE_DIR=/app/Baselines/neutube
    ENV PYTHONPATH=/app/Baselines/neutube/neurolabi/python/module:$PYTHONPATH
    WORKDIR /app

    # ============================================================
    # Copy project files
    # ============================================================
    COPY Utilities/ /app/Utilities/
    COPY Baselines/skeletonization-for-gradient-based-optimization/ /app/Baselines/diffskel/
    COPY Skeletonization_Suite/ /app/Skeletonization_Suite/
    COPY Vectorization/ /app/Vectorization/

    # Download DM2D model files from Google Drive
    RUN pip install gdown \
        && mkdir -p /app/Skeletonization_Suite/models \
        && gdown --folder https://drive.google.com/drive/folders/1dleWViW9W3tir021gr8T4njCA8gfne67 -O /app/Skeletonization_Suite/models \
        && pip uninstall -y gdown

    # Pre-computed bwskel outputs (requires MATLAB, included for convenience)
    COPY outputs/pmd/bwskel/ /app/outputs/pmd/bwskel/
    COPY outputs/stp/bwskel/ /app/outputs/stp/bwskel/

    # Sample datasets for paper reproduction
    COPY data/ /data/

    # ============================================================
    # Compile C++ binaries
    # ============================================================
    SHELL ["/bin/bash", "--login", "-c"]

    # DM2D binaries
    RUN g++ -O3 /app/Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/ComputeGraphReconstruction.cpp \
        -o /app/Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/a.out

    RUN g++ -O3 /app/Skeletonization_Suite/DM_2D_code/DiMo2d/code/paths_src/ComputePaths.cpp \
        -o /app/Skeletonization_Suite/DM_2D_code/DiMo2d/code/paths_src/a.out

    # DIPHA
    RUN cd /app/Skeletonization_Suite/DM_2D_code/DiMo2d/code/dipha-2d-thresh \
        && rm -rf build && mkdir build && cd build \
        && cmake .. && make

    # DM++ morse_code binaries (for whole-image processing)
    RUN g++ -O3 /app/Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/src/ComputeGraphReconstruction.cpp \
        -o /app/Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/src/a.out \
        $(pkg-config --cflags --libs opencv4) || true

    RUN g++ -O3 /app/Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/paths_src/ComputePaths.cpp \
        -o /app/Skeletonization_Suite/DM++/Semantic_Segmentation_NMI/morse_code/paths_src/a.out || true

    # ============================================================
    # Entrypoint
    # ============================================================
    COPY docker-entrypoint.sh /app/docker-entrypoint.sh
    RUN chmod +x /app/docker-entrypoint.sh

    WORKDIR /app
    VOLUME ["/data", "/outputs"]

    # Init script: copy bwskel and create symlinks
    RUN printf '#!/bin/bash\ncp -rn /app/outputs/* /outputs/ 2>/dev/null || true\nln -sf /data /app/data 2>/dev/null || true\nrm -rf /app/outputs && ln -sf /outputs /app/outputs 2>/dev/null || true\nexec "$@"\n' > /app/init.sh \
        && chmod +x /app/init.sh

    ENTRYPOINT ["/app/init.sh", "micromamba", "run", "-n", "wholebrain", "/app/docker-entrypoint.sh"]
    CMD ["help"]
