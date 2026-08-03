# ------------------------------------------------------------------------------
# CUDA + PyTorch + gsplat + nerfstudio + COLMAP (built from source, CUDA-enabled)
# Now also includes fastmap and JupyterLab for RunPod
# ------------------------------------------------------------------------------

ARG CUDA_VERSION=12.4.1
ARG CUDA_ARCHITECTURES="89"
ARG COLMAP_VERSION="3.9.1"

# ==============================================================================
# Stage 1: builder — compiles CUDA-enabled COLMAP.
# Needs the "devel" CUDA image (has nvcc) + the full build toolchain.
# None of this toolchain ends up in the final image.
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS colmap-builder
ARG CUDA_ARCHITECTURES
ARG COLMAP_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git /tmp/colmap \
    && cmake -B /tmp/colmap/build -S /tmp/colmap -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DCUDA_ENABLED=ON \
    && ninja -C /tmp/colmap/build install \
    && rm -rf /tmp/colmap

# ==============================================================================
# Stage 2: final runtime image
# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04
ARG CUDA_ARCHITECTURES

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------------------------
# Build args (kept from your original for flexibility)
# ------------------------------------------------------------------------------
ARG TORCH_VERSION=2.4.1
ARG TORCH_CUDA_TAG=cu124
# Dotted-decimal form for PyTorch (e.g. "8.9"), separate from CUDA_ARCHITECTURES
# above which CMake wants as "89". Keep this in sync with CUDA_ARCHITECTURES —
# e.g. 89 -> "8.9", 86 -> "8.6", multiple archs -> "8.6;8.9".
ARG TORCH_CUDA_ARCH_LIST="8.9"

# Python deps
ARG NUMPY_PKG="numpy"
ARG PIL_PKG="pillow"
ARG OPENCV_PKG="opencv-python-headless"
ARG TQDM_PKG="tqdm"

# Core libs
ARG GSPLAT_PKG="gsplat"
ARG NERFSTUDIO_PKG="nerfstudio"

# ------------------------------------------------------------------------------
# System deps (minimal) + COLMAP's *runtime* shared libs (no compilers/headers
# beyond what these -dev packages happen to carry the .so files in — this is
# still far lighter than the builder stage's full toolchain).
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y universe \
    && apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libgl1 \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libsqlite3-dev \
    libglew-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Bring in the CUDA-enabled COLMAP built in stage 1
COPY --from=colmap-builder /opt/colmap /opt/colmap
ENV PATH="/opt/colmap/bin:$PATH"

# ------------------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------------------------
# Install PyTorch (CUDA matched)
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

# ------------------------------------------------------------------------------
# Install gsplat + nerfstudio + deps
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    ${GSPLAT_PKG} \
    ${NERFSTUDIO_PKG} \
    ${NUMPY_PKG} \
    ${PIL_PKG} \
    ${OPENCV_PKG} \
    ${TQDM_PKG} \
    imageio imageio-ffmpeg scikit-image lpips rich tyro

# ------------------------------------------------------------------------------
# CUDA tuning for ADA GPUs — set BEFORE building tiny-cuda-nn below, and kept
# as a persistent ENV since nerfstudio also reads this at training time for
# its own JIT-compiled CUDA kernels.
# ------------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ------------------------------------------------------------------------------
# Optional: tiny-cuda-nn (performance boost for nerfstudio)
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || true

# ------------------------------------------------------------------------------
# NEW: Install fastmap from GitHub (fixed01)
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/pals-ttic/fastmap.git

# ------------------------------------------------------------------------------
# NEW: Install JupyterLab for RunPod
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir jupyterlab

# ------------------------------------------------------------------------------
# NEW: Create a startup script for JupyterLab (RunPod friendly)
#   - Uses JUPYTER_TOKEN env var, generates a random token if not set
#   - Binds to all interfaces, allows root, disables browser auto-launch
# ------------------------------------------------------------------------------
RUN echo '#!/bin/bash\n\
if [ -z "$JUPYTER_TOKEN" ]; then\n\
    JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")\n\
    echo "Generated JUPYTER_TOKEN: $JUPYTER_TOKEN"\n\
fi\n\
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="$JUPYTER_TOKEN"\n\
' > /usr/local/bin/start-jupyter.sh && chmod +x /usr/local/bin/start-jupyter.sh

# ------------------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------------------
RUN python -c "import torch; print('Torch:', torch.__version__)"
RUN python -c "import gsplat; print('gsplat OK')"
RUN python -c "import nerfstudio; print('nerfstudio OK')"
RUN python -c "import fastmap; print('fastmap OK')"
RUN colmap -h > /dev/null 2>&1 || (echo "COLMAP binary failed to run" && exit 1)

WORKDIR /workspace

# ------------------------------------------------------------------------------
# NEW: Default command – start JupyterLab. To get a bash shell, override:
#   docker run -it --entrypoint bash <image>
# ------------------------------------------------------------------------------
CMD ["/usr/local/bin/start-jupyter.sh"]
