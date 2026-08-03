# ==============================================================================
# Combined Dockerfile: Minimal CUDA + PyTorch + gsplat + nerfstudio + FastMap
# ==============================================================================

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
ARG CUDA_ARCHITECTURES="89"
ARG TORCH_VERSION=2.4.1
ARG TORCH_CUDA_TAG=cu124
ARG OIIO_VERSION=v2.5.9.0
ARG COLMAP_GIT_REF=main
ARG FASTMAP_GIT_REF=main
ARG PYRENDER_GIT_REF=main
ARG INSTALL_COLMAP=true
ARG BUILD_CUDA_KERNELS=true

# Python package arguments
ARG NUMPY_PKG="numpy"
ARG PIL_PKG="pillow"
ARG OPENCV_PKG="opencv-python-headless"
ARG TQDM_PKG="tqdm"
ARG GSPLAT_PKG="gsplat"
ARG NERFSTUDIO_PKG="nerfstudio"

# ------------------------------------------------------------------------------
# BUILDER STAGE
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_VERSION
ARG UBUNTU_VERSION
ARG CUDA_ARCHITECTURES
ARG TORCH_VERSION
ARG TORCH_CUDA_TAG
ARG OIIO_VERSION
ARG COLMAP_GIT_REF
ARG FASTMAP_GIT_REF
ARG PYRENDER_GIT_REF
ARG INSTALL_COLMAP
ARG BUILD_CUDA_KERNELS
ARG NUMPY_PKG
ARG PIL_PKG
ARG OPENCV_PKG
ARG TQDM_PKG
ARG GSPLAT_PKG
ARG NERFSTUDIO_PKG

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    cmake ninja-build ffmpeg \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-thread-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libqt5svg5-dev \
    libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev \
    libopencv-dev libgl1 \
    libopenexr-dev libtiff-dev libpng-dev libjpeg-turbo8-dev libwebp-dev \
    libraw-dev libssl-dev zlib1g \
    libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip "cmake>=3.28" "ninja>=1.11"

# ------------------------------------------------------------------------------
# Install PyTorch
# ------------------------------------------------------------------------------
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

WORKDIR /opt/src

# ------------------------------------------------------------------------------
# OpenImageIO (Required for COLMAP)
# ------------------------------------------------------------------------------
RUN if [ "${INSTALL_COLMAP}" = "true" ]; then \
      git clone --depth 1 --branch ${OIIO_VERSION} https://github.com/AcademySoftwareFoundation/OpenImageIO.git oiio && \
      cd oiio && mkdir -p build && cd build && \
      /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_TOOLS=ON -DUSE_PYTHON=OFF && \
      /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf oiio; \
    fi

# ------------------------------------------------------------------------------
# COLMAP
# ------------------------------------------------------------------------------
RUN if [ "${INSTALL_COLMAP}" = "true" ]; then \
      git clone --depth 1 --branch ${COLMAP_GIT_REF} https://github.com/colmap/colmap.git && \
      cd colmap && mkdir -p build && cd build && \
      /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DCUDA_ENABLED=ON && \
      /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf colmap; \
    fi

# ------------------------------------------------------------------------------
# Install gsplat + nerfstudio + FastMap + dependencies
# ------------------------------------------------------------------------------
RUN /opt/venv/bin/pip install --no-cache-dir \
    ${GSPLAT_PKG} \
    ${NERFSTUDIO_PKG} \
    ${NUMPY_PKG} \
    ${PIL_PKG} \
    ${OPENCV_PKG} \
    ${TQDM_PKG} \
    jupyterlab \
    pyrender \
    imageio imageio-ffmpeg scikit-image lpips rich tyro \
    trimesh "pyglet<2" pyyaml dacite loguru prettytable psutil


# Optional tiny-cuda-nn
RUN /opt/venv/bin/pip install --no-cache-dir \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch || true

# FastMap repository clone
RUN git clone --depth 1 --branch ${FASTMAP_GIT_REF} https://github.com/pals-ttic/fastmap.git /opt/fastmap

# Optional custom CUDA kernels for FastMap
RUN if [ "${BUILD_CUDA_KERNELS}" = "true" ]; then \
      cd /opt/fastmap && \
      TORCH_CUDA_ARCH_LIST="$(echo ${CUDA_ARCHITECTURES} | sed 's/;/ /g' | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')" \
      /opt/venv/bin/python setup.py build_ext --inplace; \
    fi

ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCHITECTURES}

# Cleanup build caches to save space
RUN /opt/venv/bin/pip uninstall -y cmake ninja || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete

# ------------------------------------------------------------------------------
# RUNTIME STAGE
# ------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG USERNAME=fastmap
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-venv python3-pip ffmpeg \
    libjpeg-turbo8 libpng16-16 libtiff5 libwebp-dev libopenexr-dev libraw-dev \
    libssl3 zlib1g libsqlite3-0 libgomp1 libstdc++6 \
    libsm6 libice6 libxext6 libxrender1 libgl1 \
    libosmesa6 libglu1-mesa freeglut3 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} && \
    mkdir -p /workspace && chown ${USERNAME}:${USERNAME} /workspace

WORKDIR /workspace

COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastmap /opt/fastmap

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig || true && \
    chown -R root:root /usr/local/bin /usr/local/lib /opt/fastmap && \
    chmod -R a+rX /usr/local/lib /opt/fastmap

# ------------------------------------------------------------------------------
# NEW: Create a startup script for JupyterLab (RunPod friendly)
# ------------------------------------------------------------------------------
RUN echo '#!/bin/bash\n\
if [ -z "$JUPYTER_TOKEN" ]; then\n\
    JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")\n\
    echo "Generated JUPYTER_TOKEN: $JUPYTER_TOKEN"\n\
fi\n\
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="$JUPYTER_TOKEN"\n\
' > /usr/local/bin/start-jupyter.sh && \
    chmod +x /usr/local/bin/start-jupyter.sh && \
    chown root:root /usr/local/bin/start-jupyter.sh



USER ${USERNAME}
ENV HOME=/home/${USERNAME}

# Sanity checks
RUN python -c "import torch; print('Torch:', torch.__version__)" || true
RUN python -c "import gsplat; print('gsplat OK')" || true
RUN python -c "import nerfstudio; print('nerfstudio OK')" || true
RUN python3 -c "import sys; sys.path.insert(0, '/opt/fastmap'); import fastmap; print('fastmap OK')" || true

# ------------------------------------------------------------------------------
# NEW: Default command – start JupyterLab. 
# ------------------------------------------------------------------------------
CMD ["/usr/local/bin/start-jupyter.sh"]
