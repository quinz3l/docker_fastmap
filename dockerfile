# ==============================================================================
# FastMap (pals-ttic/fastmap) — fast PyTorch-based structure-from-motion
# https://github.com/pals-ttic/fastmap
#
# Multi-stage build (Ada-only by default, SM 8.9). Reuses the proven fixes
# from the COLMAP/GLOMAP/OpenSplat Dockerfile in this repo:
#   - OpenImageIO must be built from source (apt's version on Ubuntu 22.04 is
#     too old for current COLMAP) AND explicitly installed to /usr/local
#     (its CMake defaults to a local dist/ folder otherwise, which is the bug
#     that broke that build originally).
#   - CMAKE_CUDA_ARCHITECTURES must be quoted in RUN commands, or the shell
#     splits a multi-value list like "86;89" into separate commands.
#   - libqt5svg5-dev is required alongside qtbase5-dev/libqt5opengl5-dev.
#
# Usage:
#   docker build -t fastmap:ada \
#     --build-arg CUDA_ARCHITECTURES="89" \
#     --build-arg TORCH_CUDA_ARCH_LIST="8.9" .
#
# Build args of note:
#   INSTALL_COLMAP=true   Set to "false" to skip building COLMAP entirely if
#                          you already have matching .db files and only need
#                          FastMap's pose estimation step (saves ~30+ min).
#   BUILD_CUDA_KERNELS=true  FastMap's custom CUDA kernels are optional but
#                          "highly recommended" by upstream for speed.
# ==============================================================================

ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.4.1
ARG CUDA_ARCHITECTURES="89"
ARG TORCH_CUDA_TAG=cu124
ARG TORCH_VERSION=2.4.1
ARG OIIO_VERSION=v2.5.9.0
ARG COLMAP_GIT_REF=main
ARG FASTMAP_GIT_REF=main
ARG PYRENDER_GIT_REF=main
ARG INSTALL_COLMAP=true
ARG BUILD_CUDA_KERNELS=true

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# Re-declare: ARGs before the first FROM are NOT visible inside a build stage
# unless re-declared here. (This is the exact bug that broke the sibling
# Dockerfile's first build — OIIO_VERSION silently evaluated to empty.)
ARG CUDA_VERSION
ARG CUDA_ARCHITECTURES
ARG TORCH_CUDA_TAG
ARG TORCH_VERSION
ARG OIIO_VERSION
ARG COLMAP_GIT_REF
ARG FASTMAP_GIT_REF
ARG PYRENDER_GIT_REF
ARG INSTALL_COLMAP
ARG BUILD_CUDA_KERNELS

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates build-essential pkg-config \
    python3 python3-pip python3-venv python3-dev \
    cmake ninja-build \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libboost-filesystem-dev libboost-thread-dev \
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libqt5svg5-dev \
    libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev \
    libopencv-dev \
    libopenexr-dev libtiff-dev libpng-dev libjpeg-turbo8-dev libwebp-dev \
    libraw-dev libssl-dev zlib1g \
    libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip "cmake>=3.28" "ninja>=1.11"

# ------------------------------------------------------------------------------
# PyTorch first — everything else (COLMAP build is independent; FastMap's own
# deps and CUDA kernel compile need torch present).
# ------------------------------------------------------------------------------
RUN /opt/venv/bin/pip install --no-cache-dir "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

WORKDIR /opt/src

# ------------------------------------------------------------------------------
# OpenImageIO — only needed if we're building COLMAP. MUST install to
# /usr/local explicitly (see header note).
# ------------------------------------------------------------------------------
RUN if [ "${INSTALL_COLMAP}" = "true" ]; then \
      git clone --depth 1 --branch ${OIIO_VERSION} https://github.com/AcademySoftwareFoundation/OpenImageIO.git oiio && \
      cd oiio && mkdir -p build && cd build && \
      /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOIIO_BUILD_TESTS=OFF -DOIIO_BUILD_TOOLS=ON -DUSE_PYTHON=OFF && \
      /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf oiio; \
    fi

# ------------------------------------------------------------------------------
# COLMAP — optional. FastMap only needs `colmap feature_extractor` and
# `colmap exhaustive_matcher` to produce the .db file it consumes; skip this
# entirely with --build-arg INSTALL_COLMAP=false if you already have one.
# ------------------------------------------------------------------------------
RUN if [ "${INSTALL_COLMAP}" = "true" ]; then \
      git clone --depth 1 --branch ${COLMAP_GIT_REF} https://github.com/colmap/colmap.git && \
      cd colmap && mkdir -p build && cd build && \
      /opt/venv/bin/cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DCUDA_ENABLED=ON && \
      /opt/venv/bin/ninja && /opt/venv/bin/ninja install && cd /opt/src && rm -rf colmap; \
    fi

# ------------------------------------------------------------------------------
# FastMap itself, its Python deps, and the pyrender fork it requires.
# pyrender needs libosmesa6-dev (installed above) for headless/offscreen
# rendering in a container with no physical display.
# ------------------------------------------------------------------------------
RUN /opt/venv/bin/pip install --no-cache-dir \
    trimesh "pyglet<2" pyyaml dacite loguru prettytable psutil

RUN /opt/venv/bin/pip install --no-cache-dir \
    "git+https://github.com/jiahaoli95/pyrender.git@${PYRENDER_GIT_REF}"

RUN git clone --depth 1 --branch ${FASTMAP_GIT_REF} https://github.com/pals-ttic/fastmap.git /opt/fastmap

# Optional custom CUDA kernels — upstream says "highly recommended" for speed.
# Uses the same CUDA_ARCHITECTURES as everything else above.
RUN if [ "${BUILD_CUDA_KERNELS}" = "true" ]; then \
      cd /opt/fastmap && \
      TORCH_CUDA_ARCH_LIST="$(echo ${CUDA_ARCHITECTURES} | sed 's/;/ /g' | sed 's/\([0-9]\)\([0-9]\)/\1.\2/g')" \
      /opt/venv/bin/python setup.py build_ext --inplace; \
    fi

# Shrink the image: build tools no longer needed once everything's compiled.
RUN /opt/venv/bin/pip uninstall -y cmake ninja || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /usr/local/lib /opt/fastmap -name "*.so*" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# ==============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG USERNAME=fastmap
ARG USER_UID=1000
ARG USER_GID=1000

ENV PATH=/opt/venv/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64 \
    PYOPENGL_PLATFORM=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-venv python3-pip \
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

USER ${USERNAME}
ENV HOME=/home/${USERNAME}

# Non-fatal sanity checks
RUN (command -v colmap >/dev/null 2>&1 && colmap -h >/dev/null 2>&1) || true
RUN python3 -c "import sys; sys.path.insert(0, '/opt/fastmap'); import torch, fastmap; print('torch.cuda:', torch.version.cuda)" || true

CMD ["/bin/bash"]
