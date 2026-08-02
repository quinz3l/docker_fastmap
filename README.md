# FastMap Docker Image

A containerized build of [FastMap](https://github.com/pals-ttic/fastmap) — a fast, PyTorch-based structure-from-motion pipeline — along with its COLMAP-based preprocessing dependency, built and tested on RunPod GPU pods.

## Attribution

This image builds and packages the following third-party projects. All credit for the underlying software belongs to their respective authors; this repo only provides the Docker packaging.

| Project | Repository | Used for |
|---|---|---|
| FastMap | [pals-ttic/fastmap](https://github.com/pals-ttic/fastmap) | Core pose estimation / structure-from-motion pipeline |
| pyrender (fork) | [jiahaoli95/pyrender](https://github.com/jiahaoli95/pyrender) | Mesh/point-cloud visualization, required by FastMap |
| COLMAP | [colmap/colmap](https://github.com/colmap/colmap) | Feature extraction + matching (`feature_extractor`, `exhaustive_matcher`) to produce the `.db` file FastMap consumes |
| OpenImageIO | [AcademySoftwareFoundation/OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | Hard build dependency of COLMAP; built from source since Ubuntu 22.04's apt version is too old for current COLMAP |

If you use FastMap in your own work, please cite it and follow the license terms in its repository — this README doesn't substitute for that.

## What's in the image

- CUDA 12.4.1 (devel in the builder stage, runtime in the final image)
- PyTorch 2.4.1 (cu124)
- COLMAP, built from source (optional — see build args below)
- FastMap, cloned from source, with its optional custom CUDA kernels compiled
- pyrender (the `jiahaoli95` fork FastMap's own README specifies), trimesh, pyglet, and FastMap's other Python dependencies

## Build

```bash
docker build -t fastmap:ada \
  --build-arg CUDA_ARCHITECTURES="89" \
  --build-arg TORCH_CUDA_ARCH_LIST="8.9" \
  .
```

Build args:

| Arg | Default | Notes |
|---|---|---|
| `CUDA_VERSION` | `12.4.1` | Must stay in sync with `TORCH_CUDA_TAG` |
| `TORCH_VERSION` | `2.4.1` | |
| `TORCH_CUDA_TAG` | `cu124` | |
| `CUDA_ARCHITECTURES` | `89` (Ada) | Semicolon-separate for multiple, e.g. `"86;89"` |
| `INSTALL_COLMAP` | `true` | Set to `false` to skip the COLMAP/OpenImageIO build entirely if you already have matching `.db` files — saves 30+ minutes |
| `BUILD_CUDA_KERNELS` | `true` | FastMap's own README calls this "optional but highly recommended for speed" |

## Run

```bash
docker run --gpus all -it -v /workspace:/workspace fastmap:ada
```

FastMap lives at `/opt/fastmap` inside the container. For the full pipeline (per [FastMap's README](https://github.com/pals-ttic/fastmap)):

```bash
colmap feature_extractor --database_path db.db --image_path images/
colmap exhaustive_matcher --database_path db.db
python /opt/fastmap/run.py --data_dir . --headless
```

The `--headless` flag matters — this container has no display, and pyrender needs `PYOPENGL_PLATFORM=osmesa` (already set as a default env var in the image) to render offscreen.

## GPU architecture reference

| GPU | Architecture | Value |
|---|---|---|
| A4500 | Ampere | `86` |
| RTX 4090, RTX 2000 Ada, L4, L40S | Ada Lovelace | `89` |
| H100 | Hopper | `90` |
| RTX 5090 | Blackwell | `120` — needs CUDA ≥ 12.8, incompatible with this image's default CUDA 12.4.1 base |

## Notes on hard-won fixes baked into this Dockerfile

- **OpenImageIO must be explicitly installed to `/usr/local`.** Its CMake defaults `CMAKE_INSTALL_PREFIX` to a local `dist/` folder inside its own source tree if not overridden — silently breaking COLMAP's `find_package(OpenImageIO)` otherwise.
- **`CMAKE_CUDA_ARCHITECTURES` must be quoted** in every `RUN` command that uses it. Left unquoted, the shell splits a multi-value list like `86;89` into separate commands at the semicolon.
- **`libqt5svg5-dev` is required** alongside `qtbase5-dev`/`libqt5opengl5-dev` for COLMAP's current build, even with the GUI disabled.
- **pyrender is installed without a pinned branch** (`pip install git+https://github.com/jiahaoli95/pyrender.git`), matching FastMap's own documented command exactly — the repo's default branch isn't `main`.
