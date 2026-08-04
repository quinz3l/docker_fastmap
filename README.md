splat-pipeline
A CUDA + PyTorch container for 3D reconstruction and Gaussian splatting, built with COLMAP, gsplat, nerfstudio, and FastMap. Designed to run as a RunPod GPU pod with a persistent /workspace network volume, and built automatically via GitHub Actions into GitHub Container Registry (ghcr.io).
What’s in the image
	•	Builder stage: CUDA devel base, OpenImageIO (built from source — required by COLMAP’s CMake config), COLMAP (built from source with CUDA support), PyTorch/torchvision/torchaudio, gsplat, nerfstudio, FastMap (cloned + optional custom CUDA kernel build), tiny-cuda-nn (optional, best-effort).
	•	Runtime stage: Slim CUDA runtime base with just nvcc + build-essential added back (needed because gsplat JIT-compiles some kernels on first use), plus all the shared libraries COLMAP needs at link time.
	•	Ships as a single image with JupyterLab as the default entrypoint, and RunPod’s Web Terminal execs directly into the same running container — no extra sshd needed.
Repo layout
