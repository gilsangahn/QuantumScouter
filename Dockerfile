# Base image with CUDA 12.2 support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    git build-essential nano curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Install JAX with CUDA 12 support
RUN pip3 install "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Quantum & Scientific Libraries (PennyLane with GPU support)
# 'import-ipynb' is required for importing notebooks
RUN pip3 install pennylane pennylane-lightning[gpu] \
    optax scipy matplotlib scikit-learn \
    pillow seaborn notebook jupyterlab \
    import-ipynb

# Enable 64-bit precision for JAX
ENV JAX_ENABLE_X64=1

WORKDIR /workspace
