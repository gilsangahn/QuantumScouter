FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y \
    python3 python3-pip python3-dev \
    git build-essential nano curl

RUN pip3 install --upgrade pip

RUN pip3 install "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install pennylane pennylane-lightning[gpu]

RUN pip3 install optax scipy matplotlib

RUN pip3 install scikit-learn

RUN pip3 install pillow

RUN pip3 install seaborn

RUN pip3 install notebook jupyterlab

ENV JAX_ENABLE_X64=1

WORKDIR /workspace
