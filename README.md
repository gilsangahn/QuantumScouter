# QuantumScouter

This repository contains the source code and datasets for the paper  
    **"QuantumScouter: Reinforcement Learning-Based Optimization of Variational
Quantum Circuits for Differential Cryptanalysis"**.

## Directory Structure
```
.  
├── Dockerfile         
├── requirements.txt
├── datasets/  
│   ├── generator_SPECK.py/                   
│   └── generator_SIMON.py/
├── RL_QML/
│   ├── model_a_definition.ipynb
│   ├── model_a_excution.ipynb
│   ├── model_b_definition.ipynb
│   ├── model_b_excution.ipynb 
│   ├── model_c_definition.ipynb                    
│   └── model_c_excution.ipynb
└── evaluation/  
    └── eval.ipynb            
```
## Requirements

To ensure reproducibility, we provide both a Docker environment (recommended) and a requirements file.

### Option 1: Docker (Recommended)
We provide a Dockerfile based on `nvidia/cuda:12.2.2` to guarantee the exact GPU environment used in our experiments.

1. Build the image (Don't forget the dot '.' at the end!):
   ```bash
   docker build -t quantumscouter .
   
2. Run the container (with GPU support):
   ```bash
   docker run --gpus all -it --rm -v $(pwd):/workspace quantumscouter bash 

### Option 2: Manual Installation (pip)
If you prefer to set up the environment locally, please use Python 3.10+ and CUDA 12.x.
```bash
pip install -r requirements.txt
