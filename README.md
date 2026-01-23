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

1. Build the image:
   ```bash
   docker build -t quantum-scouter .
