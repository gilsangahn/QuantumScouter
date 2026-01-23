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

You can set up the environment manually via `pip` or use the provided Docker image for a reproducible environment.
