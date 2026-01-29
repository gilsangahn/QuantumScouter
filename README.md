# QuantumScouter

This repository contains the source code and datasets for the paper:  
    **"QuantumScouter: Reinforcement Learning-Based Optimization of Variational
Quantum Circuits for Differential Cryptanalysis"**.

## 📌 Overview
QuantumScouter is a novel framework that utilizes **Deep Reinforcement Learning (DRL)** to automatically optimize the architecture of Variational Quantum Circuits (VQC) for cryptographic tasks. This repository includes:
- **Model A (Baseline)**: **Quantum CNN (QCNN)** based on Supervised Entanglement Learning (SEL).
- **Model B**: A hybrid approach combining **RL** with Transfer Learning from **SEL**.
- **Model C**: A **Pure RL** approach training from scratch without transfer learning.
- **Generalization Tests**: Experiments on **Parity** and **Iris** datasets to verify versatility.

---

## 📂 Directory Structure
```
.  
├── Dockerfile         
├── requirements.txt
├── datasets/  
│   ├── generator_speck.py/                   
│   └── generator_simon.py/
│   ├── parity.txt/
│   ├── iris.txt/
├── experiments/
│   ├── model_a_definition_execution_sel_qcnn.ipynb
│   ├── model_b_definition_sel_rl.ipynb
│   ├── model_b_execution_sel_rl.ipynb 
│   ├── model_c_definition_rl.ipynb
│   ├── model_c_execution_rl.ipynb
│   ├── parity_definition.ipynb
│   ├── parity_execution.ipynb
│   ├── iris_definition.ipynb                   
│   └── iris_execution.ipynb
└── evaluation/  
    └── eval.ipynb            
```
---

## 💾 Dataset Details

The `datasets/` directory contains scripts and text files required to train and evaluate the models.

* **`generator_speck.py`**: A Python script that generates plaintext pairs and corresponding labels (distinguisher data) for the **SPECK32/64** lightweight block cipher.
* **`generator_simon.py`**: A Python script that generates training data for the **SIMON32/64** lightweight block cipher.
* **`parity.txt`**: A dataset for the **4-bit Parity Problem**, used as a toy example to test the agent's ability to learn simple boolean functions.
* **`iris.txt`**: The **Iris Classification** dataset, pre-processed for quantum embedding, used to benchmark the generalization capability of the VQC.

---

## 🚀 Experiment Workflow

Our codebase follows a modular design pattern, separating the **Environment Definition** from the **Training Execution** to ensure clarity and reusability.

### 1. Definition Files (`*_definition.ipynb`)
These files define the core components of the experiment:
* **Quantum Environment**: The PennyLane-based quantum circuit environment.
* **RL Agent**: The Deep Q-Network (DQN) architecture.
* **Reward Function**: The logic for calculating rewards based on accuracy, cost, and circuit efficiency.
* **Ansatz Construction**: Functions to dynamically build quantum circuits from the agent's actions.

### 2. Execution Files (`*_execution.ipynb`)
These files handle the actual training process:
* **Hyperparameter Setup**: Configuration of learning rate, batch size, gamma, etc.
* **Training Loop**: The main RL loop (Interaction $\rightarrow$ Storage $\rightarrow$ Optimization).
* **Logging**: Saving training metrics (Reward, Accuracy) and models to `.pkl` files.

---

## 🛠️ Requirements & Installation

To ensure reproducibility, we provide both a Docker environment (recommended) and a `requirements.txt` file.

### Option 1: Docker (Recommended)
We provide a `Dockerfile` based on `nvidia/cuda:12.2.2` to guarantee the exact GPU environment used in our experiments.

1.  **Build the Docker image:**
    (Don't forget the dot `.` at the end!)
    ```bash
    docker build -t quantumscouter .
    ```

2.  **Run the container:**
    This command mounts the current directory to `/workspace` and enables GPU access.
    ```bash
    docker run --gpus all -it --rm -v $(pwd):/workspace quantumscouter bash
    ```

### Option 2: Manual Installation (pip)
If you prefer to set up the environment locally, please ensure you are using **Python 3.10+** and **CUDA 12.x**.

```bash
pip install -r requirements.txt
