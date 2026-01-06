# Quantum Measurement Tomography (QMT) and POVM Reconstruction

This project provides two different approaches for reconstructing Positive Operator-Valued Measures (POVM) in Quantum Measurement Tomography (QMT): **Maximum Likelihood Estimation (MLE)** and **Neural Networks (NN)**. Additionally, it includes a data generation script for simulating experimental data under different noise levels.

---

## 📁 File Structure

```
.
├── MLE.py                    # POVM reconstruction based on Maximum Likelihood Estimation
├── NN-QMT.py                 # POVM reconstruction based on Neural Networks
└── simulated data.py         # Generation of simulated data with depolarizing noise
```

---

## 📊 Method Overview

### 1. **MLE Method (`MLE.py`)**
- **Principle**: Parameterizes POVM elements using Cholesky decomposition and optimizes by minimizing the negative log-likelihood function.
- **Features**:
  - Traditional optimization method with stable convergence
  - Includes regularization to prevent overfitting
  - Enforces completeness and positive semidefiniteness constraints
- **Outputs**:
  - Reconstructed POVM elements
  - Fidelity calculation (compared to theoretical POVM)
  - Reconstruction error evaluation (MSE)

### 2. **Neural Network Method (`NN-QMT.py`)**
- **Principle**: Uses PyTorch to build a neural network that directly parameterizes POVM elements and trains using MSE loss.
- **Features**:
  - End-to-end training without manual optimization design
  - Includes early stopping to prevent overfitting
  - Enforces POVM constraints during training
- **Outputs**:
  - Training/validation loss curves
  - Reconstructed POVM elements
  - Fidelity and reconstruction error metrics

### 3. **Data Generation Script (`simulated data.py`)**
- **Function**: Adds depolarizing noise to ideal experimental data to simulate noise effects in real experiments.
- **Supported noise levels**: 0.1 to 1.0 (in steps of 0.1)
- **Output format**: Consistent with original data format for direct use in the two reconstruction methods.

---

## ⚙️ Dependencies

### Python Version
- Python 3.13+

### Main Libraries
```
numpy
scipy
matplotlib
torch (required only for NN-QMT.py)
tqdm
```

Install via:
```bash
pip install numpy scipy matplotlib torch tqdm
```

---

## 🚀 Quick Start

### Run MLE Reconstruction
```bash
python MLE.py
```

### Run Neural Network Reconstruction
```bash
python NN-QMT.py
```

### Generate Simulated Data
```bash
python simulated\ data.py
```

---

## 📈 Output Description

### Common Outputs:
- Four reconstructed POVM elements (4×4 complex matrices)
- Fidelity comparison (against theoretical POVM)
- Average reconstruction error (MSE)
- POVM property verification (completeness, positive semidefiniteness, Hermiticity)

### Additional Outputs (Neural Network only):
- Training/validation loss curves
- Training time statistics
- Early stopping notifications (if triggered)

---

## 🔧 Customization

### Parameter Modification (MLE example)
```python
reconstructor = MLE_POVM_Reconstructor(
    povm_size=4,      # Number of POVM elements
    state_dim=4       # Quantum state dimension (4 for two-qubit)
)
povm_elements = reconstructor.fit(
    data, 
    max_iter=2000,    # Maximum iterations
    method='L-BFGS-B' # Optimization method
)
```

### Using Custom Data
- Data format should be a list, each row like: `['0,+', 100, 200, 300, 400]`
- The first two characters indicate two-qubit preparation bases (e.g., `0`, `1`, `+`, `-`, `+i`, `-i`)
- The following four numbers are counts for corresponding POVM outcomes (total counts are recommended to be normalized)

---

## 📌 Notes

1. **Data Types**: All density matrices and POVM elements are complex matrices; real and imaginary parts should be handled separately in computations.
2. **Numerical Stability**: Regularization and small offsets are added to avoid division by zero or log(0) errors.
3. **Constraints**: Both methods enforce POVM constraints: completeness, positive semidefiniteness, and Hermiticity.
4. **Runtime**: The neural network method typically requires longer training time; GPU acceleration is recommended for faster execution.

---


## 🛠️ Future Extensions

- Add more noise models (e.g., amplitude damping, phase damping)
- Implement other optimization algorithms (e.g., gradient descent, conjugate gradient)
- Support POVM reconstruction for more qubits
- Provide visualization modules (e.g., Bloch sphere representation)

---

## 📄 License

This project code is for academic research use only. Please acknowledge the source when using.

---

For any questions or suggestions, feel free to open an Issue or contact the author.
