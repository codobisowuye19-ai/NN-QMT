import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
import scipy.linalg as la
import time
warnings.filterwarnings('ignore')

class POVMReconstructionNet(nn.Module):
    def __init__(self, povm_size=4, state_dim=4):
        super(POVMReconstructionNet, self).__init__()
        self.povm_size = povm_size
        self.state_dim = state_dim
        
        # Use complex parameters to represent POVM elements
        # Each POVM element is a 4x4 complex matrix, represented by real and imaginary parts separately
        self.povm_real = nn.Parameter(torch.randn(povm_size, state_dim, state_dim))
        self.povm_imag = nn.Parameter(torch.randn(povm_size, state_dim, state_dim))
        
        # Initialize POVM as equal division of identity matrix
        with torch.no_grad():
            identity = torch.eye(state_dim) / povm_size
            for i in range(povm_size):
                self.povm_real[i] = identity.clone()
                self.povm_imag[i] = torch.zeros_like(identity)
    
    def get_povm_elements(self):
        """Get POVM elements (complex matrices)"""
        povm_elements = []
        for i in range(self.povm_size):
            M_real = self.povm_real[i]
            M_imag = self.povm_imag[i]
            # Ensure Hermiticity: M = (M_real + 1j*M_imag) should be Hermitian
            # So we take M = (M_real + M_real.T)/2 + 1j*(M_imag - M_imag.T)/2
            M_real_sym = (M_real + M_real.T) / 2
            M_imag_sym = (M_imag - M_imag.T) / 2
            M = M_real_sym + 1j * M_imag_sym
            povm_elements.append(M.detach().numpy())
        return povm_elements
    
    def enforce_povm_constraints(self):
        """Enforce POVM constraints: positive semidefiniteness and completeness"""
        with torch.no_grad():
            povm_elements = []
            
            # Get current POVM elements and ensure positive semidefiniteness
            for i in range(self.povm_size):
                M_real = self.povm_real[i]
                M_imag = self.povm_imag[i]
                M_real_sym = (M_real + M_real.T) / 2
                M_imag_sym = (M_imag - M_imag.T) / 2
                M = M_real_sym + 1j * M_imag_sym
                
                # Convert to numpy for eigenvalue decomposition
                M_np = M.numpy()
                eigvals, eigvecs = np.linalg.eigh(M_np)
                eigvals = np.maximum(eigvals, 0)  # Ensure non-negative eigenvalues
                M_positive = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
                
                povm_elements.append(M_positive)
            
            # Normalize to satisfy completeness relation
            completeness = sum(povm_elements)
            # Use matrix square root for normalization
            sqrt_completeness = np.linalg.inv(sqrtm(completeness))
            
            normalized_povm = []
            for M in povm_elements:
                M_normalized = sqrt_completeness @ M @ sqrt_completeness.conj().T
                normalized_povm.append(M_normalized)
            
            # Update parameters
            for i, M in enumerate(normalized_povm):
                self.povm_real[i] = torch.tensor(M.real, dtype=torch.float32)
                self.povm_imag[i] = torch.tensor(M.imag, dtype=torch.float32)

def sqrtm(matrix):
    """Compute matrix square root"""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T

class QuantumMeasurementDataset(Dataset):
    def __init__(self, data):
        self.probe_states = []
        self.measurements = []
        
        # Define single-qubit states
        zero = np.array([[1], [0]], dtype=complex)
        one = np.array([[0], [1]], dtype=complex)
        plus = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
        minus = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)
        plus_i = np.array([[1/np.sqrt(2)], [1j/np.sqrt(2)]], dtype=complex)
        minus_i = np.array([[1/np.sqrt(2)], [-1j/np.sqrt(2)]], dtype=complex)
        
        state_dict = {
            '0': zero, '1': one, '+': plus, '-': minus, 
            '+i': plus_i, '-i': minus_i
        }
        
        for row in data:
            # Parse probe states
            state1, state2 = row[0].split(',')
            psi1 = state_dict[state1]
            psi2 = state_dict[state2]
            
            # Construct two-qubit state
            psi = np.kron(psi1, psi2)
            rho = psi @ psi.conj().T  # Density matrix
            
            # Normalize measurement frequencies
            freqs = np.array(row[1:], dtype=float)
            probabilities = freqs / np.sum(freqs)
            
            self.probe_states.append(rho)
            self.measurements.append(probabilities)
    
    def __len__(self):
        return len(self.probe_states)
    
    def __getitem__(self, idx):
        rho = self.probe_states[idx]
        # Convert complex density matrix to real representation
        rho_real = np.stack([rho.real, rho.imag], axis=-1)
        return torch.tensor(rho_real, dtype=torch.float32), torch.tensor(self.measurements[idx], dtype=torch.float32)

def quantum_trace(rho_real, M_real, M_imag):
    """Calculate quantum trace Tr(rho * M)"""
    batch_size = rho_real.shape[0]
    
    # Extract real and imaginary parts
    rho_real_part = rho_real[:, :, :, 0]  # shape: (batch, 4, 4)
    rho_imag_part = rho_real[:, :, :, 1]  # shape: (batch, 4, 4)
    
    # Calculate Tr(rho * M) = Tr(rho_real * M_real - rho_imag * M_imag) + i*Tr(rho_real * M_imag + rho_imag * M_real)
    # But we only need the real part since measurement probabilities are real
    trace_real = torch.einsum('bij,bij->b', rho_real_part, M_real) - torch.einsum('bij,bij->b', rho_imag_part, M_imag)
    
    return trace_real

def compute_loss(model, data_loader):
    """Compute loss on dataset"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_rho, batch_probs in data_loader:
            # Calculate predicted probabilities
            pred_probs = []
            for i in range(4):
                M_real = model.povm_real[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                M_imag = model.povm_imag[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                trace = quantum_trace(batch_rho, M_real, M_imag)
                pred_probs.append(trace)
            
            pred_probs = torch.stack(pred_probs, dim=1)
            
            # Ensure probabilities are positive and sum to 1
            pred_probs = torch.softmax(pred_probs, dim=1)
            
            # Calculate loss
            loss = nn.functional.mse_loss(pred_probs, batch_probs)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def fidelity_matrix(A, B):
    """
    Calculate fidelity between two positive semidefinite matrices A and B
    F(A, B) = [Tr(sqrt(sqrt(A) * B * sqrt(A)))]^2
    """
    # Ensure matrices are Hermitian
    A = (A + A.conj().T) / 2
    B = (B + B.conj().T) / 2
    
    # Ensure positive semidefiniteness (set negative eigenvalues to zero)
    eigvals_A, eigvecs_A = la.eigh(A)
    eigvals_A = np.maximum(eigvals_A, 0)
    A_pos = eigvecs_A @ np.diag(eigvals_A) @ eigvecs_A.conj().T
    
    eigvals_B, eigvecs_B = la.eigh(B)
    eigvals_B = np.maximum(eigvals_B, 0)
    B_pos = eigvecs_B @ np.diag(eigvals_B) @ eigvecs_B.conj().T
    
    # Compute matrix square root
    sqrt_A = la.sqrtm(A_pos)
    
    # Compute sqrt_A * B * sqrt_A
    M = sqrt_A @ B_pos @ sqrt_A
    
    # Compute square root of M
    sqrt_M = la.sqrtm(M)
    
    # Calculate trace and square it
    trace = np.trace(sqrt_M)
    fidelity = np.abs(trace)**2  # Take absolute value to ensure real number
    
    return np.real(fidelity)  # Return real part

def calculate_fidelities(povm_elements):
    """
    Calculate fidelities between reconstructed POVM and theoretical POVM
    
    Parameters:
    povm_elements: List of reconstructed POVM elements
    
    Returns:
    fidelities: List of four fidelities
    avg_fidelity: Average fidelity
    """
    # Theoretical POVM matrices
    M0_th = np.array([
        [0, 0, 0, 0],
        [0, 0.1, 0.3, 0],
        [0, 0.3, 0.9, 0],
        [0, 0, 0, 0]
    ], dtype=complex)
    
    M1_th = np.array([
        [0, 0, 0, 0],
        [0, 0.9, -0.3, 0],
        [0, -0.3, 0.1, 0],
        [0, 0, 0, 0]
    ], dtype=complex)
    
    M2_th = np.array([
        [0.1, 0, 0, 0.3],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0.3, 0, 0, 0.9]
    ], dtype=complex)
    
    M3_th = np.array([
        [0.9, 0, 0, -0.3],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.3, 0, 0, 0.1]
    ], dtype=complex)
    
    theoretical_povm = [M0_th, M1_th, M2_th, M3_th]
    
    print("\n" + "="*50)
    print("Fidelity Calculation")
    print("="*50)
    
    fidelities = []
    
    for i in range(4):
        F = fidelity_matrix(povm_elements[i], theoretical_povm[i])
        fidelities.append(F)
        print(f"F(M_{i}_rec, M_{i}_th) = {F:.6f}")
    
    avg_fidelity = np.mean(fidelities)
    print("-" * 50)
    print(f"Average Fidelity = {avg_fidelity:.6f}")
    
    # Output summary
    print("\nSummary:")
    print(f"Minimum Fidelity: {min(fidelities):.6f} (M{fidelities.index(min(fidelities))})")
    print(f"Maximum Fidelity: {max(fidelities):.6f} (M{fidelities.index(max(fidelities))})")
    
    return fidelities, avg_fidelity

def main():
    # Record total start time
    total_start_time = time.time()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Your data
    data =  [
    ['0,0', 9, 11, 93, 807],
    ['0,1', 89, 793, 8, 11],
    ['0,+', 45, 429, 35, 402],
    ['0,-', 41, 416, 49, 426],
    ['0,+i', 52, 398, 50, 407],
    ['0,-i', 55, 393, 45, 396],
    ['1,0', 803, 84, 14, 7],
    ['1,1', 8, 12, 806, 92],
    ['1,+', 407, 47, 413, 53],
    ['1,-', 407, 58, 415, 41],
    ['1,+i', 390, 50, 394, 43],
    ['1,-i', 397, 44, 418, 52],
    ['+,0', 409, 45, 42, 423],
    ['+,1', 46, 386, 401, 53],
    ['+,+', 359, 92, 353, 103],
    ['+,-', 96, 377, 97, 353],
    ['+,+i', 213, 205, 199, 203],
    ['+,-i', 217, 201, 215, 205],
    ['-,0', 412, 41, 43, 395],
    ['-,1', 47, 381, 402, 35],
    ['-,+', 107, 362, 99, 366],
    ['-,-', 362, 94, 369, 85],
    ['-,+i', 205, 220, 212, 206],
    ['-,-i', 210, 201, 191, 212],
    ['+i,0', 402, 52, 40, 409],
    ['+i,1', 56, 394, 414, 56],
    ['+i,+', 217, 223, 214, 229],
    ['+i,-', 222, 209, 223, 204],
    ['+i,+i', 376, 102, 99, 374],
    ['+i,-i', 94, 368, 366, 108],
    ['-i,0', 413, 51, 42, 392],
    ['-i,1', 53, 403, 401, 49],
    ['-i,+', 230, 202, 216, 219],
    ['-i,-', 201, 224, 233, 222],
    ['-i,+i', 107, 381, 385, 95],
    ['-i,-i', 381, 108, 88, 376]
    ]
    

    print("="*60)
    print("Neural Network POVM Reconstruction Started")
    print("="*60)
    
    # Record data preparation start time
    data_prep_start = time.time()
    
    # Create full dataset
    full_dataset = QuantumMeasurementDataset(data)
    
    print(f"Dataset Size: {len(full_dataset)}")
    print("Note: All data will be used for training (no validation set)")
    
    # Create DataLoader
    train_loader = DataLoader(full_dataset, batch_size=4, shuffle=True)
    
    data_prep_time = time.time() - data_prep_start
    print(f"Data Preparation Time: {data_prep_time:.2f} seconds")
    
    # Create model
    model = POVMReconstructionNet(povm_size=4, state_dim=4)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    
    # Train model
    num_epochs = 100
    train_losses = []
    
    print("\nTraining Started...")
    print("Epoch | Training Loss")
    print("-" * 30)
    
    # Record training start time
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Training phase
        for batch_rho, batch_probs in train_loader:
            optimizer.zero_grad()
            
            # Calculate predicted probabilities
            pred_probs = []
            for i in range(4):
                M_real = model.povm_real[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                M_imag = model.povm_imag[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                trace = quantum_trace(batch_rho, M_real, M_imag)
                pred_probs.append(trace)
            
            pred_probs = torch.stack(pred_probs, dim=1)
            
            # Ensure probabilities are positive and sum to 1
            pred_probs = torch.softmax(pred_probs, dim=1)
            
            # Calculate loss
            loss = nn.functional.mse_loss(pred_probs, batch_probs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Print training loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:5d} | {avg_train_loss:.6f}")
        
        # Enforce POVM constraints every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.enforce_povm_constraints()
    
    # Enforce POVM constraints at the last epoch
    if num_epochs % 100 != 0:
        model.enforce_povm_constraints()
    
    # Calculate total training time
    training_time = time.time() - training_start_time
    
    print(f"\nTotal Training Time: {training_time:.2f} seconds")
    print(f"Training Epochs: {num_epochs}")
    
    # Analyze training process
    print("\n" + "="*50)
    print("Training Process Analysis")
    print("="*50)
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Minimum Training Loss: {min(train_losses):.6f}")
    
    # Record evaluation start time
    evaluation_start_time = time.time()
    
    # Get final POVM elements
    povm_elements = model.get_povm_elements()
    
    # Output POVM elements
    print("\n" + "="*50)
    print("Reconstructed POVM Elements")
    print("="*50)
    for i, M in enumerate(povm_elements):
        print(f"\nM_{i}:")
        print(np.round(M, 4))
        print(f"Trace: {np.trace(M).real:.6f}")
    
    # Verify POVM properties
    verify_povm_properties(povm_elements)
    
    # Evaluate model performance on full dataset
    print("\n" + "="*50)
    print("Model Performance Evaluation")
    print("="*50)
    
    # Calculate MSE on training set
    train_mse = compute_loss(model, train_loader)
    print(f"Training Set MSE: {train_mse:.6f}")
    
    # Calculate fidelities
    fidelities, avg_fidelity = calculate_fidelities(povm_elements)
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"\nModel Evaluation Time: {evaluation_time:.2f} seconds")
    
    # Calculate total runtime
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Runtime Summary")
    print("="*60)
    print(f"Data Preparation Time: {data_prep_time:.2f} seconds")
    print(f"Total Training Time: {training_time:.2f} seconds")
    print(f"Model Evaluation Time: {evaluation_time:.2f} seconds")
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"Total Runtime: {total_time/60:.2f} minutes")
    
    # Print completion message
    print("\n" + "="*60)
    print("Neural Network POVM Reconstruction Completed")
    print("="*60)
    
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def verify_povm_properties(povm_elements):
    """Verify POVM properties"""
    print("\n" + "="*50)
    print("POVM Properties Verification")
    print("="*50)
    
    # 1. Check completeness relation
    completeness = sum(povm_elements)
    print(f"\nCompleteness Relation Check (∑M_i = I):")
    print(f"∑M_i = \n{np.round(completeness, 4)}")
    print(f"Norm difference from identity matrix: {np.linalg.norm(completeness - np.eye(4)):.6f}")
    
    # 2. Check positive semidefiniteness
    print("\nPositive Semidefiniteness Check:")
    all_positive = True
    for i, M in enumerate(povm_elements):
        eigenvalues = np.linalg.eigvalsh(M)
        min_eigenvalue = np.min(eigenvalues)
        print(f"  M_{i} minimum eigenvalue: {min_eigenvalue:.6f}", end="")
        if min_eigenvalue < -1e-6:  # Allow small numerical error
            print("  ❌ Not positive semidefinite!")
            all_positive = False
        else:
            print("  ✓")
    
    # 3. Check Hermiticity
    print("\nHermiticity Check:")
    all_hermitian = True
    for i, M in enumerate(povm_elements):
        hermitian_diff = np.linalg.norm(M - M.conj().T)
        print(f"  M_{i} Hermiticity difference: {hermitian_diff:.6f}", end="")
        if hermitian_diff > 1e-6:  # Allow small numerical error
            print("  ❌ Not Hermitian!")
            all_hermitian = False
        else:
            print("  ✓")
    
    # Summary
    print("\n" + "-"*30)
    print("Verification Summary:")
    if all_positive and all_hermitian and np.linalg.norm(completeness - np.eye(4)) < 0.01:
        print("✅ POVM satisfies all basic properties")
    else:
        print("⚠️ POVM may not satisfy some properties")

if __name__ == "__main__":
    main()