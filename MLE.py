import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time  # Add time module

class MLE_POVM_Reconstructor:
    def __init__(self, povm_size=4, state_dim=4):
        """
        Initialize MLE POVM reconstructor
        
        Parameters:
        povm_size: Number of POVM elements
        state_dim: State dimension (4 for two-qubit system)
        """
        self.povm_size = povm_size
        self.state_dim = state_dim
        
        # Initialize POVM elements as equal partitions of identity matrix
        self.povm_elements = [np.eye(state_dim, dtype=complex) / povm_size for _ in range(povm_size)]
    
    def get_cholesky_params(self):
        """
        Parameterize POVM elements using Cholesky decomposition
        Return vector of real optimization parameters
        """
        params = []
        for i in range(self.povm_size - 1):
            # Create lower triangular matrix for each POVM element (complex)
            # For n x n matrix, need n*(n+1) real parameters
            L_real = np.random.randn(self.state_dim, self.state_dim)
            L_imag = np.random.randn(self.state_dim, self.state_dim)
            
            # Ensure lower triangular matrix (zeros above diagonal)
            L_real = np.tril(L_real)
            L_imag = np.tril(L_imag)
            
            params.extend(L_real.flatten())
            params.extend(L_imag.flatten())
        
        return np.array(params)
    
    def params_to_povm(self, params):
        """
        Convert parameter vector to POVM elements
        
        Parameters:
        params: Real parameter vector
        
        Returns:
        List of POVM elements
        """
        # Each POVM element requires 2 * state_dim * state_dim parameters
        elem_size = 2 * self.state_dim * self.state_dim
        
        povm_elements = []
        param_idx = 0
        
        for i in range(self.povm_size - 1):
            # Extract real and imaginary parts
            L_real_flat = params[param_idx:param_idx + self.state_dim**2]
            param_idx += self.state_dim**2
            
            L_imag_flat = params[param_idx:param_idx + self.state_dim**2]
            param_idx += self.state_dim**2
            
            # Reshape to matrix and ensure lower triangular
            L_real = np.tril(L_real_flat.reshape(self.state_dim, self.state_dim))
            L_imag = np.tril(L_imag_flat.reshape(self.state_dim, self.state_dim))
            
            # Construct lower triangular matrix L
            L = L_real + 1j * L_imag
            
            # Construct positive semidefinite matrix: M = L @ L^†
            M = L @ L.conj().T
            
            # Ensure trace is real
            M = (M + M.conj().T) / 2
            
            povm_elements.append(M)
        
        # Last POVM element computed via completeness relation
        completeness_sum = sum(povm_elements)
        M_last = np.eye(self.state_dim) - completeness_sum
        
        # Ensure last element is positive semidefinite
        eigvals, eigvecs = la.eigh(M_last)
        eigvals = np.maximum(eigvals, 0)
        M_last = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        povm_elements.append(M_last)
        
        return povm_elements
    
    def compute_probabilities(self, rho, povm_elements):
        """
        Compute probabilities for given density matrix and POVM elements
        
        Parameters:
        rho: Density matrix
        povm_elements: List of POVM elements
        
        Returns:
        Probability vector
        """
        probs = []
        for M in povm_elements:
            prob = np.real(np.trace(rho @ M))
            probs.append(max(prob, 0))  # Ensure non-negative probability
        
        # Normalize (numerical stability)
        probs = np.array(probs)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        
        return probs
    
    def negative_log_likelihood(self, params, probe_states, measurements):
        """
        Compute negative log-likelihood function
        
        Parameters:
        params: Parameter vector
        probe_states: List of probe states
        measurements: List of measurement counts
        
        Returns:
        Negative log-likelihood value
        """
        # Convert parameters to POVM elements
        povm_elements = self.params_to_povm(params)
        
        nll = 0.0  # Negative log-likelihood
        
        for rho, counts in zip(probe_states, measurements):
            # Compute theoretical probabilities
            probs = self.compute_probabilities(rho, povm_elements)
            
            # Add small value to avoid log(0)
            epsilon = 1e-10
            probs = np.maximum(probs, epsilon)
            
            # Compute log-likelihood
            total_counts = np.sum(counts)
            
            # Multinomial log-likelihood
            log_likelihood = np.sum(counts * np.log(probs))
            
            # Add negative sign (since we minimize)
            nll -= log_likelihood / total_counts  # Normalization
        
        # Add regularization term for numerical stability
        reg_term = 0.001 * np.sum(params**2)
        nll += reg_term
        
        return nll
    
    def prepare_data(self, data):
        """
        Prepare data
        
        Parameters:
        data: Raw data list
        
        Returns:
        probe_states: List of probe states
        measurements: List of measurement counts
        """
        probe_states = []
        measurements = []
        
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
            # Parse probe state
            state1, state2 = row[0].split(',')
            psi1 = state_dict[state1]
            psi2 = state_dict[state2]
            
            # Construct two-qubit state
            psi = np.kron(psi1, psi2)
            rho = psi @ psi.conj().T  # Density matrix
            
            # Get measurement counts
            counts = np.array(row[1:], dtype=float)
            
            probe_states.append(rho)
            measurements.append(counts)
        
        return probe_states, measurements
    
    def fit(self, data, max_iter=1000, method='L-BFGS-B'):
        """
        Fit POVM using MLE
        
        Parameters:
        data: Raw data
        max_iter: Maximum number of iterations
        method: Optimization method
        
        Returns:
        Reconstructed POVM elements
        """
        # Prepare data
        probe_states, measurements = self.prepare_data(data)
        
        # Initialize parameters
        initial_params = self.get_cholesky_params()
        
        print("Starting maximum likelihood estimation optimization...")
        start_time = time.time()  # Record optimization start time
        
        # Define objective function
        def objective(params):
            return self.negative_log_likelihood(params, probe_states, measurements)
        
        # Optimize parameters
        result = minimize(
            objective,
            initial_params,
            method=method,
            options={
                'maxiter': max_iter,
                'disp': True,
                'gtol': 1e-8,
                'ftol': 1e-8,
            }
        )
        
        optimization_time = time.time() - start_time  # Calculate optimization time
        print(f"Optimization completed: {result.message}")
        print(f"Final negative log-likelihood value: {result.fun:.6f}")
        print(f"Optimization runtime: {optimization_time:.2f} seconds")
        
        # Get optimal POVM elements
        optimal_povm = self.params_to_povm(result.x)
        
        # Verify and normalize POVM elements
        optimal_povm = self.normalize_povm(optimal_povm)
        
        self.povm_elements = optimal_povm
        
        return optimal_povm
    
    def normalize_povm(self, povm_elements):
        """
        Normalize POVM elements to ensure completeness relation
        
        Parameters:
        povm_elements: List of POVM elements
        
        Returns:
        Normalized list of POVM elements
        """
        # Compute current sum
        completeness_sum = sum(povm_elements)
        
        # Compute inverse square root
        eigvals, eigvecs = la.eigh(completeness_sum)
        eigvals = np.maximum(eigvals, 1e-10)  # Avoid division by zero
        sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
        
        # Normalize each POVM element
        normalized_povm = []
        for M in povm_elements:
            M_norm = sqrt_inv @ M @ sqrt_inv.conj().T
            normalized_povm.append(M_norm)
        
        return normalized_povm
    
    def verify_povm_properties(self, povm_elements=None):
        """
        Verify properties of POVM
        
        Parameters:
        povm_elements: List of POVM elements to verify, uses self.povm_elements if None
        """
        if povm_elements is None:
            povm_elements = self.povm_elements
        
        print("Verifying POVM properties:")
        
        # 1. Check completeness relation
        completeness = sum(povm_elements)
        identity = np.eye(self.state_dim, dtype=complex)
        
        print(f"Completeness relation check (∑M_i = I):")
        print(f"Trace of ∑M_i: {np.trace(completeness):.6f}")
        print(f"Norm difference with identity matrix: {la.norm(completeness - identity):.6f}")
        
        # 2. Check positive semidefiniteness
        print("\nPositive semidefiniteness check:")
        for i, M in enumerate(povm_elements):
            eigenvalues = la.eigvalsh(M)
            min_eigenvalue = np.min(eigenvalues)
            print(f"M_{i} minimum eigenvalue: {min_eigenvalue:.6f}")
            print(f"M_{i} trace: {np.trace(M.real):.6f}")
        
        # 3. Check Hermiticity
        print("\nHermiticity check:")
        for i, M in enumerate(povm_elements):
            hermitian_diff = la.norm(M - M.conj().T)
            print(f"M_{i} Hermiticity difference: {hermitian_diff:.6f}")
    
    def evaluate_reconstruction(self, data):
        """
        Evaluate reconstruction accuracy
        
        Parameters:
        data: Raw data
        
        Returns:
        Average MSE
        """
        probe_states, measurements = self.prepare_data(data)
        
        total_mse = 0
        total_samples = len(probe_states)
        
        for rho, counts in zip(probe_states, measurements):
            # Compute theoretical probabilities
            probs_theory = self.compute_probabilities(rho, self.povm_elements)
            
            # Compute empirical probabilities
            probs_empirical = counts / np.sum(counts)
            
            # Compute MSE
            mse = np.mean((probs_theory - probs_empirical) ** 2)
            total_mse += mse
        
        avg_mse = total_mse / total_samples
        print(f"Average reconstruction MSE: {avg_mse:.6f}")
        
        return avg_mse
    
    def print_povm_elements(self, povm_elements=None):
        """
        Print POVM elements
        
        Parameters:
        povm_elements: List of POVM elements to print, uses self.povm_elements if None
        """
        if povm_elements is None:
            povm_elements = self.povm_elements
        
        print("\nReconstructed POVM elements:")
        for i, M in enumerate(povm_elements):
            print(f"\nM_{i}:")
            # Print real and imaginary parts
            print("Real part:")
            print(np.round(M.real, 10))
            print("Imaginary part:")
            print(np.round(M.imag, 10))
            print(f"Trace: {np.trace(M.real):.6f}")


def fidelity_matrix(A, B):
    """
    Compute fidelity between two positive semidefinite matrices A and B
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
    
    # Compute trace and square it
    trace = np.trace(sqrt_M)
    fidelity = np.abs(trace)**2  # Take absolute value to ensure real
    
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
    print(f"Average fidelity = {avg_fidelity:.6f}")
    
    # Output summary
    print("\nSummary:")
    print(f"Minimum fidelity: {min(fidelities):.6f} (M{fidelities.index(min(fidelities))})")
    print(f"Maximum fidelity: {max(fidelities):.6f} (M{fidelities.index(max(fidelities))})")
    
    return fidelities, avg_fidelity


def main():
    # Record total start time
    total_start_time = time.time()
    
    # Data
    data = [
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
    
    # Create complete dataset
    
    print("="*60)
    print("MLE POVM Reconstruction Started")
    print("="*60)
    
    # Create MLE reconstructor
    reconstructor = MLE_POVM_Reconstructor(povm_size=4, state_dim=4)
    
    # Fit POVM (includes internal time measurement)
    povm_elements = reconstructor.fit(data, max_iter=2000)
    
    # Print POVM elements
    reconstructor.print_povm_elements()
    
    # Verify POVM properties
    reconstructor.verify_povm_properties()
    
    # Evaluate reconstruction accuracy
    reconstructor.evaluate_reconstruction(data)
    
    # Calculate fidelities
    calculate_fidelities(povm_elements)
    
    # Calculate total runtime
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Runtime Summary")
    print("="*60)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total runtime: {total_time/60:.2f} minutes")
    
    # Print completion message
    print("\n" + "="*60)
    print("MLE POVM Reconstruction Completed")
    print("="*60)


if __name__ == "__main__":
    main()