import numpy as np

def generate_depolarizing_data_simple(data, depolarizing_prob):
    """
    Generate depolarizing noise data for quantum measurement tomography data, 
    maintaining the same format as the original data
    
    Parameters:
    data: Original data list, format: [label, HH_counts, HV_counts, VH_counts, VV_counts]
    depolarizing_prob: Depolarizing probability (between 0-1)
    
    Returns:
    Noisy data list, maintaining 36-row format
    """
    
    noisy_data = []
    
    for row in data:
        label = row[0]
        counts = np.array(row[1:], dtype=float)
        total_counts = np.sum(counts)
        
        if total_counts > 0:
            # Calculate original probabilities
            probabilities = counts / total_counts
            
            # Apply depolarizing noise: P_noisy = (1 - ε) * P_original + ε * 0.25
            noisy_probs = (1 - depolarizing_prob) * probabilities + depolarizing_prob * 0.25
            
            # Convert back to counts and round
            noisy_counts = np.round(noisy_probs * total_counts).astype(int)
            
            # Adjust total (keep as 1000)
            current_total = np.sum(noisy_counts)
            if current_total != total_counts:
                # Adjust the largest count to maintain total
                max_idx = np.argmax(noisy_counts)
                noisy_counts[max_idx] += (total_counts - current_total)
            
            noisy_data.append([label] + noisy_counts.tolist())
    
    return noisy_data

# Original data
original_data = [
    ['0,0', 0, 0, 100, 900],
    ['0,1', 100, 900, 0, 0],
    ['0,+', 50, 450, 50, 450],
    ['0,-', 50, 450, 50, 450],
    ['0,+i', 50, 450, 50, 450],
    ['0,-i', 50, 450, 50, 450],
    ['1,0', 900, 100, 0, 0],
    ['1,1', 0, 0, 900, 100],
    ['1,+', 450, 50, 450, 50],
    ['1,-', 450, 50, 450, 50],
    ['1,+i', 450, 50, 450, 50],
    ['1,-i', 450, 50, 450, 50],
    ['+,0', 450, 50, 50, 450],
    ['+,1', 50, 450, 450, 50],
    ['+,+', 400, 100, 400, 100],
    ['+,-', 100, 400, 100, 400],
    ['+,+i', 250, 250, 250, 250],
    ['+,-i', 250, 250, 250, 250],
    ['-,0', 450, 50, 50, 450],
    ['-,1', 50, 450, 450, 50],
    ['-,+', 100, 400, 100, 400],
    ['-,-', 400, 100, 400, 100],
    ['-,+i', 250, 250, 250, 250],
    ['-,-i', 250, 250, 250, 250],
    ['+i,0', 450, 50, 50, 450],
    ['+i,1', 50, 450, 450, 50],
    ['+i,+', 250, 250, 250, 250],
    ['+i,-', 250, 250, 250, 250],
    ['+i,+i', 400, 100, 100, 400],
    ['+i,-i', 100, 400, 400, 100],
    ['-i,0', 450, 50, 50, 450],
    ['-i,1', 50, 450, 450, 50],
    ['-i,+', 250, 250, 250, 250],
    ['-i,-', 250, 250, 250, 250],
    ['-i,+i', 100, 400, 400, 100],
    ['-i,-i', 400, 100, 100, 400]
]

if __name__ == "__main__":
    print("Quantum Measurement Tomography Data - Depolarizing Noise Generation")
    print("=" * 60)
    
    # Define all required noise levels
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Generate data for each noise level
    for noise_level in noise_levels:
        # Generate depolarizing noise data
        noisy_data = generate_depolarizing_data_simple(original_data, noise_level)
        
        # Calculate some statistics
        total_counts = sum(sum(row[1:]) for row in noisy_data)
        
        # Output complete data
        print(f"\n{'='*60}")
        print(f"Complete data for noise level {noise_level:.1f} (Python format):")
        print(f"Total counts: {total_counts}, Data rows: {len(noisy_data)}")
        print(f"{'='*60}")
        print(f"data_depolarizing_{int(noise_level*10)} = [")
        for i, row in enumerate(noisy_data):
            comma = "," if i < len(noisy_data) - 1 else ""
            print(f"    ['{row[0]}', {row[1]}, {row[2]}, {row[3]}, {row[4]}]{comma}")
        print("]")
    
    # Add a summary showing changes in the first measurement basis under different noise levels
    print(f"\n{'='*60}")
    print("Changes in the first measurement basis '0,0' under different noise levels:")
    print(f"{'='*60}")
    print(f"{'Noise Level':<10} {'HH':<6} {'HV':<6} {'VH':<6} {'VV':<6} {'Total Counts':<8}")
    print("-" * 50)
    
    for noise_level in noise_levels:
        noisy_data = generate_depolarizing_data_simple(original_data, noise_level)
        first_row = noisy_data[0]  # Data for '0,0'
        total_counts = sum(first_row[1:])
        print(f"{noise_level:<10.1f} {first_row[1]:<6} {first_row[2]:<6} {first_row[3]:<6} {first_row[4]:<6} {total_counts:<8}")
    
    print(f"\n{'='*60}")
    print("Notes:")
    print("1. Data for each noise level maintains 36 rows, same as original format")
    print("2. Total count per row remains 1000")
    print("3. Noise level 1.0 represents complete depolarization, all measurement results approach uniform distribution")
    print("4. Data variable name format: data_depolarizing_1 represents noise level 0.1")
    print(f"{'='*60}")