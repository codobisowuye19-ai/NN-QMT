import numpy as np

def generate_depolarizing_data_simple(data, depolarizing_prob):
    """
    为量子测量层析数据生成去极化噪声数据，保持与原数据相同的格式
    
    参数:
    data: 原始数据列表，格式为 [标签, HH_counts, HV_counts, VH_counts, VV_counts]
    depolarizing_prob: 去极化概率 (0-1之间)
    
    返回:
    带噪声的数据列表，保持36行格式
    """
    
    noisy_data = []
    
    for row in data:
        label = row[0]
        counts = np.array(row[1:], dtype=float)
        total_counts = np.sum(counts)
        
        if total_counts > 0:
            # 计算原始概率
            probabilities = counts / total_counts
            
            # 应用去极化噪声：P_noisy = (1 - ε) * P_original + ε * 0.25
            noisy_probs = (1 - depolarizing_prob) * probabilities + depolarizing_prob * 0.25
            
            # 转换回计数并四舍五入
            noisy_counts = np.round(noisy_probs * total_counts).astype(int)
            
            # 调整总数（保持为1000）
            current_total = np.sum(noisy_counts)
            if current_total != total_counts:
                # 调整最大的计数以保持总数
                max_idx = np.argmax(noisy_counts)
                noisy_counts[max_idx] += (total_counts - current_total)
            
            noisy_data.append([label] + noisy_counts.tolist())
    
    return noisy_data

# 原始数据
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
    print("量子测量层析数据 - 去极化噪声生成")
    print("=" * 60)
    
    # 定义所有需要的噪声水平
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # 为每个噪声水平生成数据
    for noise_level in noise_levels:
        # 生成去极化噪声数据
        noisy_data = generate_depolarizing_data_simple(original_data, noise_level)
        
        # 计算一些统计信息
        total_counts = sum(sum(row[1:]) for row in noisy_data)
        
        # 输出完整数据
        print(f"\n{'='*60}")
        print(f"噪声水平 {noise_level:.1f} 的完整数据 (Python格式):")
        print(f"总计数: {total_counts}, 数据行数: {len(noisy_data)}")
        print(f"{'='*60}")
        print(f"data_depolarizing_{int(noise_level*10)} = [")
        for i, row in enumerate(noisy_data):
            comma = "," if i < len(noisy_data) - 1 else ""
            print(f"    ['{row[0]}', {row[1]}, {row[2]}, {row[3]}, {row[4]}]{comma}")
        print("]")
    
    # 添加一个总结，显示不同噪声水平下第一个测量基的变化
    print(f"\n{'='*60}")
    print("不同噪声水平下第一个测量基 '0,0' 的变化:")
    print(f"{'='*60}")
    print(f"{'噪声水平':<10} {'HH':<6} {'HV':<6} {'VH':<6} {'VV':<6} {'总计数':<8}")
    print("-" * 50)
    
    for noise_level in noise_levels:
        noisy_data = generate_depolarizing_data_simple(original_data, noise_level)
        first_row = noisy_data[0]  # '0,0' 的数据
        total_counts = sum(first_row[1:])
        print(f"{noise_level:<10.1f} {first_row[1]:<6} {first_row[2]:<6} {first_row[3]:<6} {first_row[4]:<6} {total_counts:<8}")
    
    print(f"\n{'='*60}")
    print("注意:")
    print("1. 每个噪声水平的数据都保持36行，与原数据格式相同")
    print("2. 每行总计数保持为1000")
    print("3. 噪声水平1.0表示完全去极化，所有测量结果都接近均匀分布")
    print("4. 数据变量名格式: data_depolarizing_1 表示噪声水平0.1")
    print(f"{'='*60}")