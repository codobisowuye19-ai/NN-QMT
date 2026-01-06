import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time  # 添加时间模块

class MLE_POVM_Reconstructor:
    def __init__(self, povm_size=4, state_dim=4):
        """
        初始化MLE POVM重建器
        
        参数:
        povm_size: POVM元素数量
        state_dim: 状态维度（对于两比特系统为4）
        """
        self.povm_size = povm_size
        self.state_dim = state_dim
        
        # 初始化POVM元素为单位矩阵的等分
        self.povm_elements = [np.eye(state_dim, dtype=complex) / povm_size for _ in range(povm_size)]
    
    def get_cholesky_params(self):
        """
        使用Cholesky分解参数化POVM元素
        返回可优化的实数参数向量
        """
        params = []
        for i in range(self.povm_size - 1):
            # 为每个POVM元素创建下三角矩阵（复数）
            # 对于n x n矩阵，需要n*(n+1)个实数参数
            L_real = np.random.randn(self.state_dim, self.state_dim)
            L_imag = np.random.randn(self.state_dim, self.state_dim)
            
            # 确保下三角矩阵（对角线以上为0）
            L_real = np.tril(L_real)
            L_imag = np.tril(L_imag)
            
            params.extend(L_real.flatten())
            params.extend(L_imag.flatten())
        
        return np.array(params)
    
    def params_to_povm(self, params):
        """
        将参数向量转换为POVM元素
        
        参数:
        params: 实数参数向量
        
        返回:
        POVM元素列表
        """
        # 每个POVM元素需要2 * state_dim * state_dim个参数
        elem_size = 2 * self.state_dim * self.state_dim
        
        povm_elements = []
        param_idx = 0
        
        for i in range(self.povm_size - 1):
            # 提取实部和虚部
            L_real_flat = params[param_idx:param_idx + self.state_dim**2]
            param_idx += self.state_dim**2
            
            L_imag_flat = params[param_idx:param_idx + self.state_dim**2]
            param_idx += self.state_dim**2
            
            # 重塑为矩阵并确保下三角
            L_real = np.tril(L_real_flat.reshape(self.state_dim, self.state_dim))
            L_imag = np.tril(L_imag_flat.reshape(self.state_dim, self.state_dim))
            
            # 构建下三角矩阵L
            L = L_real + 1j * L_imag
            
            # 构建半正定矩阵：M = L @ L^†
            M = L @ L.conj().T
            
            # 确保迹为实数
            M = (M + M.conj().T) / 2
            
            povm_elements.append(M)
        
        # 最后一个POVM元素通过完备性关系计算
        completeness_sum = sum(povm_elements)
        M_last = np.eye(self.state_dim) - completeness_sum
        
        # 确保最后一个元素是半正定的
        eigvals, eigvecs = la.eigh(M_last)
        eigvals = np.maximum(eigvals, 0)
        M_last = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        povm_elements.append(M_last)
        
        return povm_elements
    
    def compute_probabilities(self, rho, povm_elements):
        """
        计算给定密度矩阵和POVM元素的概率
        
        参数:
        rho: 密度矩阵
        povm_elements: POVM元素列表
        
        返回:
        概率向量
        """
        probs = []
        for M in povm_elements:
            prob = np.real(np.trace(rho @ M))
            probs.append(max(prob, 0))  # 确保概率非负
        
        # 归一化（数值稳定性）
        probs = np.array(probs)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum
        
        return probs
    
    def negative_log_likelihood(self, params, probe_states, measurements):
        """
        计算负对数似然函数
        
        参数:
        params: 参数向量
        probe_states: 探针态列表
        measurements: 测量计数列表
        
        返回:
        负对数似然值
        """
        # 将参数转换为POVM元素
        povm_elements = self.params_to_povm(params)
        
        nll = 0.0  # 负对数似然
        
        for rho, counts in zip(probe_states, measurements):
            # 计算理论概率
            probs = self.compute_probabilities(rho, povm_elements)
            
            # 添加小值避免log(0)
            epsilon = 1e-10
            probs = np.maximum(probs, epsilon)
            
            # 计算对数似然
            total_counts = np.sum(counts)
            
            # 多项分布的对数似然
            log_likelihood = np.sum(counts * np.log(probs))
            
            # 添加负号（因为我们要最小化）
            nll -= log_likelihood / total_counts  # 归一化
        
        # 添加正则化项以确保数值稳定性
        reg_term = 0.001 * np.sum(params**2)
        nll += reg_term
        
        return nll
    
    def prepare_data(self, data):
        """
        准备数据
        
        参数:
        data: 原始数据列表
        
        返回:
        probe_states: 探针态列表
        measurements: 测量计数列表
        """
        probe_states = []
        measurements = []
        
        # 定义单比特态
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
            # 解析探针态
            state1, state2 = row[0].split(',')
            psi1 = state_dict[state1]
            psi2 = state_dict[state2]
            
            # 构建两比特态
            psi = np.kron(psi1, psi2)
            rho = psi @ psi.conj().T  # 密度矩阵
            
            # 获取测量计数
            counts = np.array(row[1:], dtype=float)
            
            probe_states.append(rho)
            measurements.append(counts)
        
        return probe_states, measurements
    
    def fit(self, data, max_iter=1000, method='L-BFGS-B'):
        """
        使用MLE拟合POVM
        
        参数:
        data: 原始数据
        max_iter: 最大迭代次数
        method: 优化方法
        
        返回:
        重建的POVM元素
        """
        # 准备数据
        probe_states, measurements = self.prepare_data(data)
        
        # 初始化参数
        initial_params = self.get_cholesky_params()
        
        print("开始最大似然估计优化...")
        start_time = time.time()  # 记录优化开始时间
        
        # 定义目标函数
        def objective(params):
            return self.negative_log_likelihood(params, probe_states, measurements)
        
        # 优化参数
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
        
        optimization_time = time.time() - start_time  # 计算优化时间
        print(f"优化完成: {result.message}")
        print(f"最终负对数似然值: {result.fun:.6f}")
        print(f"优化运行时间: {optimization_time:.2f}秒")
        
        # 获取最优POVM元素
        optimal_povm = self.params_to_povm(result.x)
        
        # 验证并归一化POVM元素
        optimal_povm = self.normalize_povm(optimal_povm)
        
        self.povm_elements = optimal_povm
        
        return optimal_povm
    
    def normalize_povm(self, povm_elements):
        """
        归一化POVM元素以确保完备性关系
        
        参数:
        povm_elements: POVM元素列表
        
        返回:
        归一化的POVM元素列表
        """
        # 计算当前的和
        completeness_sum = sum(povm_elements)
        
        # 计算平方根逆
        eigvals, eigvecs = la.eigh(completeness_sum)
        eigvals = np.maximum(eigvals, 1e-10)  # 避免除零
        sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
        
        # 归一化每个POVM元素
        normalized_povm = []
        for M in povm_elements:
            M_norm = sqrt_inv @ M @ sqrt_inv.conj().T
            normalized_povm.append(M_norm)
        
        return normalized_povm
    
    def verify_povm_properties(self, povm_elements=None):
        """
        验证POVM的性质
        
        参数:
        povm_elements: 要验证的POVM元素列表，如果为None则使用self.povm_elements
        """
        if povm_elements is None:
            povm_elements = self.povm_elements
        
        print("验证POVM性质:")
        
        # 1. 检查完备性关系
        completeness = sum(povm_elements)
        identity = np.eye(self.state_dim, dtype=complex)
        
        print(f"完备性关系检查 (∑M_i = I):")
        print(f"∑M_i 的迹: {np.trace(completeness):.6f}")
        print(f"与单位矩阵的差异范数: {la.norm(completeness - identity):.6f}")
        
        # 2. 检查半正定性
        print("\n半正定性检查:")
        for i, M in enumerate(povm_elements):
            eigenvalues = la.eigvalsh(M)
            min_eigenvalue = np.min(eigenvalues)
            print(f"M_{i} 的最小特征值: {min_eigenvalue:.6f}")
            print(f"M_{i} 的迹: {np.trace(M.real):.6f}")
        
        # 3. 检查厄米性
        print("\n厄米性检查:")
        for i, M in enumerate(povm_elements):
            hermitian_diff = la.norm(M - M.conj().T)
            print(f"M_{i} 的厄米性差异: {hermitian_diff:.6f}")
    
    def evaluate_reconstruction(self, data):
        """
        评估重建精度
        
        参数:
        data: 原始数据
        
        返回:
        平均MSE
        """
        probe_states, measurements = self.prepare_data(data)
        
        total_mse = 0
        total_samples = len(probe_states)
        
        for rho, counts in zip(probe_states, measurements):
            # 计算理论概率
            probs_theory = self.compute_probabilities(rho, self.povm_elements)
            
            # 计算经验概率
            probs_empirical = counts / np.sum(counts)
            
            # 计算MSE
            mse = np.mean((probs_theory - probs_empirical) ** 2)
            total_mse += mse
        
        avg_mse = total_mse / total_samples
        print(f"平均重建MSE: {avg_mse:.6f}")
        
        return avg_mse
    
    def print_povm_elements(self, povm_elements=None):
        """
        打印POVM元素
        
        参数:
        povm_elements: 要打印的POVM元素列表，如果为None则使用self.povm_elements
        """
        if povm_elements is None:
            povm_elements = self.povm_elements
        
        print("\n重建的POVM元素:")
        for i, M in enumerate(povm_elements):
            print(f"\nM_{i}:")
            # 打印实部和虚部
            print("实部:")
            print(np.round(M.real, 10))
            print("虚部:")
            print(np.round(M.imag, 10))
            print(f"迹: {np.trace(M.real):.6f}")


def fidelity_matrix(A, B):
    """
    计算两个半正定矩阵A和B之间的保真度
    F(A, B) = [Tr(sqrt(sqrt(A) * B * sqrt(A)))]^2
    """
    # 确保矩阵是厄米的
    A = (A + A.conj().T) / 2
    B = (B + B.conj().T) / 2
    
    # 确保半正定性（将负特征值设为零）
    eigvals_A, eigvecs_A = la.eigh(A)
    eigvals_A = np.maximum(eigvals_A, 0)
    A_pos = eigvecs_A @ np.diag(eigvals_A) @ eigvecs_A.conj().T
    
    eigvals_B, eigvecs_B = la.eigh(B)
    eigvals_B = np.maximum(eigvals_B, 0)
    B_pos = eigvecs_B @ np.diag(eigvals_B) @ eigvecs_B.conj().T
    
    # 计算矩阵平方根
    sqrt_A = la.sqrtm(A_pos)
    
    # 计算 sqrt_A * B * sqrt_A
    M = sqrt_A @ B_pos @ sqrt_A
    
    # 计算M的平方根
    sqrt_M = la.sqrtm(M)
    
    # 计算迹并平方
    trace = np.trace(sqrt_M)
    fidelity = np.abs(trace)**2  # 取绝对值确保实数
    
    return np.real(fidelity)  # 返回实部


def calculate_fidelities(povm_elements):
    """
    计算重建POVM与理论POVM之间的保真度
    
    参数:
    povm_elements: 重建的POVM元素列表
    
    返回:
    fidelities: 四个保真度的列表
    avg_fidelity: 平均保真度
    """
    # 理论POVM矩阵
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
    print("保真度计算")
    print("="*50)
    
    fidelities = []
    
    for i in range(4):
        F = fidelity_matrix(povm_elements[i], theoretical_povm[i])
        fidelities.append(F)
        print(f"F(M_{i}_rec, M_{i}_th) = {F:.6f}")
    
    avg_fidelity = np.mean(fidelities)
    print("-" * 50)
    print(f"平均保真度 = {avg_fidelity:.6f}")
    
    # 输出总结
    print("\n总结:")
    print(f"最小保真度: {min(fidelities):.6f} (M{fidelities.index(min(fidelities))})")
    print(f"最大保真度: {max(fidelities):.6f} (M{fidelities.index(max(fidelities))})")
    
    return fidelities, avg_fidelity


def main():
    # 记录总开始时间
    total_start_time = time.time()
    
    # 数据
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
    
    # 创建完整数据集
    
    print("="*60)
    print("MLE POVM重建开始")
    print("="*60)
    
    # 创建MLE重建器
    reconstructor = MLE_POVM_Reconstructor(povm_size=4, state_dim=4)
    
    # 拟合POVM（包含内部时间测量）
    povm_elements = reconstructor.fit(data, max_iter=2000)
    
    # 打印POVM元素
    reconstructor.print_povm_elements()
    
    # 验证POVM性质
    reconstructor.verify_povm_properties()
    
    # 评估重建精度
    reconstructor.evaluate_reconstruction(data)
    
    # 计算保真度
    calculate_fidelities(povm_elements)
    
    # 计算总运行时间
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("运行时间总结")
    print("="*60)
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"总运行时间: {total_time/60:.2f}分钟")
    
    # 打印完成信息
    print("\n" + "="*60)
    print("MLE POVM重建完成")
    print("="*60)


if __name__ == "__main__":
    main()