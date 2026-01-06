import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import warnings
import scipy.linalg as la
import time  # 添加时间模块
warnings.filterwarnings('ignore')

class POVMReconstructionNet(nn.Module):
    def __init__(self, povm_size=4, state_dim=4):
        super(POVMReconstructionNet, self).__init__()
        self.povm_size = povm_size
        self.state_dim = state_dim
        
        # 使用复数参数表示POVM元素
        # 每个POVM元素是一个4x4复数矩阵，用实部和虚部分开表示
        self.povm_real = nn.Parameter(torch.randn(povm_size, state_dim, state_dim))
        self.povm_imag = nn.Parameter(torch.randn(povm_size, state_dim, state_dim))
        
        # 初始化POVM为单位矩阵的等分
        with torch.no_grad():
            identity = torch.eye(state_dim) / povm_size
            for i in range(povm_size):
                self.povm_real[i] = identity.clone()
                self.povm_imag[i] = torch.zeros_like(identity)
    
    def get_povm_elements(self):
        """获取POVM元素（复数矩阵）"""
        povm_elements = []
        for i in range(self.povm_size):
            M_real = self.povm_real[i]
            M_imag = self.povm_imag[i]
            # 确保厄米性：M = (M_real + 1j*M_imag) 应该是厄米矩阵
            # 所以我们取 M = (M_real + M_real.T)/2 + 1j*(M_imag - M_imag.T)/2
            M_real_sym = (M_real + M_real.T) / 2
            M_imag_sym = (M_imag - M_imag.T) / 2
            M = M_real_sym + 1j * M_imag_sym
            povm_elements.append(M.detach().numpy())
        return povm_elements
    
    def enforce_povm_constraints(self):
        """强制POVM约束：半正定性和完备性"""
        with torch.no_grad():
            povm_elements = []
            
            # 获取当前POVM元素并确保半正定性
            for i in range(self.povm_size):
                M_real = self.povm_real[i]
                M_imag = self.povm_imag[i]
                M_real_sym = (M_real + M_real.T) / 2
                M_imag_sym = (M_imag - M_imag.T) / 2
                M = M_real_sym + 1j * M_imag_sym
                
                # 转换为numpy进行特征值分解
                M_np = M.numpy()
                eigvals, eigvecs = np.linalg.eigh(M_np)
                eigvals = np.maximum(eigvals, 0)  # 确保非负特征值
                M_positive = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
                
                povm_elements.append(M_positive)
            
            # 归一化以满足完备性关系
            completeness = sum(povm_elements)
            # 使用矩阵平方根进行归一化
            sqrt_completeness = np.linalg.inv(sqrtm(completeness))
            
            normalized_povm = []
            for M in povm_elements:
                M_normalized = sqrt_completeness @ M @ sqrt_completeness.conj().T
                normalized_povm.append(M_normalized)
            
            # 更新参数
            for i, M in enumerate(normalized_povm):
                self.povm_real[i] = torch.tensor(M.real, dtype=torch.float32)
                self.povm_imag[i] = torch.tensor(M.imag, dtype=torch.float32)

def sqrtm(matrix):
    """计算矩阵平方根"""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.conj().T

class QuantumMeasurementDataset(Dataset):
    def __init__(self, data):
        self.probe_states = []
        self.measurements = []
        
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
            
            # 归一化测量频率
            freqs = np.array(row[1:], dtype=float)
            probabilities = freqs / np.sum(freqs)
            
            self.probe_states.append(rho)
            self.measurements.append(probabilities)
    
    def __len__(self):
        return len(self.probe_states)
    
    def __getitem__(self, idx):
        rho = self.probe_states[idx]
        # 将复数密度矩阵转换为实数表示
        rho_real = np.stack([rho.real, rho.imag], axis=-1)
        return torch.tensor(rho_real, dtype=torch.float32), torch.tensor(self.measurements[idx], dtype=torch.float32)

def quantum_trace(rho_real, M_real, M_imag):
    """计算量子迹 Tr(rho * M)"""
    batch_size = rho_real.shape[0]
    
    # 提取实部和虚部
    rho_real_part = rho_real[:, :, :, 0]  # shape: (batch, 4, 4)
    rho_imag_part = rho_real[:, :, :, 1]  # shape: (batch, 4, 4)
    
    # 计算 Tr(rho * M) = Tr(rho_real * M_real - rho_imag * M_imag) + i*Tr(rho_real * M_imag + rho_imag * M_real)
    # 但我们只需要实数部分，因为测量概率是实数
    trace_real = torch.einsum('bij,bij->b', rho_real_part, M_real) - torch.einsum('bij,bij->b', rho_imag_part, M_imag)
    
    return trace_real

def compute_val_loss(model, val_loader):
    """计算验证集上的损失"""
    model.eval()
    val_loss = 0
    with torch.no_grad():  # 关键！不计算梯度
        for batch_rho, batch_probs in val_loader:
            # 计算预测概率
            pred_probs = []
            for i in range(4):
                M_real = model.povm_real[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                M_imag = model.povm_imag[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                trace = quantum_trace(batch_rho, M_real, M_imag)
                pred_probs.append(trace)
            
            pred_probs = torch.stack(pred_probs, dim=1)
            
            # 确保概率为正且和为1
            pred_probs = torch.softmax(pred_probs, dim=1)
            
            # 计算损失
            loss = nn.functional.mse_loss(pred_probs, batch_probs)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

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
    
    # 设置随机种子，确保结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 您的数据
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
    
    print("="*60)
    print("神经网络POVM重建开始")
    print("="*60)
    
    # 记录数据准备开始时间
    data_prep_start = time.time()
    
    # 创建完整数据集
    full_dataset = QuantumMeasurementDataset(data)
    
    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(full_dataset))  # 29个样本
    val_size = len(full_dataset) - train_size   # 7个样本
    
    print(f"数据集大小: {len(full_dataset)}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可重复
    )
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    data_prep_time = time.time() - data_prep_start
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    
    # 创建模型
    model = POVMReconstructionNet(povm_size=4, state_dim=4)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # 训练模型
    num_epochs = 100
    train_losses = []
    val_losses = []
    
    # 早停机制参数
    best_val_loss = float('inf')
    patience = 10  # 容忍多少个epoch验证损失不下降
    patience_counter = 0
    best_model_state = None
    
    print("\n开始训练...")
    print("Epoch | 训练损失 | 验证损失 | 状态")
    print("-" * 40)
    
    # 记录训练开始时间
    training_start_time = time.time()
    
    # 记录每个epoch的时间（用于计算平均epoch时间）
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        epoch_loss = 0
        
        # 训练阶段
        for batch_rho, batch_probs in train_loader:
            optimizer.zero_grad()
            
            # 计算预测概率
            pred_probs = []
            for i in range(4):
                M_real = model.povm_real[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                M_imag = model.povm_imag[i].unsqueeze(0).expand(batch_rho.shape[0], -1, -1)
                trace = quantum_trace(batch_rho, M_real, M_imag)
                pred_probs.append(trace)
            
            pred_probs = torch.stack(pred_probs, dim=1)
            
            # 确保概率为正且和为1
            pred_probs = torch.softmax(pred_probs, dim=1)
            
            # 计算损失
            loss = nn.functional.mse_loss(pred_probs, batch_probs)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 每100个epoch强制执行POVM约束
        if (epoch + 1) % 100 == 0:
            constraint_start = time.time()
            model.enforce_povm_constraints()
            constraint_time = time.time() - constraint_start
            if epoch == 99:  # 只打印第一次约束执行的时间
                print(f"POVM约束执行时间: {constraint_time:.4f}秒")
        
        # 计算验证损失（每10个epoch或最后几个epoch）
        if (epoch + 1) % 10 == 0 or epoch >= num_epochs - 20:
            avg_val_loss = compute_val_loss(model, val_loader)
            val_losses.append(avg_val_loss)
            
            # 记录最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # 每50个epoch打印一次最佳结果
                if (epoch + 1) % 10 == 0 :
                    print(f"{epoch+1:5d} | {avg_train_loss:.6f} | {avg_val_loss:.6f} | 新最佳 ✓")
            else:
                patience_counter += 1
                
                # 检查是否应该早停
                if patience_counter >= patience:
                    print(f"\n早停触发! 在 epoch {epoch+1} 停止训练")
                    print(f"最佳验证损失: {best_val_loss:.6f} (在 epoch {epoch+1-patience})")
                    
                    # 加载最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            # 每50个epoch打印一次进度
            if (epoch + 1) % 10 == 0 and patience_counter > 0:
                print(f"{epoch+1:5d} | {avg_train_loss:.6f} | {avg_val_loss:.6f} |")
        
        # 训练结束前强制计算一次验证损失
        if epoch == num_epochs - 1:
            avg_val_loss = compute_val_loss(model, val_loader)
            val_losses.append(avg_val_loss)
            print(f"{epoch+1:5d} | {avg_train_loss:.6f} | {avg_val_loss:.6f} | 最终结果")
        
        # 记录epoch时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
    
    # 计算训练总时间
    training_time = time.time() - training_start_time
    
    # 计算平均epoch时间
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
    
    print(f"\n训练总时间: {training_time:.2f}秒")
    print(f"实际训练轮数: {len(epoch_times)}")
    print(f"平均每个epoch时间: {avg_epoch_time:.4f}秒")
    
    # 分析训练过程
    print("\n" + "="*50)
    print("训练过程分析")
    print("="*50)
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 判断过拟合/欠拟合
    loss_gap = val_losses[-1] - train_losses[-1]
    if loss_gap > 0.05 and train_losses[-1] < 0.02:
        print("⚠️  警告：检测到可能过拟合（验证损失显著高于训练损失）")
    elif val_losses[-1] > 0.1:
        print("⚠️  警告：可能欠拟合（验证损失仍然很高）")
    else:
        print("✅  训练过程正常")
    
    # 记录评估开始时间
    evaluation_start_time = time.time()
    
    # 获取最终的POVM元素
    povm_elements = model.get_povm_elements()
    
    # 输出POVM元素
    print("\n" + "="*50)
    print("重建的POVM元素")
    print("="*50)
    for i, M in enumerate(povm_elements):
        print(f"\nM_{i}:")
        print(np.round(M, 4))
        print(f"迹: {np.trace(M).real:.6f}")
    
    # 验证POVM性质
    verify_povm_properties(povm_elements)
    
    # 在完整数据集上评估模型性能
    print("\n" + "="*50)
    print("模型性能评估")
    print("="*50)
    
    # 计算训练集和验证集的MSE
    train_mse = compute_val_loss(model, train_loader)
    val_mse = compute_val_loss(model, val_loader)
    
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"验证集 MSE: {val_mse:.6f}")
    
    # 计算测试集（即完整数据集）的MSE
    test_loader = DataLoader(full_dataset, batch_size=8, shuffle=False)
    test_mse = compute_val_loss(model, test_loader)
    print(f"完整数据集 MSE: {test_mse:.6f}")
    
    # 计算保真度
    fidelities, avg_fidelity = calculate_fidelities(povm_elements)
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"\n模型评估时间: {evaluation_time:.2f}秒")
    
    # 计算总运行时间
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("运行时间总结")
    print("="*60)
    print(f"数据准备时间: {data_prep_time:.2f}秒")
    print(f"训练总时间: {training_time:.2f}秒")
    print(f"模型评估时间: {evaluation_time:.2f}秒")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"总运行时间: {total_time/60:.2f}分钟")
    
    # 打印完成信息
    print("\n" + "="*60)
    print("神经网络POVM重建完成")
    print("="*60)

def verify_povm_properties(povm_elements):
    """验证POVM的性质"""
    print("\n" + "="*50)
    print("POVM性质验证")
    print("="*50)
    
    # 1. 检查完备性关系
    completeness = sum(povm_elements)
    print(f"\n完备性关系检查 (∑M_i = I):")
    print(f"∑M_i = \n{np.round(completeness, 4)}")
    print(f"与单位矩阵的差异范数: {np.linalg.norm(completeness - np.eye(4)):.6f}")
    
    # 2. 检查半正定性
    print("\n半正定性检查:")
    all_positive = True
    for i, M in enumerate(povm_elements):
        eigenvalues = np.linalg.eigvalsh(M)
        min_eigenvalue = np.min(eigenvalues)
        print(f"  M_{i} 的最小特征值: {min_eigenvalue:.6f}", end="")
        if min_eigenvalue < -1e-6:  # 允许小的数值误差
            print("  ❌ 不是半正定!")
            all_positive = False
        else:
            print("  ✓")
    
    # 3. 检查厄米性
    print("\n厄米性检查:")
    all_hermitian = True
    for i, M in enumerate(povm_elements):
        hermitian_diff = np.linalg.norm(M - M.conj().T)
        print(f"  M_{i} 的厄米性差异: {hermitian_diff:.6f}", end="")
        if hermitian_diff > 1e-6:  # 允许小的数值误差
            print("  ❌ 不是厄米矩阵!")
            all_hermitian = False
        else:
            print("  ✓")
    
    # 总结
    print("\n" + "-"*30)
    print("验证总结:")
    if all_positive and all_hermitian and np.linalg.norm(completeness - np.eye(4)) < 0.01:
        print("✅  POVM满足所有基本性质")
    else:
        print("⚠️  POVM可能不满足某些性质")

if __name__ == "__main__":
    main()