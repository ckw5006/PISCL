import math, torch, torch.nn as nn, torch.optim as optim
import os

import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import griddata  # 保留数值解插值（用于剖面图对比）

# ---------- 0. 设备与超参 ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NU = 0.01  # viscosity ν
NX, NT = 100, 100  # eval grid for plotting
N_F = 10000  # collocation (physics) points
N_IC_BC = 1000  # IC + BC points / epoch
EPOCHS = 5000  # 保持原测试用epoch数
LR = 1e-4  # 保持原学习率
PRINT_EVERY = 100  # 调整为1，方便观察少量epoch的训练过程

# 时间范围限制
T_MIN, T_MAX = 0.0, 1.0

# 激活函数列表（待测试）
ACTIVATION_FUNCTIONS = ['cosine']


# ========== 新增：NU转文件名标签函数 ==========
def nu_to_tag(nu):
    nu_str = f"{nu}".replace('.', 'p')
    if 'e' in nu_str:
        base, exp = nu_str.split('e')
        exp = int(exp)
        nu_str = f"{float(base) * 10 ** exp:.10f}".replace('.', 'p')
    nu_str = nu_str.rstrip('0').rstrip('p') if 'p' in nu_str else nu_str
    return nu_str


# ---------- 1. 初始条件 ----------
def u_IC(x):
    return -torch.sin(math.pi * x)


# ---------- 2. 通用PINN模型（参数总数≈5100） ----------
class PINN_Burgers(nn.Module):
    """
    通用PINN结构，参数总数≈5100，支持多种激活函数：
    结构：2(x,t) → 80 → 60 → 1(u)
    参数计算：
    - Linear(2,80): 2*80 +80 = 240
    - Linear(80,60): 80*60 +60 = 4860
    - Linear(60,1): 60*1 +1 = 61
    总计：240+4860+61=5161 ≈ 5100
    """

    def __init__(self, activation='tanh'):
        super().__init__()
        self.activation_name = activation

        # 定义网络层
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

        # 激活函数映射
        self.activation = self._get_activation(activation)

    def _get_activation(self, name):
        """返回对应激活函数"""
        if name == 'cosine':
            return lambda x: torch.cos(x)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'softplus':
            return nn.Softplus()
        elif name == 'swish':
            return lambda x: x * torch.sigmoid(x)  # Swish激活
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数：{name}")

    def forward(self, xt):
        """xt: (N,2) → u: (N,1)"""
        x = self.fc1(xt)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


# ========== 统计可训练参数数量（保留） ==========
def count_trainable_params(model):
    total_params = 0
    print(f"\n========== {model.activation_name} 可训练参数明细 ==========")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"参数名称: {name:20s} | 形状: {str(param.shape):15s} | 数量: {param_count:6d}")
    print("=" * 50)
    print(f"总可训练参数数量: {total_params:,}")
    print("=" * 50)
    return total_params


# ---------- 3. 采样函数（保持不变） ----------
def sample_collocation(n):
    x = torch.rand(n, 1, device=DEVICE) * 2.0 - 1.0  # [-1,1]
    t = torch.rand(n, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
    return torch.cat([x, t], dim=1).requires_grad_(True)


def sample_ic_bc(n):
    n_half = n // 2
    # 初始条件 (t=0)
    x_ic = torch.rand(n_half, 1, device=DEVICE) * 2.0 - 1.0
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(math.pi * x_ic)
    # 边界条件 (x=±1)
    t_bc = torch.rand(n_half, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
    x_bc = torch.where(torch.rand(n_half, 1, device=DEVICE) < 0.5,
                       -torch.ones_like(t_bc),
                       torch.ones_like(t_bc))
    u_bc = torch.zeros_like(x_bc)
    xt = torch.cat([torch.cat([x_ic, t_ic], dim=1),
                    torch.cat([x_bc, t_bc], dim=1)], dim=0)
    u = torch.cat([u_ic, u_bc], dim=0)
    return xt, u


# ---------- 4. CSV读取函数（保持不变） ----------
def read_vec_csv(path):
    try:
        arr = np.loadtxt(path, delimiter=',')
    except ValueError:
        arr = np.loadtxt(path, delimiter=',', skiprows=1)
    return np.atleast_1d(arr).squeeze()


def read_mat_csv(path):
    try:
        mat = np.loadtxt(path, delimiter=',')
    except ValueError:
        mat = np.loadtxt(path, delimiter=',', skiprows=1)
    return np.array(mat)


# ---------- 5. 训练函数 ----------
def train_model(model, activation_name):
    opt = optim.Adam(model.parameters(), lr=LR)
    ckpt_path = f'pinn_burgers_{activation_name}.pth'

    # 加载模型（如果存在）
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f'已加载 {ckpt_path}')
    else:
        print(f'开始训练 {activation_name} 激活函数的PINN模型...')

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()

        # PDE残差计算
        xt_f = sample_collocation(N_F)
        u = model(xt_f)
        grads = torch.autograd.grad(u, xt_f, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, xt_f, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        f_res = u_t + u * u_x - NU * u_xx
        loss_f = torch.mean(f_res ** 2)

        # 初值+边界损失
        xt_ib, u_true_ib = sample_ic_bc(N_IC_BC)
        u_pred_ib = model(xt_ib)
        loss_ib = torch.mean((u_pred_ib - u_true_ib) ** 2)

        loss = loss_f + loss_ib
        loss.backward()
        opt.step()

        if epoch % PRINT_EVERY == 0:
            print(
                f'[{activation_name}] Epoch {epoch:6d} | Loss {loss.item():.3e} | PDE {loss_f.item():.3e} | IC+BC {loss_ib.item():.3e}')

    # 保存模型
    torch.save(model.state_dict(), ckpt_path)
    print(f"{activation_name} 模型训练完成，保存为 {ckpt_path}")
    return model


# ---------- 6. 预测函数 ----------
def predict_model(model):
    x_eval = torch.linspace(-1.0, 1.0, NX, device=DEVICE)
    t_eval = torch.linspace(T_MIN, T_MAX, NT, device=DEVICE)
    X, T = torch.meshgrid(x_eval, t_eval, indexing='ij')
    XT_eval = torch.stack([X.flatten(), T.flatten()], dim=-1).to(DEVICE)

    with torch.no_grad():
        U_pred = model(XT_eval).view(NX, NT).cpu().numpy()

    return x_eval.cpu().numpy(), t_eval.cpu().numpy(), U_pred


# ---------- 7. 时间剖面图绘制（保持原逻辑） ----------
def plot_time_profiles(activation_name, x_pinn, t_pinn, U_pred, num_matched, x_num=None, t_num=None, U_num=None):
    step = 0.4
    t_plot = np.arange(T_MIN, T_MAX + 1e-12, step)
    t_plot = t_plot[(t_plot >= T_MIN) & (t_plot <= T_MAX)]
    print(f"\n[{activation_name}] 待绘制时间点: {t_plot}")

    if len(t_plot) == 0:
        print(f'[WARN] 训练区间 [{T_MIN}, {T_MAX}] 内无满足步长的时间点，跳过剖面图绘制。')
        return

    # 绘图逻辑与原代码完全一致
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(t_plot))]
    legend_items = []

    # 绘制数值解（如果有）
    if x_num is not None and t_num is not None and U_num is not None:
        for i, (tt, c) in enumerate(zip(t_plot, colors)):
            diffs = np.abs(t_num - tt)
            min_diff = diffs.min() if len(diffs) > 0 else 1e9
            if min_diff < 0.01:
                idx_num = np.argmin(diffs)
                num_matched[tt] = t_num[idx_num]
                num_line = plt.plot(
                    x_num, U_num[idx_num],
                    lw=2.0, color=c, linestyle='-', alpha=0.5,
                )[0]
                legend_items.append((num_line, f"Numeric t={tt:.2f}"))
                print(f"[{activation_name}] 时间点 {tt:.2f} 匹配数值解 {t_num[idx_num]:.2f}（误差 {min_diff:.4f}）")
            else:
                num_matched[tt] = None
                print(f"[{activation_name}] 时间点 {tt:.2f} 无匹配数值解（最小误差 {min_diff:.4f}）")

    # 绘制PINN预测解
    for i, (tt, c) in enumerate(zip(t_plot, colors)):
        x_grid = torch.tensor(x_pinn.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        t_grid = torch.full((x_grid.shape[0], 1), float(tt), device=DEVICE)
        xt_pair = torch.cat([x_grid, t_grid], dim=1)

        with torch.no_grad():
            u_pinn_tt = model(xt_pair).cpu().numpy().flatten()

        pinn_line = plt.plot(
            x_pinn, u_pinn_tt,
            lw=1.8, color=c, linestyle='--', alpha=1.0
        )[0]
        legend_items.append((pinn_line, f"PINN ({activation_name}) t={tt:.2f}"))

    # 图例与样式（原逻辑）
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x,t)', fontsize=12)
    plt.title(f'Burgers Equation Time Profiles (Activation: {activation_name}, T∈[{T_MIN}, {T_MAX}])', fontsize=14)
    plt.grid(True, alpha=0.3)

    leg = plt.legend(
        [item[0] for item in legend_items],
        [item[1] for item in legend_items],
        loc='upper right',
        ncol=2,
        fontsize=9,
        framealpha=0.2,
        fancybox=True,
        shadow=False
    )
    for text in leg.get_texts():
        text.set_color('black')
        text.set_weight('bold')

    plt.tight_layout()
    save_path = f'burgers_pinn_{activation_name}_profiles.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"[{activation_name}] 时间剖面图已保存 → {save_path}")
    return num_matched


# ---------- 主流程 ----------
if __name__ == "__main__":
    # 读取数值解（如果有）
    tag = nu_to_tag(NU)
    x_file = f'x_axis_{tag}.csv'
    t_file = f't_sel_{tag}.csv'
    U_file = f'U_profiles_{tag}.csv'
    has_num = all(os.path.exists(p) for p in [x_file, t_file, U_file])
    x_num, t_num, U_num = None, None, None

    if has_num:
        x_num = read_vec_csv(x_file)
        t_num = read_vec_csv(t_file)
        U_num = read_mat_csv(U_file)
        if U_num.shape == (x_num.size, t_num.size):
            U_num = U_num.T
        if U_num.shape != (t_num.size, x_num.size):
            raise RuntimeError(f'U_profiles_{tag}.csv 维度不匹配: {U_num.shape} vs ({t_num.size}, {x_num.size})')
        mask = t_num <= (T_MAX + 1e-12)
        t_num = t_num[mask]
        U_num = U_num[mask, :]
        print(f"\n数值解时间点: {t_num}")

    # 遍历所有激活函数测试
    for activation in ACTIVATION_FUNCTIONS:
        print(f"\n==================== 测试激活函数：{activation} ====================")
        # 初始化模型
        model = PINN_Burgers(activation=activation).to(DEVICE)
        # 统计参数
        count_trainable_params(model)
        # 训练模型
        model = train_model(model, activation)
        # 预测
        x_pinn, t_pinn, U_pred = predict_model(model)
        # 绘制剖面图
        num_matched = {}
        plot_time_profiles(activation, x_pinn, t_pinn, U_pred, num_matched, x_num, t_num, U_num)
        print(f"\n[{activation}] 数值解匹配结果: {num_matched}")