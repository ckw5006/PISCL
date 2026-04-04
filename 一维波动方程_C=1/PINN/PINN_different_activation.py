import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ==============================================================================
# 【核心配置区】完全与参考代码风格一致
# ==============================================================================
# ---------------------- 1. 时空维度参数 ----------------------
X_MIN, X_MAX = 0.0, 2 * np.pi
T_MIN, T_MAX = 0.0, 10.0

# ---------------------- 2. PDE核心参数 ----------------------
C = 1.0

# ---------------------- 3. 训练参数 ----------------------
TRAIN_EPOCHS = 2000
TRAIN_LR = 1e-1
TRAIN_N_COLLOC = 200
TRAIN_N_IC = 200
TRAIN_N_BC = 200
TRAIN_PRINT_EVERY = 1000

# ---------------------- 5. 可视化参数（完全统一） ----------------------
FONT_FAMILY = 'Times New Roman'
FONT_SIZE_BASE = 20
FONT_SIZE_TITLE = 35
FONT_SIZE_LABEL = 40
FONT_SIZE_TICK = 20
FONT_SIZE_LEGEND = 20

LEGEND_ALPHA = 0.5
LEGEND_LOC = 'upper right'
LEGEND_NCOL = 2
LEGEND_LW = 2

VIS_NX = 200
VIS_NT = 200

FIG_DPI = 400
FIG_SIZE_3D = (14, 6)
FIG_SIZE_SLICE = (10, 6)
FIG_SIZE_ERROR = (6, 4)

# ==============================================================================

# ========== 1. 已知初始波形 ==========
def u_IC(x):
    return torch.sin(x) if isinstance(x, torch.Tensor) else np.sin(x)

# ========== 2. PINN 模型 ==========
class PINN_MLP(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(2, 5)
        self.activation = activation
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        out = self.fc1(xt)
        out = self.activation(out)
        out = self.fc2(out)
        return out

# ========== 3. 统计可训练参数（统一格式） ==========
def count_trainable_params(model, model_name="PINN"):
    total_params = 0
    print("=" * 50)
    print(f"{model_name} 可训练参数明细：")
    print("=" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_params += param_num
            shape_str = str(list(param.shape))
            print(f"{name:<20} | 形状：{shape_str:<10} | 数量：{param_num}")
    print("=" * 50)
    print(f"{model_name} 可训练参数总数：{total_params}")
    print("=" * 50)
    return total_params

# ========== 4. PDE 残差 ==========
def pde_residual(net, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = net(x, t)
    ones = torch.ones_like(u)
    u_t = autograd.grad(u, t, ones, create_graph=True)[0]
    u_x = autograd.grad(u, x, ones, create_graph=True)[0]
    u_tt = autograd.grad(u_t, t, ones, create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, ones, create_graph=True)[0]
    return u_tt - C ** 2 * u_xx

# ========== 5. 初值 & 边界损失 ==========
def loss_ic(net, n_ic=TRAIN_N_IC):
    x = torch.rand(n_ic, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t0 = torch.zeros_like(x)
    x.requires_grad_(True)
    t0.requires_grad_(True)
    u = net(x, t0)
    u_t = autograd.grad(u, t0, torch.ones_like(u), create_graph=True)[0]
    return ((u - u_IC(x)) ** 2).mean() + (u_t ** 2).mean()

def loss_bc_periodic(net, n_bc=TRAIN_N_BC):
    t = torch.rand(n_bc, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    xl = torch.zeros_like(t)
    xr = torch.zeros_like(t) + X_MAX
    ul = net(xl, t)
    ur = net(xr, t)

    xl.requires_grad_(True)
    ulx = net(xl, t)
    ul_x = autograd.grad(ulx, xl, torch.ones_like(ulx), create_graph=True)[0]

    xr.requires_grad_(True)
    urx = net(xr, t)
    ur_x = autograd.grad(urx, xr, torch.ones_like(urx), create_graph=True)[0]

    return ((ul - ur) ** 2).mean() + ((ul_x - ur_x) ** 2).mean()

# ========== 6. ✅ 训练函数已修改：自动保存损失最小的模型 ==========
def train(net, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, n_colloc=TRAIN_N_COLLOC, print_every=TRAIN_PRINT_EVERY):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print(f"开始训练（总轮数：{epochs}）...")
    start_time = time.time()

    # ====================== 关键修改 ======================
    best_loss = float('inf')
    best_state = None
    # ======================================================

    for ep in range(1, epochs + 1):
        x_col = torch.rand(n_colloc, 1, device=device) * (X_MAX - X_MIN) + X_MIN
        t_col = torch.rand(n_colloc, 1, device=device) * (T_MAX - T_MIN) + T_MIN

        R = pde_residual(net, x_col, t_col)
        l_pde = (R ** 2).mean()
        l_ic = loss_ic(net)
        l_bc = loss_bc_periodic(net)
        loss = l_pde + l_ic + l_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ====================== 关键修改 ======================
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = net.state_dict()
        # ======================================================

        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:5d} | Total={loss.item():.2e} | PDE={l_pde.item():.2e} | Best Loss={best_loss:.2e}")

    # 加载最优模型
    net.load_state_dict(best_state)
    print(f"\n✅ 已自动加载 损失最小 的模型！Best Loss = {best_loss:.4e}")

    end_time = time.time()
    total_time = end_time - start_time
    total_time_min = total_time / 60
    print("=" * 50)
    print(f"训练完成！总训练时长：")
    print(f"  - 秒：{total_time:.2f} s")
    print(f"  - 分钟：{total_time_min:.2f} min")
    print("=" * 50)

# ========== 7. 解析解 ==========
def u_true(x, t):
    return np.sin(x) * np.cos(C * t)

# ========== 8. 误差指标计算（统一） ==========
def calculate_error_metrics(u_pred, u_ref):
    pred_flat = u_pred.flatten()
    ref_flat = u_ref.flatten()
    abs_error = np.abs(pred_flat - ref_flat)
    mse = np.mean((pred_flat - ref_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    max_ae = np.max(abs_error)
    l2_norm_abs = np.linalg.norm(pred_flat - ref_flat, ord=2)
    l2_norm_ref = np.linalg.norm(ref_flat, ord=2)
    l2_norm_rel = l2_norm_abs / l2_norm_ref if l2_norm_ref != 0 else np.inf
    abs_rel_error = np.abs((pred_flat - ref_flat) / (ref_flat + 1e-10))
    rmsre = np.sqrt(np.mean(abs_rel_error ** 2))

    metrics = {
        "MSE (均方误差)": mse, "RMSE (根均方误差)": rmse, "MAE (平均绝对误差)": mae,
        "Max AE (最大绝对误差)": max_ae, "L2 Norm (绝对L2范数)": l2_norm_abs,
        "Relative L2 Norm (相对L2范数)": l2_norm_rel, "RMSRE (均方根相对误差)": rmsre
    }

    print("=" * 60)
    print("全局误差指标（基于所有采样点）：")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:<30} | {value:.6e}")
    print("=" * 60)
    return metrics

# ========== 9. 3D 对比图（完全统一） ==========
def plot_3d_comparison(X, T, u_pred, u_ref):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE

    fig = plt.figure(figsize=FIG_SIZE_3D)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, T, u_ref, cmap='viridis')
    ax1.set_title(f"Analytic Solution (C={C})", fontsize=FONT_SIZE_TITLE)
    ax1.set_xlabel("x", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("t", fontsize=FONT_SIZE_LABEL)
    ax1.set_zlabel("u", fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, T, u_pred, cmap='plasma')
    ax2.set_title("PINN Prediction", fontsize=FONT_SIZE_TITLE)
    ax2.set_xlabel("x", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("t", fontsize=FONT_SIZE_LABEL)
    ax2.set_zlabel("u", fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    plt.savefig("pinn_3d_comparison.png", dpi=FIG_DPI)
    # plt.show()

# ========== 10. 统一可视化函数（和参考代码完全一致） ==========
def visualize(net, activation_name):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE

    net.eval()
    x = np.linspace(X_MIN, X_MAX, VIS_NX)
    t = np.linspace(T_MIN, T_MAX, VIS_NT)
    X, T = np.meshgrid(x, t)

    xx = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = net(xx, tt).cpu().numpy().reshape(VIS_NT, VIS_NX)
    u_ref = u_true(X, T)

    calculate_error_metrics(u_pred, u_ref)
    plot_3d_comparison(X, T, u_pred, u_ref)

    # 时间切片对比图
    plt.figure(figsize=FIG_SIZE_SLICE)
    time_slices = [0.0, 0.99, 0.49, 1.50]
    colors = ['r', 'b', 'g', 'm']

    for t_val, color in zip(time_slices, colors):
        tidx = int((t_val - T_MIN) / (T_MAX - T_MIN) * (VIS_NT - 1))
        tidx = np.clip(tidx, 0, VIS_NT - 1)
        plt.plot(x, u_ref[tidx], color=color, lw=2, label=f"Ref t={t_val:.1f}", alpha=0.5)
        plt.plot(x, u_pred[tidx], color=color, ls='--', lw=1.6, label=f"PINN t={t_val:.1f}")

    plt.xlabel("x", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("u", fontsize=FONT_SIZE_LABEL)
    plt.title(f"PINN vs Analytic Solution (C={C})", fontsize=FONT_SIZE_TITLE)
    plt.legend(ncol=LEGEND_NCOL, loc=LEGEND_LOC, fontsize=FONT_SIZE_LEGEND, framealpha=LEGEND_ALPHA, fancybox=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"pinn_{activation_name}_slices.png", dpi=FIG_DPI)
    # plt.show()

    # 误差热力图
    err = np.abs(u_pred - u_ref)
    plt.figure(figsize=FIG_SIZE_ERROR)
    plt.pcolormesh(X, T, err, shading='auto')
    plt.colorbar()
    plt.xlabel("x", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("t", fontsize=FONT_SIZE_LABEL)
    plt.title("Absolute Error", fontsize=FONT_SIZE_TITLE)
    plt.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    plt.savefig("pinn_abs_error.png", dpi=FIG_DPI)
    # plt.show()

# ========== 主程序 ==========
if __name__ == "__main__":
    activation_configs = {
        'Cosine': lambda x: torch.cos(x),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Swish': lambda x: x * torch.sigmoid(x),
        'Softplus': nn.Softplus()
    }

    for act_name, act_func in activation_configs.items():
        print("\n" + "=" * 60)
        print(f"训练 {act_name} 激活函数 PINN 模型...")
        print("=" * 60)
        model = PINN_MLP(activation=act_func).to(device)
        count_trainable_params(model, f"PINN-{act_name}")
        train(model)
        visualize(model, act_name)