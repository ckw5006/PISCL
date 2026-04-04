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
# 【核心配置区】所有可调整参数集中在此处
# ==============================================================================
# ---------------------- 1. 时空维度参数 ----------------------
X_MIN, X_MAX = 0.0, 2 * np.pi
T_MIN, T_MAX = 0.0, 10.0

# ---------------------- 2. PDE核心参数 ----------------------
C = 1.0

# ---------------------- 3. 模型参数 ----------------------
K_SPACE = 2
M_TIME = 2

# ---------------------- 4. 训练参数 ----------------------
TRAIN_EPOCHS = 2000
TRAIN_LR = 1e-4
TRAIN_N_COLLOC = 800
TRAIN_N_IC = 200
TRAIN_N_BC = 200
TRAIN_PRINT_EVERY = 100

# ---------------------- 5. 可视化参数 ----------------------
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

# ---------------------- 6. 模型保存/加载路径 ----------------------
MODEL_SAVE_PATH = "best_model.pth"


# ==============================================================================

# ========== 1. 已知初始波形 ==========
def u_IC(x):
    return torch.sin(x) if isinstance(x, torch.Tensor) else np.sin(x)


# ========== 2. LF-FPINN 模型（支持维度自适应加载） ==========
class PDE_LF_FPINN(nn.Module):
    def __init__(self, K_space=K_SPACE, M_time=M_TIME):
        super().__init__()
        self.K_space = K_space
        self.n_basis = 2 * K_space
        self.M_time = M_time

        m = torch.arange(1, M_time + 1, dtype=torch.float32)
        self.freq_t = nn.Parameter(m.clone())
        self.time_linear = nn.Linear(2 * M_time, self.n_basis)

    def spatial_basis(self, x):
        x = x.view(-1, 1)
        k = torch.arange(1, self.K_space + 1, device=x.device).view(1, -1)
        sins = torch.sin(k * x)
        coss = torch.cos(k * x)
        return torch.cat([sins, coss], dim=1)


    def time_coeff(self, t):
        t = t.view(-1, 1)
        ft = self.freq_t.view(1, -1)
        phase = t @ ft
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        g = self.time_linear(feats)
        return t * g

    def forward(self, x, t):
        phi = self.spatial_basis(x)
        a = self.time_coeff(t)
        delta = (phi * a).sum(dim=1, keepdim=True)
        return u_IC(x) + delta


# ========== 统计可训练参数 ==========
def count_trainable_params(model):
    total_params = 0
    print("=" * 50)
    print("LF-FPINN 可训练参数明细：")
    print("=" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_num = param.numel()
            total_params += param_num
            shape_str = str(list(param.shape))
            print(f"{name:<20} | 形状：{shape_str:<10} | 数量：{param_num}")
    print("=" * 50)
    print(f"可训练参数总数：{total_params}")
    print("=" * 50)
    return total_params


# ========== ✅ 修复：误差指标计算 ==========
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
    abs_rel_error = np.abs((pred_flat - ref_flat) / (ref_flat + 1e-10))  # 修复笔误
    rmsre = np.sqrt(np.mean(abs_rel_error ** 2))

    metrics = {
        "MSE (均方误差)": mse,
        "RMSE (根均方误差)": rmse,
        "MAE (平均绝对误差)": mae,
        "Max AE (最大绝对误差)": max_ae,
        "L2 Norm (绝对L2范数)": l2_norm_abs,
        "Relative L2 Norm (相对L2范数)": l2_norm_rel,
        "RMSRE (均方根相对误差)": rmsre
    }

    print("=" * 60)
    print("全局误差指标（基于所有采样点）：")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:<30} | {value:.6e}")
    print("=" * 60)
    return metrics


# ========== PDE 残差 ==========
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


# ========== 初值 & 边界损失 ==========
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


# ====================== 【保存最优模型】 ======================
def save_best_model(model, save_path=MODEL_SAVE_PATH):
    save_dict = {
        "K_space": model.K_space,
        "M_time": model.M_time,
        "state_dict": model.state_dict()
    }
    torch.save(save_dict, save_path)
    print(f"\n✅ 最优模型已保存：{save_path}")
    print(f"   维度 K={model.K_space}, M={model.M_time}")


# ====================== 【自动加载模型】 ======================
def load_model_auto(model, load_path=MODEL_SAVE_PATH):
    if not os.path.exists(load_path):
        print(f"⚠️  未找到模型文件，从头训练")
        return model

    checkpoint = torch.load(load_path, map_location=device)
    K_old = checkpoint["K_space"]
    M_old = checkpoint["M_time"]
    K_new = model.K_space
    M_new = model.M_time

    print(f"\n🔍 加载模型：旧维度 K={K_old}, M={M_old} | 当前模型 K={K_new}, M={M_new}")

    old_state = checkpoint["state_dict"]
    new_state = model.state_dict()

    for key in old_state:
        if key not in new_state:
            continue
        p_old = old_state[key]
        p_new = new_state[key]

        if p_old.shape == p_new.shape:
            new_state[key] = p_old
        else:
            try:
                if key == "freq_t":
                    new_state[key][:M_old] = p_old
                elif key == "time_linear.weight":
                    new_state[key][:2 * K_old, :2 * M_old] = p_old[:2 * K_old, :2 * M_old]
                elif key == "time_linear.bias":
                    new_state[key][:2 * K_old] = p_old[:2 * K_old]
            except:
                print(f"⚠️  参数 {key} 形状不匹配，跳过加载")
                pass

    model.load_state_dict(new_state)
    print(f"✅ 模型自动加载完成（小维度 → 大维度 自适应）")
    return model


# ========== 训练函数（保存最优） ==========
def train(net, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, n_colloc=TRAIN_N_COLLOC, print_every=TRAIN_PRINT_EVERY):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print(f"开始训练（总轮数：{epochs}）...")
    start_time = time.time()

    best_loss = float('inf')
    best_state = None

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

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = net.state_dict()

        if ep % print_every == 0 or ep == 1:
            print(f"Epoch {ep:5d} | Total={loss.item():.2e} | PDE={l_pde.item():.2e} | Best Loss={best_loss:.2e}")

    net.load_state_dict(best_state)
    save_best_model(net)
    print(f"\n✅ 已加载最优模型，Best Loss = {best_loss:.4e}")

    end_time = time.time()
    total_time = end_time - start_time
    print("=" * 50)
    print(f"训练完成！总时长：{total_time:.2f} s")
    print("=" * 50)


# ========== 解析解 ==========
def u_true(x, t):
    return np.sin(x) * np.cos(C * t)


# ========== 3D 可视化 ==========
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
    ax2.set_title("PISCL Prediction", fontsize=FONT_SIZE_TITLE)
    ax2.set_xlabel("x", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("t", fontsize=FONT_SIZE_LABEL)
    ax2.set_zlabel("u", fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)

    plt.tight_layout()
    plt.savefig("3d_comparison.png", dpi=FIG_DPI)


# ========== 可视化 ==========
def visualize(net, nx=VIS_NX, nt=VIS_NT):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE

    net.eval()
    x = np.linspace(X_MIN, X_MAX, nx)
    t = np.linspace(T_MIN, T_MAX, nt)
    X, T = np.meshgrid(x, t)

    xx = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        u_pred = net(xx, tt).cpu().numpy().reshape(nt, nx)

    u_ref = u_true(X, T)
    error_metrics = calculate_error_metrics(u_pred, u_ref)

    plot_3d_comparison(X, T, u_pred, u_ref)

    plt.figure(figsize=FIG_SIZE_SLICE)
    slices = [0.0, 0.25, 0.5, 0.99]
    colors = ['r', 'b', 'g', 'm']
    for frac, color in zip(slices, colors):
        tidx = int(frac * (nt - 1))
        plt.plot(x, u_ref[tidx], color=color, lw=2, label=f"Ref t={t[tidx]:.2f}", alpha=0.5)
        plt.plot(x, u_pred[tidx], color=color, ls='--', lw=1.6, label=f"Pred t={t[tidx]:.2f}")
    plt.xlabel("x", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("u", fontsize=FONT_SIZE_LABEL)
    plt.title(f"PISCL vs Analytic Solution (C={C})", fontsize=FONT_SIZE_TITLE)
    plt.legend(ncol=LEGEND_NCOL, loc=LEGEND_LOC, fontsize=FONT_SIZE_LEGEND, framealpha=LEGEND_ALPHA, fancybox=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("compare_slices_same_color.png", dpi=FIG_DPI)

    err = np.abs(u_pred - u_ref)
    plt.figure(figsize=FIG_SIZE_ERROR)
    plt.pcolormesh(X, T, err, shading='auto')
    plt.colorbar()
    plt.xlabel("x", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("t", fontsize=FONT_SIZE_LABEL)
    plt.title("Absolute Error", fontsize=FONT_SIZE_TITLE)
    plt.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    plt.savefig("abs_error.png", dpi=FIG_DPI)


# ========== 主程序（支持自动加载） ==========
if __name__ == "__main__":
    # 1. 创建模型
    model = PDE_LF_FPINN(K_space=K_SPACE, M_time=M_TIME).to(device)

    # 2. 自动加载已有模型（小维度 → 大维度 自适应）
    model = load_model_auto(model)

    # 3. 统计参数 & 训练
    total_params = count_trainable_params(model)
    train(model)

    # 4. 绘图
    visualize(model)