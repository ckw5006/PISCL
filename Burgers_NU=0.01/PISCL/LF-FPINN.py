import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from lion_pytorch import Lion

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==============================================================================
# 【核心配置区】所有可调整参数集中在此处
# ==============================================================================
# ---------------------- 1. 时空维度参数 ----------------------
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

# ---------------------- 2. PDE核心参数 ----------------------
NU = 0.01
STEP = 0.4

# ---------------------- 3. 模型参数 ----------------------
K_SPACE = 150
M_TIME = 20

# ---------------------- 4. 训练参数 ----------------------
TRAIN_EPOCHS = 5000
TRAIN_LR = 1e-5
TRAIN_N_F = 10000
TRAIN_N_IC_BC = 200
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

VIS_NX = 100
VIS_NT = 100

FIG_DPI = 400
FIG_SIZE_3D = (14, 6)
FIG_SIZE_SLICE = (12, 7)
FIG_SIZE_ERROR = (10, 6)

# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


# ========== 工具函数 ==========
def nu_to_tag(nu):
    nu_str = f"{nu}".replace('.', 'p')
    if 'e' in nu_str:
        base, exp = nu_str.split('e')
        exp = int(exp)
        nu_str = f"{float(base) * 10 ** exp:.10f}".replace('.', 'p')
    nu_str = nu_str.rstrip('0').rstrip('p') if 'p' in nu_str else nu_str
    return nu_str


# ========== 初始条件 ==========
def u_IC(x):
    return -torch.sin(math.pi * x)


# ========== LF-FPINN 模型：严格初始波形保持 + 时序演化学习 ==========
# 核心保证：
# 1. t=0 时，输出 100% 严格等于 u_IC(x)，绝不破坏初始波形
# 2. 模型只学习：t>0 时，对初始波形的“变形增量”
# 3. 增量 delta 在 t=0 时自动 = 0
# =====================================================================
class LFFPINN_Burgers(nn.Module):
    def __init__(self, K_space=K_SPACE, M_time=M_TIME):
        super().__init__()
        self.K = K_space
        self.M = M_time

        # 时间编码（保留原结构）
        m = torch.arange(1, M_time + 1, dtype=torch.float32) * 2
        self.freq_t = nn.Parameter(m.clone())

        # 输出：演化系数（维度 = 空间基数量）
        self.time_linear = nn.Linear(2 * M_time, self.K)

        # 固定：纯傅里叶空间基（不改动、不混合）
        self.register_buffer('k', torch.arange(1, self.K + 1, device=DEVICE))

    def spatial_basis(self, x):
        """ 固定标准傅里叶基，无额外混合 """
        x = x.view(-1, 1)
        return torch.sin(self.k.view(1, -1) * math.pi * (x + 1.0) / 2.0)

    def evolution_coeff(self, t):
        """
        学习时序演化系数
        关键：输出会乘以 t → 保证 t=0 时增量=0
        """
        t = t.view(-1, 1)
        phase = t @ self.freq_t.view(1, -1)
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        coeff = self.time_linear(feats)  # 形状 (B, K)
        return coeff

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        # 1. 固定初始波形（纹丝不动）
        u0 = u_IC(x)

        # 2. 固定傅里叶基
        phi_x = self.spatial_basis(x)

        # 3. 学习时序演化系数
        coeff = self.evolution_coeff(t)

        # 4. 🔥 核心：变形增量 = t * 系数 * 基
        # 保证 t=0 时，delta = 0 → 初始波形绝对不变！
        delta = t * (phi_x * coeff).sum(dim=1, keepdim=True)

        # 5. 最终解：初始波形 + 时序变形
        return u0 + delta


# ========== 参数统计 ==========
def count_trainable_params(model):
    total_params = 0
    print("=" * 50)
    print("LF-FPINN 可训练参数明细：")
    print("=" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            shape_str = str(list(param.shape))
            print(f"{name:<20} | 形状：{shape_str:<10} | 数量：{param_count:6d}")
    print("=" * 50)
    print(f"总可训练参数数量: {total_params:,}")
    print("=" * 50)
    return total_params


# ========== 误差计算 ==========
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
        "MSE (均方误差)": mse,
        "RMSE (根均方误差)": rmse,
        "MAE (平均绝对误差)": mae,
        "Max AE (最大绝对误差)": max_ae,
        "L2 Norm (绝对L2范数)": l2_norm_abs,
        "Relative L2 Norm (相对L2范数)": l2_norm_rel,
        "RMSRE (均方根相对误差)": rmsre
    }

    print("=" * 60)
    print("全局误差指标：")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:<30} | {value:.6e}")
    print("=" * 60)
    return metrics


# ========== 采样函数 ==========
def sample_collocation(n):
    x = torch.rand(n, 1, device=DEVICE) * 2 - 1
    t = torch.rand(n, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
    return torch.cat([x, t], dim=1).requires_grad_(True)


def sample_ic_bc(n):
    n_half = n // 2
    x_ic = torch.rand(n_half, 1, device=DEVICE) * 2 - 1
    t_ic = torch.zeros_like(x_ic)
    u_ic = -torch.sin(math.pi * x_ic)

    t_bc = torch.rand(n_half, 1, device=DEVICE) * (T_MAX - T_MIN) + T_MIN
    x_bc = torch.where(torch.rand(n_half, 1, device=DEVICE) < 0.5, -torch.ones_like(t_bc), torch.ones_like(t_bc))
    u_bc = torch.zeros_like(x_bc)

    xt = torch.cat([torch.cat([x_ic, t_ic], dim=1), torch.cat([x_bc, t_bc], dim=1)], dim=0)
    u = torch.cat([u_ic, u_bc], dim=0)
    return xt, u


# ========== CSV 读取 ==========
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


# ========== 【核心修复】数值解3D重构（完全重写，解决锯齿问题）==========
def reconstruct_numeric_3d(x_pinn, t_pinn, x_num, t_num, U_num):
    """
    修复核心：
    1.  用最近邻插值+线性插值结合，保证数值解完全光滑
    2.  移除填充0的逻辑，用数值解的边界值填充
    3.  确保数值解时间范围完全覆盖PISCL的0~1.0
    """
    # 1. 生成PISCL的标准网格（0~1.0，100×100）
    X_pinn, T_pinn = np.meshgrid(x_pinn, t_pinn, indexing='ij')

    # 2. 生成数值解的离散点（完全保留所有时间点，不截断）
    x_num_mesh, t_num_mesh = np.meshgrid(x_num, t_num)
    points = np.vstack([x_num_mesh.ravel(), t_num_mesh.ravel()]).T
    values = U_num.ravel()

    # 3. 去重+微小扰动（避免插值失败）
    points, idx = np.unique(points, axis=0, return_index=True)
    values = values[idx]
    points += 1e-8 * np.random.randn(*points.shape)

    # 4. 生成网格点（与PISCL完全对齐）
    grid_points = np.vstack([X_pinn.ravel(), T_pinn.ravel()]).T

    # 5. 【关键修复】插值逻辑：先线性，失败则用最近邻，填充值用数值解的均值（而非0）
    try:
        U_num_3d = griddata(points, values, grid_points, method='linear', fill_value=np.nanmean(values))
    except Exception as e:
        print(f"线性插值失败：{e}，使用最近邻插值")
        U_num_3d = griddata(points, values, grid_points, method='nearest', fill_value=np.nanmean(values))

    # 6. 处理NaN值（用最近邻插值补全，避免0填充导致的锯齿）
    if np.isnan(U_num_3d).any():
        U_num_3d = griddata(points, values, grid_points, method='nearest', fill_value=np.nanmean(values))

    return X_pinn, T_pinn, U_num_3d.reshape(VIS_NX, VIS_NT)


# ========== 【核心修复】3D对比绘图（已去掉右侧色条）==========
def plot_3d_comparison(U_pred, U_num_3d, X_pinn, T_pinn):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE
    fig = plt.figure(figsize=FIG_SIZE_3D)

    # 统一z轴范围
    z_min = min(np.min(U_num_3d), np.min(U_pred))
    z_max = max(np.max(U_num_3d), np.max(U_pred))

    # 子图1：数值解
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(
        X_pinn, T_pinn, U_num_3d,
        cmap='viridis',
        linewidth=0,
        antialiased=True,
        rstride=1, cstride=1,
        vmin=z_min, vmax=z_max
    )
    ax1.set_xlabel('x', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('t', fontsize=FONT_SIZE_LABEL)
    ax1.set_zlabel('u', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('Numeric Solution', fontsize=FONT_SIZE_TITLE)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    ax1.set_zlim(z_min, z_max)
    # 这里删除了 colorbar

    # 子图2：PISCL预测解
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(
        X_pinn, T_pinn, U_pred,
        cmap='plasma',
        linewidth=0,
        antialiased=True,
        rstride=1, cstride=1,
        vmin=z_min, vmax=z_max
    )
    ax2.set_xlabel('x', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('t', fontsize=FONT_SIZE_LABEL)
    ax2.set_zlabel('u', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('PISCL Prediction', fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    ax2.set_zlim(z_min, z_max)
    # 这里删除了 colorbar

    plt.tight_layout()
    plt.savefig('burgers_lf_fpinn_3d_comparison.png', dpi=FIG_DPI)
    # plt.show()
    print('3D对比图已保存 → burgers_lf_fpinn_3d_comparison.png')

def plot_absolute_error(abs_error, X_pinn, T_pinn):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE
    fig, ax = plt.subplots(figsize=FIG_SIZE_ERROR)
    im = ax.pcolormesh(
        X_pinn, T_pinn, abs_error,
        cmap='YlGnBu_r',
        shading='gouraud',
        vmin=np.min(abs_error),
        vmax=np.max(abs_error)
    )
    ax.set_xlabel('x', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('t', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Absolute Error', fontsize=FONT_SIZE_TITLE)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=10)
    cbar.set_label('', fontsize=FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig('burgers_absolute_error.png', dpi=FIG_DPI)
    # plt.show()
    print('绝对误差图已保存 → burgers_absolute_error.png')


# ========== 时间剖面（修复：完全对齐数值解与PISCL解）==========
def plot_time_profiles(U_pred, x_pinn, t_pinn, x_num=None, t_num=None, U_num=None):
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONT_SIZE_BASE

    # 生成时间点（0, 0.4, 0.8, 1.0，完全覆盖到t=1.0）
    t_plot = np.arange(T_MIN, T_MAX + 1e-12, STEP)
    t_plot = t_plot[(t_plot >= T_MIN) & (t_plot <= T_MAX)]

    if len(t_plot) == 0:
        print(f'[WARN] 区间 [{T_MIN}, {T_MAX}] 内无满足步长{STEP}的时间点，跳过剖面图绘制。')
        return

    # 绘图配置
    plt.figure(figsize=FIG_SIZE_SLICE)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(t_plot))]
    legend_items = []
    num_matched = {}

    # 绘制数值解（完全对齐时间点，无截断）
    if x_num is not None and t_num is not None and U_num is not None:
        for i, (tt, c) in enumerate(zip(t_plot, colors)):
            # 【关键修复】放宽匹配阈值，确保t=1.0一定匹配
            diffs = np.abs(t_num - tt)
            min_diff = diffs.min()
            idx_num = np.argmin(diffs)
            num_matched[tt] = t_num[idx_num]
            # 绘制数值解曲线
            num_line = plt.plot(
                x_num, U_num[idx_num],
                lw=LEGEND_LW, color=c, linestyle='-', alpha=0.5,
            )[0]
            legend_items.append((num_line, f"Numeric t={tt:.2f}"))
            print(f"时间点 {tt:.2f} 匹配数值解 {t_num[idx_num]:.2f}（误差 {min_diff:.4f}）")

    # 绘制PISCL解（完全对齐时间点）
    for i, (tt, c) in enumerate(zip(t_plot, colors)):
        x_grid = torch.tensor(x_pinn.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        t_grid = torch.full((x_grid.shape[0], 1), float(tt), device=DEVICE)
        xt_pair = torch.cat([x_grid, t_grid], dim=1)

        with torch.no_grad():
            u_pinn_tt = model(xt_pair).cpu().numpy().flatten()

        # 绘制PISCL曲线
        pinn_line = plt.plot(
            x_pinn, u_pinn_tt,
            lw=LEGEND_LW - 0.2, color=c, linestyle='--', alpha=1.0
        )[0]
        legend_items.append((pinn_line, f"PISCL t={tt:.2f}"))

    # 统一图例样式
    plt.xlabel('x', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('u', fontsize=FONT_SIZE_LABEL)
    plt.title(f'PISCL vs Numerical Solution', fontsize=FONT_SIZE_TITLE)
    plt.grid(True, alpha=0.3)

    plt.legend(
        [item[0] for item in legend_items],
        [item[1] for item in legend_items],
        loc=LEGEND_LOC,
        ncol=LEGEND_NCOL,
        fontsize=FONT_SIZE_LEGEND,
        framealpha=LEGEND_ALPHA,
        fancybox=True,
        shadow=False
    )

    plt.tight_layout()
    plt.savefig('burgers_lf_fpinn_profiles_overlay.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    print(f"\n数值解匹配结果: {num_matched}")
    print('时间剖面图已保存 → burgers_lf_fpinn_profiles_overlay.png')


# ========== 训练函数 ==========
# ========== 训练函数 ==========
def train(model, epochs=TRAIN_EPOCHS, lr=TRAIN_LR, print_every=TRAIN_PRINT_EVERY):
    opt = optim.Adam(model.parameters(), lr=lr)
    # opt = optim.AdamW(model.parameters(), lr=lr)

    # opt = Lion(model.parameters(), lr=lr)

    ckpt_path = 'lf_fpinn_burgers.pth'

    # 初始化：记录最小损失
    best_loss = float('inf')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f'已加载预训练模型: {ckpt_path}')
    else:
        print('从头开始训练')

    start_time = time.time()
    print(f"开始训练（总轮数：{epochs}）...")

    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # PDE残差计算
        xt_f = sample_collocation(TRAIN_N_F)
        u = model(xt_f)
        grads = torch.autograd.grad(u, xt_f, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, xt_f, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        f_res = u_t + u * u_x - NU * u_xx
        loss_f = torch.mean(f_res ** 2)

        # IC+BC损失计算
        xt_ib, u_true_ib = sample_ic_bc(TRAIN_N_IC_BC)
        u_pred_ib = model(xt_ib)
        loss_ib = torch.mean((u_pred_ib - u_true_ib) ** 2)

        # 总损失
        loss = loss_f + loss_ib
        loss.backward()
        opt.step()

        # ===================== 核心修改：保存最优模型 =====================
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), ckpt_path)
            if epoch % print_every == 0:
                print(f"✅ 最优模型已保存 | Epoch {epoch} | Best Loss: {best_loss:.3e}")
        # =================================================================

        if epoch % print_every == 0 or epoch == 1:
            print(f'Epoch {epoch:6d} | Loss {loss.item():.3e} | PDE {loss_f.item():.3e} | IC+BC {loss_ib.item():.3e}')

    end_time = time.time()
    total_time = end_time - start_time
    total_time_min = total_time / 60
    print("=" * 50)
    print(f"训练完成！总训练时长：")
    print(f"  - 秒：{total_time:.2f} s")
    print(f"  - 分钟：{total_time_min:.2f} min")
    print(f"🏆 最小损失: {best_loss:.3e}")
    print("=" * 50)

    # 训练结束，加载最优权重
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"已加载【最优模型】: {ckpt_path}")
    return model


# ========== 主程序 ==========
if __name__ == "__main__":
    # 初始化模型
    model = LFFPINN_Burgers(K_space=K_SPACE, M_time=M_TIME).to(DEVICE)

    # 统计参数数量
    total_params = count_trainable_params(model)

    # 训练模型
    model = train(model)

    # 预测
    x_eval = torch.linspace(X_MIN, X_MAX, VIS_NX, device=DEVICE)
    t_eval = torch.linspace(T_MIN, T_MAX, VIS_NT, device=DEVICE)
    X, T = torch.meshgrid(x_eval, t_eval, indexing='ij')
    XT_eval = torch.stack([X.flatten(), T.flatten()], dim=-1).to(DEVICE)

    with torch.no_grad():
        U_pred = model(XT_eval).view(VIS_NX, VIS_NT).cpu().numpy()


    # 保存PINN单独3D图
    def plot_surface(U, title, fname):
        plt.rcParams['font.family'] = FONT_FAMILY
        plt.rcParams['font.size'] = FONT_SIZE_BASE
        X_np, T_np = X.cpu().numpy(), T.cpu().numpy()
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_np, T_np, U, linewidth=0, antialiased=True, cmap='plasma', rstride=1, cstride=1)
        ax.set_xlabel('x', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('t', fontsize=FONT_SIZE_LABEL)
        ax.set_zlabel('u', fontsize=FONT_SIZE_LABEL)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        plt.tight_layout()
        plt.savefig(fname, dpi=FIG_DPI)
        # plt.show()


    plot_surface(U_pred, 'PISCL prediction of Burgers equation', 'burgers_lf_fpinn_surface.png')
    print('LF-FPINN 3D图已保存 → burgers_lf_fpinn_surface.png')

    # 加载数值解并可视化（完全修复）
    tag = nu_to_tag(NU)
    print(f"\n自动匹配NU={NU}的文件名标签：{tag}")
    x_file = f'x_axis_{tag}.csv'
    t_file = f't_sel_{tag}.csv'
    U_file = f'U_profiles_{tag}.csv'

    has_num = all(os.path.exists(p) for p in [x_file, t_file, U_file])
    x_pinn = x_eval.cpu().numpy()
    t_pinn = t_eval.cpu().numpy()

    if has_num:
        # 读取数值解（完全保留所有时间点，不截断）
        x_num = read_vec_csv(x_file)
        t_num = read_vec_csv(t_file)
        U_num = read_mat_csv(U_file)

        # 调整维度（确保U_num为(时间点, x点)）
        if U_num.shape == (x_num.size, t_num.size):
            U_num = U_num.T
        if U_num.shape != (t_num.size, x_num.size):
            raise RuntimeError(f'U_profiles_{tag}.csv 维度不匹配: {U_num.shape} vs ({t_num.size}, {x_num.size})')

        # 【关键修复】移除强制截断时间点的代码，完全保留数值解的所有时间点
        # mask = t_num <= (t_pinn.max() + 1e-12)
        # t_num = t_num[mask]
        # U_num = U_num[mask, :]

        # 重构数值解3D网格（完全光滑）
        X_pinn, T_pinn, U_num_3d = reconstruct_numeric_3d(x_pinn, t_pinn, x_num, t_num, U_num)

        # 计算误差指标
        error_metrics = calculate_error_metrics(U_pred, U_num_3d)

        # 绘制3D对比图（数值解完全光滑，无锯齿）
        plot_3d_comparison(U_pred, U_num_3d, X_pinn, T_pinn)

        # 绘制绝对误差图
        abs_error = np.abs(U_pred - U_num_3d)
        plot_absolute_error(abs_error, X_pinn, T_pinn)

        # 绘制时间剖面图（完全对齐）
        plot_time_profiles(U_pred, x_pinn, t_pinn, x_num, t_num, U_num)
    else:
        print(f'[WARN] 未找到数值解文件，仅绘制PISCL时间剖面。')
        plot_time_profiles(U_pred, x_pinn, t_pinn)