# -*- coding: utf-8 -*-
"""
Baseline PINN — dimensionless 1D ADE: dC*/dt* + CFL dC*/dx* - Pe d2C*/dx*2 = 0.

Pe = D * T_seconds / L^2 (computed; D in m^2/s). CFL = U * T_MAX / L (U in m/d, T_MAX in d).
Network: (x*, t*) -> C* in [0,1]. Training: L-BFGS on a regular mesh.
Analytical: Ogata–Banks with D in m^2/day = D_SI * 86400, t in days.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erfc, erfcx
from tqdm import trange

# =============================================================================
# Plot style
# =============================================================================
mpl.rcParams["figure.dpi"] = 800
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#FF5F05",
        "#13294B",
        "#009FD4",
        "#FCB316",
        "#006230",
        "#007E8E",
        "#5C0E41",
        "#7D3E13",
    ]
)
plt.rcParams["font.family"] = "Times New Roman"

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# PyTorch device / dtype
# =============================================================================
seed = 1234567
torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(device))
DTYPE = torch.get_default_dtype()

# =============================================================================
# Physical parameters (primary) and derived Pe, CFL
# =============================================================================
L = 100.0  # m
T_MAX = 1200.0  # d
D_M2_S = 1e-6  # m^2/s
SECONDS_PER_DAY = 86400.0
C0 = 5.0  # kg/m^3 reference for dimensional plots

# Configurable velocity (m/d); CFL = U * T_MAX / L
U = 0.05 # from 0.01 to 0.05

T_SECONDS = T_MAX * SECONDS_PER_DAY
PE = D_M2_S * T_SECONDS / (L**2)
CFL = U * T_MAX / L
D_M2_PER_DAY = D_M2_S * SECONDS_PER_DAY

num_layers = 4
num_neurons = 16
activation = nn.Tanh

num_epochs_lbfgs = 1000
lr_lbfgs = 1.0

mesh_nx_pde = 50
mesh_nt_pde = 50
mesh_n_ic = 50
mesh_n_bc = 50

weight_pde = 1.0
weight_ic = 1.0
weight_inlet_bc = 1.0
weight_outlet_bc = 1.0

save_model = True
collocation_scatter_ms = 3

times_days = np.array([100.0, 300.0, 600.0, 900.0, 1200.0])
num_points = 5001
X_MAX = 100.0

t_scale = T_MAX
t_final_star = T_MAX / t_scale  # 1.0

print("Run configuration:")
print(f"  L={L} m, T_MAX={T_MAX} d, D={D_M2_S} m^2/s, U={U} m/d, C0={C0}")
print(f"  PE (computed) = D*T_s/L^2 = {PE:.12g}  (matches 0.010368: {np.isclose(PE, 0.010368)})")
print(f"  CFL = U*T_MAX/L = {CFL:.12g}")
print(f"  D (m^2/d) for Ogata = {D_M2_PER_DAY:.12g}")
print(f"  L-BFGS epochs={num_epochs_lbfgs}, mesh PDE={mesh_nx_pde}x{mesh_nt_pde}")

# =============================================================================
# Neural network
# =============================================================================


class PINN_Transp(nn.Module):
    def __init__(self, num_layers, num_neurons, activation_cls):
        super().__init__()
        layers = [nn.Linear(2, num_neurons), activation_cls()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation_cls())
        layers.append(nn.Linear(num_neurons, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def init_weights(self):
        gain = nn.init.calculate_gain("tanh")
        for param in self.parameters():
            if param.ndim >= 2:
                nn.init.xavier_uniform_(param, gain=gain)
            else:
                nn.init.zeros_(param)

    def forward(self, x_star, tau_star):
        return self.net(torch.cat([x_star, tau_star], dim=1))


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


# =============================================================================
# Collocation mesh (NumPy) and tensors
# =============================================================================
tf = float(t_final_star)
x_1d = np.linspace(0.0, 1.0, mesh_nx_pde)
tau_1d = np.linspace(0.0, tf, mesh_nt_pde)
gx, gtau = np.meshgrid(x_1d, tau_1d, indexing="ij")
x_pde_np = gx.reshape(-1)
tau_pde_np = gtau.reshape(-1)

x_ic_np = np.linspace(0.0, 1.0, mesh_n_ic)
tau_ic_np = np.zeros_like(x_ic_np)
tau_bc_np = np.linspace(0.0, tf, mesh_n_bc)
x_bc_in_np = np.zeros_like(tau_bc_np)
x_bc_out_np = np.ones_like(tau_bc_np)

train_x_pde = torch.tensor(x_pde_np.reshape(-1, 1), dtype=DTYPE, device=device, requires_grad=True)
train_tau_pde = torch.tensor(tau_pde_np.reshape(-1, 1), dtype=DTYPE, device=device, requires_grad=True)
train_x_ic = torch.tensor(x_ic_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_tau_ic = torch.tensor(tau_ic_np.reshape(-1, 1), dtype=DTYPE, device=device)
c_ic_target = torch.zeros((mesh_n_ic, 1), dtype=DTYPE, device=device)
train_x_in = torch.tensor(x_bc_in_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_tau_in = torch.tensor(tau_bc_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_x_out = torch.tensor(x_bc_out_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_tau_out = torch.tensor(tau_bc_np.reshape(-1, 1), dtype=DTYPE, device=device)

# =============================================================================
# Train
# =============================================================================
model = PINN_Transp(num_layers, num_neurons, activation).to(device)
model.init_weights()
print("Total parameters:", sum(p.numel() for p in model.parameters()))

obj = []

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=lr_lbfgs,
    max_iter=1,
    history_size=50,
    line_search_fn="strong_wolfe",
)
print(optimizer)

t_bar = trange(num_epochs_lbfgs, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")


def closure():
    optimizer.zero_grad(set_to_none=True)
    c_pde = model(train_x_pde, train_tau_pde)
    dc_dtau = gradients(c_pde, train_tau_pde)
    dc_dx = gradients(c_pde, train_x_pde)
    d2c_dx2 = gradients(dc_dx, train_x_pde)
    residual = dc_dtau + CFL * dc_dx - PE * d2c_dx2
    pde_loss = (residual**2).mean()

    c_ic_pred = model(train_x_ic, train_tau_ic)
    ic_loss = ((c_ic_pred - c_ic_target) ** 2).mean()

    c_in_pred = model(train_x_in, train_tau_in)
    in_loss = ((c_in_pred - 1.0) ** 2).mean()

    c_out_pred = model(train_x_out, train_tau_out)
    out_loss = ((c_out_pred - 0.0) ** 2).mean()

    total_loss = (
        weight_pde * pde_loss
        + weight_ic * ic_loss
        + weight_inlet_bc * in_loss
        + weight_outlet_bc * out_loss
    )
    total_loss.backward()
    closure.latest = (
        total_loss.item(),
        pde_loss.item(),
        ic_loss.item(),
        in_loss.item(),
        out_loss.item(),
    )
    t_bar.set_description(
        "loss : %.3e  mse_pde %.3e  mse_ic %.3e  mse_bc_l %.3e  mse_bc_r %.3e"
        % (total_loss.item(), pde_loss.item(), ic_loss.item(), in_loss.item(), out_loss.item())
    )
    t_bar.refresh()
    return total_loss


for _ in t_bar:
    model.train()
    optimizer.step(closure)
    obj.append(list(closure.latest))
t_bar.close()

# =============================================================================
# Collocation plot
# =============================================================================
fig_colloc = plt.figure(figsize=(10, 6), tight_layout=True)
ax_c = fig_colloc.add_subplot(111)
ax_c.set_title("Fixed regular collocation mesh", fontsize=14)
ms = collocation_scatter_ms
ax_c.scatter(x_pde_np, tau_pde_np, s=ms, alpha=1.0, label="PDE")
ax_c.scatter(x_ic_np, tau_ic_np, s=ms * 1.2, alpha=1.0, label="IC", marker="x")
ax_c.scatter(x_bc_in_np, tau_bc_np, s=ms, alpha=1.0, label="Inlet")
ax_c.scatter(x_bc_out_np, tau_bc_np, s=ms, alpha=1.0, marker="^", label="Outlet")
ax_c.set_xlabel(r"Dimensionless distance $x^* = x/L$", fontsize=12)
ax_c.set_ylabel(r"Normalized time $t^* = t/T_{\max}$", fontsize=12)
ax_c.grid(True, alpha=0.35)
ax_c.minorticks_on()
ax_c.legend(loc="best", fontsize=10)
plt.savefig(os.path.join(results_dir, "baseline_cfl_pinn_collocation_points.png"))
plt.close(fig_colloc)

# =============================================================================
# Concentration and loss plots
# =============================================================================
model.eval().cpu()
xp = torch.tensor(np.linspace(0, X_MAX, num_points).reshape(-1, 1) / L, dtype=DTYPE)
x_line = np.linspace(0, X_MAX, num_points)

fig = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig.add_subplot(111)
fig.suptitle("Concentration profiles at selected times (days)", fontsize=14)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
ax.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

for idx, ti in enumerate(times_days):
    color = colors[idx % len(colors)]
    if ti <= 0:
        c_an = np.zeros_like(x_line, dtype=np.float64)
    else:
        sqrt_dt = np.sqrt(D_M2_PER_DAY * ti)
        term1 = erfc((x_line - U * ti) / (2.0 * sqrt_dt))
        ux_over_d = U * x_line / D_M2_PER_DAY
        b = (x_line + U * ti) / (2.0 * sqrt_dt)
        exponent = ux_over_d - b**2
        term2 = np.exp(np.clip(exponent, -745.0, 700.0)) * erfcx(b)
        c_an = (C0 / 2.0) * (term1 + term2)
        c_an = np.nan_to_num(c_an, nan=0.0, posinf=0.0, neginf=0.0)
    tau_eval = torch.tensor((ti / t_scale) * np.ones(num_points).reshape(-1, 1), dtype=DTYPE)
    c_pred = model(xp, tau_eval).detach().numpy().flatten() * C0
    ax.plot(x_line, c_pred, linewidth=2, linestyle="-", color=color)
    ax.plot(x_line, c_an, linewidth=2, linestyle="--", color=color, alpha=0.7)
    ax.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{ti:.1f}")

ax.set_xlabel("Distance $x$ (m)", fontsize=12)
ax.set_ylabel(r"Concentration $C$ (kg/m$^3$)", fontsize=12)
ax.grid()
ax.minorticks_on()
legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.25),
    ncol=4,
    frameon=False,
    fontsize=10,
    labelspacing=0.5,
    columnspacing=1.2,
)
for text in legend.get_texts():
    text.set_color("black")
    text.set_alpha(1.0)
plt.savefig(os.path.join(results_dir, "baseline_cfl_pinn_concentration.png"))
plt.close(fig)

fig3 = plt.figure(figsize=(10, 6), tight_layout=True)
ax3 = fig3.add_subplot(111)
obj_arr = np.array(obj)
ax3.plot(obj_arr[:, 0], "-b", label="Total")
ax3.plot(obj_arr[:, 1], color="goldenrod", label="PDE")
ax3.plot(obj_arr[:, 2], "k", label="IC", alpha=0.35)
ax3.plot(obj_arr[:, 3], "m", label="Inlet BC")
ax3.plot(obj_arr[:, 4], "r", label="Outlet BC")
ax3.set_yscale("log", base=10)
ax3.set_xlabel("L-BFGS step")
ax3.set_ylabel("Loss")
ax3.minorticks_on()
ax3.grid()
ax3.legend(loc="best", fontsize=8)
plt.savefig(os.path.join(results_dir, "baseline_cfl_pinn_loss.png"))
plt.close(fig3)

if save_model:
    torch.save(model.state_dict(), os.path.join(results_dir, "baseline_cfl_pinn_model.pt"))

print("Saved outputs to:", results_dir)
