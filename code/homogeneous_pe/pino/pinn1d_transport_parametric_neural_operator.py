# -*- coding: utf-8 -*-
"""
Parametric neural operator (DeepONet-style) — same PDE as the parametric PINN.

Branch encodes log(Pe); trunk maps (x*, t*). Training: L-BFGS on the same fixed mesh.
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
# Configuration
# =============================================================================
seed = 1234567
torch_dtype = torch.float64

U_ref = 0.05
L_ref = 100.0
T_phys = 1000.0
time_scale_mode = "physical"
D_min = 0.001
D_max = 0.1

sensor_count = 1
branch_architecture = [sensor_count, 16, 16, 16]
trunk_architecture = [2, 16, 16, 16]
activation_cls = nn.Tanh

num_epochs_lbfgs = 1000
lr_lbfgs = 1.0

mesh_nx_pde = 60
mesh_nt_pde = 60
mesh_nlog_pde = 60
mesh_ic_nx = 60
mesh_ic_nlog = 60
mesh_bc_nt = 60
mesh_bc_nlog = 60

weight_pde = 1.0
weight_ic = 1.0
weight_inlet_bc = 1.0
weight_outlet_bc = 1.0

save_model = True
collocation_scatter_ms = 3
n_cycle_colors = 8

times_tstar = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
Pe_plot_values = [50.0, 500.0, 5000.0]
num_spatial_points = 500
x_plot_max_star = 1.0

# =============================================================================
# Device and scales
# =============================================================================
torch.set_default_dtype(torch_dtype)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch_dtype

pe_min = (U_ref * L_ref) / D_max
pe_max = (U_ref * L_ref) / D_min
log_pe_min = float(np.log(pe_min))
log_pe_max = float(np.log(pe_max))
print(f"Pe range: [{pe_min:g}, {pe_max:g}]")

t_adv = L_ref / U_ref
if time_scale_mode == "advective":
    t_scale = t_adv
    coeff_x_scale = 1.0
    coeff_xx_factor = 1.0
elif time_scale_mode == "physical":
    t_scale = T_phys
    coeff_x_scale = U_ref * T_phys / L_ref
    coeff_xx_factor = coeff_x_scale
else:
    t_scale = T_phys
    coeff_x_scale = U_ref * T_phys / L_ref
    coeff_xx_factor = coeff_x_scale

t_final_star = T_phys / t_scale

# =============================================================================
# DeepONet
# =============================================================================


class DeepONetParametric(nn.Module):
    def __init__(self, branch_arch, trunk_arch, activation):
        super().__init__()
        branch_layers = []
        for i in range(len(branch_arch) - 1):
            in_f, out_f = branch_arch[i], branch_arch[i + 1]
            branch_layers.append(nn.Linear(in_f, out_f))
            if i < len(branch_arch) - 2:
                branch_layers.append(activation())
        self.branch = nn.Sequential(*branch_layers)

        trunk_layers = []
        for i in range(len(trunk_arch) - 1):
            in_f, out_f = trunk_arch[i], trunk_arch[i + 1]
            trunk_layers.append(nn.Linear(in_f, out_f))
            if i < len(trunk_arch) - 2:
                trunk_layers.append(activation())
        self.trunk = nn.Sequential(*trunk_layers)

        gain = nn.init.calculate_gain("tanh")
        for mod in (self.branch, self.trunk):
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x_star, t_star, branch_input):
        pts = torch.cat([x_star, t_star], dim=1)
        b_vec = self.branch(branch_input)
        t_vec = self.trunk(pts)
        return torch.sigmoid((b_vec * t_vec).sum(dim=-1, keepdim=True))


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
t_1d = np.linspace(0.0, tf, mesh_nt_pde)
log_1d = np.linspace(log_pe_min, log_pe_max, mesh_nlog_pde)
ilog_1d = np.arange(mesh_nlog_pde, dtype=np.int64)
gx, gt, glog = np.meshgrid(x_1d, t_1d, log_1d, indexing="ij")
_, _, gil = np.meshgrid(x_1d, t_1d, ilog_1d, indexing="ij")
x_star_pde_np = gx.reshape(-1)
t_star_pde_np = gt.reshape(-1)
log_pe_pde_np = glog.reshape(-1)
ilog_pde_flat = gil.reshape(-1)
pde_colors = [f"C{int(i) % n_cycle_colors}" for i in ilog_pde_flat]

x_ic_1d = np.linspace(0.0, 1.0, mesh_ic_nx)
log_ic_1d = np.linspace(log_pe_min, log_pe_max, mesh_ic_nlog)
gxi, gli = np.meshgrid(x_ic_1d, log_ic_1d, indexing="ij")
x_star_ic_np = gxi.reshape(-1)
t_star_ic_np = np.zeros_like(x_star_ic_np)
log_pe_ic_np = gli.reshape(-1)

t_bc_1d = np.linspace(0.0, tf, mesh_bc_nt)
log_bc_1d = np.linspace(log_pe_min, log_pe_max, mesh_bc_nlog)
gtb, glb = np.meshgrid(t_bc_1d, log_bc_1d, indexing="ij")
t_star_inlet_np = gtb.reshape(-1)
log_pe_inlet_np = glb.reshape(-1)
x_star_inlet_np = np.zeros_like(t_star_inlet_np)
x_star_outlet_np = np.ones_like(t_star_inlet_np)
t_star_outlet_np = t_star_inlet_np.copy()
log_pe_outlet_np = log_pe_inlet_np.copy()

train_x_pde = torch.tensor(x_star_pde_np.reshape(-1, 1), dtype=DTYPE, device=device, requires_grad=True)
train_t_pde = torch.tensor(t_star_pde_np.reshape(-1, 1), dtype=DTYPE, device=device, requires_grad=True)
train_log_pe_pde = torch.tensor(log_pe_pde_np.reshape(-1, 1), dtype=DTYPE, device=device)
branch_pde = train_log_pe_pde.expand(-1, sensor_count)

train_x_ic = torch.tensor(x_star_ic_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_t_ic = torch.tensor(t_star_ic_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_log_pe_ic = torch.tensor(log_pe_ic_np.reshape(-1, 1), dtype=DTYPE, device=device)
branch_ic = train_log_pe_ic.expand(-1, sensor_count)

train_x_in = torch.tensor(x_star_inlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_t_in = torch.tensor(t_star_inlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_log_pe_in = torch.tensor(log_pe_inlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
branch_in = train_log_pe_in.expand(-1, sensor_count)

train_x_out = torch.tensor(x_star_outlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_t_out = torch.tensor(t_star_outlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
train_log_pe_out = torch.tensor(log_pe_outlet_np.reshape(-1, 1), dtype=DTYPE, device=device)
branch_out = train_log_pe_out.expand(-1, sensor_count)

# =============================================================================
# Train
# =============================================================================
model = DeepONetParametric(branch_architecture, trunk_architecture, activation_cls).to(device)
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
    c_pde = model(train_x_pde, train_t_pde, branch_pde)
    dC_dt = gradients(c_pde, train_t_pde)
    dC_dx = gradients(c_pde, train_x_pde)
    d2C_dx2 = gradients(dC_dx, train_x_pde)
    pe_pde = torch.exp(train_log_pe_pde)
    residual = dC_dt + coeff_x_scale * dC_dx - (coeff_xx_factor / pe_pde) * d2C_dx2
    pde_loss = torch.mean(residual**2)
    ic_loss = torch.mean((model(train_x_ic, train_t_ic, branch_ic) - 0.0) ** 2)
    inlet_loss = torch.mean((model(train_x_in, train_t_in, branch_in) - 1.0) ** 2)
    outlet_loss = torch.mean((model(train_x_out, train_t_out, branch_out) - 0.0) ** 2)
    total_loss = (
        weight_pde * pde_loss
        + weight_ic * ic_loss
        + weight_inlet_bc * inlet_loss
        + weight_outlet_bc * outlet_loss
    )
    total_loss.backward()
    closure.latest = (
        total_loss.item(),
        pde_loss.item(),
        ic_loss.item(),
        inlet_loss.item(),
        outlet_loss.item(),
    )
    t_bar.set_description(
        "loss : %.3e  mse_pde %.3e  mse_ic %.3e  mse_in %.3e  mse_out %.3e"
        % closure.latest
    )
    t_bar.refresh()
    return total_loss


for _ in t_bar:
    model.train()
    optimizer.step(closure)
    obj.append(list(closure.latest))
t_bar.close()

# =============================================================================
# Collocation figure
# =============================================================================
ms = collocation_scatter_ms
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
axes2[0].scatter(x_star_pde_np, t_star_pde_np, c=pde_colors, s=ms, alpha=1.0)
axes2[0].set_title(r"PDE points: $x^*$ vs $t^*$ (color by $\log Pe$ index)")
axes2[0].set_xlabel(r"$x^*$")
axes2[0].set_ylabel(r"$\hat{t}$" if time_scale_mode == "physical" else r"$t^*$")
axes2[0].grid(True, alpha=0.3)

axes2[1].scatter(
    x_star_ic_np,
    log_pe_ic_np,
    s=ms,
    alpha=1.0,
    marker="x",
    color="C0",
    label="IC",
)
axes2[1].scatter(
    t_star_inlet_np,
    log_pe_inlet_np,
    s=ms,
    alpha=1.0,
    color="C1",
    label="Inlet BC",
)
axes2[1].set_xlabel(r"$x^*$ or $t^*$")
axes2[1].set_ylabel(r"$\log Pe$")
axes2[1].set_title(r"IC ($x^*$, $\log Pe$) and inlet ($t^*$, $\log Pe$)")
axes2[1].grid(True, alpha=0.3)
axes2[1].legend(loc="best", fontsize=8)
fig2.suptitle("Fixed regular collocation mesh", fontsize=14)
plt.savefig(os.path.join(results_dir, "pino_collocation_points.png"), bbox_inches="tight")
plt.close(fig2)

# =============================================================================
# Concentration and loss plots
# =============================================================================
x_plot = np.linspace(0.0, x_plot_max_star, num_spatial_points, dtype=np.float64)
x_plot_t = torch.tensor(x_plot.reshape(-1, 1), dtype=DTYPE, device=device)
pe_plot_values = list(Pe_plot_values)
times_plot = list(times_tstar)
if len(pe_plot_values) == 0:
    pe_plot_values = [10.0]

if len(pe_plot_values) <= 4:
    nrows = int(np.ceil(len(pe_plot_values) / 2))
    ncols = 2 if len(pe_plot_values) > 1 else 1
else:
    nrows = 2
    ncols = int(np.ceil(len(pe_plot_values) / 2))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), constrained_layout=True)
axes_arr = np.array(axes).reshape(-1)
colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for idx, pe_val in enumerate(pe_plot_values):
    ax = axes_arr[idx]
    log_pe_val = float(np.log(pe_val))
    d_eff = coeff_xx_factor / float(pe_val)
    branch_eval = torch.full((len(x_plot), sensor_count), log_pe_val, dtype=DTYPE, device=device)
    for j, t_star in enumerate(times_plot):
        color = colors_cycle[j % len(colors_cycle)]
        t_t = torch.full((len(x_plot), 1), float(t_star), dtype=DTYPE, device=device)
        model.eval()
        with torch.no_grad():
            c_op = model(x_plot_t, t_t, branch_eval).detach().cpu().numpy().flatten()
        if t_star <= 0:
            c_ana = np.zeros_like(x_plot, dtype=np.float64)
        else:
            sqrt_dt = np.sqrt(d_eff * t_star)
            term1 = erfc((x_plot - coeff_x_scale * t_star) / (2.0 * sqrt_dt))
            ux_over_d = coeff_x_scale * x_plot / d_eff
            b = (x_plot + coeff_x_scale * t_star) / (2.0 * sqrt_dt)
            exponent = ux_over_d - b**2
            term2 = np.exp(np.clip(exponent, -745.0, 700.0)) * erfcx(b)
            c_ana = 0.5 * (term1 + term2)
            c_ana = np.nan_to_num(c_ana, nan=0.0, posinf=0.0, neginf=0.0)
        ax.plot(x_plot, c_op, linewidth=2, linestyle="-", color=color)
        ax.plot(x_plot, c_ana, linewidth=2, linestyle="--", color=color, alpha=0.7)
    ax.set_title(f"Pe = {pe_val:g}")
    ax.set_xlim(0.0, x_plot_max_star)
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$C^*$")
    ax.grid(True, alpha=0.3)
    for j, t_star in enumerate(times_plot):
        color = colors_cycle[j % len(colors_cycle)]
        ax.plot([], [], marker="s", markersize=7, linestyle="None", color=color, label=f"t*={t_star:g}")
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label="Operator")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")
    legend = ax.legend(loc="upper right", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color("black")
        text.set_alpha(1.0)

for k in range(len(pe_plot_values), len(axes_arr)):
    axes_arr[k].axis("off")

fig.suptitle("Parametric neural operator: concentration profiles", fontsize=14)
conc_path = os.path.join(results_dir, "pino_concentration.png")
plt.savefig(conc_path, bbox_inches="tight")
plt.close(fig)

fig3, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
epochs = np.arange(1, len(obj) + 1)
h = np.array(obj, dtype=np.float64)
ax3.plot(epochs, h[:, 0], color="black", linewidth=1.6, label="Total")
ax3.plot(epochs, h[:, 1], color="goldenrod", linewidth=1.0, alpha=0.85, label="PDE")
ax3.plot(epochs, h[:, 2], color="gray", linewidth=1.0, alpha=0.6, label="IC")
ax3.plot(epochs, h[:, 3], color="purple", linewidth=1.0, alpha=0.75, label="Inlet")
ax3.plot(epochs, h[:, 4], color="crimson", linewidth=1.0, alpha=0.75, label="Outlet")
ax3.set_yscale("log")
ax3.set_xlabel("L-BFGS step")
ax3.set_ylabel("Loss")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="best", fontsize=9, frameon=False)
loss_path = os.path.join(results_dir, "pino_loss.png")
plt.savefig(loss_path, bbox_inches="tight")
plt.close(fig3)

if save_model:
    torch.save(model.state_dict(), os.path.join(results_dir, "pino_model.pt"))

print("Saved:", conc_path, os.path.join(results_dir, "pino_collocation_points.png"), loss_path)
