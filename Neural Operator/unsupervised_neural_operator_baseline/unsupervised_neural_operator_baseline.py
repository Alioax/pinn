"""
Unsupervised Neural Operator (DeepONet) for 1D Advection-Dispersion (dimensionless).

Trains the operator log Pe -> C*(x*, t*) using only physics-informed losses (no data).
The dimensionless PDE is:
    C*_t* + C*_x* - (1/Pe) C*_{x*x*} = 0

with boundary/initial conditions:
    C*(0, t*) = 1  (inlet)
    C*(1, t*) = 0  (outlet, far-field approximation)
    C*(x*, 0) = 0  (initial condition)

The DeepONet is trained with PDE residual, IC, and BC losses only; analytical solution
is used only for plot overlay. All parameters are configurable at the top of the file.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from torch.autograd import grad
from pathlib import Path
from tqdm import tqdm

# Import analytical solution from sibling folder (for plot overlay only)
sys.path.append(str(Path(__file__).parent.parent / "analytical_solution"))
from analytical_solution import analytical_solution

# Visualization settings
mpl.rcParams["figure.dpi"] = 800
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#FF5F05", "#13294B", "#009FD4", "#FCB316",
           "#006230", "#007E8E", "#5C0E41", "#7D3E13"]
)

# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics parameters (dimensionless)
Pe_min = 1
Pe_max = 1e5
t_final_star = 1.0

# Model architecture (DeepONet)
# Branch: encodes log Pe only (single scalar) -> keep small (1 layer, 6 neurons)
# Trunk: larger to represent sharp spatial fronts (3 layers, 32 neurons)
branch_layers = 1
branch_neurons = 6
trunk_layers = 3
trunk_neurons = 32
latent_dim = 32
activation_name = "Tanh"

# Stratified collocation: if True, oversample near boundaries (x*=0,1) and high Pe
use_stratified_sampling = True

# Training parameters (more collocation = finer resolution + better Pe coverage)
num_epochs = 10000
lr = 0.001
num_collocation = 500 * 500
num_ic = 250
num_bc = 250
weight_pde = 1
weight_ic = 1
weight_inlet_bc = 1
weight_outlet_bc = 1

# Plotting parameters (dimensionless)
times_tstar = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
pe_values_to_plot = [10, 50, 1e2, 500, 1e3, 1e4, 1e5]
num_spatial_points = 500
plot_dpi = 800

# Derived parameters
logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)

# ============================================================================
# Model setup
# ============================================================================

torch.manual_seed(123456789)
np.random.seed(123456789)

activation_map = {
    "Tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
}
activation_cls = activation_map[activation_name]
gain = nn.init.calculate_gain("tanh")


def sample_collocation_points(stratified):
    """Sample PDE/IC/BC collocation points.

    Stratified collocation sampling: instead of drawing all PDE points uniformly
    in (x*, t*, log Pe), we reserve fixed fractions for "important" regions:
    - 50% uniform over the domain
    - 25% near boundaries (x* in [0, 0.1] or [0.9, 1]) where BCs and sharp
      fronts matter
    - 25% in the upper half of log Pe (high Pe), where the solution has
      steeper gradients and needs more resolution

    This gives the residual loss more examples in regions where the solution
    varies quickly, often improving accuracy (especially for high Pe).
    """
    if not stratified:
        x_pde = np.random.rand(num_collocation, 1).astype(np.float32)
        t_pde = np.random.rand(num_collocation, 1).astype(np.float32) * t_final_star
        log_pe_pde = (np.random.rand(num_collocation, 1).astype(np.float32) *
                      (logPe_max - logPe_min) + logPe_min)
    else:
        # Stratified: 50% uniform, 25% near boundaries, 25% high Pe (each point has consistent (x,t,log_pe))
        n_unif = num_collocation // 2
        n_bound = num_collocation // 4
        n_highpe = num_collocation - n_unif - 2 * n_bound
        x_pde = np.vstack([
            np.random.rand(n_unif, 1).astype(np.float32),
            np.random.rand(n_bound, 1).astype(np.float32) * 0.1,
            np.random.rand(n_bound, 1).astype(np.float32) * 0.1 + 0.9,
            np.random.rand(n_highpe, 1).astype(np.float32),
        ])
        t_pde = np.random.rand(num_collocation, 1).astype(np.float32) * t_final_star
        log_pe_pde = np.vstack([
            (np.random.rand(n_unif, 1).astype(np.float32) * (logPe_max - logPe_min) + logPe_min),
            (np.random.rand(2 * n_bound, 1).astype(np.float32) * (logPe_max - logPe_min) + logPe_min),
            (np.random.rand(n_highpe, 1).astype(np.float32) * 0.5 + 0.5) * (logPe_max - logPe_min) + logPe_min,
        ])
        perm = np.random.permutation(num_collocation)
        x_pde = x_pde[perm]
        t_pde = t_pde[perm]
        log_pe_pde = log_pe_pde[perm]

    x_ic = np.random.rand(num_ic, 1).astype(np.float32)
    t_ic = np.zeros((num_ic, 1), dtype=np.float32)
    log_pe_ic = (np.random.rand(num_ic, 1).astype(np.float32) *
                 (logPe_max - logPe_min) + logPe_min)

    x_inlet = np.zeros((num_bc, 1), dtype=np.float32)
    t_inlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    log_pe_inlet = (np.random.rand(num_bc, 1).astype(np.float32) *
                    (logPe_max - logPe_min) + logPe_min)

    x_outlet = np.ones((num_bc, 1), dtype=np.float32)
    t_outlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    log_pe_outlet = (np.random.rand(num_bc, 1).astype(np.float32) *
                     (logPe_max - logPe_min) + logPe_min)

    return (x_pde, t_pde, log_pe_pde,
            x_ic, t_ic, log_pe_ic,
            x_inlet, t_inlet, log_pe_inlet,
            x_outlet, t_outlet, log_pe_outlet)


class DeepONet(nn.Module):
    """DeepONet: branch(log Pe) and trunk(x*, t*) with inner product + sigmoid -> C*.
    Forward supports pointwise (training) and grid (plotting) modes.
    """

    def __init__(self):
        super(DeepONet, self).__init__()
        branch_layers_list = []
        in_f = 1
        for _ in range(branch_layers):
            branch_layers_list.append(nn.Linear(in_f, branch_neurons))
            branch_layers_list.append(activation_cls())
            in_f = branch_neurons
        branch_layers_list.append(nn.Linear(in_f, latent_dim))
        self.branch = nn.Sequential(*branch_layers_list)

        trunk_layers_list = []
        in_f = 2
        for _ in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(in_f, trunk_neurons))
            trunk_layers_list.append(activation_cls())
            in_f = trunk_neurons
        trunk_layers_list.append(nn.Linear(in_f, latent_dim))
        self.trunk = nn.Sequential(*trunk_layers_list)

        self._init_weights()

    def _init_weights(self):
        for m in self.branch:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, log_pe, x_star, t_star):
        """
        Pointwise: log_pe (N, 1), x_star (N, 1), t_star (N, 1) -> (N,).
        Grid: log_pe (B, 1), x_star (P,), t_star (P,) -> (B, P).
        """
        x_ = x_star.squeeze(-1) if x_star.dim() > 1 else x_star
        t_ = t_star.squeeze(-1) if t_star.dim() > 1 else t_star
        points = torch.stack([x_, t_], dim=-1)
        b = self.branch(log_pe)
        t = self.trunk(points)
        if log_pe.shape[0] == points.shape[0]:
            return torch.sigmoid((b * t).sum(dim=-1))
        return torch.sigmoid(torch.mm(b, t.t()))


model = DeepONet()

# ============================================================================
# Plot Collocation Points (Pretraining)
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

(x_star_pde, t_star_pde, log_pe_pde,
 x_star_ic, t_star_ic, log_pe_ic,
 x_star_inlet, t_star_inlet, log_pe_inlet,
 x_star_outlet, t_star_outlet, log_pe_outlet) = sample_collocation_points(use_stratified_sampling)

plt.figure(figsize=(6.5, 4))
plt.scatter(x_star_pde, t_star_pde, s=1, alpha=1, color="C1", label="PDE", edgecolors="none")
plt.scatter(x_star_ic, t_star_ic, s=1, alpha=1, color="C0", label="Initial Condition", edgecolors="none")
plt.scatter(x_star_inlet, t_star_inlet, s=1, alpha=1, color="C2", label="Inlet BC", edgecolors="none")
plt.scatter(x_star_outlet, t_star_outlet, s=1, alpha=1, color="C3", label="Outlet BC", edgecolors="none")
plt.xlabel("x* (dimensionless)", fontsize=12)
plt.ylabel("t* (dimensionless)", fontsize=12)
plt.title("Collocation Points Distribution", fontsize=14, pad=20)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False,
           fontsize=10, markerscale=5, handletextpad=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

collocation_plot_path = results_dir / "collocation_points.png"
plt.savefig(str(collocation_plot_path), dpi=plot_dpi, bbox_inches="tight")
print(f"Collocation points plot saved to: {collocation_plot_path}")
collocation_pdf_path = results_dir / "collocation_points.pdf"
plt.savefig(str(collocation_pdf_path), format="pdf", bbox_inches="tight")
print(f"Collocation points PDF saved to: {collocation_pdf_path}")
plt.close()

plt.figure(figsize=(6.5, 4))
plt.scatter(t_star_pde, log_pe_pde, s=1, alpha=1, color="C1", edgecolors="none")
plt.xlabel("t* (dimensionless)", fontsize=12)
plt.ylabel("log(Pe)", fontsize=12)
plt.title("Parametric Coverage (t* vs log Pe)", fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pe_plot_path = results_dir / "collocation_points_logpe.png"
plt.savefig(str(pe_plot_path), dpi=plot_dpi, bbox_inches="tight")
print(f"Parametric coverage plot saved to: {pe_plot_path}")
pe_plot_pdf_path = results_dir / "collocation_points_logpe.pdf"
plt.savefig(str(pe_plot_pdf_path), format="pdf", bbox_inches="tight")
print(f"Parametric coverage PDF saved to: {pe_plot_pdf_path}")
plt.close()

# ============================================================================
# Training Loop
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lr / 10)

losses = {
    "total": [],
    "pde": [],
    "ic": [],
    "inlet_bc": [],
    "outlet_bc": [],
}

pbar = tqdm(range(num_epochs), desc="Training DeepONet")
for epoch in pbar:
    (x_star_pde, t_star_pde, log_pe_pde,
     x_star_ic, t_star_ic, log_pe_ic,
     x_star_inlet, t_star_inlet, log_pe_inlet,
     x_star_outlet, t_star_outlet, log_pe_outlet) = sample_collocation_points(use_stratified_sampling)

    x_star_pde_tensor = torch.tensor(x_star_pde, requires_grad=True)
    t_star_pde_tensor = torch.tensor(t_star_pde, requires_grad=True)
    log_pe_pde_tensor = torch.tensor(log_pe_pde)
    C_star_pde = model(log_pe_pde_tensor, x_star_pde_tensor, t_star_pde_tensor)

    dC_dt_star = grad(C_star_pde, t_star_pde_tensor, grad_outputs=torch.ones_like(C_star_pde),
                      create_graph=True, retain_graph=True)[0]
    dC_dx_star = grad(C_star_pde, x_star_pde_tensor, grad_outputs=torch.ones_like(C_star_pde),
                      create_graph=True, retain_graph=True)[0]
    d2C_dx2_star = grad(dC_dx_star, x_star_pde_tensor, grad_outputs=torch.ones_like(dC_dx_star),
                        create_graph=True, retain_graph=True)[0]

    pe_pde = torch.exp(log_pe_pde_tensor)
    pde_residual = dC_dt_star + dC_dx_star - (1.0 / pe_pde) * d2C_dx2_star
    pde_loss = torch.mean(pde_residual ** 2)

    x_star_ic_tensor = torch.tensor(x_star_ic)
    t_star_ic_tensor = torch.tensor(t_star_ic)
    log_pe_ic_tensor = torch.tensor(log_pe_ic)
    C_star_ic = model(log_pe_ic_tensor, x_star_ic_tensor, t_star_ic_tensor)
    ic_loss = nn.MSELoss()(C_star_ic, torch.zeros_like(C_star_ic))

    x_star_inlet_tensor = torch.tensor(x_star_inlet)
    t_star_inlet_tensor = torch.tensor(t_star_inlet)
    log_pe_inlet_tensor = torch.tensor(log_pe_inlet)
    C_star_inlet = model(log_pe_inlet_tensor, x_star_inlet_tensor, t_star_inlet_tensor)
    inlet_loss = nn.MSELoss()(C_star_inlet, torch.ones_like(C_star_inlet))

    x_star_outlet_tensor = torch.tensor(x_star_outlet)
    t_star_outlet_tensor = torch.tensor(t_star_outlet)
    log_pe_outlet_tensor = torch.tensor(log_pe_outlet)
    C_star_outlet = model(log_pe_outlet_tensor, x_star_outlet_tensor, t_star_outlet_tensor)
    outlet_loss = nn.MSELoss()(C_star_outlet, torch.zeros_like(C_star_outlet))

    total_loss = (weight_pde * pde_loss +
                  weight_ic * ic_loss +
                  weight_inlet_bc * inlet_loss +
                  weight_outlet_bc * outlet_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    losses["total"].append(total_loss.item())
    losses["pde"].append(pde_loss.item())
    losses["ic"].append(ic_loss.item())
    losses["inlet_bc"].append(inlet_loss.item())
    losses["outlet_bc"].append(outlet_loss.item())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        pbar.set_postfix({
            "Loss": f"{losses['total'][-1]:.4e}",
            "PDE": f"{losses['pde'][-1]:.4e}",
            "IC": f"{losses['ic'][-1]:.4e}",
            "Inlet": f"{losses['inlet_bc'][-1]:.4e}",
            "Outlet": f"{losses['outlet_bc'][-1]:.4e}",
        })

pbar.close()

# Save trained model state
model_path = results_dir / "unsupervised_neural_operator_baseline_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# ============================================================================
# Loss plots
# ============================================================================

fig, axes = plt.subplot_mosaic("A;Z;B;C;D;E", figsize=(6, 12))

axes["A"].plot(losses["total"])
axes["Z"].plot(losses["total"], lw=1)
axes["B"].plot(losses["pde"])
axes["C"].plot(losses["ic"])
axes["D"].plot(losses["inlet_bc"])
axes["E"].plot(losses["outlet_bc"])

axes["A"].set_ylabel("total")
axes["Z"].set_ylabel("total")
axes["B"].set_ylabel("pde")
axes["C"].set_ylabel("ic")
axes["D"].set_ylabel("inlet_bc")
axes["E"].set_ylabel("outlet_bc")

for key, ax in axes.items():
    ax.spines[["right", "top"]].set_visible(False)
    if key != "Z":
        ax.set_yscale("log")

plot_path = results_dir / "loss.png"
plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
plot_pdf_path = results_dir / "loss.pdf"
plt.savefig(str(plot_pdf_path), format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Loss plot saved to: {plot_path}")

# ============================================================================
# Concentration profile plots (dimensionless)
# ============================================================================

x_star_plot = np.linspace(0, 1, num_spatial_points)
x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32)

model.eval()
with torch.no_grad():
    for pe_plot in pe_values_to_plot:
        log_pe_plot = np.log(pe_plot)
        log_pe_plot_tensor = torch.tensor([[log_pe_plot]], dtype=torch.float32)

        plt.figure(figsize=(5, 4))
        plt.plot([], [], linewidth=2, linestyle="-", color="black", label="DeepONet")
        plt.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

        for idx, t_star in enumerate(reversed(times_tstar)):
            color = f"C{idx % 8}"

            t_star_plot_tensor = torch.full((num_spatial_points,), t_star, dtype=torch.float32)
            pred = model(log_pe_plot_tensor, x_star_plot_tensor, t_star_plot_tensor)
            C_star = pred.squeeze().cpu().numpy()
            plt.plot(x_star_plot, C_star, linewidth=2, linestyle="-", color=color)

            U_fake = 1.0
            C0_fake = 1.0
            D_fake = 1.0 / pe_plot
            C_analytical = analytical_solution(
                x_star_plot, t_star,
                U_param=U_fake, D_param=D_fake, C_0_param=C0_fake,
            )
            plt.plot(x_star_plot, C_analytical, linewidth=2, linestyle="--", color=color, alpha=0.7)
            plt.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"t* = {t_star}")

        plt.xlabel("x* (dimensionless)", fontsize=12)
        plt.ylabel("C* (dimensionless)", fontsize=12)
        plt.title(f"C* profiles (Pe = {pe_plot:g})", fontsize=12)

        legend = plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.6),
            ncol=4,
            frameon=False,
            fontsize=10,
            labelspacing=0.5,
            columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color("black")
            text.set_alpha(1.0)

        plt.xlim(0, 1)
        plt.ylim(-0.25, 1.1)

        ax_ = plt.gca()
        grid_alpha = 0.3
        grid_color = "black"
        grid_linewidth = 0.4

        ax_.grid(True, axis="x", alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)

        x_ticks = ax_.get_xticks()
        xgridlines = ax_.get_xgridlines()
        indices_to_hide = []
        for i, tick_pos in enumerate(x_ticks):
            if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - 1.0) < 1e-6:
                indices_to_hide.append(i)
        for i in indices_to_hide:
            if i < len(xgridlines):
                xgridlines[i].set_visible(False)

        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)
        ax_.spines["bottom"].set_visible(True)
        ax_.spines["left"].set_visible(True)
        ax_.spines["bottom"].set_color(grid_color)
        ax_.spines["bottom"].set_alpha(grid_alpha)
        ax_.spines["bottom"].set_linewidth(grid_linewidth)
        ax_.spines["left"].set_color(grid_color)
        ax_.spines["left"].set_alpha(grid_alpha)
        ax_.spines["left"].set_linewidth(grid_linewidth)

        ax_.tick_params(axis="x", which="major",
                        colors=grid_color,
                        width=grid_linewidth,
                        length=0)
        ax_.tick_params(axis="y", which="major",
                        colors=grid_color,
                        width=grid_linewidth,
                        length=0)

        for label in ax_.get_xticklabels():
            label.set_color("black")
            label.set_alpha(1.0)
        for label in ax_.get_yticklabels():
            label.set_color("black")
            label.set_alpha(1.0)

        ax_.xaxis.label.set_color("black")
        ax_.xaxis.label.set_alpha(1.0)
        ax_.yaxis.label.set_color("black")
        ax_.yaxis.label.set_alpha(1.0)

        plt.tight_layout()

        plot_path = results_dir / f"{pe_plot} Cstar_profiles.png"
        plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")
        plot_pdf_path = results_dir / f"{pe_plot} Cstar_profiles.pdf"
        plt.savefig(str(plot_pdf_path), format="pdf", bbox_inches="tight")
        print(f"PDF saved to: {plot_pdf_path}")

        plt.close()

print("Done.")
