"""
Parametric PINN plotting for 1D Advection-Dispersion.

Loads a trained checkpoint and generates concentration profile plots.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Import analytical solution from sibling folder
import sys
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

model_load_name = "pinn_parametric_30000.pt"
model_load_name = "pinn_parametric_20000_800_1e-0_3x16.pt"
model_load_name = "pinn_parametric_30000_200_1e-0_3x16.pt"
times_tstar = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]  # Dimensionless times to plot
num_spatial_points = 500
plot_dpi = 800
pe_values_to_plot = [10, 50, 1e2, 500, 1e3, 1e4, 1e5]

# ============================================================================
# Neural Network Architecture
# ============================================================================

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def get_activation(name):
    activations = {
        "Tanh": torch.nn.Tanh,
        "ReLU": torch.nn.ReLU,
        "SiLU": torch.nn.SiLU,
        "GELU": torch.nn.GELU,
        "ELU": torch.nn.ELU,
        "LeakyReLU": torch.nn.LeakyReLU,
        "Sigmoid": torch.nn.Sigmoid,
        "Softplus": torch.nn.Softplus,
        "Sin": Sine,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation_name: {name}")
    return activations[name]

class PINN(nn.Module):
    """Neural network that takes (x*, t*, log Pe) and outputs dimensionless concentration C*."""

    def __init__(self, num_layers, num_neurons, activation_cls):
        super().__init__()
        layer_sizes = [3] + [num_neurons] * num_layers + [1]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_cls())
        self.net = nn.Sequential(*layers)

        gain = nn.init.calculate_gain("tanh")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x_star, t_star, log_pe):
        inputs = torch.cat([x_star, t_star, log_pe], dim=1)
        return self.net(inputs)

# ============================================================================
# Load Model
# ============================================================================

script_dir = Path(__file__).parent
models_dir = script_dir / "models"
model_load_path = models_dir / model_load_name

if not model_load_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {model_load_path}")

checkpoint = torch.load(model_load_path, map_location="cpu")
config = checkpoint.get("config")
state_dict = checkpoint.get("state_dict")
losses = checkpoint.get("losses")

if config is None or state_dict is None:
    raise ValueError("Checkpoint is missing required keys: 'config' and/or 'state_dict'.")

activation_cls = get_activation(config["activation_name"])
model = PINN(config["num_layers"], config["num_neurons"], activation_cls)
model.load_state_dict(state_dict)
model.eval()

# ============================================================================
# Derived parameters from checkpoint config
# ============================================================================

Pe_min = config["Pe_min"]
Pe_max = config["Pe_max"]
t_final_star = config.get("t_final_star")
if t_final_star is None:
    raise ValueError("Checkpoint config missing t_final_star.")

if max(times_tstar) > t_final_star:
    raise ValueError("times_tstar exceeds t_final_star from checkpoint.")

# ============================================================================
# Output directory
# ============================================================================

plots_dir = script_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Plot Losses (if available)
# ============================================================================

if isinstance(losses, dict) and losses.get("total"):
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

    loss_plot_path = plots_dir / "loss.png"
    plt.savefig(str(loss_plot_path), dpi=plot_dpi, bbox_inches="tight")
    plt.close()

# ============================================================================
# Plot Concentration Profiles
# ============================================================================

x_star_plot = np.linspace(0.0, 1.0, num_spatial_points)

for pe_plot in pe_values_to_plot:
    log_pe_plot = np.log(pe_plot)
    plt.figure(figsize=(5, 4))

    plt.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
    plt.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

    for idx, t_star in enumerate(reversed(times_tstar)):
        color = f"C{idx % 8}"
        t_star_plot = t_star

        x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32).reshape(-1, 1)
        t_star_plot_tensor = torch.tensor(t_star_plot, dtype=torch.float32).reshape(-1, 1)
        log_pe_plot_tensor = torch.tensor(log_pe_plot, dtype=torch.float32).reshape(-1, 1)
        if t_star_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
            t_star_plot_tensor = t_star_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)
        if log_pe_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
            log_pe_plot_tensor = log_pe_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)

        with torch.no_grad():
            C_star_pinn = model(x_star_plot_tensor, t_star_plot_tensor, log_pe_plot_tensor)
        C_pinn = C_star_pinn.cpu().numpy()
        plt.plot(x_star_plot, C_pinn, linewidth=2, linestyle="-", color=color)

        U_fake = 1.0
        C0_fake = 1.0
        D_fake = 1.0 / pe_plot
        C_analytical = analytical_solution(x_star_plot, t_star, U_param=U_fake, D_param=D_fake, C_0_param=C0_fake)
        plt.plot(x_star_plot, C_analytical, linewidth=2, linestyle="--", color=color, alpha=0.7)
        plt.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"t* = {t_star}")

    plt.xlabel("x*", fontsize=12)
    plt.ylabel("C*", fontsize=12)

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
    plt.ylim(-0.1, 1.1)

    ax = plt.gca()
    grid_alpha = 0.3
    grid_color = "black"
    grid_linewidth = 0.4

    ax.grid(True, axis="x", alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)

    x_ticks = ax.get_xticks()
    xgridlines = ax.get_xgridlines()
    indices_to_hide = []
    for i, tick_pos in enumerate(x_ticks):
        if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - 1.0) < 1e-6:
            indices_to_hide.append(i)
    for i in indices_to_hide:
        if i < len(xgridlines):
            xgridlines[i].set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["bottom"].set_alpha(grid_alpha)
    ax.spines["bottom"].set_linewidth(grid_linewidth)
    ax.spines["left"].set_color(grid_color)
    ax.spines["left"].set_alpha(grid_alpha)
    ax.spines["left"].set_linewidth(grid_linewidth)

    ax.tick_params(axis="x", which="major", colors=grid_color, width=grid_linewidth, length=0)
    ax.tick_params(axis="y", which="major", colors=grid_color, width=grid_linewidth, length=0)

    for label in ax.get_xticklabels():
        label.set_color("black")
        label.set_alpha(1.0)
    for label in ax.get_yticklabels():
        label.set_color("black")
        label.set_alpha(1.0)

    ax.xaxis.label.set_color("black")
    ax.xaxis.label.set_alpha(1.0)
    ax.yaxis.label.set_color("black")
    ax.yaxis.label.set_alpha(1.0)

    plt.tight_layout()

    plot_path = plots_dir / f"{pe_plot}_Cstar_profiles.png"
    plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()
