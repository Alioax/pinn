"""
Plotting script for parametric PINN baseline.

Loads the trained model from results and recreates the plots.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Visualization settings
mpl.rcParams["figure.dpi"] = 800
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#FF5F05", "#13294B", "#009FD4", "#FCB316",
           "#006230", "#007E8E", "#5C0E41", "#7D3E13"]
)

# ============================================================================
# Configuration - keep consistent with training
# ============================================================================

Pe_min = 1
Pe_max = 1e5
t_final_star = 1.0

num_layers = 3
num_neurons = 16
activation_name = "Tanh"

num_spatial_points = 500
plot_dpi = 800
times_tstar = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
pe_values_to_plot = [10, 50, 1e2, 500, 1e3, 1e4, 1e5]

logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)

# ============================================================================
# Model setup (must match training)
# ============================================================================

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

if activation_name not in activation_map:
    raise ValueError(f"Unsupported activation_name: {activation_name}")

activation_cls = activation_map[activation_name]

layers = []
in_features = 3
for i in range(num_layers):
    layers.append(nn.Linear(in_features, num_neurons))
    layers.append(activation_cls())
    in_features = num_neurons
layers.append(nn.Linear(num_neurons, 1))
model = nn.Sequential(*layers)

model_path = Path(__file__).parent / "results" / "pinn_parametric_baseline_model.pt"
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ============================================================================
# Plot Concentration Profiles (dimensionless)
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

x_star_plot = np.linspace(0, 1, num_spatial_points)

for pe_plot in pe_values_to_plot:
    log_pe_plot = np.log(pe_plot)
    plt.figure(figsize=(5, 4))

    for idx, t_star in enumerate(reversed(times_tstar)):
        color = f"C{idx % 8}"

        x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32).reshape(-1, 1)
        t_star_plot_tensor = torch.tensor(t_star, dtype=torch.float32).reshape(-1, 1)
        log_pe_plot_tensor = torch.tensor(log_pe_plot, dtype=torch.float32).reshape(-1, 1)
        if t_star_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
            t_star_plot_tensor = t_star_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)
        if log_pe_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
            log_pe_plot_tensor = log_pe_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)

        with torch.no_grad():
            C_star_pinn = model(torch.cat([x_star_plot_tensor, t_star_plot_tensor, log_pe_plot_tensor], dim=1))
        C_star = C_star_pinn.cpu().numpy()
        plt.plot(x_star_plot, C_star, linewidth=2, linestyle="-", color=color)

    plt.xlabel("x* (dimensionless)", fontsize=12)
    plt.ylabel("C* (dimensionless)", fontsize=12)
    plt.title(f"C* profiles (Pe = {pe_plot:g})", fontsize=12)

    plt.xlim(0, 1)
    plt.ylim(-0.25, 1.1)

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

    ax.tick_params(axis="x", which="major",
                   colors=grid_color,
                   width=grid_linewidth,
                   length=0)
    ax.tick_params(axis="y", which="major",
                   colors=grid_color,
                   width=grid_linewidth,
                   length=0)

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

    plot_path = results_dir / f"{pe_plot} Cstar_profiles.png"
    plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    plot_pdf_path = results_dir / f"{pe_plot} Cstar_profiles.pdf"
    plt.savefig(str(plot_pdf_path), format="pdf", bbox_inches="tight")
    print(f"PDF saved to: {plot_pdf_path}")

    plt.close()
