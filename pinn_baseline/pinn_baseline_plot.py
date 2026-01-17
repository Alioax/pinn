"""
Plotting script for baseline PINN.

Loads the trained model from results and recreates the plots.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

# Import analytical solution from sibling folder
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
# Configuration - keep consistent with training
# ============================================================================

# Physics parameters
U = 0.1
D = 1e-7 * 86400
C0 = 5.0
L = 100.0
T_phys = 1000.0

# Model architecture
num_layers = 3
num_neurons = 16
activation = torch.nn.Tanh

# Plotting parameters
times_days = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
num_spatial_points = 500
plot_dpi = 800

# Derived parameters
T = L / U
Pe = (U * L) / D

# ============================================================================
# Model setup (must match training)
# ============================================================================

layers = []
in_features = 2
for i in range(num_layers):
    layers.append(nn.Linear(in_features, num_neurons))
    layers.append(activation())
    in_features = num_neurons
layers.append(nn.Linear(num_neurons, 1))
model = nn.Sequential(*layers)

model_path = Path(__file__).parent / "results" / "pinn_baseline_model.pt"
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ============================================================================
# Plot Concentration Profiles
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

x_plot = np.linspace(0, L, num_spatial_points)
plt.figure(figsize=(5, 3.5))

plt.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
plt.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

for idx, t_days in enumerate(reversed(times_days)):
    color = f"C{idx % 8}"

    x_star_plot = x_plot / L
    t_star_plot = t_days / T

    x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32).reshape(-1, 1)
    t_star_plot_tensor = torch.tensor(t_star_plot, dtype=torch.float32).reshape(-1, 1)
    if t_star_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
        t_star_plot_tensor = t_star_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)

    with torch.no_grad():
        C_star_pinn = model(torch.cat([x_star_plot_tensor, t_star_plot_tensor], dim=1))
    C_pinn = (C_star_pinn * C0).cpu().numpy()
    plt.plot(x_plot, C_pinn, linewidth=2, linestyle="-", color=color)

    C_analytical = analytical_solution(x_plot, t_days)
    plt.plot(x_plot, C_analytical, linewidth=2, linestyle="--", color=color, alpha=0.7)

    plt.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{t_days} days")

plt.xlabel("Distance x (m)", fontsize=12)
plt.ylabel("Concentration C (kg/m^3)", fontsize=12)

legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.45),
                    ncol=4, frameon=False, fontsize=10,
                    labelspacing=0.5, columnspacing=1.2)
for text in legend.get_texts():
    text.set_color("black")
    text.set_alpha(1.0)

plt.xlim(0, L)
plt.ylim(-0.25, C0 * 1.1)

ax = plt.gca()
grid_alpha = 0.3
grid_color = "black"
grid_linewidth = 0.4

ax.grid(True, axis="x", alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)

x_ticks = ax.get_xticks()
xgridlines = ax.get_xgridlines()
indices_to_hide = []
for i, tick_pos in enumerate(x_ticks):
    if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - L) < 1e-6:
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

plot_path = results_dir / "pinn_baseline_concentration_profiles.png"
plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches="tight")
print(f"Plot saved to: {plot_path}")

pdf_path = results_dir / "pinn_baseline_concentration_profiles.pdf"
plt.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
print(f"PDF copy saved to: {pdf_path}")

plt.close()
