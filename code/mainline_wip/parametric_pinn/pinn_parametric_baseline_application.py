"""
Application script for the parametric PINN baseline.

Loads the trained parametric model, reads the baseline physical parameters,
and produces concentration profiles in physical units.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import ast
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

Pe_min = 1
Pe_max = 1e5
t_final_star = 1.0

num_layers = 4
num_neurons = 16
activation_name = "Tanh"

num_spatial_points = 500
plot_dpi = 800
baseline_params_path = Path(__file__).parent.parent / "pinn_baseline" / "pinn_baseline.py"

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
# Read baseline physical parameters without executing training code.
def _safe_eval_expr(expr):
    node = ast.parse(expr, mode="eval")
    for subnode in ast.walk(node):
        if not isinstance(subnode, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num,
                                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError(f"Unsupported expression: {expr}")
    return eval(compile(node, filename="<ast>", mode="eval"))


def load_baseline_params(path):
    targets = {"U", "D", "C0", "L", "T_phys", "times_days"}
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            left, right = line.split("=", 1)
            key = left.strip()
            if key not in targets:
                continue
            expr = right.split("#", 1)[0].strip()
            if not expr:
                continue
            if key == "times_days":
                values[key] = ast.literal_eval(expr)
            else:
                values[key] = _safe_eval_expr(expr)
    missing = targets - values.keys()
    if missing:
        raise ValueError(f"Missing baseline parameters: {sorted(missing)}")
    return values


baseline_params = load_baseline_params(baseline_params_path)
U = baseline_params["U"]
D = baseline_params["D"]
C0 = baseline_params["C0"]
L = baseline_params["L"]
T_phys = baseline_params["T_phys"]
times_days = baseline_params["times_days"]

T = L / U
Pe = (U * L) / D
t_final_star = T_phys / T
log_pe_plot = np.log(Pe)

if not (Pe_min <= Pe <= Pe_max):
    raise ValueError(f"Baseline Pe={Pe:g} outside trained range [{Pe_min}, {Pe_max}].")

# ============================================================================
# Plot Concentration Profiles (physical units)
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

x_plot = np.linspace(0, L, num_spatial_points)
x_star_plot = x_plot / L

plt.figure(figsize=(4.5, 4))

plt.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
plt.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

for idx, t_days in enumerate(reversed(times_days)):
    color = f"C{idx % 8}"
    t_star = t_days / T

    x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32).reshape(-1, 1)
    t_star_plot_tensor = torch.tensor(t_star, dtype=torch.float32).reshape(-1, 1)
    log_pe_plot_tensor = torch.tensor(log_pe_plot, dtype=torch.float32).reshape(-1, 1)
    if t_star_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
        t_star_plot_tensor = t_star_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)
    if log_pe_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
        log_pe_plot_tensor = log_pe_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)

    with torch.no_grad():
        C_star_pinn = model(torch.cat([x_star_plot_tensor, t_star_plot_tensor, log_pe_plot_tensor], dim=1))
    C_pinn = (C_star_pinn * C0).cpu().numpy()
    plt.plot(x_plot, C_pinn, linewidth=2, linestyle="-", color=color)

    C_analytical = analytical_solution(x_plot, t_days, U_param=U, D_param=D, C_0_param=C0)
    plt.plot(x_plot, C_analytical, linewidth=2, linestyle="--", color=color, alpha=0.7)
    plt.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{t_days} days")

plt.xlabel("Distance x (m)", fontsize=12)
plt.ylabel("Concentration C (kg/m^3)", fontsize=12)

legend = plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4),
                    ncol=3, frameon=False, fontsize=10,
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

plot_pdf_path = results_dir / "pinn_parametric_application_baseline_physical.pdf"
plt.savefig(str(plot_pdf_path), format="pdf", bbox_inches="tight", bbox_extra_artists=(legend,))
print(f"PDF saved to: {plot_pdf_path}")

reports_pdf_dir = Path(r"C:\Research\PINN\Code Base\reports\report 2 - PINN Update\figs")
reports_pdf_dir.mkdir(parents=True, exist_ok=True)
reports_pdf_path = reports_pdf_dir / "pinn_parametric_application_baseline_physical.pdf"
plt.savefig(str(reports_pdf_path), format="pdf", bbox_inches="tight", bbox_extra_artists=(legend,))
print(f"PDF saved to: {reports_pdf_path}")

plt.close()
