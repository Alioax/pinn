"""
Plot 3D scatter of collocation points for the parametric PINN.

Uses the same configuration and RNG seed as pinn_parametric_baseline.py
so the distribution matches the training setup. Axes: x* (space), t* (time), log Pe.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Match parametric PINN config
Pe_min = 1
Pe_max = 1e5
t_final_star = 1.0
num_collocation = 1000 * 1000
num_ic = 1000
num_bc = 1000

logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)

script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

# Reproducible points (same seed as main script)
np.random.seed(123456789)

# PDE collocation
x_star_pde = np.random.rand(num_collocation, 1).astype(np.float32)
t_star_pde = np.random.rand(num_collocation, 1).astype(np.float32) * t_final_star
log_pe_pde = (
    np.random.rand(num_collocation, 1).astype(np.float32) * (logPe_max - logPe_min)
    + logPe_min
)

# Initial condition (t* = 0)
x_star_ic = np.random.rand(num_ic, 1).astype(np.float32)
t_star_ic = np.zeros((num_ic, 1), dtype=np.float32)
log_pe_ic = (
    np.random.rand(num_ic, 1).astype(np.float32) * (logPe_max - logPe_min) + logPe_min
)

# Inlet (x* = 0)
x_star_inlet = np.zeros((num_bc, 1), dtype=np.float32)
t_star_inlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
log_pe_inlet = (
    np.random.rand(num_bc, 1).astype(np.float32) * (logPe_max - logPe_min) + logPe_min
)

# Outlet (x* = 1)
x_star_outlet = np.ones((num_bc, 1), dtype=np.float32)
t_star_outlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
log_pe_outlet = (
    np.random.rand(num_bc, 1).astype(np.float32) * (logPe_max - logPe_min) + logPe_min
)

# Visualization (match main script style)
mpl.rcParams["figure.dpi"] = 800
plt.rcParams["font.family"] = "Times New Roman"
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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Subsample PDE points for clearer 3D plot (optional; set to 1 to show all)
pde_subsample = max(1, num_collocation // 5000)
ax.scatter(
    x_star_pde[::pde_subsample].ravel(),
    t_star_pde[::pde_subsample].ravel(),
    log_pe_pde[::pde_subsample].ravel(),
    s=2,
    alpha=0.9,
    color="C1",
    label="PDE",
    edgecolors="none",
)
ax.scatter(
    x_star_ic.ravel(),
    t_star_ic.ravel(),
    log_pe_ic.ravel(),
    s=4,
    alpha=0.9,
    color="C0",
    label="Initial Condition",
    edgecolors="none",
)
ax.scatter(
    x_star_inlet.ravel(),
    t_star_inlet.ravel(),
    log_pe_inlet.ravel(),
    s=4,
    alpha=0.9,
    color="C2",
    label="Inlet BC",
    edgecolors="none",
)
ax.scatter(
    x_star_outlet.ravel(),
    t_star_outlet.ravel(),
    log_pe_outlet.ravel(),
    s=4,
    alpha=0.9,
    color="C3",
    label="Outlet BC",
    edgecolors="none",
)

ax.set_xlabel("x* (dimensionless)", fontsize=11)
ax.set_ylabel("t* (dimensionless)", fontsize=11)
ax.set_zlabel("log Pe (Péclet)", fontsize=11, labelpad=8)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    fontsize=9,
    markerscale=3,
    frameon=False,
)
plt.tight_layout(pad=2.0)
# Extra padding so z-axis label is not clipped by figure frame
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)

png_path = results_dir / "collocation_points_3d.png"
pdf_path = results_dir / "collocation_points_3d.pdf"
plt.savefig(str(png_path), dpi=800, bbox_inches="tight", pad_inches=0.35)
print(f"3D collocation plot saved to: {png_path}")
plt.savefig(str(pdf_path), format="pdf", bbox_inches="tight", pad_inches=0.35)
print(f"3D collocation PDF saved to: {pdf_path}")
plt.close()

# 2D version: x* vs t* (ignore Peclet) — match pinn_baseline collocation style
fig2, ax2 = plt.subplots(figsize=(6.5, 4))
ax2.scatter(
    x_star_pde[::pde_subsample].ravel(),
    t_star_pde[::pde_subsample].ravel(),
    s=2,
    alpha=0.9,
    color="C1",
    label="PDE",
    edgecolors="none",
)
ax2.scatter(
    x_star_ic.ravel(),
    t_star_ic.ravel(),
    s=4,
    alpha=0.9,
    color="C0",
    label="Initial Condition",
    edgecolors="none",
)
ax2.scatter(
    x_star_inlet.ravel(),
    t_star_inlet.ravel(),
    s=4,
    alpha=0.9,
    color="C2",
    label="Inlet BC",
    edgecolors="none",
)
ax2.scatter(
    x_star_outlet.ravel(),
    t_star_outlet.ravel(),
    s=4,
    alpha=0.9,
    color="C3",
    label="Outlet BC",
    edgecolors="none",
)
ax2.set_xlabel("x* (dimensionless)", fontsize=12)
ax2.set_ylabel("t* (dimensionless)", fontsize=12)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=4,
    frameon=False,
    fontsize=10,
    markerscale=5,
    handletextpad=0.5,
)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
png_2d = results_dir / "collocation_points_2d.png"
pdf_2d = results_dir / "collocation_points_2d.pdf"
plt.savefig(str(png_2d), dpi=800, bbox_inches="tight", pad_inches=0.35)
print(f"2D collocation plot saved to: {png_2d}")
plt.savefig(str(pdf_2d), format="pdf", bbox_inches="tight", pad_inches=0.35)
print(f"2D collocation PDF saved to: {pdf_2d}")
plt.close()
