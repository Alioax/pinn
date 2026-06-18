# -*- coding: utf-8 -*-
"""
Physics-informed heterogeneous neural operator (DeepONet-style): train + COMSOL validation.

Branch encodes dimensionless zone CFL (CFL1..CFL4) with CFL_i = u_i T_max / L;
trunk maps (x*, t*). PDE residual uses physical u for piecewise CFL(x*).
PDE: dC*/dt* + CFL(x*; u1..u4) dC*/dx* - Pe d2C*/dx*^2 = 0 on four zones (20+20+20+40 m).

Edit the configuration block below, then run this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

_HETERO_ROOT = Path(__file__).resolve().parent.parent
_PINO_DIR = Path(__file__).resolve().parent
for _p in (_HETERO_ROOT, _PINO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deeponet import DeepONetParametric, build_deeponet  # noqa: E402
from utils.comsol_4zones import list_parameter_combos, load_case  # noqa: E402
from utils.grf_sampling import (  # noqa: E402
    GRFTrainingBatch,
    GRF_CORR_LENGTHS_DEFAULT,
    MixedTrainConfig,
    PIECEWISE_ZONE_COUNTS_DEFAULT,
    branch_cfl_from_grf_sensor_u,
    branch_cfl_from_zone_case,
    default_sensor_xstar,
    interpolate_u_xstar,
    load_or_generate_grf_train_cases,
    load_grf_cases_npz,
    zone_u_at_xstar,
)
from utils.lhc_sampling import (  # noqa: E402
    TrainDesign,
    comsol_validation_u_grid,
    load_or_generate_train_u_cases as load_train_u_cases_for_design,
    load_u_cases_csv,
)
from utils.zone_velocity import (  # noqa: E402
    L_DEFAULT,
    PE_DEFAULT,
    T_MAX_DEFAULT,
    XSTAR_TICKS,
    ZONE_INTERFACE_XSTAR,
    cfl_from_u,
    cfl_from_u_np,
    piecewise_cfl_from_branch,
    zone_index_xstar,
    zone_index_xstar_np,
)

# =============================================================================
# Configuration
# =============================================================================
seed = 1234567
torch_dtype = torch.float32

L = L_DEFAULT
T_MAX = T_MAX_DEFAULT
PE = PE_DEFAULT

U_LO = 0.01
U_HI = 0.05
N_LHC_TRAIN = 2500
LHC_EXCLUDE_COMSOL_GRID = True
LHC_EXCLUDE_ATOL = 1e-9

sensor_count = 4
ARCH_PRESETS: dict[str, tuple[list[int], list[int]]] = {
    "default": ([sensor_count, 16, 16, 32], [2, 32, 32, 32, 32]),
    "dense": ([sensor_count, 64, 64, 128], [2, 64, 64, 64, 64, 128]),
    "w64": ([sensor_count, 64, 64, 64], [2, 64, 64, 64, 64]),
}
branch_architecture = ARCH_PRESETS["default"][0]
trunk_architecture = ARCH_PRESETS["default"][1]
activation_cls = nn.Tanh

num_epochs_lbfgs = 1000
lr_lbfgs = 1
lbfgs_max_iter = 1  # PyTorch LBFGS max_iter per outer step
early_stop_patience = 0  # 0 = disabled; stop after N steps with no total-loss improvement

mesh_nx_pde = 50
mesh_nt_pde = 50
mesh_ic_nx = mesh_nx_pde
mesh_bc_nt = mesh_nt_pde

# Extra PDE points in narrow bands around zone interfaces (20, 40, 60 m -> x* 0.2, 0.4, 0.6).
interface_band_half_width = 0.025
mesh_nx_interface_per_band = 12

weight_pde = 1.0
weight_ic = 1.0
weight_inlet_bc = 1.0
weight_outlet_bc = 1.0
weight_interface_c = 1.0
weight_interface_flux = 1.0

# Half-width for left/right interface collocation (zone trunks only).
interface_collocation_eps = 1e-3

save_model = True
collocation_scatter_ms = 3
collocation_xstar_axis_pad = 0.05
n_cycle_colors = 4

times_tstar = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
num_spatial_points = 500
x_plot_max_star = 1.0

PLOT_U_CASES: list[tuple[float, float, float, float]] = [
    (0.01, 0.01, 0.01, 0.01),
    (0.05, 0.05, 0.05, 0.05),
    (0.01, 0.03, 0.05, 0.01),
    (0.05, 0.01, 0.01, 0.01),
    (0.01, 0.01, 0.01, 0.05),
]

run_training = True
run_comsol_validation = True
validate_only = False
reload_lhc_train_cases = False

COMSOL_DATA_PATH = (_HETERO_ROOT / "data" / "comsol_4zones.txt").resolve()
COMSOL_TIMES_DAYS = np.array([100.0, 300.0, 600.0, 900.0, 1200.0], dtype=float)
COMSOL_C_REF = 1.0

RESULTS_DIR = _PINO_DIR / "results"
COMSOL_VALIDATION_DIR = RESULTS_DIR / "comsol_validation"
MODEL_PATH = RESULTS_DIR / "pino_heterogeneous_model.pt"
U_TRAIN_CASES_PATH = RESULTS_DIR / f"lhc_train_u{N_LHC_TRAIN}.csv"

# Set by parse_cli() when the script is invoked with batch flags.
train_design: TrainDesign = "lhc"
n_train_requested = N_LHC_TRAIN
n_corner_anchors = 16
arch_preset = "default"
trunk_mode = "single"
skip_validation_plots = False
batch_mode = False

# GRF (PINO_2) mode — set by --design grf
media_mode: str = "zone"
n_sensors_requested = 100
n_grf_requested = 300
n_piecewise_per_zones = 50
piecewise_zone_counts = PIECEWISE_ZONE_COUNTS_DEFAULT
grf_corr_lengths = GRF_CORR_LENGTHS_DEFAULT
grf_grid_n = 201
min_zone_frac: float | None = None
sensor_xstar: np.ndarray | None = None

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

XSTAR_TICKS_NP = np.array(XSTAR_TICKS, dtype=np.float64)


def _grid_x_only(ax: plt.Axes, *, alpha: float = 0.3) -> None:
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=alpha)
    ax.yaxis.grid(False)


def _apply_xstar_ticks_x(ax: plt.Axes, *, pad: float = 0.0) -> None:
    ax.set_xlim(-pad, 1.0 + pad)
    ax.set_xticks(XSTAR_TICKS_NP)


def _apply_xstar_ticks_y(ax: plt.Axes, *, pad: float = 0.0) -> None:
    ax.set_ylim(-pad, 1.0 + pad)
    ax.set_yticks(XSTAR_TICKS_NP)


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]


def branch_cfl_from_u(u: np.ndarray) -> np.ndarray:
    """Zone CFL branch features (N, 4) from physical u (N, 4) in m/d."""
    return cfl_from_u_np(u, l_m=L, t_max_d=T_MAX)


def load_or_generate_train_u_cases() -> np.ndarray:
    """Training media for the active design, shape (N, 4)."""
    if train_design == "capacity":
        exclude_comsol = False
    elif batch_mode:
        exclude_comsol = True
    else:
        exclude_comsol = LHC_EXCLUDE_COMSOL_GRID
    return load_train_u_cases_for_design(
        train_design,
        n_train_requested,
        U_TRAIN_CASES_PATH,
        u_lo=U_LO,
        u_hi=U_HI,
        seed=seed,
        exclude_comsol_grid=exclude_comsol,
        exclude_atol=LHC_EXCLUDE_ATOL,
        reload=reload_lhc_train_cases,
        n_corner_anchors=n_corner_anchors,
    )


def _pde_collocation_from_u_cases(
    x_1d: np.ndarray,
    t_1d: np.ndarray,
    u_cases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build PDE collocation from 1D x*, t* grids and (N, 4) training velocities."""
    n_u = u_cases.shape[0]
    gx, gt, gu = np.meshgrid(x_1d, t_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    idx = gu.reshape(-1)
    branch = u_cases[idx]
    return (
        gx.reshape(-1),
        gt.reshape(-1),
        branch,
        branch[:, 0],
        branch[:, 1],
        branch[:, 2],
    )


def _interface_band_x_1d() -> np.ndarray:
    """x* samples in bands [xi - w, xi + w] for each zone interface."""
    bands: list[np.ndarray] = []
    for xi in ZONE_INTERFACE_XSTAR:
        lo = max(0.0, float(xi) - interface_band_half_width)
        hi = min(1.0, float(xi) + interface_band_half_width)
        bands.append(np.linspace(lo, hi, mesh_nx_interface_per_band, dtype=np.float64))
    return np.concatenate(bands)


def pde_collocation_x_star_1d() -> np.ndarray:
    """Unique x* used in PDE training: bulk linspace plus interface-band points."""
    x_bulk = np.linspace(0.0, 1.0, mesh_nx_pde, dtype=np.float64)
    return np.unique(np.concatenate([x_bulk, _interface_band_x_1d()]))


def comsol_c_star_on_x_star(
    x_star: np.ndarray,
    x_m_comsol: np.ndarray,
    c_comsol: np.ndarray,
) -> np.ndarray:
    """Interpolate COMSOL C onto a target x* grid (dimensionless)."""
    x_star_comsol = x_m_comsol / L
    c_star = c_comsol / COMSOL_C_REF
    return np.interp(x_star, x_star_comsol, c_star)


def branch_cfl_vector_for_validation(
    u_case: tuple[float, float, float, float],
) -> np.ndarray:
    """Branch CFL features for COMSOL validation (zone or GRF sensor encoding)."""
    if media_mode == "grf":
        if sensor_xstar is None:
            raise RuntimeError("sensor_xstar not set for GRF validation")
        return branch_cfl_from_zone_case(u_case, sensor_xstar, l_m=L, t_max_d=T_MAX)
    physical = np.array([list(u_case)], dtype=np.float64).reshape(1, 4)
    return branch_cfl_from_u(physical).flatten()


def branch_tensor_from_u_case(
    u_case: tuple[float, float, float, float],
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cfl_branch = branch_cfl_vector_for_validation(u_case).reshape(1, -1)
    return torch.tensor(
        np.repeat(cfl_branch, n, axis=0), dtype=dtype, device=device
    )


def u_case_tag(u_case: tuple[float, float, float, float]) -> str:
    return "_".join(f"{v:g}".replace(".", "p") for v in u_case)


def _match_comsol_times(
    series: dict[float, np.ndarray], requested: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    for target in requested:
        hit = None
        for t_key, values in series.items():
            if np.isclose(float(t_key), float(target), rtol=0.0, atol=1e-10):
                hit = (float(t_key), values)
                break
        if hit is None:
            raise KeyError(f"No COMSOL slice at t={target} d.")
        out.append(hit)
    return out


def _l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref)
    return float(num / den) if den > 0 else float(num)


def _pde_residual_1d(
    model: DeepONetParametric,
    x_star: np.ndarray,
    t_star: float,
    u_case: tuple[float, float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """PDE residual r = C_t + CFL(x*) C_x - Pe C_xx at fixed t* (same as training)."""
    n_x = x_star.size
    x_t = torch.tensor(
        x_star.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
    )
    t_t = torch.full((n_x, 1), t_star, dtype=dtype, device=device, requires_grad=True)
    branch = branch_tensor_from_u_case(u_case, n_x, device=device, dtype=dtype)
    c = model(x_t, t_t, branch)
    dC_dt = gradients(c, t_t)
    dC_dx = gradients(c, x_t)
    d2C_dx2 = gradients(dC_dx, x_t)
    if media_mode == "grf":
        u_np = zone_u_at_xstar(x_star, u_case, l_m=L)
        u_phys = torch.tensor(u_np.reshape(-1, 1), dtype=dtype, device=device)
        cfl_local = cfl_from_u(u_phys, l_m=L, t_max_d=T_MAX)
    else:
        u_phys = torch.tensor(
            np.tile(np.array(u_case, dtype=np.float64), (n_x, 1)),
            dtype=dtype,
            device=device,
        )
        cfl_local = piecewise_cfl_from_branch(x_t, u_phys, l_m=L, t_max_d=T_MAX)
    residual = dC_dt + cfl_local * dC_dx - PE * d2C_dx2
    return residual.detach().cpu().numpy().flatten()


def comsol_case_mean_l2(
    *,
    model: DeepONetParametric,
    device: torch.device,
    dtype: torch.dtype,
    x_m: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    u_case: tuple[float, float, float, float],
    x_star: np.ndarray | None = None,
) -> float:
    """Mean relative L2 of C* vs COMSOL over the five reference times."""
    if x_star is None:
        x_star = pde_collocation_x_star_1d()
    n_x = x_star.size
    x_t = torch.tensor(x_star.reshape(-1, 1), dtype=dtype, device=device)
    branch = branch_tensor_from_u_case(u_case, n_x, device=device, dtype=dtype)

    l2_sum = 0.0
    matched = _match_comsol_times(comsol_by_time, COMSOL_TIMES_DAYS)
    model.eval()
    for t_days, c_comsol in matched:
        t_star = t_days / T_MAX
        t_t = torch.full((n_x, 1), t_star, dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(x_t, t_t, branch).cpu().numpy().flatten()
        c_ref = comsol_c_star_on_x_star(x_star, x_m, c_comsol)
        l2_sum += _l2_rel(c_pred, c_ref)
    return l2_sum / len(matched)


def plot_comsol_case(
    *,
    model: DeepONetParametric,
    device: torch.device,
    dtype: torch.dtype,
    x_m: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    u_case: tuple[float, float, float, float],
    out_png: Path,
    x_star: np.ndarray | None = None,
) -> float:
    if x_star is None:
        x_star = pde_collocation_x_star_1d()
    n_x = x_star.size
    x_t = torch.tensor(x_star.reshape(-1, 1), dtype=dtype, device=device)
    branch = branch_tensor_from_u_case(u_case, n_x, device=device, dtype=dtype)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pde_lw = 1.0
    pde_alpha = 1.0
    pde_ylim = (-0.5, 0.5)
    fig, (ax, ax_pde) = plt.subplots(
        2,
        1,
        figsize=(8, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("none")
    for a in (ax, ax_pde):
        a.set_facecolor("none")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label="PINO")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="COMSOL")
    ax_pde.plot(
        [],
        [],
        linewidth=pde_lw,
        linestyle="-",
        color="gray",
        alpha=pde_alpha,
        label=r"PDE residual",
    )

    matched = _match_comsol_times(comsol_by_time, COMSOL_TIMES_DAYS)
    model.eval()
    l2_sum = 0.0
    for idx, (t_days, c_comsol) in enumerate(matched):
        t_star = t_days / T_MAX
        t_t = torch.full((n_x, 1), t_star, dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(x_t, t_t, branch).cpu().numpy().flatten()
        c_ref = comsol_c_star_on_x_star(x_star, x_m, c_comsol)
        l2_sum += _l2_rel(c_pred, c_ref)
        color = colors[idx % len(colors)]
        ax.plot(x_star, c_pred, linewidth=2, linestyle="-", color=color)
        ax.plot(x_star, c_ref, linewidth=2, linestyle="--", color=color, alpha=0.7)
        ax.plot(
            [],
            [],
            marker="s",
            markersize=7,
            linestyle="None",
            color=color,
            label=rf"$t$={t_days:.0f} d",
        )
        with torch.enable_grad():
            r_pde = _pde_residual_1d(
                model,
                x_star,
                t_star,
                u_case,
                device=device,
                dtype=dtype,
            )
        ax_pde.plot(
            x_star,
            r_pde,
            linewidth=pde_lw,
            linestyle="-",
            color=color,
            alpha=pde_alpha,
        )

    mean_l2 = l2_sum / len(matched)
    u1, u2, u3, u4 = u_case
    ax.set_title(rf"$u=({u1:g},{u2:g},{u3:g},{u4:g})$ m/d, mean rel. $L_2$={mean_l2:.3e}")
    ax.set_ylabel(r"$C^*$")
    ax_pde.set_xlabel(r"$x^*$")
    ax_pde.set_ylabel(r"PDE residual")
    ax_pde.set_ylim(*pde_ylim)
    _apply_xstar_ticks_x(ax_pde)
    _grid_x_only(ax)
    _grid_x_only(ax_pde)
    ax.tick_params(labelbottom=False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_pde.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper right",
        frameon=False,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return mean_l2


def validate_against_comsol(
    model: DeepONetParametric,
    device: torch.device,
    dtype: torch.dtype,
    *,
    data_file: Path | None = None,
    out_dir: Path | None = None,
    save_plots: bool = True,
) -> list[dict[str, object]]:
    if data_file is None:
        data_file = COMSOL_DATA_PATH
    if out_dir is None:
        out_dir = COMSOL_VALIDATION_DIR
    if not data_file.is_file():
        raise FileNotFoundError(f"COMSOL data not found: {data_file}")

    u_cases = list_parameter_combos(data_file)
    x_star_val = pde_collocation_x_star_1d()
    print(
        f"COMSOL validation: {len(u_cases)} media -> {out_dir}  "
        f"(x* collocation n={x_star_val.size})"
    )

    rows: list[dict[str, object]] = []
    for u_case in tqdm(u_cases, desc="COMSOL vs PINO"):
        x_m, comsol_by_time = load_case(data_file, u_case)
        out_png = out_dir / f"compare_comsol_pino_{u_case_tag(u_case)}.png"
        if save_plots:
            mean_l2 = plot_comsol_case(
                model=model,
                device=device,
                dtype=dtype,
                x_m=x_m,
                comsol_by_time=comsol_by_time,
                u_case=u_case,
                out_png=out_png,
                x_star=x_star_val,
            )
            plot_path = str(out_png)
        else:
            mean_l2 = comsol_case_mean_l2(
                model=model,
                device=device,
                dtype=dtype,
                x_m=x_m,
                comsol_by_time=comsol_by_time,
                u_case=u_case,
                x_star=x_star_val,
            )
            plot_path = ""
        rows.append(
            {
                "u1": u_case[0],
                "u2": u_case[1],
                "u3": u_case[2],
                "u4": u_case[3],
                "mean_rel_l2": mean_l2,
                "plot": plot_path,
            }
        )

    summary_path = out_dir / "comsol_validation_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["u1", "u2", "u3", "u4", "mean_rel_l2", "plot"]
        )
        writer.writeheader()
        writer.writerows(rows)

    l2_vals = np.array([float(r["mean_rel_l2"]) for r in rows], dtype=np.float64)
    print(
        f"COMSOL validation summary: n={len(l2_vals)}  "
        f"mean={l2_vals.mean():.6e}  max={l2_vals.max():.6e}  min={l2_vals.min():.6e}"
    )
    print(f"Wrote {summary_path}")
    return rows


def zone_local_cfl_from_u(
    x_star: torch.Tensor,
    u_phys: torch.Tensor,
) -> torch.Tensor:
    """Zone-constant CFL at each collocation point (zone trunks / per-zone PDE)."""
    cfl_all = cfl_from_u(u_phys, l_m=L, t_max_d=T_MAX)
    zidx = zone_index_xstar(x_star).long().view(-1)
    return cfl_all.gather(1, zidx.unsqueeze(1))


def build_collocation_tensors(
    device: torch.device,
    dtype: torch.dtype,
    u_cases: np.ndarray,
    *,
    include_interface: bool = False,
):
    tf = float(T_MAX / T_MAX)

    x_1d = np.linspace(0.0, 1.0, mesh_nx_pde)
    t_1d = np.linspace(0.0, tf, mesh_nt_pde)
    (
        x_bulk_np,
        t_bulk_np,
        u_bulk_np,
        u1_bulk_np,
        u2_bulk_np,
        u3_bulk_np,
    ) = _pde_collocation_from_u_cases(x_1d, t_1d, u_cases)

    x_if_1d = _interface_band_x_1d()
    (
        x_if_np,
        t_if_np,
        u_if_np,
        u1_if_np,
        _u2_if_np,
        _u3_if_np,
    ) = _pde_collocation_from_u_cases(x_if_1d, t_1d, u_cases)

    x_star_pde_np = np.concatenate([x_bulk_np, x_if_np])
    t_star_pde_np = np.concatenate([t_bulk_np, t_if_np])
    branch_pde_np = np.concatenate([u_bulk_np, u_if_np])
    u1_pde_np = np.concatenate([u1_bulk_np, u1_if_np])
    u2_pde_np = np.concatenate([u2_bulk_np, _u2_if_np])
    u3_pde_np = np.concatenate([u3_bulk_np, _u3_if_np])
    u4_pde_np = branch_pde_np[:, 3]

    n_bulk = x_bulk_np.size
    n_interface = x_if_np.size
    is_interface_pde_np = np.concatenate(
        [
            np.zeros(n_bulk, dtype=bool),
            np.ones(n_interface, dtype=bool),
        ]
    )

    zone_pde_np = zone_index_xstar_np(x_star_pde_np, l_m=L)
    pde_colors = [f"C{int(i) % n_cycle_colors}" for i in zone_pde_np]

    n_u = u_cases.shape[0]
    x_ic_1d = np.linspace(0.0, 1.0, mesh_ic_nx)
    gxi, gui = np.meshgrid(x_ic_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    x_star_ic_np = gxi.reshape(-1)
    t_star_ic_np = np.zeros_like(x_star_ic_np)
    branch_ic_np = u_cases[gui.reshape(-1)]

    t_bc_1d = np.linspace(0.0, tf, mesh_bc_nt)
    gtb, gub = np.meshgrid(t_bc_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    t_star_inlet_np = gtb.reshape(-1)
    branch_in_np = u_cases[gub.reshape(-1)]
    x_star_inlet_np = np.zeros_like(t_star_inlet_np)
    x_star_outlet_np = np.ones_like(t_star_inlet_np)
    t_star_outlet_np = t_star_inlet_np.copy()
    branch_out_np = branch_in_np.copy()

    target_tail = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    i_ref = int(np.argmin(np.linalg.norm(u_cases[:, 1:] - target_tail, axis=1)))
    u_ref = u_cases[i_ref]
    slice_mask = np.all(
        np.isclose(branch_pde_np, u_ref, rtol=0.0, atol=1e-12), axis=1
    )
    if not np.any(slice_mask):
        slice_mask = np.ones_like(x_star_pde_np, dtype=bool)

    tensors: dict[str, torch.Tensor] = {
        "x_pde": torch.tensor(
            x_star_pde_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
        ),
        "t_pde": torch.tensor(
            t_star_pde_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
        ),
        "branch_pde": torch.tensor(
            branch_cfl_from_u(branch_pde_np), dtype=dtype, device=device
        ),
        "u_pde": torch.tensor(branch_pde_np, dtype=dtype, device=device),
        "x_ic": torch.tensor(x_star_ic_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_ic": torch.tensor(t_star_ic_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_ic": torch.tensor(
            branch_cfl_from_u(branch_ic_np), dtype=dtype, device=device
        ),
        "x_in": torch.tensor(x_star_inlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_in": torch.tensor(t_star_inlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_in": torch.tensor(
            branch_cfl_from_u(branch_in_np), dtype=dtype, device=device
        ),
        "x_out": torch.tensor(x_star_outlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_out": torch.tensor(t_star_outlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_out": torch.tensor(
            branch_cfl_from_u(branch_out_np), dtype=dtype, device=device
        ),
    }

    if include_interface:
        eps = interface_collocation_eps
        x_left_parts: list[np.ndarray] = []
        x_right_parts: list[np.ndarray] = []
        t_if_parts: list[np.ndarray] = []
        u_if_parts: list[np.ndarray] = []
        z_left_parts: list[np.ndarray] = []
        for i_if, xi in enumerate(ZONE_INTERFACE_XSTAR):
            x_l = max(0.0, float(xi) - eps)
            x_r = min(1.0, float(xi) + eps)
            gtb, gub = np.meshgrid(t_bc_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
            t_flat = gtb.reshape(-1)
            u_flat = u_cases[gub.reshape(-1)]
            n_pts = t_flat.size
            x_left_parts.append(np.full(n_pts, x_l, dtype=np.float64))
            x_right_parts.append(np.full(n_pts, x_r, dtype=np.float64))
            t_if_parts.append(t_flat)
            u_if_parts.append(u_flat)
            z_left_parts.append(np.full(n_pts, i_if, dtype=np.int64))

        x_if_l_np = np.concatenate(x_left_parts)
        x_if_r_np = np.concatenate(x_right_parts)
        t_if_np = np.concatenate(t_if_parts)
        u_if_np = np.concatenate(u_if_parts)
        z_if_left_np = np.concatenate(z_left_parts)
        tensors.update(
            {
                "x_if_l": torch.tensor(
                    x_if_l_np.reshape(-1, 1),
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                ),
                "x_if_r": torch.tensor(
                    x_if_r_np.reshape(-1, 1),
                    dtype=dtype,
                    device=device,
                    requires_grad=True,
                ),
                "t_if": torch.tensor(
                    t_if_np.reshape(-1, 1), dtype=dtype, device=device
                ),
                "branch_if": torch.tensor(
                    branch_cfl_from_u(u_if_np), dtype=dtype, device=device
                ),
                "u_if": torch.tensor(u_if_np, dtype=dtype, device=device),
                "z_if_left": torch.tensor(
                    z_if_left_np, dtype=torch.long, device=device
                ),
            }
        )

    plot_data = {
        "x_star_pde_np": x_star_pde_np,
        "t_star_pde_np": t_star_pde_np,
        "u1_pde_np": u1_pde_np,
        "pde_colors": pde_colors,
        "slice_mask": slice_mask,
        "u_ref_case": u_ref,
        "is_interface_pde_np": is_interface_pde_np,
        "n_pde_bulk": n_bulk,
        "n_pde_interface": n_interface,
        "n_pde": x_star_pde_np.size,
    }
    return tensors, plot_data


def _pde_collocation_from_grf_cases(
    x_1d: np.ndarray,
    t_1d: np.ndarray,
    batch: GRFTrainingBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build PDE collocation from 1D x*, t* grids and GRF training batch."""
    n_u = batch.sensor_u.shape[0]
    gx, gt, gu = np.meshgrid(x_1d, t_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    case_idx = gu.reshape(-1)
    x_flat = gx.reshape(-1)
    t_flat = gt.reshape(-1)
    grid_rows = batch.grid_u[case_idx]
    u_interp = np.empty(x_flat.size, dtype=np.float64)
    for i, (xi, row) in enumerate(zip(x_flat, grid_rows, strict=True)):
        u_interp[i] = interpolate_u_xstar(xi, batch.grid_x, row)
    return x_flat, t_flat, case_idx, u_interp.reshape(-1, 1)


def build_grf_collocation_tensors(
    device: torch.device,
    dtype: torch.dtype,
    batch: GRFTrainingBatch,
):
    """Collocation tensors for GRF training (bulk PDE mesh only, no interface bands)."""
    tf = float(T_MAX / T_MAX)
    x_1d = np.linspace(0.0, 1.0, mesh_nx_pde)
    t_1d = np.linspace(0.0, tf, mesh_nt_pde)

    x_star_pde_np, t_star_pde_np, case_idx_pde, u_pde_np = _pde_collocation_from_grf_cases(
        x_1d, t_1d, batch
    )
    sensor_pde_np = batch.sensor_u[case_idx_pde]

    zone_pde_np = zone_index_xstar_np(x_star_pde_np, l_m=L)
    pde_colors = [f"C{int(i) % n_cycle_colors}" for i in zone_pde_np]

    n_u = batch.sensor_u.shape[0]
    x_ic_1d = np.linspace(0.0, 1.0, mesh_ic_nx)
    gxi, gui = np.meshgrid(x_ic_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    x_star_ic_np = gxi.reshape(-1)
    t_star_ic_np = np.zeros_like(x_star_ic_np)
    case_ic_np = gui.reshape(-1)
    sensor_ic_np = batch.sensor_u[case_ic_np]

    t_bc_1d = np.linspace(0.0, tf, mesh_bc_nt)
    gtb, gub = np.meshgrid(t_bc_1d, np.arange(n_u, dtype=np.int64), indexing="ij")
    t_star_inlet_np = gtb.reshape(-1)
    case_bc_np = gub.reshape(-1)
    sensor_bc_np = batch.sensor_u[case_bc_np]
    x_star_inlet_np = np.zeros_like(t_star_inlet_np)
    x_star_outlet_np = np.ones_like(t_star_inlet_np)
    t_star_outlet_np = t_star_inlet_np.copy()

    i_ref = 0
    slice_mask = case_idx_pde == i_ref
    if not np.any(slice_mask):
        slice_mask = np.ones_like(x_star_pde_np, dtype=bool)
    u_ref_sensors = batch.sensor_u[i_ref]
    k_show = min(4, u_ref_sensors.size)
    u_ref_display = tuple(float(v) for v in u_ref_sensors[:k_show])

    tensors: dict[str, torch.Tensor] = {
        "x_pde": torch.tensor(
            x_star_pde_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
        ),
        "t_pde": torch.tensor(
            t_star_pde_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
        ),
        "branch_pde": torch.tensor(
            branch_cfl_from_grf_sensor_u(sensor_pde_np), dtype=dtype, device=device
        ),
        "u_pde": torch.tensor(u_pde_np, dtype=dtype, device=device),
        "x_ic": torch.tensor(x_star_ic_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_ic": torch.tensor(t_star_ic_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_ic": torch.tensor(
            branch_cfl_from_grf_sensor_u(sensor_ic_np), dtype=dtype, device=device
        ),
        "x_in": torch.tensor(x_star_inlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_in": torch.tensor(t_star_inlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_in": torch.tensor(
            branch_cfl_from_grf_sensor_u(sensor_bc_np), dtype=dtype, device=device
        ),
        "x_out": torch.tensor(x_star_outlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "t_out": torch.tensor(t_star_outlet_np.reshape(-1, 1), dtype=dtype, device=device),
        "branch_out": torch.tensor(
            branch_cfl_from_grf_sensor_u(sensor_bc_np), dtype=dtype, device=device
        ),
    }

    plot_data = {
        "x_star_pde_np": x_star_pde_np,
        "t_star_pde_np": t_star_pde_np,
        "u1_pde_np": sensor_pde_np[:, 0],
        "pde_colors": pde_colors,
        "slice_mask": slice_mask,
        "u_ref_case": u_ref_display,
        "is_interface_pde_np": np.zeros(x_star_pde_np.size, dtype=bool),
        "n_pde_bulk": x_star_pde_np.size,
        "n_pde_interface": 0,
        "n_pde": x_star_pde_np.size,
        "grf_mode": True,
    }
    return tensors, plot_data


def compute_interface_losses(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """C and flux continuity at zone interfaces (zone-trunk models)."""
    x_l = tensors["x_if_l"]
    x_r = tensors["x_if_r"]
    t = tensors["t_if"]
    branch = tensors["branch_if"]
    u_phys = tensors["u_if"]
    z_left = tensors["z_if_left"]

    c_l = model(x_l, t, branch)
    c_r = model(x_r, t, branch)
    continuity_loss = torch.mean((c_l - c_r) ** 2)

    dC_dx_l = gradients(c_l, x_l)
    dC_dx_r = gradients(c_r, x_r)
    cfl_all = cfl_from_u(u_phys, l_m=L, t_max_d=T_MAX)
    cfl_l = cfl_all.gather(1, z_left.unsqueeze(1))
    cfl_r = cfl_all.gather(1, (z_left + 1).unsqueeze(1))
    flux_l = cfl_l * c_l - PE * dC_dx_l
    flux_r = cfl_r * c_r - PE * dC_dx_r
    flux_loss = torch.mean((flux_l - flux_r) ** 2)
    return continuity_loss, flux_loss


def compute_physics_loss(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    use_zone_pde: bool = False,
    use_grf_pde: bool = False,
    use_interface_losses: bool = False,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    c_pde = model(tensors["x_pde"], tensors["t_pde"], tensors["branch_pde"])
    dC_dt = gradients(c_pde, tensors["t_pde"])
    dC_dx = gradients(c_pde, tensors["x_pde"])
    d2C_dx2 = gradients(dC_dx, tensors["x_pde"])
    if use_zone_pde:
        cfl_local = zone_local_cfl_from_u(tensors["x_pde"], tensors["u_pde"])
    elif use_grf_pde:
        cfl_local = cfl_from_u(tensors["u_pde"], l_m=L, t_max_d=T_MAX)
    else:
        cfl_local = piecewise_cfl_from_branch(
            tensors["x_pde"], tensors["u_pde"], l_m=L, t_max_d=T_MAX
        )
    residual = dC_dt + cfl_local * dC_dx - PE * d2C_dx2
    pde_loss = torch.mean(residual**2)
    ic_loss = torch.mean(
        (model(tensors["x_ic"], tensors["t_ic"], tensors["branch_ic"]) - 0.0) ** 2
    )
    inlet_loss = torch.mean(
        (model(tensors["x_in"], tensors["t_in"], tensors["branch_in"]) - 1.0) ** 2
    )
    outlet_loss = torch.mean(
        (model(tensors["x_out"], tensors["t_out"], tensors["branch_out"]) - 0.0) ** 2
    )

    interface_c_loss = torch.tensor(0.0, dtype=pde_loss.dtype, device=pde_loss.device)
    interface_flux_loss = torch.tensor(
        0.0, dtype=pde_loss.dtype, device=pde_loss.device
    )
    if use_interface_losses:
        interface_c_loss, interface_flux_loss = compute_interface_losses(model, tensors)

    total_loss = (
        weight_pde * pde_loss
        + weight_ic * ic_loss
        + weight_inlet_bc * inlet_loss
        + weight_outlet_bc * outlet_loss
        + weight_interface_c * interface_c_loss
        + weight_interface_flux * interface_flux_loss
    )
    metrics = (
        total_loss.item(),
        pde_loss.item(),
        ic_loss.item(),
        inlet_loss.item(),
        outlet_loss.item(),
        interface_c_loss.item(),
        interface_flux_loss.item(),
    )
    return total_loss, metrics


def _loss_improved(current: float, best: float, *, rtol: float, atol: float) -> bool:
    if not np.isfinite(best):
        return True
    margin = atol + rtol * max(abs(best), 1.0)
    return current < best - margin


def train_model(
    model: nn.Module,
    device: torch.device,
    tensors: dict[str, torch.Tensor],
    *,
    use_zone_pde: bool = False,
    use_grf_pde: bool = False,
    use_interface_losses: bool = False,
) -> tuple[list[list[float]], dict[str, object]]:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    print(optimizer)
    if early_stop_patience > 0:
        stop_rtol = 1e-12 if torch_dtype == torch.float64 else 1e-8
        print(
            f"Early stopping: patience={early_stop_patience} L-BFGS steps "
            f"(no total-loss improvement, rtol={stop_rtol:g})"
        )
    else:
        stop_rtol = 0.0

    history: list[list[float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    early_stopped = False

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = compute_physics_loss(
            model,
            tensors,
            use_zone_pde=use_zone_pde,
            use_grf_pde=use_grf_pde,
            use_interface_losses=use_interface_losses,
        )
        total_loss.backward()
        closure.latest = metrics
        return total_loss

    t_bar = trange(
        num_epochs_lbfgs,
        desc="L-BFGS",
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{remaining} > {elapsed}]"
        ),
    )
    for epoch in t_bar:
        model.train()
        optimizer.step(closure)
        history.append(list(closure.latest))
        total_loss = float(closure.latest[0])
        if _loss_improved(total_loss, best_loss, rtol=stop_rtol, atol=0.0):
            best_loss = total_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
                early_stopped = True
                t_bar.set_postfix_str("early stop", refresh=False)
                print(
                    f"\nEarly stopping at L-BFGS step {epoch + 1}/{num_epochs_lbfgs}: "
                    f"total loss unchanged for {early_stop_patience} steps "
                    f"(best={best_loss:.6e})."
                )
                break
    t_bar.close()

    summary: dict[str, object] = {
        "num_epochs_lbfgs_requested": num_epochs_lbfgs,
        "num_epochs_lbfgs_actual": len(history),
        "early_stop_patience": early_stop_patience,
        "early_stopped": early_stopped,
        "best_total_loss": best_loss,
    }
    return history, summary


def plot_collocation_points(plot_data: dict, *, n_train_media: int) -> Path:
    """Write collocation mesh figure (no model / training required)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ms = collocation_scatter_ms

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    sm = plot_data["slice_mask"]
    axes2[0].scatter(
        plot_data["x_star_pde_np"][sm],
        plot_data["t_star_pde_np"][sm],
        c=[plot_data["pde_colors"][i] for i in np.where(sm)[0]],
        s=ms,
        alpha=0.6,
    )
    u_ref = plot_data["u_ref_case"]
    if plot_data.get("grf_mode"):
        ref_title = (
            rf"PDE: $x^*$ vs $t^*$ (GRF case 0), "
            rf"sensor $u_1..u_{len(u_ref)}$=({', '.join(f'{v:g}' for v in u_ref)}) m/d"
        )
    else:
        ref_title = (
            rf"PDE: $x^*$ vs $t^*$ (one LHC medium), "
            rf"$u=({u_ref[0]:g},{u_ref[1]:g},{u_ref[2]:g},{u_ref[3]:g})$ m/d"
        )
    axes2[0].set_title(ref_title)
    axes2[0].set_xlabel(r"$x^*$")
    axes2[0].set_ylabel(r"$t^* = t/T_{\max}$")
    _apply_xstar_ticks_x(axes2[0], pad=collocation_xstar_axis_pad)
    _grid_x_only(axes2[0])
    axes2[1].scatter(
        plot_data["u1_pde_np"][sm],
        plot_data["x_star_pde_np"][sm],
        c=[plot_data["pde_colors"][i] for i in np.where(sm)[0]],
        s=ms,
        alpha=0.6,
    )
    axes2[1].set_xlabel(r"$u_1$ (m/d)")
    axes2[1].set_ylabel(r"$x^*$")
    axes2[1].set_title(r"Parameter slice: $u_1$ vs $x^*$ (color by zone)")
    _apply_xstar_ticks_y(axes2[1], pad=collocation_xstar_axis_pad)
    _grid_x_only(axes2[1])
    fig2.suptitle(
        f"Training collocation ({n_train_media} media, one slice shown)",
        fontsize=14,
    )
    coll_path = RESULTS_DIR / "pino_heterogeneous_collocation_points.png"
    plt.savefig(coll_path, bbox_inches="tight")
    plt.close(fig2)
    return coll_path


def plot_training_figures(
    model: DeepONetParametric,
    device: torch.device,
    dtype: torch.dtype,
    history: list[list[float]],
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_plot = np.linspace(0.0, x_plot_max_star, num_spatial_points, dtype=np.float64)
    x_plot_t = torch.tensor(x_plot.reshape(-1, 1), dtype=dtype, device=device)
    n_panels = len(PLOT_U_CASES)
    nrows = int(np.ceil(n_panels / 2))
    ncols = 2 if n_panels > 1 else 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), constrained_layout=True
    )
    axes_arr = np.array(axes).reshape(-1)
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    model.eval()
    for idx, u_case in enumerate(PLOT_U_CASES):
        ax = axes_arr[idx]
        u1, u2, u3, u4 = u_case
        branch_eval = branch_tensor_from_u_case(
            (u1, u2, u3, u4), len(x_plot), device=device, dtype=dtype
        )
        for j, t_star in enumerate(times_tstar):
            color = colors_cycle[j % len(colors_cycle)]
            t_t = torch.full((len(x_plot), 1), float(t_star), dtype=dtype, device=device)
            with torch.no_grad():
                c_op = model(x_plot_t, t_t, branch_eval).cpu().numpy().flatten()
            ax.plot(x_plot, c_op, linewidth=2, linestyle="-", color=color)
        ax.set_title(rf"$u=({u1:g},{u2:g},{u3:g},{u4:g})$ m/d")
        ax.set_xlim(0.0, x_plot_max_star)
        ax.set_xlabel(r"$x^*$")
        ax.set_ylabel(r"$C^*$")
        ax.grid(True, alpha=0.3)
        for j, t_star in enumerate(times_tstar):
            color = colors_cycle[j % len(colors_cycle)]
            ax.plot(
                [],
                [],
                marker="s",
                markersize=7,
                linestyle="None",
                color=color,
                label=rf"$t^*$={t_star:g}",
            )
        ax.plot([], [], linewidth=2, linestyle="-", color="black", label="Operator")
        legend = ax.legend(loc="upper right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color("black")
            text.set_alpha(1.0)

    for k in range(len(PLOT_U_CASES), len(axes_arr)):
        axes_arr[k].axis("off")
    fig.suptitle("Heterogeneous PINO: concentration profiles", fontsize=14)
    conc_path = RESULTS_DIR / "pino_heterogeneous_concentration.png"
    plt.savefig(conc_path, bbox_inches="tight")
    plt.close(fig)

    fig3, ax3 = plt.subplots(figsize=(8, 6), constrained_layout=True)
    epochs = np.arange(1, len(history) + 1)
    h = np.array(history, dtype=np.float64)
    ax3.plot(epochs, h[:, 0], color="black", linewidth=1.6, label="Total")
    ax3.plot(epochs, h[:, 1], color="goldenrod", linewidth=1.0, alpha=0.85, label="PDE")
    ax3.plot(epochs, h[:, 2], color="gray", linewidth=1.0, alpha=0.6, label="IC")
    ax3.plot(epochs, h[:, 3], color="purple", linewidth=1.0, alpha=0.75, label="Inlet")
    ax3.plot(epochs, h[:, 4], color="crimson", linewidth=1.0, alpha=0.75, label="Outlet")
    if h.shape[1] > 5 and (np.any(h[:, 5] > 0) or np.any(h[:, 6] > 0)):
        ax3.plot(
            epochs,
            h[:, 5],
            color="teal",
            linewidth=1.0,
            alpha=0.75,
            label="Interface C",
        )
        ax3.plot(
            epochs,
            h[:, 6],
            color="navy",
            linewidth=1.0,
            alpha=0.75,
            label="Interface flux",
        )
    ax3.set_yscale("log")
    ax3.set_xlabel("L-BFGS step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9, frameon=False)
    loss_path = RESULTS_DIR / "pino_heterogeneous_loss.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close(fig3)
    return conc_path, loss_path


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-informed heterogeneous PINO: train and COMSOL validation."
    )
    parser.add_argument(
        "--design",
        choices=["capacity", "lhc", "maximin", "anchored", "grf"],
        default=None,
        help="Training-parameter design (Report 5 batch mode); grf = PINO_2 GRF fields.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Number of LHC training tuples for lhc/maximin/anchored (default 500 in batch mode).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Results subfolder relative to this script, e.g. results/exp_B_lhc_N500.",
    )
    parser.add_argument(
        "--skip-validation-plots",
        action="store_true",
        help="Write comsol_validation_summary.csv only; skip per-case PNGs.",
    )
    parser.add_argument(
        "--reload-train-cases",
        action="store_true",
        help="Regenerate train_u_cases.csv even if it already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load checkpoint and run COMSOL validation (with plots unless --skip-validation-plots).",
    )
    parser.add_argument(
        "--arch",
        choices=["default", "dense", "w64"],
        default=None,
        help=(
            "Network width preset (default: [K,16,16,32]/[2,32,32,32,32]; "
            "dense: [K,64,64,128]/[2,64,64,64,64,128]; "
            "w64: [K,64,64,64]/[2,64,64,64,64])."
        ),
    )
    parser.add_argument(
        "--n-corner-anchors",
        type=int,
        default=None,
        help="Boundary anchor count for anchored design (default 16 cube corners).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Torch floating-point dtype (default float32).",
    )
    parser.add_argument(
        "--trunk-mode",
        choices=["single", "zone"],
        default=None,
        help="Trunk architecture: one shared trunk or one trunk per zone.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop L-BFGS after N steps with no total-loss improvement (0 = run all "
            "num_epochs_lbfgs steps)."
        ),
    )
    parser.add_argument(
        "--lr-lbfgs",
        type=float,
        default=None,
        help="L-BFGS learning rate (default 1).",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=None,
        metavar="N",
        help="PyTorch LBFGS max_iter per outer step (default 1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="L-BFGS outer steps (num_epochs_lbfgs, default 1000).",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=None,
        metavar="K",
        help="GRF branch sensor count on uniform x* grid [0,1] (default 100).",
    )
    parser.add_argument(
        "--n-grf",
        type=int,
        default=None,
        metavar="N",
        help="GRF smooth-field training cases (default 300; piecewise fills remainder).",
    )
    parser.add_argument(
        "--n-piecewise-per-zones",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Piecewise cases per zone count in --piecewise-zone-counts "
            "(default 50 -> 200 piecewise + 300 GRF = 500 total)."
        ),
    )
    parser.add_argument(
        "--piecewise-zone-counts",
        type=str,
        default=None,
        metavar="Z_LIST",
        help=(
            "Comma-separated piecewise zone counts, e.g. 1,2,3,4,5 "
            f"(default {','.join(str(v) for v in PIECEWISE_ZONE_COUNTS_DEFAULT)})."
        ),
    )
    parser.add_argument(
        "--min-zone-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Minimum zone width as fraction of domain length in x* "
            "(e.g. 0.1 = 10%% of domain / 10 of K=100 sensors). "
            "Enables 1-zone uniform fields; omit for legacy iface-margin sampling."
        ),
    )
    parser.add_argument(
        "--grf-corr-length",
        type=float,
        default=None,
        help="Single GRF correlation length in x* (overrides --grf-corr-lengths).",
    )
    parser.add_argument(
        "--grf-corr-lengths",
        type=str,
        default=None,
        metavar="ELL_LIST",
        help=(
            "Comma-separated GRF correlation lengths in x*; n_grf is split evenly "
            f"(default {','.join(str(v) for v in GRF_CORR_LENGTHS_DEFAULT)})."
        ),
    )
    parser.add_argument(
        "--grf-grid-n",
        type=int,
        default=None,
        help="GRF velocity field grid points on [0,1] (default 201).",
    )
    return parser.parse_args()


def _parse_grf_corr_lengths(text: str) -> tuple[float, ...]:
    values = tuple(float(s.strip()) for s in text.split(",") if s.strip())
    if not values:
        raise ValueError("grf-corr-lengths must list at least one positive value")
    if any(v <= 0 for v in values):
        raise ValueError("each grf correlation length must be positive")
    return values


def _parse_piecewise_zone_counts(text: str) -> tuple[int, ...]:
    values = tuple(int(s.strip()) for s in text.split(",") if s.strip())
    if not values:
        raise ValueError("piecewise-zone-counts must list at least one positive integer")
    if any(v < 1 for v in values):
        raise ValueError("each piecewise zone count must be >= 1")
    return values


def _apply_arch_preset(name: str) -> None:
    global branch_architecture, trunk_architecture, arch_preset
    if name not in ARCH_PRESETS:
        raise ValueError(f"Unknown arch preset: {name!r}")
    arch_preset = name
    preset_branch, trunk_architecture = ARCH_PRESETS[name]
    if media_mode == "grf":
        branch_architecture = [sensor_count, *preset_branch[1:]]
    else:
        branch_architecture = list(preset_branch)


def _apply_grf_branch_arch(k: int, *, arch: str = "default") -> None:
    """Set branch input width to K sensor CFL features; keep trunk latent dim in sync."""
    global sensor_count
    sensor_count = k
    _apply_arch_preset(arch)


def _load_run_meta_for_validate() -> bool:
    """If run_meta.json exists, restore arch/dtype/trunk_mode for validate-only."""
    meta_path = RESULTS_DIR / "run_meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    branch = meta.get("branch_architecture")
    trunk = meta.get("trunk_architecture")
    if not branch or not trunk:
        return False
    global branch_architecture, trunk_architecture, arch_preset, torch_dtype, trunk_mode
    global lr_lbfgs, lbfgs_max_iter, early_stop_patience
    global media_mode, sensor_count, sensor_xstar, grf_corr_lengths, grf_grid_n
    global n_sensors_requested, n_grf_requested, n_piecewise_per_zones, piecewise_zone_counts
    global min_zone_frac
    branch_architecture = list(branch)
    trunk_architecture = list(trunk)
    arch_preset = str(meta.get("arch_preset", "custom"))
    dtype_name = str(meta.get("dtype", "float32"))
    if dtype_name == "float64":
        torch_dtype = torch.float64
    else:
        torch_dtype = torch.float32
    trunk_mode = str(meta.get("trunk_mode", "single"))
    if "lr_lbfgs" in meta:
        lr_lbfgs = float(meta["lr_lbfgs"])
    if "lbfgs_max_iter" in meta:
        lbfgs_max_iter = int(meta["lbfgs_max_iter"])
    if "early_stop_patience" in meta:
        early_stop_patience = int(meta["early_stop_patience"])
    media_mode = str(meta.get("media_mode", "zone"))
    if media_mode == "grf":
        sensor_xstar = np.asarray(meta["sensor_xstar"], dtype=np.float64)
        sensor_count = int(meta.get("n_sensors", sensor_xstar.size))
        if isinstance(meta.get("grf_corr_lengths"), list):
            grf_corr_lengths = tuple(float(v) for v in meta["grf_corr_lengths"])
        elif "grf_corr_length" in meta:
            grf_corr_lengths = (float(meta["grf_corr_length"]),)
        else:
            grf_corr_lengths = GRF_CORR_LENGTHS_DEFAULT
        grf_grid_n = int(meta.get("grf_grid_n", 201))
        n_sensors_requested = int(meta.get("n_sensors_requested", sensor_count))
        if "n_grf" in meta:
            n_grf_requested = int(meta["n_grf"])
            n_piecewise_per_zones = int(meta.get("n_piecewise_per_zones", 50))
        if isinstance(meta.get("piecewise_zone_counts"), list):
            piecewise_zone_counts = tuple(int(v) for v in meta["piecewise_zone_counts"])
        if "min_zone_frac" in meta:
            min_zone_frac = float(meta["min_zone_frac"])
    return True


def apply_cli(args: argparse.Namespace) -> None:
    global RESULTS_DIR, COMSOL_VALIDATION_DIR, MODEL_PATH, U_TRAIN_CASES_PATH
    global train_design, n_train_requested, n_corner_anchors, skip_validation_plots
    global batch_mode, reload_lhc_train_cases, run_training, validate_only
    global torch_dtype, trunk_mode, early_stop_patience, lr_lbfgs, lbfgs_max_iter
    global num_epochs_lbfgs, media_mode, n_sensors_requested, grf_corr_lengths
    global grf_grid_n, sensor_xstar, sensor_count, branch_architecture
    global n_grf_requested, n_piecewise_per_zones, n_train_requested, piecewise_zone_counts
    global min_zone_frac

    if args.reload_train_cases:
        reload_lhc_train_cases = True

    if args.skip_validation_plots:
        skip_validation_plots = True

    if args.validate_only:
        validate_only = True
        run_training = False
        if not args.skip_validation_plots:
            skip_validation_plots = False

    if args.design is not None:
        train_design = args.design  # type: ignore[assignment]
        if train_design == "capacity":
            n_train_requested = 81
        elif train_design == "grf":
            media_mode = "grf"
            if args.n_sensors is not None:
                n_sensors_requested = args.n_sensors
            if args.n_grf is not None:
                n_grf_requested = args.n_grf
            if args.n_piecewise_per_zones is not None:
                n_piecewise_per_zones = args.n_piecewise_per_zones
            if args.piecewise_zone_counts is not None:
                piecewise_zone_counts = _parse_piecewise_zone_counts(
                    args.piecewise_zone_counts
                )
            if args.min_zone_frac is not None:
                min_zone_frac = args.min_zone_frac
            if args.n_train is not None:
                n_train_requested = args.n_train
            else:
                n_train_requested = (
                    n_grf_requested
                    + len(piecewise_zone_counts) * n_piecewise_per_zones
                )
            if args.grf_corr_length is not None:
                grf_corr_lengths = (args.grf_corr_length,)
            elif args.grf_corr_lengths is not None:
                grf_corr_lengths = _parse_grf_corr_lengths(args.grf_corr_lengths)
            if args.grf_grid_n is not None:
                grf_grid_n = args.grf_grid_n
            sensor_xstar = default_sensor_xstar(
                n_sensors_requested, include_interfaces=False
            )
            _apply_grf_branch_arch(
                sensor_xstar.size,
                arch=args.arch if args.arch is not None else "default",
            )
        elif args.n_train is not None:
            n_train_requested = args.n_train
        else:
            n_train_requested = 500
    elif args.n_train is not None:
        n_train_requested = args.n_train

    if args.n_corner_anchors is not None:
        n_corner_anchors = args.n_corner_anchors

    if args.design is not None or args.out_dir is not None:
        batch_mode = True

    if args.out_dir is not None:
        RESULTS_DIR = _PINO_DIR / args.out_dir
    else:
        RESULTS_DIR = _PINO_DIR / "results"

    MODEL_PATH = RESULTS_DIR / "pino_heterogeneous_model.pt"

    if args.validate_only:
        _load_run_meta_for_validate()

    if args.arch is not None:
        if media_mode == "grf" and args.design != "grf":
            # validate-only restore may have set GRF arch from meta
            _apply_grf_branch_arch(sensor_count, arch=args.arch)
        elif media_mode != "grf":
            _apply_arch_preset(args.arch)

    if args.trunk_mode == "zone" and media_mode == "grf":
        raise ValueError("--trunk-mode zone is not supported with --design grf")

    if args.dtype is not None:
        torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.trunk_mode is not None:
        trunk_mode = args.trunk_mode

    if args.early_stop_patience is not None:
        early_stop_patience = args.early_stop_patience

    if args.lr_lbfgs is not None:
        lr_lbfgs = args.lr_lbfgs

    if args.lbfgs_max_iter is not None:
        lbfgs_max_iter = args.lbfgs_max_iter

    if args.epochs is not None:
        num_epochs_lbfgs = args.epochs

    if batch_mode:
        if media_mode == "grf":
            U_TRAIN_CASES_PATH = RESULTS_DIR / "train_grf_cases.npz"
        else:
            U_TRAIN_CASES_PATH = RESULTS_DIR / "train_u_cases.csv"
        if skip_validation_plots:
            COMSOL_VALIDATION_DIR = RESULTS_DIR
        else:
            COMSOL_VALIDATION_DIR = RESULTS_DIR / "comsol_validation"
    else:
        COMSOL_VALIDATION_DIR = RESULTS_DIR / "comsol_validation"
        U_TRAIN_CASES_PATH = RESULTS_DIR / f"lhc_train_u{N_LHC_TRAIN}.csv"


def write_run_meta(
    *,
    wall_clock_s: float,
    n_train_actual: int,
    validation_rows: list[dict[str, object]] | None,
    training_summary: dict[str, object] | None = None,
) -> Path:
    meta: dict[str, object] = {
        "design": train_design,
        "media_mode": media_mode,
        "n_train": n_train_requested,
        "n_train_actual": n_train_actual,
        "n_corner_anchors": n_corner_anchors if train_design == "anchored" else None,
        "arch_preset": arch_preset,
        "trunk_mode": trunk_mode,
        "branch_architecture": branch_architecture,
        "trunk_architecture": trunk_architecture,
        "seed": seed,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "num_epochs_lbfgs_requested": num_epochs_lbfgs,
        "early_stop_patience": early_stop_patience,
        "lr_lbfgs": lr_lbfgs,
        "lbfgs_max_iter": lbfgs_max_iter,
        "wall_clock_s": round(wall_clock_s, 3),
        "out_dir": str(RESULTS_DIR.relative_to(_PINO_DIR)),
    }
    if media_mode == "grf" and sensor_xstar is not None:
        meta["n_sensors"] = int(sensor_xstar.size)
        meta["n_sensors_requested"] = n_sensors_requested
        meta["sensor_xstar"] = sensor_xstar.tolist()
        meta["grf_corr_lengths"] = list(grf_corr_lengths)
        meta["grf_grid_n"] = grf_grid_n
        meta["n_grf"] = n_grf_requested
        meta["n_piecewise_per_zones"] = n_piecewise_per_zones
        meta["piecewise_zone_counts"] = list(piecewise_zone_counts)
        if min_zone_frac is not None:
            meta["min_zone_frac"] = min_zone_frac
    if training_summary:
        meta.update(training_summary)
    if validation_rows:
        l2_vals = np.array(
            [float(r["mean_rel_l2"]) for r in validation_rows], dtype=np.float64
        )
        meta["mean_rel_l2"] = float(l2_vals.mean())
        meta["max_rel_l2"] = float(l2_vals.max())
        meta["min_rel_l2"] = float(l2_vals.min())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = RESULTS_DIR / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


def main() -> None:
    global reload_lhc_train_cases

    args = parse_cli()
    apply_cli(args)

    t_start = time.perf_counter()
    validation_rows: list[dict[str, object]] | None = None
    training_summary: dict[str, object] | None = None

    torch.set_default_dtype(torch_dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)} ({device}, dtype={dtype})")
    else:
        print(f"Using CPU (CUDA not available, dtype={dtype})")

    print(f"PE (computed) = {PE:.12g}")
    print(
        f"design={train_design}  media_mode={media_mode}  "
        f"n_train_requested={n_train_requested}  "
        f"n_corner_anchors={n_corner_anchors}  arch={arch_preset}  "
        f"trunk_mode={trunk_mode}  batch_mode={batch_mode}"
    )
    print(f"branch={branch_architecture}  trunk={trunk_architecture}")
    print(f"results_dir={RESULTS_DIR}")
    n_u = 0
    grf_batch: GRFTrainingBatch | None = None
    if validate_only:
        print(f"validate_only: loading checkpoint from {MODEL_PATH}")
    elif media_mode == "grf":
        if sensor_xstar is None:
            raise RuntimeError("GRF mode requires sensor_xstar")
        mixed_config = MixedTrainConfig(
            n_grf=n_grf_requested,
            n_piecewise_per_zone_count=n_piecewise_per_zones,
            piecewise_zone_counts=piecewise_zone_counts,
            grf_corr_lengths=grf_corr_lengths,
            min_zone_frac=min_zone_frac,
        )
        if n_train_requested != mixed_config.n_total:
            raise ValueError(
                f"n_train_requested={n_train_requested} does not match mixed batch "
                f"size {mixed_config.n_total} "
                f"(n_grf={n_grf_requested}, "
                f"n_piecewise={mixed_config.n_piecewise})"
            )
        grf_batch = load_or_generate_grf_train_cases(
            U_TRAIN_CASES_PATH,
            sensor_xstar,
            mixed_config=mixed_config,
            u_lo=U_LO,
            u_hi=U_HI,
            grid_n=grf_grid_n,
            seed=seed,
            reload=reload_lhc_train_cases,
            l_m=L,
            t_max_d=T_MAX,
        )
        n_u = grf_batch.sensor_u.shape[0]
        cfl_train = branch_cfl_from_grf_sensor_u(grf_batch.sensor_u)
        if mixed_config.n_grf > 0:
            grf_counts = mixed_config.grf_counts_per_length()
            ell_desc = ", ".join(
                f"{n}@{ell:g}" for ell, n in zip(grf_corr_lengths, grf_counts, strict=True)
            )
            train_desc = (
                f"{mixed_config.n_grf} GRF ({ell_desc}) + {mixed_config.n_piecewise} piecewise "
                f"({n_piecewise_per_zones} each for zones "
                f"{list(piecewise_zone_counts)}) = {n_u} total; "
                f"K={sensor_xstar.size} uniform sensors, grid_n={grf_grid_n}"
            )
        else:
            zone_desc = (
                f"{mixed_config.n_piecewise} piecewise "
                f"({n_piecewise_per_zones} each for zones {list(piecewise_zone_counts)})"
            )
            if min_zone_frac is not None:
                zone_desc += f", min_zone_frac={min_zone_frac:g}"
            train_desc = (
                f"{zone_desc} = {n_u} total; "
                f"K={sensor_xstar.size} uniform sensors, grid_n={grf_grid_n}"
            )
        print(f"Training: {train_desc}")
        print(f"  saved/loaded: {U_TRAIN_CASES_PATH}")
        print(f"  branch CFL range [{cfl_train.min():g}, {cfl_train.max():g}]")
        print(
            f"PDE mesh: bulk {mesh_nx_pde}x{mesh_nt_pde}x{n_u} "
            f"(no interface bands in GRF mode)"
        )
    else:
        u_train = load_or_generate_train_u_cases()
        cfl_train = branch_cfl_from_u(u_train)
        n_u = u_train.shape[0]
        if train_design == "capacity":
            train_desc = f"{n_u} COMSOL grid tuples (capacity / in-sample)"
        elif train_design == "anchored":
            train_desc = (
                f"{n_train_requested} LHC + {n_corner_anchors} boundary anchors "
                f"-> {n_u} unique tuples (anchored)"
            )
        else:
            train_desc = f"{n_u} {train_design} samples in [{U_LO:g}, {U_HI:g}] m/d"
        print(f"Training: {train_desc}")
        print(f"  saved/loaded: {U_TRAIN_CASES_PATH}")
        print(f"  branch CFL range [{cfl_train.min():g}, {cfl_train.max():g}]")
        n_if_per = mesh_nx_interface_per_band * len(ZONE_INTERFACE_XSTAR)
        print(
            f"PDE mesh: bulk {mesh_nx_pde}x{mesh_nt_pde}x{n_u} + interface bands "
            f"({n_if_per} x* / band, +/-{interface_band_half_width:g} around "
            f"{ZONE_INTERFACE_XSTAR})"
        )
    print(
        f"COMSOL validation: {len(comsol_validation_u_grid())} fixed grid cases "
        f"from {COMSOL_DATA_PATH.name} -> {COMSOL_VALIDATION_DIR}"
    )
    print(
        f"run_training={run_training}  run_comsol_validation={run_comsol_validation}  "
        f"validate_only={validate_only}  skip_validation_plots={skip_validation_plots}"
    )
    if run_training and early_stop_patience > 0:
        print(
            f"L-BFGS early stopping: patience={early_stop_patience} "
            f"(max steps={num_epochs_lbfgs})"
        )
    if run_training:
        print(
            f"L-BFGS: lr={lr_lbfgs:g}  max_iter={lbfgs_max_iter}  "
            f"outer_steps={num_epochs_lbfgs}"
        )

    use_zone_trunks = trunk_mode == "zone"
    model = build_deeponet(
        trunk_mode, branch_architecture, trunk_architecture, activation_cls
    ).to(device)

    if run_training:
        if media_mode == "grf":
            if grf_batch is None:
                raise RuntimeError("GRF batch missing for training")
            tensors, plot_data = build_grf_collocation_tensors(device, dtype, grf_batch)
            use_zone_pde = False
            use_grf_pde = True
            use_interface_losses = False
        else:
            tensors, plot_data = build_collocation_tensors(
                device, dtype, u_train, include_interface=use_zone_trunks
            )
            use_zone_pde = use_zone_trunks
            use_grf_pde = False
            use_interface_losses = use_zone_trunks
        print(
            f"PDE collocation points: {plot_data['n_pde']:,} "
            f"(bulk {plot_data['n_pde_bulk']:,}, interface {plot_data['n_pde_interface']:,})"
        )
        coll_path = plot_collocation_points(plot_data, n_train_media=n_u)
        print(f"Collocation mesh plot (pre-train): {coll_path}")
        if use_interface_losses:
            n_if_pts = tensors["x_if_l"].shape[0]
            print(
                f"Zone trunks: per-zone PDE CFL + interface losses "
                f"({n_if_pts:,} interface collocation pairs)"
            )
        history, training_summary = train_model(
            model,
            device,
            tensors,
            use_zone_pde=use_zone_pde,
            use_grf_pde=use_grf_pde,
            use_interface_losses=use_interface_losses,
        )
        conc_path, loss_path = plot_training_figures(model, device, dtype, history)
        if save_model:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        print("Training saved:", conc_path, loss_path, MODEL_PATH)
    else:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"run_training=False but no checkpoint at {MODEL_PATH}"
            )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print(f"Loaded checkpoint: {MODEL_PATH}")

    if run_comsol_validation:
        validation_rows = validate_against_comsol(
            model,
            device,
            dtype,
            save_plots=not skip_validation_plots,
        )

    if validate_only and n_u == 0:
        if media_mode == "grf" and U_TRAIN_CASES_PATH.is_file():
            n_u = int(load_grf_cases_npz(U_TRAIN_CASES_PATH).sensor_u.shape[0])
        elif U_TRAIN_CASES_PATH.is_file() and U_TRAIN_CASES_PATH.suffix == ".csv":
            n_u = int(load_u_cases_csv(U_TRAIN_CASES_PATH).shape[0])

    if validation_rows is not None or not validate_only:
        meta_path = write_run_meta(
            wall_clock_s=time.perf_counter() - t_start,
            n_train_actual=n_u,
            validation_rows=validation_rows,
            training_summary=training_summary,
        )
        print(f"Run metadata: {meta_path}")


if __name__ == "__main__":
    main()
