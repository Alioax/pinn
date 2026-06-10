# -*- coding: utf-8 -*-
"""
Physics-informed heterogeneous neural operator (DeepONet-style): train + COMSOL validation.

Branch encodes dimensionless zone CFL (CFL1..CFL4) with CFL_i = u_i T_max / L;
trunk maps (x*, t*). PDE residual uses physical u for piecewise CFL(x*).
PDE: dC*/dt* + CFL(x*; u1..u4) dC*/dx* - Pe d2C*/dx*^2 = 0 on four zones (20+20+20+40 m).

Edit the configuration block below, then run this file.
"""

from __future__ import annotations

import csv
import sys
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

from deeponet import DeepONetParametric  # noqa: E402
from utils.comsol_4zones import list_parameter_combos, load_case  # noqa: E402
from utils.lhc_sampling import (  # noqa: E402
    comsol_validation_u_grid,
    generate_lhc_u_samples,
    load_u_cases_csv,
    save_u_cases_csv,
)
from utils.zone_velocity import (  # noqa: E402
    L_DEFAULT,
    PE_DEFAULT,
    T_MAX_DEFAULT,
    XSTAR_TICKS,
    ZONE_INTERFACE_XSTAR,
    cfl_from_u_np,
    piecewise_cfl_from_branch,
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
N_LHC_TRAIN = 500
LHC_EXCLUDE_COMSOL_GRID = True
LHC_EXCLUDE_ATOL = 1e-9

sensor_count = 4
branch_architecture = [sensor_count, 16, 16, 32]
trunk_architecture = [2, 32, 32, 32, 32]
# branch_architecture = [sensor_count, 64, 64, 128]
# trunk_architecture  = [2, 64, 64, 64, 64, 128]
activation_cls = nn.Tanh

num_epochs_lbfgs = 1000
lr_lbfgs = 1

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

save_model = True
collocation_scatter_ms = 3
collocation_xstar_axis_pad = 0.05
n_cycle_colors = 4

times_tstar = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
num_spatial_points = 5000
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
reload_lhc_train_cases = False

COMSOL_DATA_PATH = (_HETERO_ROOT / "data" / "comsol_4zones.txt").resolve()
COMSOL_TIMES_DAYS = np.array([100.0, 300.0, 600.0, 900.0, 1200.0], dtype=float)
COMSOL_C_REF = 1.0

RESULTS_DIR = _PINO_DIR / "results"
COMSOL_VALIDATION_DIR = RESULTS_DIR / "comsol_validation"
MODEL_PATH = RESULTS_DIR / "pino_heterogeneous_model.pt"
U_TRAIN_CASES_PATH = RESULTS_DIR / f"lhc_train_u{N_LHC_TRAIN}.csv"

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
    """Training media: ``N_LHC_TRAIN`` LHC samples in [U_LO, U_HI]^4, shape (N, 4)."""
    if reload_lhc_train_cases and U_TRAIN_CASES_PATH.is_file():
        U_TRAIN_CASES_PATH.unlink()
    if U_TRAIN_CASES_PATH.is_file():
        u_cases = load_u_cases_csv(U_TRAIN_CASES_PATH)
        if u_cases.shape[0] != N_LHC_TRAIN:
            raise ValueError(
                f"{U_TRAIN_CASES_PATH} has {u_cases.shape[0]} rows; "
                f"expected N_LHC_TRAIN={N_LHC_TRAIN}. Set reload_lhc_train_cases=True."
            )
        return u_cases

    exclude = comsol_validation_u_grid() if LHC_EXCLUDE_COMSOL_GRID else None
    u_cases = generate_lhc_u_samples(
        N_LHC_TRAIN,
        U_LO,
        U_HI,
        seed=seed,
        exclude_near=exclude,
        exclude_atol=LHC_EXCLUDE_ATOL,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_u_cases_csv(U_TRAIN_CASES_PATH, u_cases)
    return u_cases


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


def branch_tensor_from_u_case(
    u_case: tuple[float, float, float, float],
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    physical = np.array([list(u_case)], dtype=np.float64).reshape(1, 4)
    cfl_branch = branch_cfl_from_u(physical)
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
    u_phys = torch.tensor(
        np.tile(np.array(u_case, dtype=np.float64), (n_x, 1)),
        dtype=dtype,
        device=device,
    )
    c = model(x_t, t_t, branch)
    dC_dt = gradients(c, t_t)
    dC_dx = gradients(c, x_t)
    d2C_dx2 = gradients(dC_dx, x_t)
    cfl_local = piecewise_cfl_from_branch(x_t, u_phys, l_m=L, t_max_d=T_MAX)
    residual = dC_dt + cfl_local * dC_dx - PE * d2C_dx2
    return residual.detach().cpu().numpy().flatten()


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

    l2_sum = 0.0
    matched = _match_comsol_times(comsol_by_time, COMSOL_TIMES_DAYS)
    model.eval()
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
    data_file: Path = COMSOL_DATA_PATH,
    out_dir: Path = COMSOL_VALIDATION_DIR,
) -> list[dict[str, object]]:
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
        rows.append(
            {
                "u1": u_case[0],
                "u2": u_case[1],
                "u3": u_case[2],
                "u4": u_case[3],
                "mean_rel_l2": mean_l2,
                "plot": str(out_png),
            }
        )

    summary_path = out_dir / "comsol_validation_summary.csv"
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


def build_collocation_tensors(
    device: torch.device,
    dtype: torch.dtype,
    u_cases: np.ndarray,
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

    tensors = {
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


def compute_physics_loss(
    model: DeepONetParametric,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[float, float, float, float, float]]:
    c_pde = model(tensors["x_pde"], tensors["t_pde"], tensors["branch_pde"])
    dC_dt = gradients(c_pde, tensors["t_pde"])
    dC_dx = gradients(c_pde, tensors["x_pde"])
    d2C_dx2 = gradients(dC_dx, tensors["x_pde"])
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
    total_loss = (
        weight_pde * pde_loss
        + weight_ic * ic_loss
        + weight_inlet_bc * inlet_loss
        + weight_outlet_bc * outlet_loss
    )
    metrics = (
        total_loss.item(),
        pde_loss.item(),
        ic_loss.item(),
        inlet_loss.item(),
        outlet_loss.item(),
    )
    return total_loss, metrics


def train_model(
    model: DeepONetParametric,
    device: torch.device,
    tensors: dict[str, torch.Tensor],
) -> list[list[float]]:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    print(optimizer)

    history: list[list[float]] = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = compute_physics_loss(model, tensors)
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
    for _ in t_bar:
        model.train()
        optimizer.step(closure)
        history.append(list(closure.latest))
    t_bar.close()
    return history


def plot_collocation_points(plot_data: dict) -> Path:
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
    axes2[0].set_title(
        rf"PDE: $x^*$ vs $t^*$ (one LHC medium), "
        rf"$u=({u_ref[0]:g},{u_ref[1]:g},{u_ref[2]:g},{u_ref[3]:g})$ m/d"
    )
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
        f"LHC training collocation ({N_LHC_TRAIN} media, one slice shown)",
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
    ax3.set_yscale("log")
    ax3.set_xlabel("L-BFGS step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9, frameon=False)
    loss_path = RESULTS_DIR / "pino_heterogeneous_loss.png"
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close(fig3)
    return conc_path, loss_path


def main() -> None:
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
    u_train = load_or_generate_train_u_cases()
    cfl_train = branch_cfl_from_u(u_train)
    print(
        f"Training: {N_LHC_TRAIN} LHC samples in [{U_LO:g}, {U_HI:g}] m/d "
        f"(exclude COMSOL 3^4 grid: {LHC_EXCLUDE_COMSOL_GRID})"
    )
    print(f"  saved/loaded: {U_TRAIN_CASES_PATH}")
    print(
        f"  branch CFL range [{cfl_train.min():g}, {cfl_train.max():g}]"
    )
    n_u = u_train.shape[0]
    n_if_per = mesh_nx_interface_per_band * len(ZONE_INTERFACE_XSTAR)
    print(
        f"PDE mesh: bulk {mesh_nx_pde}x{mesh_nt_pde}x{n_u} + interface bands "
        f"({n_if_per} x* / band, +/-{interface_band_half_width:g} around "
        f"{ZONE_INTERFACE_XSTAR})"
    )
    print(
        f"COMSOL validation: {len(comsol_validation_u_grid())} fixed grid cases "
        f"from {COMSOL_DATA_PATH.name}"
    )
    print(f"run_training={run_training}  run_comsol_validation={run_comsol_validation}")

    model = DeepONetParametric(
        branch_architecture, trunk_architecture, activation_cls
    ).to(device)

    if run_training:
        tensors, plot_data = build_collocation_tensors(device, dtype, u_train)
        print(
            f"PDE collocation points: {plot_data['n_pde']:,} "
            f"(bulk {plot_data['n_pde_bulk']:,}, interface {plot_data['n_pde_interface']:,})"
        )
        coll_path = plot_collocation_points(plot_data)
        print(f"Collocation mesh plot (pre-train): {coll_path}")
        history = train_model(model, device, tensors)
        conc_path, loss_path = plot_training_figures(model, device, dtype, history)
        if save_model:
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
        validate_against_comsol(model, device, dtype)


if __name__ == "__main__":
    main()
