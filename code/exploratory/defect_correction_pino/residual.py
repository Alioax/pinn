# -*- coding: utf-8 -*-
"""Load PINO checkpoint and evaluate PDE residuals on a dense (x*, t*) grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_EXPLORE_DIR = Path(__file__).resolve().parent
_HETERO_ROOT = _EXPLORE_DIR.parent.parent / "heterogeneous"
_PINO_DIR = _HETERO_ROOT / "pino_heterogeneous"
for _p in (_HETERO_ROOT, _PINO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deeponet import DeepONetParametric, build_deeponet  # noqa: E402
from utils.zone_velocity import (  # noqa: E402
    L_DEFAULT,
    PE_DEFAULT,
    T_MAX_DEFAULT,
    cfl_from_u_np,
    piecewise_cfl_from_branch,
    zone_index_xstar_np,
)

L = L_DEFAULT
T_MAX = T_MAX_DEFAULT
PE = PE_DEFAULT

UCases = tuple[float, float, float, float]


def gradients(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=retain_graph,
        retain_graph=retain_graph,
    )[0]


def branch_cfl_from_u(u: np.ndarray) -> np.ndarray:
    return cfl_from_u_np(u, l_m=L, t_max_d=T_MAX)


def branch_tensor_from_u_case(
    u_case: UCases,
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


def piecewise_cfl_np(
    x_star: np.ndarray,
    u_case: UCases,
    *,
    l_m: float = L,
    t_max_d: float = T_MAX,
) -> np.ndarray:
    """Piecewise zone CFL at each x* (numpy)."""
    u = np.asarray(u_case, dtype=np.float64).reshape(4)
    cfl_zone = cfl_from_u_np(u.reshape(1, 4), l_m=l_m, t_max_d=t_max_d).flatten()
    zidx = zone_index_xstar_np(x_star, l_m=l_m)
    return cfl_zone[zidx]


def load_model_from_checkpoint(
    checkpoint_dir: Path,
    *,
    device: torch.device | None = None,
) -> tuple[DeepONetParametric, torch.dtype, dict[str, Any]]:
    """Restore DeepONet from run_meta.json + pino_heterogeneous_model.pt."""
    checkpoint_dir = checkpoint_dir.resolve()
    meta_path = checkpoint_dir / "run_meta.json"
    model_path = checkpoint_dir / "pino_heterogeneous_model.pt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    branch = list(meta["branch_architecture"])
    trunk = list(meta["trunk_architecture"])
    dtype_name = str(meta.get("dtype", "float32"))
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    trunk_mode = str(meta.get("trunk_mode", "single"))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_deeponet(trunk_mode, branch, trunk, nn.Tanh)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, dtype, meta


def evaluate_model_on_grid(
    model: DeepONetParametric,
    x_star: np.ndarray,
    t_star: np.ndarray,
    u_case: UCases,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8192,
) -> np.ndarray:
    """Evaluate PINO C* on a tensor-product grid; returns shape (nx, nt)."""
    x_star = np.asarray(x_star, dtype=np.float64)
    t_star = np.asarray(t_star, dtype=np.float64)
    xx, tt = np.meshgrid(x_star, t_star, indexing="ij")
    x_flat = xx.ravel()
    t_flat = tt.ravel()
    n_pts = x_flat.size
    out = np.empty(n_pts, dtype=np.float64)

    branch_np = branch_cfl_from_u(np.array([list(u_case)], dtype=np.float64))
    branch_base = torch.tensor(branch_np, dtype=dtype, device=device)

    with torch.no_grad():
        for start in range(0, n_pts, batch_size):
            end = min(start + batch_size, n_pts)
            n = end - start
            x_t = torch.tensor(
                x_flat[start:end].reshape(-1, 1), dtype=dtype, device=device
            )
            t_t = torch.tensor(
                t_flat[start:end].reshape(-1, 1), dtype=dtype, device=device
            )
            branch = branch_base.expand(n, -1)
            c = model(x_t, t_t, branch)
            out[start:end] = c.detach().cpu().numpy().flatten()

    return out.reshape(x_star.size, t_star.size)


def pde_residual_autograd_slice(
    model: DeepONetParametric,
    x_star: np.ndarray,
    t_star_val: float,
    u_case: UCases,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """PDE residual r = C_t + CFL(x*) C_x - Pe C_xx at fixed t* (training operator)."""
    n_x = x_star.size
    x_t = torch.tensor(
        x_star.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True
    )
    t_t = torch.full((n_x, 1), t_star_val, dtype=dtype, device=device, requires_grad=True)
    branch = branch_tensor_from_u_case(u_case, n_x, device=device, dtype=dtype)
    u_phys = torch.tensor(
        np.tile(np.array(u_case, dtype=np.float64), (n_x, 1)),
        dtype=dtype,
        device=device,
    )
    c = model(x_t, t_t, branch)
    dC_dt = gradients(c, t_t, retain_graph=True)
    dC_dx = gradients(c, x_t, retain_graph=True)
    d2C_dx2 = gradients(dC_dx, x_t)
    cfl_local = piecewise_cfl_from_branch(x_t, u_phys, l_m=L, t_max_d=T_MAX)
    residual = dC_dt + cfl_local * dC_dx - PE * d2C_dx2
    return residual.detach().cpu().numpy().flatten()


def pde_residual_autograd_grid(
    model: DeepONetParametric,
    x_star: np.ndarray,
    t_star: np.ndarray,
    u_case: UCases,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Autograd PDE residual on full grid; shape (nx, nt)."""
    x_star = np.asarray(x_star, dtype=np.float64)
    t_star = np.asarray(t_star, dtype=np.float64)
    r = np.zeros((x_star.size, t_star.size), dtype=np.float64)
    model.eval()
    for j, t_val in enumerate(t_star):
        with torch.enable_grad():
            r[:, j] = pde_residual_autograd_slice(
                model, x_star, float(t_val), u_case, device=device, dtype=dtype
            )
    return r


def pde_residual_fd_grid(
    c: np.ndarray,
    x_star: np.ndarray,
    t_star: np.ndarray,
    u_case: UCases,
    *,
    pe: float = PE,
) -> np.ndarray:
    """
    Finite-difference PDE residual on interior points (central x, backward t).
    Boundary/time-slice rows left at zero (not in metric mask).
    """
    c = np.asarray(c, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    t_star = np.asarray(t_star, dtype=np.float64)
    nx, nt = c.shape
    dx = x_star[1] - x_star[0]
    dt = t_star[1] - t_star[0]
    cfl = piecewise_cfl_np(x_star, u_case)
    r = np.zeros_like(c)

    for j in range(1, nt):
        for i in range(1, nx - 1):
            ct = (c[i, j] - c[i, j - 1]) / dt
            cx = (c[i + 1, j] - c[i - 1, j]) / (2.0 * dx)
            cxx = (c[i + 1, j] - 2.0 * c[i, j] + c[i - 1, j]) / (dx * dx)
            r[i, j] = ct + cfl[i] * cx - pe * cxx
    return r


def residual_interior_mask(c: np.ndarray) -> np.ndarray:
    """Boolean mask for interior (x*, t*) nodes used in FD residual metrics."""
    nx, nt = c.shape
    mask = np.zeros((nx, nt), dtype=bool)
    mask[1 : nx - 1, 1:nt] = True
    return mask


def residual_stats(r: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        mask = np.abs(r) > 0
        if not mask.any():
            mask = np.ones_like(r, dtype=bool)
    vals = np.abs(r[mask])
    return {
        "mean_abs": float(np.mean(vals)),
        "max_abs": float(np.max(vals)),
        "rms": float(np.sqrt(np.mean(vals * vals))),
    }


def l2_rel(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref)
    return float(num / den) if den > 0 else float(num)
