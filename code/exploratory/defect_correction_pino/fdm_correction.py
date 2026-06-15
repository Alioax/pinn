# -*- coding: utf-8 -*-
"""Implicit FDM solver for L[delta] = -r with homogeneous BC/IC."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from residual import PE, UCases, piecewise_cfl_np, pde_residual_fd_grid, residual_stats


def _build_interior_operator(
    n_interior: int,
    cfl_interior: np.ndarray,
    dx: float,
    dt: float,
    pe: float,
) -> sparse.csr_matrix:
    """
    Backward-Euler discretization of
      delta_t + c(x) delta_x - Pe delta_xx = -r
    on interior x nodes (Dirichlet delta=0 at boundaries).
    """
    inv_dt = 1.0 / dt
    inv_2dx = 1.0 / (2.0 * dx)
    inv_dx2 = 1.0 / (dx * dx)

    diag = np.full(n_interior, inv_dt + 2.0 * pe * inv_dx2, dtype=np.float64)
    lower = np.zeros(n_interior - 1, dtype=np.float64)
    upper = np.zeros(n_interior - 1, dtype=np.float64)

    for j in range(n_interior):
        c_i = cfl_interior[j]
        if j > 0:
            lower[j - 1] = -c_i * inv_2dx - pe * inv_dx2
        if j < n_interior - 1:
            upper[j] = c_i * inv_2dx - pe * inv_dx2

    return sparse.diags(
        [lower, diag, upper],
        offsets=[-1, 0, 1],
        format="csr",
    )


def solve_correction_fdm(
    r: np.ndarray,
    x_star: np.ndarray,
    t_star: np.ndarray,
    u_case: UCases,
    *,
    pe: float = PE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Solve L[delta] = -r with delta=0 on x*=0,1 and t*=0.

    Parameters
    ----------
    r : ndarray, shape (nx, nt)
        Source field (PINO autograd residual on same grid).
    x_star, t_star : 1d arrays
        Uniform grids spanning [0, 1].

    Returns
    -------
    delta : ndarray, shape (nx, nt)
    info : dict with solve_wall_s and discrete residual stats
    """
    r = np.asarray(r, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    t_star = np.asarray(t_star, dtype=np.float64)
    nx, nt = r.shape
    if nx != x_star.size or nt != t_star.size:
        raise ValueError("r shape must match x_star and t_star sizes")

    dx = x_star[1] - x_star[0]
    dt = t_star[1] - t_star[0]
    n_interior = nx - 2
    if n_interior < 1:
        raise ValueError("Need at least 3 x nodes for interior solve")

    cfl = piecewise_cfl_np(x_star, u_case)
    cfl_interior = cfl[1:-1]
    A = _build_interior_operator(n_interior, cfl_interior, dx, dt, pe)

    delta = np.zeros((nx, nt), dtype=np.float64)
    delta_int = np.zeros(n_interior, dtype=np.float64)

    t0 = time.perf_counter()
    for j in range(1, nt):
        rhs = -r[1:-1, j] + delta_int / dt
        delta_int = spsolve(A, rhs)
        delta[1:-1, j] = delta_int
    solve_wall_s = time.perf_counter() - t0

    # Discrete residual: L_fd[delta] + r (should be ~0 on interior)
    l_delta = pde_residual_fd_grid(delta, x_star, t_star, u_case, pe=pe)
    disc = l_delta + r
    mask = np.zeros((nx, nt), dtype=bool)
    mask[1 : nx - 1, 1:nt] = True
    disc_stats = residual_stats(disc, mask)

    return delta, {
        "solve_wall_s": float(solve_wall_s),
        "discrete_residual_mean_abs": disc_stats["mean_abs"],
        "discrete_residual_max_abs": disc_stats["max_abs"],
        "discrete_residual_rms": disc_stats["rms"],
    }
