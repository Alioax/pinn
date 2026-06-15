# -*- coding: utf-8 -*-
"""Gaussian random field (GRF) velocity sampling for PINO_2 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.zone_velocity import (
    L_DEFAULT,
    T_MAX_DEFAULT,
    ZONE_INTERFACE_XSTAR,
    cfl_from_u_np,
    zone_index_xstar_np,
)

UCases = tuple[float, float, float, float]


@dataclass(frozen=True)
class GRFTrainingBatch:
    """GRF training media: sensor velocities + fine grid fields."""

    sensor_u: np.ndarray  # (N, K) physical u at sensors (m/d)
    grid_u: np.ndarray  # (N, G) physical u on fine grid (m/d)
    grid_x: np.ndarray  # (G,) x* in [0, 1]
    sensor_x: np.ndarray  # (K,) x* sensor locations


def default_sensor_xstar(
    k: int,
    *,
    include_interfaces: bool = True,
) -> np.ndarray:
    """Fixed sensor locations on [0, 1]; optionally add zone interface x*."""
    if k < 2:
        raise ValueError("k must be >= 2")
    sensors = np.linspace(0.0, 1.0, k, dtype=np.float64)
    if include_interfaces:
        sensors = np.unique(
            np.concatenate([sensors, np.array(ZONE_INTERFACE_XSTAR, dtype=np.float64)])
        )
    return sensors


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _squared_exponential_covariance(
    x: np.ndarray,
    *,
    corr_length: float,
) -> np.ndarray:
    """Covariance matrix for GP with exp(-(x-x')^2 / (2 ell^2))."""
    if corr_length <= 0:
        raise ValueError("corr_length must be positive")
    diff = x[:, None] - x[None, :]
    return np.exp(-0.5 * (diff / corr_length) ** 2)


def draw_grf_velocity_fields(
    n: int,
    sensor_x: np.ndarray,
    *,
    u_lo: float = 0.01,
    u_hi: float = 0.05,
    corr_length: float = 0.2,
    grid_n: int = 201,
    seed: int = 0,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> GRFTrainingBatch:
    """
    Draw ``n`` smooth GRF velocity fields on a fine x* grid.

    GP draw on grid, map to [u_lo, u_hi] via sigmoid, sample sensors by interpolation.
    """
    del l_m, t_max_d  # CFL conversion happens in branch helpers
    if n < 1:
        raise ValueError("n must be >= 1")
    if grid_n < 3:
        raise ValueError("grid_n must be >= 3")

    grid_x = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    k = sensor_x.size

    rng = np.random.default_rng(seed)
    cov = _squared_exponential_covariance(grid_x, corr_length=corr_length)
    # Jitter for numerical stability of Cholesky
    cov = cov + 1e-10 * np.eye(grid_n, dtype=np.float64)
    chol = np.linalg.cholesky(cov)

    grid_u = np.empty((n, grid_n), dtype=np.float64)
    for i in range(n):
        xi = chol @ rng.standard_normal(grid_n)
        u = u_lo + (u_hi - u_lo) * _sigmoid(xi)
        grid_u[i] = np.clip(u, u_lo, u_hi)

    sensor_u = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        sensor_u[i] = interpolate_u_xstar(sensor_x, grid_x, grid_u[i])

    return GRFTrainingBatch(
        sensor_u=sensor_u,
        grid_u=grid_u,
        grid_x=grid_x,
        sensor_x=sensor_x,
    )


def interpolate_u_xstar(
    x_star: np.ndarray,
    grid_x: np.ndarray,
    grid_u_row: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of u(x*) from a GRF grid row."""
    x_star = np.asarray(x_star, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64)
    grid_u_row = np.asarray(grid_u_row, dtype=np.float64)
    return np.interp(x_star, grid_x, grid_u_row)


def branch_cfl_from_grf_sensor_u(
    sensor_u: np.ndarray,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> np.ndarray:
    """
    Branch CFL features from sensor velocities.

    sensor_u: (K,) or (N, K) physical u (m/d) -> CFL = u * T_max / L.
    """
    sensor_u = np.asarray(sensor_u, dtype=np.float64)
    if sensor_u.ndim == 1:
        return cfl_from_u_np(sensor_u.reshape(1, -1), l_m=l_m, t_max_d=t_max_d).flatten()
    return cfl_from_u_np(sensor_u, l_m=l_m, t_max_d=t_max_d)


def zone_u_at_xstar(
    x_star: np.ndarray,
    u_case: UCases,
    *,
    l_m: float = L_DEFAULT,
) -> np.ndarray:
    """Piecewise-constant zone velocity at each x* (validation / zoned media)."""
    x_star = np.asarray(x_star, dtype=np.float64)
    u = np.asarray(u_case, dtype=np.float64).reshape(4)
    zidx = zone_index_xstar_np(x_star, l_m=l_m)
    return u[zidx]


def branch_cfl_from_zone_case(
    u_case: UCases,
    sensor_x: np.ndarray,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> np.ndarray:
    """Sample zoned velocity at sensor locations and return CFL branch vector (K,)."""
    sensor_u = zone_u_at_xstar(sensor_x, u_case, l_m=l_m)
    return branch_cfl_from_grf_sensor_u(sensor_u, l_m=l_m, t_max_d=t_max_d)


def load_grf_cases_npz(path: Path) -> GRFTrainingBatch:
    """Load cached GRF training batch from ``train_grf_cases.npz``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GRF cache: {path}")
    data = np.load(path)
    return GRFTrainingBatch(
        sensor_u=np.asarray(data["sensor_u"], dtype=np.float64),
        grid_u=np.asarray(data["grid_u"], dtype=np.float64),
        grid_x=np.asarray(data["grid_x"], dtype=np.float64),
        sensor_x=np.asarray(data["sensor_x"], dtype=np.float64),
    )


def save_grf_cases_npz(path: Path, batch: GRFTrainingBatch) -> Path:
    """Persist GRF training batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sensor_u=batch.sensor_u,
        grid_u=batch.grid_u,
        grid_x=batch.grid_x,
        sensor_x=batch.sensor_x,
    )
    return path


def load_or_generate_grf_train_cases(
    npz_path: Path,
    n_train: int,
    sensor_x: np.ndarray,
    *,
    u_lo: float,
    u_hi: float,
    corr_length: float,
    grid_n: int,
    seed: int,
    reload: bool = False,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> GRFTrainingBatch:
    """Load ``train_grf_cases.npz`` or draw a new GRF batch."""
    npz_path = Path(npz_path)
    if npz_path.is_file() and not reload:
        batch = load_grf_cases_npz(npz_path)
        if batch.sensor_u.shape[0] != n_train:
            raise ValueError(
                f"Cached GRF batch has N={batch.sensor_u.shape[0]}, "
                f"expected n_train={n_train}"
            )
        if batch.sensor_x.size != sensor_x.size or not np.allclose(
            batch.sensor_x, sensor_x
        ):
            raise ValueError(
                "Cached sensor_x does not match requested sensor locations"
            )
        return batch

    batch = draw_grf_velocity_fields(
        n_train,
        sensor_x,
        u_lo=u_lo,
        u_hi=u_hi,
        corr_length=corr_length,
        grid_n=grid_n,
        seed=seed,
        l_m=l_m,
        t_max_d=t_max_d,
    )
    save_grf_cases_npz(npz_path, batch)
    return batch
