# -*- coding: utf-8 -*-
"""Latin hypercube sampling for zone velocities (training parameter design)."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.stats import qmc

UTuple = tuple[float, float, float, float]


def comsol_validation_u_grid(
    levels: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> np.ndarray:
    """Full factorial grid used in COMSOL export, shape (81, 4)."""
    combos = list(itertools.product(levels, repeat=4))
    return np.array(combos, dtype=np.float64)


def _is_near_any_row(u: np.ndarray, grid: np.ndarray, atol: float) -> bool:
    return bool(np.any(np.all(np.isclose(u, grid, rtol=0.0, atol=atol), axis=1)))


def generate_lhc_u_samples(
    n: int,
    u_lo: float,
    u_hi: float,
    *,
    seed: int,
    exclude_near: np.ndarray | None = None,
    exclude_atol: float = 1e-9,
    max_attempts: int = 20,
) -> np.ndarray:
    """
    Draw ``n`` Latin hypercube samples in [u_lo, u_hi]^4.

    If ``exclude_near`` is provided (e.g. COMSOL 3^4 grid), resample until no row
    lies within ``exclude_atol`` of any excluded tuple (up to ``max_attempts``).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if u_hi <= u_lo:
        raise ValueError("u_hi must exceed u_lo")

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    lo = np.full(4, u_lo, dtype=np.float64)
    hi = np.full(4, u_hi, dtype=np.float64)

    batch = max(n * 4, n + 64)
    for attempt in range(max_attempts):
        draw = qmc.scale(sampler.random(n=batch), lo, hi)
        if exclude_near is None or exclude_near.size == 0:
            return draw[:n].copy()

        keep: list[np.ndarray] = []
        for row in draw:
            if _is_near_any_row(row, exclude_near, exclude_atol):
                continue
            keep.append(row)
            if len(keep) >= n:
                return np.stack(keep, axis=0)

        sampler = qmc.LatinHypercube(d=4, seed=seed + 1 + attempt)

    raise RuntimeError(
        f"Could not collect {n} LHC samples away from excluded grid after "
        f"{max_attempts} attempts; relax exclude_atol or increase max_attempts."
    )


def save_u_cases_csv(path: Path, u_cases: np.ndarray) -> None:
    """Write (N, 4) zone velocities to CSV with header u1..u4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "u1,u2,u3,u4"
    np.savetxt(path, u_cases, delimiter=",", header=header, comments="")


def load_u_cases_csv(path: Path) -> np.ndarray:
    """Load zone velocities written by :func:`save_u_cases_csv`."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 4)
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 velocity columns in {path}, got shape {data.shape}")
    return data
