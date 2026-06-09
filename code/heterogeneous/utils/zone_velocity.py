# -*- coding: utf-8 -*-
"""Piecewise zone CFL on [0, L] with unequal zone lengths (COMSOL geometry)."""

from __future__ import annotations

import numpy as np
import torch

L_DEFAULT = 100.0
T_MAX_DEFAULT = 1200.0
D_M2_S_DEFAULT = 1e-6
SECONDS_PER_DAY = 86400.0

# Physical zone lengths (m): z1 [0,20), z2 [20,40), z3 [40,60), z4 [60,100].
ZONE_LENGTHS_M = (20.0, 20.0, 20.0, 40.0)


def zone_boundaries_xstar(*, l_m: float = L_DEFAULT) -> tuple[float, float, float]:
    """Interface locations in x* at 20, 40, 60 m (for L=100 m)."""
    if abs(sum(ZONE_LENGTHS_M) - l_m) > 1e-9:
        raise ValueError(
            f"ZONE_LENGTHS_M must sum to l_m; got {sum(ZONE_LENGTHS_M)} vs {l_m}"
        )
    x_end = 0.0
    out: list[float] = []
    for length in ZONE_LENGTHS_M[:-1]:
        x_end += length
        out.append(x_end / l_m)
    return tuple(out)


ZONE_BOUNDARIES_XSTAR = zone_boundaries_xstar()
ZONE_INTERFACE_XSTAR = ZONE_BOUNDARIES_XSTAR
XSTAR_TICKS = (0.0, *ZONE_BOUNDARIES_XSTAR, 1.0)


def zone_digitize_bins(*, l_m: float = L_DEFAULT) -> list[float]:
    """Bins for np.digitize(..., right=False) → zone indices 0..3."""
    b1, b2, b3 = zone_boundaries_xstar(l_m=l_m)
    return [0.0, b1, b2, b3, 1.0 + 1e-12]


def pe_from_diffusivity(
    d_m2_s: float = D_M2_S_DEFAULT,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> float:
    t_seconds = t_max_d * SECONDS_PER_DAY
    return d_m2_s * t_seconds / (l_m**2)


PE_DEFAULT = pe_from_diffusivity()


def cfl_from_u(
    u: torch.Tensor,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> torch.Tensor:
    """CFL = u * T_max / L for velocity u (m/d)."""
    return u * (t_max_d / l_m)


def cfl_from_u_np(
    u: np.ndarray,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> np.ndarray:
    """Numpy CFL = u * T_max / L (same scaling as :func:`cfl_from_u`)."""
    return u * (t_max_d / l_m)


def piecewise_cfl_xstar(
    x_star: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    u3: torch.Tensor,
    u4: torch.Tensor,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> torch.Tensor:
    """
    Zone-local CFL at each x* (x* = x/L):
      z1 [0, 20 m)   -> u1; z2 [20, 40 m) -> u2; z3 [40, 60 m) -> u3; z4 [60, L] -> u4.
    """
    b1, b2, b3 = zone_boundaries_xstar(l_m=l_m)
    cfl1 = cfl_from_u(u1, l_m=l_m, t_max_d=t_max_d)
    cfl2 = cfl_from_u(u2, l_m=l_m, t_max_d=t_max_d)
    cfl3 = cfl_from_u(u3, l_m=l_m, t_max_d=t_max_d)
    cfl4 = cfl_from_u(u4, l_m=l_m, t_max_d=t_max_d)
    x = x_star if x_star.dim() == 2 else x_star.unsqueeze(-1)
    return torch.where(
        x < b1,
        cfl1,
        torch.where(x < b2, cfl2, torch.where(x < b3, cfl3, cfl4)),
    )


def piecewise_cfl_from_branch(
    x_star: torch.Tensor,
    branch_u: torch.Tensor,
    *,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
) -> torch.Tensor:
    """branch_u shape (N, 4) with columns u1..u4 (m/d)."""
    return piecewise_cfl_xstar(
        x_star,
        branch_u[:, 0:1],
        branch_u[:, 1:2],
        branch_u[:, 2:3],
        branch_u[:, 3:4],
        l_m=l_m,
        t_max_d=t_max_d,
    )


def zone_index_xstar(x_star: torch.Tensor, *, l_m: float = L_DEFAULT) -> torch.Tensor:
    """Integer zone index 0..3 for each x*."""
    b1, b2, b3 = zone_boundaries_xstar(l_m=l_m)
    x = x_star.squeeze(-1) if x_star.dim() == 2 else x_star
    idx = torch.zeros_like(x, dtype=torch.long)
    idx = torch.where(x >= b1, torch.ones_like(idx), idx)
    idx = torch.where(x >= b2, torch.full_like(idx, 2), idx)
    idx = torch.where(x >= b3, torch.full_like(idx, 3), idx)
    return idx


def zone_index_xstar_np(x_star: np.ndarray, *, l_m: float = L_DEFAULT) -> np.ndarray:
    """Numpy zone index 0..3 (same bins as :func:`zone_digitize_bins`)."""
    z = np.digitize(x_star, bins=zone_digitize_bins(l_m=l_m), right=False) - 1
    return np.clip(z, 0, 3)
