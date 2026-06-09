# -*- coding: utf-8 -*-
"""Parser for COMSOL 4-zone velocity exports."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

UTuple = tuple[float, float, float, float]

_ROW_META = re.compile(
    r"t\s*=\s*([0-9.]+)\s*,\s*u1\s*=\s*([0-9.]+)\s*,\s*u2\s*=\s*([0-9.]+)\s*,\s*"
    r"u3\s*=\s*([0-9.]+)\s*,\s*u4\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_comsol_grid(path: Path) -> np.ndarray:
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    for i, ln in enumerate(lines):
        if ln.strip().startswith("% Grid"):
            if i + 1 >= len(lines):
                raise ValueError("Missing grid line after '% Grid'")
            return np.array([float(p) for p in lines[i + 1].split()], dtype=np.float64)
    raise ValueError(f"Could not find '% Grid' in {path}")


def _parse_data_row(line: str, n_x: int) -> np.ndarray:
    vals = np.array([float(s) for s in line.split()], dtype=np.float64)
    if vals.size == n_x + 1:
        return vals[1:]
    if vals.size == n_x:
        return vals
    if vals.size > n_x:
        return vals[-n_x:]
    raise ValueError(f"Data row has {vals.size} floats; expected at least {n_x}")


def _u_match(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def list_parameter_combos(path: Path, *, tol: float = 1e-12) -> list[UTuple]:
    """Unique (u1, u2, u3, u4) tuples appearing in the export."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    combos: set[UTuple] = set()
    for ln in lines:
        m = _ROW_META.search(ln)
        if m:
            combos.add(
                (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
            )
    return sorted(combos, key=lambda t: t)


def load_case(path: Path, u_target: UTuple, *, tol: float = 1e-12) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return (x_grid_m, {time_days: c_array}) for one velocity quadruple."""
    x_arr = parse_comsol_grid(path)
    n_x = x_arr.size
    u1_t, u2_t, u3_t, u4_t = u_target
    series: dict[float, np.ndarray] = {}

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    j = 0
    while j < len(lines):
        m = _ROW_META.search(lines[j])
        if m:
            t_val = float(m.group(1))
            u1, u2, u3, u4 = (
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
                float(m.group(5)),
            )
            if (
                _u_match(u1, u1_t, tol)
                and _u_match(u2, u2_t, tol)
                and _u_match(u3, u3_t, tol)
                and _u_match(u4, u4_t, tol)
                and j + 1 < len(lines)
            ):
                series[t_val] = _parse_data_row(lines[j + 1], n_x)
            j += 2
            continue
        j += 1

    if not series:
        raise ValueError(f"No data blocks for u={u_target} in {path}")
    return x_arr, series
