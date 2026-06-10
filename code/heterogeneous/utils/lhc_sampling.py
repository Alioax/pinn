# -*- coding: utf-8 -*-
"""Latin hypercube sampling for zone velocities (training parameter design)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import qmc

UTuple = tuple[float, float, float, float]
TrainDesign = Literal["capacity", "lhc", "maximin", "anchored"]


def comsol_validation_u_grid(
    levels: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> np.ndarray:
    """Full factorial grid used in COMSOL export, shape (81, 4)."""
    combos = list(itertools.product(levels, repeat=4))
    return np.array(combos, dtype=np.float64)


def cube_corners_u_grid(
    u_lo: float = 0.01,
    u_hi: float = 0.05,
) -> np.ndarray:
    """Cube corners at ``{u_lo, u_hi}^4``, shape (16, 4)."""
    combos = list(itertools.product((u_lo, u_hi), repeat=4))
    return np.array(combos, dtype=np.float64)


def _is_near_any_row(u: np.ndarray, grid: np.ndarray, atol: float) -> bool:
    return bool(np.any(np.all(np.isclose(u, grid, rtol=0.0, atol=atol), axis=1)))


def _scale_unit_lhc(draw: np.ndarray, u_lo: float, u_hi: float) -> np.ndarray:
    lo = np.full(4, u_lo, dtype=np.float64)
    hi = np.full(4, u_hi, dtype=np.float64)
    return qmc.scale(draw, lo, hi)


def _min_pairwise_distance(samples: np.ndarray) -> float:
    """Minimum Euclidean distance between distinct rows (maximin criterion)."""
    n = samples.shape[0]
    if n < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(n - 1):
        diff = samples[i + 1 :] - samples[i]
        dists = np.linalg.norm(diff, axis=1)
        min_dist = min(min_dist, float(dists.min()))
    return min_dist


def _collect_lhc_rows(
    sampler: qmc.LatinHypercube,
    n: int,
    u_lo: float,
    u_hi: float,
    *,
    exclude_near: np.ndarray | None,
    exclude_atol: float,
    batch: int,
) -> np.ndarray | None:
    draw = _scale_unit_lhc(sampler.random(n=batch), u_lo, u_hi)
    if exclude_near is None or exclude_near.size == 0:
        return draw[:n].copy()

    keep: list[np.ndarray] = []
    for row in draw:
        if _is_near_any_row(row, exclude_near, exclude_atol):
            continue
        keep.append(row)
        if len(keep) >= n:
            return np.stack(keep, axis=0)
    return None


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
    batch = max(n * 4, n + 64)
    for attempt in range(max_attempts):
        result = _collect_lhc_rows(
            sampler,
            n,
            u_lo,
            u_hi,
            exclude_near=exclude_near,
            exclude_atol=exclude_atol,
            batch=batch,
        )
        if result is not None:
            return result
        sampler = qmc.LatinHypercube(d=4, seed=seed + 1 + attempt)

    raise RuntimeError(
        f"Could not collect {n} LHC samples away from excluded grid after "
        f"{max_attempts} attempts; relax exclude_atol or increase max_attempts."
    )


def generate_maximin_lhc_u_samples(
    n: int,
    u_lo: float,
    u_hi: float,
    *,
    seed: int,
    exclude_near: np.ndarray | None = None,
    exclude_atol: float = 1e-9,
    n_candidates: int = 200,
    max_attempts: int = 20,
) -> np.ndarray:
    """
    Select among ``n_candidates`` LHC draws the one with largest minimum
    pairwise distance (maximin / space-filling criterion).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if u_hi <= u_lo:
        raise ValueError("u_hi must exceed u_lo")

    batch = max(n * 4, n + 64)
    best_samples: np.ndarray | None = None
    best_score = -1.0

    for k in range(n_candidates):
        sampler = qmc.LatinHypercube(d=4, seed=seed + k)
        for attempt in range(max_attempts):
            samples = _collect_lhc_rows(
                sampler,
                n,
                u_lo,
                u_hi,
                exclude_near=exclude_near,
                exclude_atol=exclude_atol,
                batch=batch,
            )
            if samples is None:
                sampler = qmc.LatinHypercube(d=4, seed=seed + k + (1 + attempt) * n_candidates)
                continue
            score = _min_pairwise_distance(samples)
            if score > best_score:
                best_score = score
                best_samples = samples
            break

    if best_samples is None:
        raise RuntimeError(
            f"Could not collect {n} maximin LHC samples away from excluded grid "
            f"after {n_candidates} candidates."
        )
    return best_samples


def _deduplicate_rows(
    rows: np.ndarray,
    *,
    exclude_near: np.ndarray | None,
    exclude_atol: float,
) -> np.ndarray:
    """Drop duplicate / near-duplicate rows; optionally skip excluded tuples."""
    kept: list[np.ndarray] = []
    for row in rows:
        if exclude_near is not None and exclude_near.size > 0:
            if _is_near_any_row(row, exclude_near, exclude_atol):
                continue
        if kept and np.any(
            np.all(np.isclose(row, np.stack(kept), rtol=0.0, atol=exclude_atol), axis=1)
        ):
            continue
        kept.append(row)
    if not kept:
        raise RuntimeError("No unique training tuples remained after deduplication.")
    return np.stack(kept, axis=0)


def generate_anchored_lhc_u_samples(
    n: int,
    u_lo: float,
    u_hi: float,
    *,
    seed: int,
    exclude_near: np.ndarray | None = None,
    exclude_atol: float = 1e-9,
    max_attempts: int = 20,
) -> np.ndarray:
    """``n`` plain LHC rows plus 16 cube corners, deduplicated."""
    lhc = generate_lhc_u_samples(
        n,
        u_lo,
        u_hi,
        seed=seed,
        exclude_near=exclude_near,
        exclude_atol=exclude_atol,
        max_attempts=max_attempts,
    )
    corners = cube_corners_u_grid(u_lo, u_hi)
    combined = np.concatenate([lhc, corners], axis=0)
    # Corners are kept even when they coincide with COMSOL grid vertices.
    return _deduplicate_rows(
        combined,
        exclude_near=None,
        exclude_atol=exclude_atol,
    )


def _generate_for_design(
    design: TrainDesign,
    n_train: int,
    u_lo: float,
    u_hi: float,
    *,
    seed: int,
    exclude_near: np.ndarray | None,
    exclude_atol: float,
) -> np.ndarray:
    if design == "capacity":
        return comsol_validation_u_grid()
    if design == "lhc":
        return generate_lhc_u_samples(
            n_train,
            u_lo,
            u_hi,
            seed=seed,
            exclude_near=exclude_near,
            exclude_atol=exclude_atol,
        )
    if design == "maximin":
        return generate_maximin_lhc_u_samples(
            n_train,
            u_lo,
            u_hi,
            seed=seed,
            exclude_near=exclude_near,
            exclude_atol=exclude_atol,
        )
    if design == "anchored":
        return generate_anchored_lhc_u_samples(
            n_train,
            u_lo,
            u_hi,
            seed=seed,
            exclude_near=exclude_near,
            exclude_atol=exclude_atol,
        )
    raise ValueError(f"Unknown design: {design!r}")


def expected_train_count(design: TrainDesign, n_train: int) -> int | None:
    """Expected row count when reloading CSV; ``None`` if count may vary."""
    if design == "capacity":
        return 81
    if design in ("lhc", "maximin"):
        return n_train
    return None  # anchored: n_train + corners minus duplicates


def load_or_generate_train_u_cases(
    design: TrainDesign,
    n_train: int,
    csv_path: Path,
    *,
    u_lo: float,
    u_hi: float,
    seed: int,
    exclude_comsol_grid: bool,
    exclude_atol: float = 1e-9,
    reload: bool = False,
) -> np.ndarray:
    """
    Load training velocities from ``csv_path`` or generate for ``design``.

    For ``capacity``, ``n_train`` is ignored (always 81 grid tuples).
    """
    if reload and csv_path.is_file():
        csv_path.unlink()

    expected = expected_train_count(design, n_train)

    if csv_path.is_file():
        u_cases = load_u_cases_csv(csv_path)
        if expected is not None and u_cases.shape[0] != expected:
            raise ValueError(
                f"{csv_path} has {u_cases.shape[0]} rows; "
                f"expected {expected} for design={design!r}, n_train={n_train}. "
                f"Use reload=True to regenerate."
            )
        return u_cases

    exclude = comsol_validation_u_grid() if exclude_comsol_grid else None
    u_cases = _generate_for_design(
        design,
        n_train,
        u_lo,
        u_hi,
        seed=seed,
        exclude_near=exclude,
        exclude_atol=exclude_atol,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_u_cases_csv(csv_path, u_cases)
    return u_cases


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
