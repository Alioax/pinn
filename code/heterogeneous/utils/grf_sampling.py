# -*- coding: utf-8 -*-
"""Gaussian random field (GRF) and piecewise velocity sampling for PINO_2 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.lhc_sampling import generate_maximin_lhc_box_samples
from utils.zone_velocity import (
    L_DEFAULT,
    T_MAX_DEFAULT,
    cfl_from_u_np,
    zone_index_xstar_np,
)

UCases = tuple[float, float, float, float]

PIECEWISE_ZONE_COUNTS_DEFAULT: tuple[int, ...] = (2, 3, 4, 5)
GRF_CORR_LENGTHS_DEFAULT: tuple[float, ...] = (0.05, 0.1)
CASE_KIND_GRF = 0
CASE_KIND_PIECEWISE = 1
ZONE9_BRANCH_DIM = 10
MAX_PADDED_ZONES = 5
MAX_PADDED_INTERFACES = 4


def _split_counts(n: int, n_parts: int) -> tuple[int, ...]:
    """Split ``n`` as evenly as possible across ``n_parts`` buckets."""
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    base, rem = divmod(n, n_parts)
    return tuple(base + (1 if i < rem else 0) for i in range(n_parts))


@dataclass(frozen=True)
class MixedTrainConfig:
    """GRF + piecewise training batch composition."""

    n_grf: int = 300
    n_piecewise_per_zone_count: int = 50
    piecewise_zone_counts: tuple[int, ...] = PIECEWISE_ZONE_COUNTS_DEFAULT
    n_piecewise_per_zone_list: tuple[int, ...] | None = None
    grf_corr_lengths: tuple[float, ...] = GRF_CORR_LENGTHS_DEFAULT
    iface_margin: float = 0.02
    min_zone_frac: float | None = None

    def grf_counts_per_length(self) -> tuple[int, ...]:
        """GRF case counts per entry in :attr:`grf_corr_lengths`."""
        if not self.grf_corr_lengths:
            raise ValueError("grf_corr_lengths must be non-empty when n_grf > 0")
        return _split_counts(self.n_grf, len(self.grf_corr_lengths))

    def piecewise_counts(self) -> tuple[int, ...]:
        """Per-zone-count sample counts (explicit list overrides scalar)."""
        if self.n_piecewise_per_zone_list is not None:
            if len(self.n_piecewise_per_zone_list) != len(self.piecewise_zone_counts):
                raise ValueError(
                    "n_piecewise_per_zone_list length must match piecewise_zone_counts"
                )
            return self.n_piecewise_per_zone_list
        return tuple(
            self.n_piecewise_per_zone_count for _ in self.piecewise_zone_counts
        )

    @property
    def n_piecewise(self) -> int:
        return sum(self.piecewise_counts())

    @property
    def n_total(self) -> int:
        return self.n_grf + self.n_piecewise


@dataclass(frozen=True)
class GRFTrainingBatch:
    """Training media: sensor velocities + fine grid fields."""

    sensor_u: np.ndarray  # (N, K) physical u at sensors (m/d)
    grid_u: np.ndarray  # (N, G) physical u on fine grid (m/d)
    grid_x: np.ndarray  # (G,) x* in [0, 1]
    sensor_x: np.ndarray  # (K,) x* sensor locations
    case_kind: np.ndarray | None = None  # (N,) 0=GRF, 1=piecewise
    case_n_zones: np.ndarray | None = None  # (N,) 0 for GRF else zone count
    case_grf_corr_length: np.ndarray | None = None  # (N,) ell for GRF, 0 for piecewise
    zone9_branch: np.ndarray | None = None  # (N, 10) padded (u,b,n_zones) branch
    piecewise_interfaces: np.ndarray | None = None  # (N, 4) padded internal interfaces
    piecewise_zone_u: np.ndarray | None = None  # (N, 5) padded zone velocities


def default_sensor_xstar(
    k: int,
    *,
    include_interfaces: bool = False,
) -> np.ndarray:
    """Uniform sensor locations on [0, 1]."""
    if k < 2:
        raise ValueError("k must be >= 2")
    return np.linspace(0.0, 1.0, k, dtype=np.float64)


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


def interpolate_u_xstar(
    x_star: np.ndarray,
    grid_x: np.ndarray,
    grid_u_row: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of u(x*) from a grid row."""
    x_star = np.asarray(x_star, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64)
    grid_u_row = np.asarray(grid_u_row, dtype=np.float64)
    return np.interp(x_star, grid_x, grid_u_row)


def encode_zone9_branch_vector(
    n_zones: int,
    zone_u: np.ndarray,
    interfaces: np.ndarray,
) -> np.ndarray:
    """
    Padded 10-dim branch: (u1..u5, b1..b4, n_zones).

    Velocities repeat u_k in trailing slots; unused boundaries are 1.0.
    """
    k = int(n_zones)
    if not (1 <= k <= MAX_PADDED_ZONES):
        raise ValueError(f"n_zones must lie in [1, {MAX_PADDED_ZONES}], got {k}")
    u = np.asarray(zone_u, dtype=np.float64).reshape(-1)
    b = np.asarray(interfaces, dtype=np.float64).reshape(-1)
    if u.size < k:
        raise ValueError(f"Expected at least {k} zone velocities, got {u.size}")
    if k > 1 and b.size < k - 1:
        raise ValueError(f"Expected at least {k - 1} interfaces, got {b.size}")

    u_pad = np.empty(MAX_PADDED_ZONES, dtype=np.float64)
    u_pad[:k] = u[:k]
    u_pad[k:] = u[k - 1]

    b_pad = np.ones(MAX_PADDED_INTERFACES, dtype=np.float64)
    if k > 1:
        b_pad[: k - 1] = b[: k - 1]

    return np.concatenate([u_pad, b_pad, [float(k)]])


def comsol_4zone_branch_vector(
    u_case: UCases,
    *,
    l_m: float = L_DEFAULT,
) -> np.ndarray:
    """Encode fixed 4-zone COMSOL validation case into the 9-vector branch form."""
    del l_m
    u = np.asarray(u_case, dtype=np.float64).reshape(4)
    interfaces = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    return encode_zone9_branch_vector(4, u, interfaces)


def piecewise_u_at_xstar(
    x_star: np.ndarray,
    interfaces: np.ndarray,
    zone_u: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant velocity at each x* from sorted interfaces and zone u."""
    x_star = np.asarray(x_star, dtype=np.float64)
    interfaces = np.asarray(interfaces, dtype=np.float64)
    zone_u = np.asarray(zone_u, dtype=np.float64)
    n_zones = zone_u.size
    if interfaces.size != n_zones - 1:
        raise ValueError(
            f"Expected {n_zones - 1} interfaces for {n_zones} zones, "
            f"got {interfaces.size}"
        )
    bins = np.concatenate([[0.0], interfaces, [1.0 + 1e-12]])
    idx = np.digitize(x_star, bins, right=False) - 1
    return zone_u[np.clip(idx, 0, n_zones - 1)]


def _fields_from_grid_rows(
    grid_u: np.ndarray,
    grid_x: np.ndarray,
    sensor_x: np.ndarray,
) -> np.ndarray:
    n = grid_u.shape[0]
    k = sensor_x.size
    sensor_u = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        sensor_u[i] = interpolate_u_xstar(sensor_x, grid_x, grid_u[i])
    return sensor_u


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw ``n`` smooth GRF velocity fields on a fine x* grid.

    Returns ``(grid_u, sensor_u, grid_x)``.
    """
    del l_m, t_max_d
    if n < 1:
        raise ValueError("n must be >= 1")
    if grid_n < 3:
        raise ValueError("grid_n must be >= 3")

    grid_x = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    sensor_x = np.asarray(sensor_x, dtype=np.float64)

    rng = np.random.default_rng(seed)
    cov = _squared_exponential_covariance(grid_x, corr_length=corr_length)
    cov = cov + 1e-10 * np.eye(grid_n, dtype=np.float64)
    chol = np.linalg.cholesky(cov)

    grid_u = np.empty((n, grid_n), dtype=np.float64)
    for i in range(n):
        xi = chol @ rng.standard_normal(grid_n)
        u = u_lo + (u_hi - u_lo) * _sigmoid(xi)
        grid_u[i] = np.clip(u, u_lo, u_hi)

    sensor_u = _fields_from_grid_rows(grid_u, grid_x, sensor_x)
    return grid_u, sensor_u, grid_x


def _zone_widths_from_simplex_params(
    raw_p: np.ndarray,
    *,
    n_zones: int,
    min_frac: float,
) -> np.ndarray:
    """Map LHC rows to zone widths in [min_frac, 1] summing to 1."""
    if n_zones < 1:
        raise ValueError("n_zones must be >= 1")
    if not (0.0 < min_frac < 1.0):
        raise ValueError("min_frac must lie in (0, 1)")
    remaining = 1.0 - n_zones * min_frac
    if remaining < -1e-12:
        raise ValueError(
            f"min_frac={min_frac:g} too large for {n_zones} zones "
            f"(need {n_zones * min_frac:g} <= 1)"
        )
    if n_zones == 1:
        return np.ones((raw_p.shape[0], 1), dtype=np.float64)
    p = np.asarray(raw_p, dtype=np.float64)
    if p.shape[1] != n_zones:
        raise ValueError(
            f"Expected {n_zones} simplex parameters, got shape {p.shape}"
        )
    p = p / p.sum(axis=1, keepdims=True)
    return min_frac + remaining * p


def _interfaces_from_zone_widths(widths: np.ndarray) -> np.ndarray:
    """Cumulative zone widths -> sorted interface locations in (0, 1)."""
    if widths.shape[1] < 2:
        return np.empty((widths.shape[0], 0), dtype=np.float64)
    return np.cumsum(widths[:, :-1], axis=1)


def draw_piecewise_velocity_fields(
    n: int,
    n_zones: int,
    sensor_x: np.ndarray,
    *,
    u_lo: float = 0.01,
    u_hi: float = 0.05,
    iface_margin: float = 0.02,
    min_zone_frac: float | None = None,
    grid_n: int = 201,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw ``n`` piecewise-constant velocity fields with ``n_zones`` zones.

    Interface locations and zone velocities are jointly sampled via maximin LHC.
    When ``min_zone_frac`` is set, zone widths are sampled with each zone at
    least that fraction of the domain (``n_zones`` may be 1 for uniform fields).
    Otherwise interfaces are sampled in ``[iface_margin, 1-iface_margin]`` (legacy).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if grid_n < 3:
        raise ValueError("grid_n must be >= 3")

    if min_zone_frac is not None:
        if n_zones < 1:
            raise ValueError("n_zones must be >= 1 when min_zone_frac is set")
        if n_zones == 1:
            d = 1
            lo = np.array([u_lo], dtype=np.float64)
            hi = np.array([u_hi], dtype=np.float64)
            params = generate_maximin_lhc_box_samples(n, lo, hi, seed=seed)
            interfaces = np.empty((n, 0), dtype=np.float64)
            zone_u = params.reshape(n, 1)
        else:
            d = n_zones + n_zones
            lo = np.concatenate(
                [
                    np.zeros(n_zones, dtype=np.float64),
                    np.full(n_zones, u_lo, dtype=np.float64),
                ]
            )
            hi = np.concatenate(
                [
                    np.ones(n_zones, dtype=np.float64),
                    np.full(n_zones, u_hi, dtype=np.float64),
                ]
            )
            params = generate_maximin_lhc_box_samples(n, lo, hi, seed=seed)
            widths = _zone_widths_from_simplex_params(
                params[:, :n_zones],
                n_zones=n_zones,
                min_frac=min_zone_frac,
            )
            interfaces = _interfaces_from_zone_widths(widths)
            zone_u = params[:, n_zones:]
    else:
        if n_zones < 2:
            raise ValueError("n_zones must be >= 2 when min_zone_frac is not set")
        if not (0.0 < iface_margin < 0.5):
            raise ValueError("iface_margin must lie in (0, 0.5)")
        n_if = n_zones - 1
        d = n_if + n_zones
        lo = np.concatenate(
            [
                np.full(n_if, iface_margin, dtype=np.float64),
                np.full(n_zones, u_lo, dtype=np.float64),
            ]
        )
        hi = np.concatenate(
            [
                np.full(n_if, 1.0 - iface_margin, dtype=np.float64),
                np.full(n_zones, u_hi, dtype=np.float64),
            ]
        )
        params = generate_maximin_lhc_box_samples(n, lo, hi, seed=seed)
        interfaces = np.sort(params[:, :n_if], axis=1)
        zone_u = params[:, n_if:]

    grid_x = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)
    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    grid_u = np.empty((n, grid_n), dtype=np.float64)
    for i in range(n):
        grid_u[i] = piecewise_u_at_xstar(grid_x, interfaces[i], zone_u[i])

    sensor_u = _fields_from_grid_rows(grid_u, grid_x, sensor_x)
    return grid_u, sensor_u, grid_x, interfaces, zone_u


def draw_mixed_train_cases(
    config: MixedTrainConfig,
    sensor_x: np.ndarray,
    *,
    u_lo: float = 0.01,
    u_hi: float = 0.05,
    grid_n: int = 201,
    seed: int = 0,
    branch_mode: str = "sensor",
) -> GRFTrainingBatch:
    """Build a mixed GRF + piecewise training batch."""
    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    if branch_mode == "zone9" and config.n_grf > 0:
        raise ValueError("branch_mode=zone9 requires n_grf=0 (piecewise-only training)")
    n_total = config.n_total
    grid_u = np.empty((n_total, grid_n), dtype=np.float64)
    case_kind = np.empty(n_total, dtype=np.int8)
    case_n_zones = np.empty(n_total, dtype=np.int8)
    case_grf_corr_length = np.zeros(n_total, dtype=np.float64)
    zone9_branch: np.ndarray | None = None
    piecewise_interfaces: np.ndarray | None = None
    piecewise_zone_u: np.ndarray | None = None
    if branch_mode == "zone9":
        zone9_branch = np.empty((n_total, ZONE9_BRANCH_DIM), dtype=np.float64)
        piecewise_interfaces = np.zeros((n_total, MAX_PADDED_INTERFACES), dtype=np.float64)
        piecewise_zone_u = np.zeros((n_total, MAX_PADDED_ZONES), dtype=np.float64)

    offset = 0
    sensor_blocks: list[np.ndarray] = []
    grid_x: np.ndarray | None = None

    if config.n_grf > 0:
        grf_counts = config.grf_counts_per_length()
        for j, (ell, n_ell) in enumerate(zip(config.grf_corr_lengths, grf_counts, strict=True)):
            if n_ell <= 0:
                continue
            g_grid, g_sensor, grid_x_draw = draw_grf_velocity_fields(
                n_ell,
                sensor_x,
                u_lo=u_lo,
                u_hi=u_hi,
                corr_length=ell,
                grid_n=grid_n,
                seed=seed + 100 * (j + 1),
            )
            if grid_x is None:
                grid_x = grid_x_draw
            grid_u[offset : offset + n_ell] = g_grid
            case_kind[offset : offset + n_ell] = CASE_KIND_GRF
            case_n_zones[offset : offset + n_ell] = 0
            case_grf_corr_length[offset : offset + n_ell] = ell
            offset += n_ell
            sensor_blocks.append(g_sensor)

    if grid_x is None:
        grid_x = np.linspace(0.0, 1.0, grid_n, dtype=np.float64)

    for j, (n_zones, n_pw) in enumerate(
        zip(config.piecewise_zone_counts, config.piecewise_counts(), strict=True)
    ):
        if n_pw <= 0:
            continue
        pw_grid, pw_sensor, _, pw_ifaces, pw_zone_u = draw_piecewise_velocity_fields(
            n_pw,
            n_zones,
            sensor_x,
            u_lo=u_lo,
            u_hi=u_hi,
            iface_margin=config.iface_margin,
            min_zone_frac=config.min_zone_frac,
            grid_n=grid_n,
            seed=seed + 1000 * (j + 1) + n_zones,
        )
        grid_u[offset : offset + n_pw] = pw_grid
        case_kind[offset : offset + n_pw] = CASE_KIND_PIECEWISE
        case_n_zones[offset : offset + n_pw] = n_zones
        if branch_mode == "zone9":
            assert zone9_branch is not None
            assert piecewise_interfaces is not None
            assert piecewise_zone_u is not None
            for i in range(n_pw):
                row = offset + i
                piecewise_zone_u[row, :n_zones] = pw_zone_u[i]
                if n_zones > 1:
                    piecewise_interfaces[row, : n_zones - 1] = pw_ifaces[i]
                zone9_branch[row] = encode_zone9_branch_vector(
                    n_zones, pw_zone_u[i], pw_ifaces[i]
                )
        offset += n_pw
        sensor_blocks.append(pw_sensor)

    sensor_u = np.concatenate(sensor_blocks, axis=0)
    return GRFTrainingBatch(
        sensor_u=sensor_u,
        grid_u=grid_u,
        grid_x=grid_x,
        sensor_x=sensor_x,
        case_kind=case_kind,
        case_n_zones=case_n_zones,
        case_grf_corr_length=case_grf_corr_length,
        zone9_branch=zone9_branch,
        piecewise_interfaces=piecewise_interfaces,
        piecewise_zone_u=piecewise_zone_u,
    )


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


def _mixed_config_matches_batch(
    batch: GRFTrainingBatch,
    config: MixedTrainConfig,
) -> bool:
    if batch.case_kind is None or batch.case_n_zones is None:
        return False
    if batch.case_grf_corr_length is None:
        return False
    if batch.sensor_u.shape[0] != config.n_total:
        return False
    n_grf = int(np.sum(batch.case_kind == CASE_KIND_GRF))
    if n_grf != config.n_grf:
        return False
    if config.n_grf > 0:
        grf_counts = config.grf_counts_per_length()
        for ell, n_expected in zip(config.grf_corr_lengths, grf_counts, strict=True):
            n_have = int(
                np.sum(
                    (batch.case_kind == CASE_KIND_GRF)
                    & np.isclose(batch.case_grf_corr_length, ell, rtol=0.0, atol=1e-12)
                )
            )
            if n_have != n_expected:
                return False
    for n_zones, n_expected in zip(
        config.piecewise_zone_counts, config.piecewise_counts(), strict=True
    ):
        n_pw = int(np.sum(batch.case_n_zones == n_zones))
        if n_pw != n_expected:
            return False
    return True


def load_grf_cases_npz(path: Path) -> GRFTrainingBatch:
    """Load cached training batch from ``train_grf_cases.npz``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GRF cache: {path}")
    data = np.load(path)
    case_kind = data["case_kind"] if "case_kind" in data else None
    case_n_zones = data["case_n_zones"] if "case_n_zones" in data else None
    case_grf_corr_length = (
        data["case_grf_corr_length"] if "case_grf_corr_length" in data else None
    )
    zone9_branch = data["zone9_branch"] if "zone9_branch" in data else None
    piecewise_interfaces = (
        data["piecewise_interfaces"] if "piecewise_interfaces" in data else None
    )
    piecewise_zone_u = data["piecewise_zone_u"] if "piecewise_zone_u" in data else None
    return GRFTrainingBatch(
        sensor_u=np.asarray(data["sensor_u"], dtype=np.float64),
        grid_u=np.asarray(data["grid_u"], dtype=np.float64),
        grid_x=np.asarray(data["grid_x"], dtype=np.float64),
        sensor_x=np.asarray(data["sensor_x"], dtype=np.float64),
        case_kind=None if case_kind is None else np.asarray(case_kind, dtype=np.int8),
        case_n_zones=None
        if case_n_zones is None
        else np.asarray(case_n_zones, dtype=np.int8),
        case_grf_corr_length=None
        if case_grf_corr_length is None
        else np.asarray(case_grf_corr_length, dtype=np.float64),
        zone9_branch=None
        if zone9_branch is None
        else np.asarray(zone9_branch, dtype=np.float64),
        piecewise_interfaces=None
        if piecewise_interfaces is None
        else np.asarray(piecewise_interfaces, dtype=np.float64),
        piecewise_zone_u=None
        if piecewise_zone_u is None
        else np.asarray(piecewise_zone_u, dtype=np.float64),
    )


def save_grf_cases_npz(path: Path, batch: GRFTrainingBatch) -> Path:
    """Persist training batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sensor_u": batch.sensor_u,
        "grid_u": batch.grid_u,
        "grid_x": batch.grid_x,
        "sensor_x": batch.sensor_x,
    }
    if batch.case_kind is not None:
        payload["case_kind"] = batch.case_kind
    if batch.case_n_zones is not None:
        payload["case_n_zones"] = batch.case_n_zones
    if batch.case_grf_corr_length is not None:
        payload["case_grf_corr_length"] = batch.case_grf_corr_length
    if batch.zone9_branch is not None:
        payload["zone9_branch"] = batch.zone9_branch
    if batch.piecewise_interfaces is not None:
        payload["piecewise_interfaces"] = batch.piecewise_interfaces
    if batch.piecewise_zone_u is not None:
        payload["piecewise_zone_u"] = batch.piecewise_zone_u
    np.savez(path, **payload)
    return path


def load_or_generate_grf_train_cases(
    npz_path: Path,
    sensor_x: np.ndarray,
    *,
    mixed_config: MixedTrainConfig,
    u_lo: float,
    u_hi: float,
    grid_n: int,
    seed: int,
    reload: bool = False,
    l_m: float = L_DEFAULT,
    t_max_d: float = T_MAX_DEFAULT,
    branch_mode: str = "sensor",
) -> GRFTrainingBatch:
    """Load ``train_grf_cases.npz`` or draw a new mixed GRF + piecewise batch."""
    del l_m, t_max_d
    npz_path = Path(npz_path)
    sensor_x = np.asarray(sensor_x, dtype=np.float64)
    n_train = mixed_config.n_total

    if npz_path.is_file() and not reload:
        batch = load_grf_cases_npz(npz_path)
        if batch.sensor_u.shape[0] != n_train:
            raise ValueError(
                f"Cached batch has N={batch.sensor_u.shape[0]}, "
                f"expected n_train={n_train}"
            )
        if batch.sensor_x.size != sensor_x.size or not np.allclose(
            batch.sensor_x, sensor_x
        ):
            raise ValueError(
                "Cached sensor_x does not match requested sensor locations"
            )
        if not _mixed_config_matches_batch(batch, mixed_config):
            raise ValueError(
                "Cached batch composition does not match requested mixed config. "
                "Use reload=True to regenerate."
            )
        if branch_mode == "zone9" and batch.zone9_branch is None:
            raise ValueError(
                "Cached batch lacks zone9_branch metadata. Use reload=True to regenerate."
            )
        return batch

    batch = draw_mixed_train_cases(
        mixed_config,
        sensor_x,
        u_lo=u_lo,
        u_hi=u_hi,
        grid_n=grid_n,
        seed=seed,
        branch_mode=branch_mode,
    )
    save_grf_cases_npz(npz_path, batch)
    return batch
