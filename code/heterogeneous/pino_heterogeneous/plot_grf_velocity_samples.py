# -*- coding: utf-8 -*-
"""
Visualize GRF velocity samples u(x*) for tuning --grf-corr-length.

Uses the same draw as PINO_2 training (squared-exponential GP + sigmoid -> [u_lo, u_hi]).

Examples (from this directory):
  python plot_grf_velocity_samples.py
  python plot_grf_velocity_samples.py --corr-length 0.1
  python plot_grf_velocity_samples.py --compare 0.05,0.1,0.2,0.4 --n 5
  python plot_grf_velocity_samples.py --corr-length 0.2 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HETERO_ROOT = Path(__file__).resolve().parent.parent
_PINO_DIR = Path(__file__).resolve().parent
for _p in (_HETERO_ROOT, _PINO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.grf_sampling import default_sensor_xstar, draw_grf_velocity_fields  # noqa: E402

U_LO_DEFAULT = 0.01
U_HI_DEFAULT = 0.05
ELL_DEFAULT = 0.2
N_DEFAULT = 10
SEED_DEFAULT = 42


def _parse_ell_list(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(s.strip()) for s in text.split(",") if s.strip()]


def _draw_fields(
    n: int,
    corr_length: float,
    *,
    u_lo: float,
    u_hi: float,
    grid_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_x = default_sensor_xstar(2)
    grid_u, _sensor_u, grid_x = draw_grf_velocity_fields(
        n,
        sensor_x,
        u_lo=u_lo,
        u_hi=u_hi,
        corr_length=corr_length,
        grid_n=grid_n,
        seed=seed,
    )
    return grid_x, grid_u


def _style_axes(ax: plt.Axes, *, u_lo: float, u_hi: float) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(u_lo - 0.002, u_hi + 0.002)
    ax.axhline(u_lo, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.axhline(u_hi, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.axhline(0.5 * (u_lo + u_hi), color="0.9", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$u$ (m/d)")
    ax.grid(True, alpha=0.25)


def plot_single_panel(
    grid_x: np.ndarray,
    grid_u: np.ndarray,
    *,
    corr_length: float,
    u_lo: float,
    u_hi: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(grid_u.shape[0]):
        ax.plot(grid_x, grid_u[i], lw=1.2, alpha=0.85, label=f"sample {i}")
    _style_axes(ax, u_lo=u_lo, u_hi=u_hi)
    ax.set_title(
        rf"GRF velocity fields ($\ell={corr_length:g}$ in $x^*$, "
        rf"$u\in[{u_lo:g},{u_hi:g}]$ m/d)"
    )
    if grid_u.shape[0] <= 12:
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    return ax


def plot_compare_panels(
    ell_values: list[float],
    *,
    n: int,
    u_lo: float,
    u_hi: float,
    grid_n: int,
    seed: int,
) -> plt.Figure:
    n_panels = len(ell_values)
    n_cols = min(2, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for idx, ell in enumerate(ell_values):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        grid_x, grid_u = _draw_fields(
            n,
            ell,
            u_lo=u_lo,
            u_hi=u_hi,
            grid_n=grid_n,
            seed=seed,
        )
        for i in range(grid_u.shape[0]):
            ax.plot(grid_x, grid_u[i], lw=1.1, alpha=0.85)
        _style_axes(ax, u_lo=u_lo, u_hi=u_hi)
        ax.set_title(rf"$\ell = {ell:g}$")
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)
    fig.suptitle(
        rf"GRF samples (same seed={seed}, n={n}, $u\in[{u_lo:g},{u_hi:g}]$ m/d)",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GRF velocity u(x*) samples for tuning correlation length."
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=N_DEFAULT,
        help=f"Number of GRF curves per panel (default {N_DEFAULT}).",
    )
    parser.add_argument(
        "--corr-length",
        "--ell",
        type=float,
        default=ELL_DEFAULT,
        dest="corr_length",
        help=f"Squared-exponential correlation length in x* (default {ELL_DEFAULT}).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        metavar="ELL_LIST",
        help="Comma-separated ell values for side-by-side panels, e.g. 0.05,0.1,0.2,0.4.",
    )
    parser.add_argument("--u-lo", type=float, default=U_LO_DEFAULT, help="Lower u bound (m/d).")
    parser.add_argument("--u-hi", type=float, default=U_HI_DEFAULT, help="Upper u bound (m/d).")
    parser.add_argument(
        "--grid-n",
        type=int,
        default=201,
        help="Fine grid resolution (default 201, same as training).",
    )
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="RNG seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: results/grf_preview_ell{ell}.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive window (blocks until closed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.u_hi <= args.u_lo:
        raise SystemExit("--u-hi must exceed --u-lo")

    compare_ells = _parse_ell_list(args.compare)
    out_dir = _PINO_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare_ells:
        fig = plot_compare_panels(
            compare_ells,
            n=args.n,
            u_lo=args.u_lo,
            u_hi=args.u_hi,
            grid_n=args.grid_n,
            seed=args.seed,
        )
        out_path = args.out or out_dir / "grf_preview_compare.png"
    else:
        grid_x, grid_u = _draw_fields(
            args.n,
            args.corr_length,
            u_lo=args.u_lo,
            u_hi=args.u_hi,
            grid_n=args.grid_n,
            seed=args.seed,
        )
        fig, _ax = plt.subplots(figsize=(8, 4.5))
        plot_single_panel(
            grid_x,
            grid_u,
            corr_length=args.corr_length,
            u_lo=args.u_lo,
            u_hi=args.u_hi,
            ax=fig.axes[0],
        )
        fig.tight_layout()
        ell_tag = str(args.corr_length).replace(".", "p")
        out_path = args.out or out_dir / f"grf_preview_ell{ell_tag}.png"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
