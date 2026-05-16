"""
3D scatter visualization of parametric PINN collocation points (x*, t*, Pe).

Sampling is uniform in (x*, t*, log Pe); the vertical axis shows Pe on a log scale.

Modes (set MODE below):
  regular         — tensor-product grid
  random-once     — single uniform sample in the hypercube
  random-sequence — N independent samples, one PNG per frame (fixed camera for GIFs)

CLI flags override the configuration block when provided.

Style matches code/mainline_wip/pinn_alpha/pinn_parametric_baseline.py (rcParams, domain).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Configuration — edit here; optional CLI overrides (see main())
# =============================================================================

# Which mode to run when you execute the script with no arguments:
#   "regular" | "random-once" | "random-sequence"
MODE = "random-sequence"

OUTPUT_DIR: Path | None = None  # None -> <this_folder>/results

Pe_min = 1.0
Pe_max = 1e5
t_final_star = 1.0
seed = 123456789

xxx = 10

# regular mesh resolution
nx = xxx
nt = xxx
npe = xxx

# random / random-sequence
num_points = xxx * xxx * xxx
num_frames = 40

# 3D view
elev = 30.0
azim = -58.0
marker_size = 8.0


def apply_parametric_style() -> None:
    """Font and color cycle aligned with parametric PINN scripts; higher DPI for exports."""
    mpl.rcParams["figure.dpi"] = 800
    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#FF5F05", "#13294B", "#009FD4", "#FCB316",
               "#006230", "#007E8E", "#5C0E41", "#7D3E13"]
    )


def regular_mesh(
    nx: int, nt: int, npe: int, t_final_star: float, log_pe_min: float, log_pe_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    t = np.linspace(0.0, t_final_star, nt, dtype=np.float64)
    lp = np.linspace(log_pe_min, log_pe_max, npe, dtype=np.float64)
    tt, xx, ll = np.meshgrid(t, x, lp, indexing="ij")
    return xx.ravel(), tt.ravel(), ll.ravel()


def random_uniform_hypercube(
    num_points: int,
    t_final_star: float,
    log_pe_min: float,
    log_pe_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.random(num_points, dtype=np.float64)
    t = rng.random(num_points, dtype=np.float64) * t_final_star
    log_pe = rng.random(num_points, dtype=np.float64) * (log_pe_max - log_pe_min) + log_pe_min
    return x, t, log_pe


def plot_collocation_3d(
    x: np.ndarray,
    t: np.ndarray,
    log_pe: np.ndarray,
    out_path: Path,
    t_final_star: float,
    pe_min: float,
    pe_max: float,
    elev: float,
    azim: float,
    marker_size: float,
    title: str | None,
) -> None:
    grid_alpha, grid_color, grid_linewidth = 0.3, "black", 0.4
    tick_label_fontsize = 11
    axis_label_fontsize = 12

    # We sample uniformly in ln(Pe) (see pinn_parametric_baseline.py),
    # but for nicer spacing we plot z = log10(Pe) on a *linear* z-axis.
    log_pe = np.asarray(log_pe, dtype=np.float64)
    log10_pe = log_pe / np.log(10.0)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

    ax.scatter(
        x,
        t,
        log10_pe,
        s=marker_size,
        c="C0",
        alpha=1.0,
        depthshade=True,
        edgecolors="none",
    )

    ax.set_xlabel(r"$x^*$", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$t^*$", fontsize=axis_label_fontsize)
    ax.set_zlabel(r"$\log_{10}\mathrm{Pe}$", fontsize=axis_label_fontsize)
    if title:
        ax.set_title(title, fontsize=11)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, t_final_star)
    log10_pe_min = float(np.log10(pe_min))
    log10_pe_max = float(np.log10(pe_max))
    ax.set_zlim(log10_pe_min, log10_pe_max)

    # Ticks as powers of 10 (10^0, 10^1, ..., 10^5).
    k_min = int(np.ceil(log10_pe_min))
    k_max = int(np.floor(log10_pe_max))
    if k_max >= k_min:
        exponents = list(range(k_min, k_max + 1))
        ax.set_zticks(exponents)
        # Avoid mathtext so font/size stays consistent; show numeric exponents.
        # Since z = log10(Pe), tick at k corresponds to Pe = 10^k.
        ax.set_zticklabels([f"1e{k}" for k in exponents])

    ax.view_init(elev=elev, azim=azim)

    # Make 3D tick labels match font/size used by axis labels.
    ax.tick_params(axis="x", which="major", labelsize=tick_label_fontsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_label_fontsize)
    ax.tick_params(axis="z", which="major", labelsize=tick_label_fontsize)
    for lab in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        lab.set_fontfamily("Times New Roman")
        lab.set_fontsize(tick_label_fontsize)

    # Dim the 3D panes (xy, xz, yz) so they don't overpower the points.
    pane_facecolor_rgba = (0.85, 0.85, 0.85, 0.12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor(pane_facecolor_rgba)
        axis.pane.set_edgecolor("0.85")
        axis.pane.set_linewidth(0.6)

    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.grid(True, alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 3D projections can clip outer tick labels when using tight bounding boxes.
    fig.subplots_adjust(right=0.98)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.25)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D collocation point scatter (parametric PINN domain). "
        "Defaults come from the configuration block at the top of this file."
    )
    parser.add_argument(
        "--mode",
        choices=("regular", "random-once", "random-sequence"),
        default=MODE,
        help="regular mesh, one random sample, or many frames for animation (default: MODE at top of file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="default: OUTPUT_DIR in file config, or <script_dir>/results if None",
    )
    parser.add_argument("--Pe-min", type=float, default=Pe_min, dest="pe_min")
    parser.add_argument("--Pe-max", type=float, default=Pe_max, dest="pe_max")
    parser.add_argument("--t-final", type=float, default=t_final_star, dest="t_final_star")
    parser.add_argument("--seed", type=int, default=seed, help="base seed for random modes")
    parser.add_argument("--nx", type=int, default=nx)
    parser.add_argument("--nt", type=int, default=nt)
    parser.add_argument("--npe", type=int, default=npe)
    parser.add_argument("--num-points", type=int, default=num_points, dest="num_points")
    parser.add_argument("--num-frames", type=int, default=num_frames, dest="num_frames")
    parser.add_argument("--elev", type=float, default=elev)
    parser.add_argument("--azim", type=float, default=azim)
    parser.add_argument("--marker-size", type=float, default=marker_size, dest="marker_size")
    args = parser.parse_args()

    apply_parametric_style()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir is not None else script_dir / "results"

    log_pe_min = float(np.log(args.pe_min))
    log_pe_max = float(np.log(args.pe_max))

    if args.mode == "regular":
        x, t, lp = regular_mesh(args.nx, args.nt, args.npe, args.t_final_star, log_pe_min, log_pe_max)
        out = output_dir / "collocation_3d_regular.png"
        plot_collocation_3d(
            x,
            t,
            lp,
            out,
            args.t_final_star,
            args.pe_min,
            args.pe_max,
            args.elev,
            args.azim,
            args.marker_size,
            title="Collocation: regular mesh",
        )
        print(f"Wrote {out} ({len(x)} points)")

    elif args.mode == "random-once":
        rng = np.random.default_rng(args.seed)
        x, t, lp = random_uniform_hypercube(
            args.num_points, args.t_final_star, log_pe_min, log_pe_max, rng
        )
        out = output_dir / "collocation_3d_random_once.png"
        plot_collocation_3d(
            x,
            t,
            lp,
            out,
            args.t_final_star,
            args.pe_min,
            args.pe_max,
            args.elev,
            args.azim,
            args.marker_size,
            title="Collocation: random (uniform)",
        )
        print(f"Wrote {out} ({len(x)} points, seed={args.seed})")

    else:
        for i in range(1, args.num_frames + 1):
            frame_seed = args.seed + i
            rng = np.random.default_rng(frame_seed)
            x, t, lp = random_uniform_hypercube(
                args.num_points, args.t_final_star, log_pe_min, log_pe_max, rng
            )
            out = output_dir / f"collocation_3d_random_{i:05d}.png"
            plot_collocation_3d(
                x,
                t,
                lp,
                out,
                args.t_final_star,
                args.pe_min,
                args.pe_max,
                args.elev,
                args.azim,
                args.marker_size,
                title=None,
            )
            print(f"Wrote {out} (seed={frame_seed})")
        print(f"Done: {args.num_frames} frames in {output_dir}")


if __name__ == "__main__":
    main()
