#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report 5 figure: COMSOL validation grid (runs C vs J, three media)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HETERO_ROOT = _REPO_ROOT / "code" / "heterogeneous"
_PINO_DIR = _HETERO_ROOT / "pino_heterogeneous"
_RESULTS_DIR = _PINO_DIR / "results"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "_src" / "report-5-heterogeneous" / "figs"
_COMSOL_DATA = _HETERO_ROOT / "data" / "comsol_4zones.txt"

# Full-width figure* on A4 extarticle twocolumn (textwidth ~18 cm).
FIGSIZE = (7.0, 4.55)
FONT_SIZE = 8
ANNOT_SIZE = 6
HEIGHT_RATIOS_INNER = (3, 1)
N_MEDIA_COLS = 3
N_RUN_ROWS = 2
RUN_BLOCK_HSPACE = 0.20
INNER_HSPACE = 0.15
INNER_WSPACE = 0.10

COMSOL_TIMES_DAYS = np.array([100.0, 300.0, 600.0, 900.0, 1200.0], dtype=float)
T_MAX = 1200.0
PDE_YLIM = (-0.5, 0.5)
PROFILE_LW = 1.5
RESIDUAL_LW = 1.0
INTERFACE_LW = 0.6

# (experiment folder, torch dtype)
RUNS: list[tuple[str, torch.dtype]] = [
    ("exp_C_maximin_N500", torch.float32),
    ("exp_J_maximin_N500_float64_lr01_maxiter20", torch.float64),
]

COL_LABEL_PAD = 0.003
CSTAR_YLIM = (-0.05, 1.05)
CSTAR_YTICKS = (0.0, 0.5, 1.0)

# Three representative media (uniform, alternating, two-step).
MEDIA: list[tuple[tuple[float, float, float, float], str]] = [
    ((0.03, 0.03, 0.03, 0.03), "uniform"),
    ((0.01, 0.05, 0.01, 0.05), "alternating"),
    ((0.05, 0.05, 0.01, 0.01), "two-step"),
]

BRANCH_ARCH = [4, 16, 16, 32]
TRUNK_ARCH = [2, 32, 32, 32, 32]


def _import_pino_helpers():
    for path in (_HETERO_ROOT, _PINO_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("pinn1d_heterogeneous_parametric_neural_operator")


def _apply_style() -> None:
    mpl.rcParams["figure.dpi"] = 800
    mpl.rcParams["savefig.dpi"] = 800
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#FF5F05",
            "#13294B",
            "#009FD4",
            "#8C1515",
            "#FCB316",
            "#006230",
            "#007E8E",
            "#5C0E41",
            "#7D3E13",
        ]
    )
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.linewidth"] = 0.6
    mpl.rcParams["xtick.major.width"] = 0.6
    mpl.rcParams["ytick.major.width"] = 0.6


def _load_model(checkpoint: Path, *, dtype: torch.dtype):
    from deeponet import build_deeponet

    device = torch.device("cpu")
    torch.set_default_dtype(dtype)
    model = build_deeponet(BRANCH_ARCH, TRUNK_ARCH, nn.Tanh).to(
        device=device, dtype=dtype
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label_text_kwargs() -> dict:
    return {"fontsize": ANNOT_SIZE, "color": "black"}


def _annotate_mean_rel(ax_c: plt.Axes, mean_l2: float) -> None:
    ax_c.text(
        0.98,
        0.98,
        rf"mean rel. {100.0 * mean_l2:.1f}$\%$",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        **_label_text_kwargs(),
    )


def _format_u_label(u_case: tuple[float, float, float, float]) -> str:
    u1, u2, u3, u4 = u_case
    return rf"$u=({u1:g},{u2:g},{u3:g},{u4:g})$ m/d"


def _place_column_labels(fig: plt.Figure, axes_c: list[list[plt.Axes]]) -> None:
    fig.canvas.draw()
    for col, (u_case, _media_label) in enumerate(MEDIA):
        pos = axes_c[0][col].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2.0,
            pos.y1 + COL_LABEL_PAD,
            _format_u_label(u_case),
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            **_label_text_kwargs(),
        )


def _hide_shared_ticks(
    axes_c: list[list[plt.Axes]],
    axes_r: list[list[plt.Axes]],
) -> None:
    for row in range(N_RUN_ROWS):
        for col in range(N_MEDIA_COLS):
            ax_c = axes_c[row][col]
            ax_r = axes_r[row][col]
            ax_c.tick_params(labelbottom=False, labelsize=FONT_SIZE)
            ax_r.tick_params(labelsize=FONT_SIZE)
            if row < N_RUN_ROWS - 1:
                ax_r.tick_params(labelbottom=False)
            if col > 0:
                ax_c.tick_params(labelleft=False)
                ax_r.tick_params(labelleft=False)
            else:
                ax_c.tick_params(labelleft=True)
                ax_r.tick_params(labelleft=True)


def _plot_panel(
    *,
    ax_c: plt.Axes,
    ax_r: plt.Axes,
    model,
    device: torch.device,
    dtype: torch.dtype,
    u_case: tuple[float, float, float, float],
    x_m: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    x_star: np.ndarray,
    pino,
    interface_x: tuple[float, ...],
    colors: list[str],
    legend_handles: list | None,
) -> tuple[float, list]:
    """Plot concentration + residual; return mean L2 and legend handles (first call only)."""
    n_x = x_star.size
    x_t = torch.tensor(x_star.reshape(-1, 1), dtype=dtype, device=device)
    branch = pino.branch_tensor_from_u_case(u_case, n_x, device=device, dtype=dtype)

    handles: list = []
    if legend_handles is None:
        handles.append(
            ax_c.plot([], [], linewidth=PROFILE_LW, linestyle="-", color="black", label="PINO")[0]
        )
        handles.append(
            ax_c.plot(
                [], [], linewidth=PROFILE_LW, linestyle="--", color="black", label="COMSOL"
            )[0]
        )
        handles.append(
            ax_r.plot(
                [],
                [],
                linewidth=RESIDUAL_LW,
                linestyle="-",
                color="gray",
                label="PDE residual",
            )[0]
        )

    matched = pino._match_comsol_times(comsol_by_time, COMSOL_TIMES_DAYS)
    l2_sum = 0.0
    model.eval()
    for idx, (t_days, c_comsol) in enumerate(matched):
        t_star = t_days / T_MAX
        t_t = torch.full((n_x, 1), t_star, dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(x_t, t_t, branch).cpu().numpy().flatten()
        c_ref = pino.comsol_c_star_on_x_star(x_star, x_m, c_comsol)
        l2_sum += pino._l2_rel(c_pred, c_ref)
        color = colors[idx % len(colors)]
        ax_c.plot(x_star, c_pred, linewidth=PROFILE_LW, linestyle="-", color=color)
        ax_c.plot(x_star, c_ref, linewidth=PROFILE_LW, linestyle="--", color=color, alpha=0.7)
        if legend_handles is None:
            handles.append(
                ax_c.plot(
                    [],
                    [],
                    marker="s",
                    markersize=5,
                    linestyle="None",
                    color=color,
                    label=rf"$t$={t_days:.0f} d",
                )[0]
            )
        with torch.enable_grad():
            r_pde = pino._pde_residual_1d(
                model, x_star, t_star, u_case, device=device, dtype=dtype
            )
        ax_r.plot(x_star, r_pde, linewidth=RESIDUAL_LW, linestyle="-", color=color, alpha=1.0)

    for xi in interface_x:
        ax_c.axvline(xi, color="#999999", linewidth=INTERFACE_LW, linestyle="--", zorder=0)

    mean_l2 = l2_sum / len(matched)
    return mean_l2, handles if legend_handles is None else legend_handles


def plot_validation_grid(*, out_pdf: Path) -> None:
    _apply_style()
    pino = _import_pino_helpers()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    interface_x = pino.ZONE_INTERFACE_XSTAR
    x_star = pino.pde_collocation_x_star_1d()
    xstar_ticks = np.array(pino.XSTAR_TICKS, dtype=np.float64)

    models: list[tuple[object, torch.device, torch.dtype]] = []
    for exp_name, dtype in RUNS:
        ckpt = _RESULTS_DIR / exp_name / "pino_heterogeneous_model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model, device = _load_model(ckpt, dtype=dtype)
        models.append((model, device, dtype))

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        N_RUN_ROWS,
        1,
        height_ratios=[1, 1],
        hspace=RUN_BLOCK_HSPACE,
        top=0.90,
        bottom=0.10,
        left=0.07,
        right=0.99,
    )

    axes_c: list[list[plt.Axes]] = [[None] * N_MEDIA_COLS for _ in range(N_RUN_ROWS)]  # type: ignore
    axes_r: list[list[plt.Axes]] = [[None] * N_MEDIA_COLS for _ in range(N_RUN_ROWS)]  # type: ignore
    ax_x_ref: plt.Axes | None = None
    block_c_ref: list[plt.Axes | None] = [None, None]
    block_r_ref: list[plt.Axes | None] = [None, None]

    legend_handles: list | None = None

    for run_row, (model, device, dtype) in enumerate(models):
        inner = outer[run_row].subgridspec(
            2,
            N_MEDIA_COLS,
            height_ratios=HEIGHT_RATIOS_INNER,
            hspace=INNER_HSPACE,
            wspace=INNER_WSPACE,
        )
        for col, (u_case, _label) in enumerate(MEDIA):
            x_m, comsol_by_time = pino.load_case(_COMSOL_DATA, u_case)

            if block_c_ref[run_row] is None:
                if ax_x_ref is None:
                    ax_c = fig.add_subplot(inner[0, col])
                    ax_r = fig.add_subplot(inner[1, col], sharex=ax_c)
                    ax_x_ref = ax_c
                else:
                    ax_c = fig.add_subplot(inner[0, col], sharex=ax_x_ref)
                    ax_r = fig.add_subplot(inner[1, col], sharex=ax_x_ref)
                block_c_ref[run_row] = ax_c
                block_r_ref[run_row] = ax_r
            else:
                ax_c = fig.add_subplot(
                    inner[0, col],
                    sharex=ax_x_ref,
                    sharey=block_c_ref[run_row],
                )
                ax_r = fig.add_subplot(
                    inner[1, col],
                    sharex=ax_x_ref,
                    sharey=block_r_ref[run_row],
                )

            axes_c[run_row][col] = ax_c
            axes_r[run_row][col] = ax_r
            _style_axes(ax_c)
            _style_axes(ax_r)

            mean_l2, legend_handles = _plot_panel(
                ax_c=ax_c,
                ax_r=ax_r,
                model=model,
                device=device,
                dtype=dtype,
                u_case=u_case,
                x_m=x_m,
                comsol_by_time=comsol_by_time,
                x_star=x_star,
                pino=pino,
                interface_x=interface_x,
                colors=colors,
                legend_handles=legend_handles,
            )
            _annotate_mean_rel(ax_c, mean_l2)

    for run_row in range(N_RUN_ROWS):
        for ax_c in axes_c[run_row]:
            ax_c.set_ylim(*CSTAR_YLIM)
            ax_c.set_yticks(CSTAR_YTICKS)
        for ax_r in axes_r[run_row]:
            ax_r.set_ylim(*PDE_YLIM)

        axes_c[run_row][0].set_ylabel(r"$C^*$", fontsize=FONT_SIZE)
        axes_r[run_row][0].set_ylabel("PDE residual", fontsize=FONT_SIZE)

    for ax_r in axes_r[-1]:
        ax_r.set_xlabel(r"$x^*$", fontsize=FONT_SIZE)

    if ax_x_ref is not None:
        ax_x_ref.set_xlim(0.0, 1.0)
        ax_x_ref.set_xticks(xstar_ticks)

    _hide_shared_ticks(axes_c, axes_r)
    _place_column_labels(fig, axes_c)

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=FONT_SIZE,
            handlelength=1.8,
            columnspacing=1.0,
            handletextpad=0.4,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    out_dir = Path(os.environ.get("REPORT5_FIGS_DIR", _FIGS_DIR)).resolve()
    out_pdf = Path(
        os.environ.get(
            "REPORT5_VALIDATION_GRID_OUT_PDF",
            str(out_dir / "report5_validation_grid.pdf"),
        )
    ).resolve()
    plot_validation_grid(out_pdf=out_pdf)


if __name__ == "__main__":
    main()
