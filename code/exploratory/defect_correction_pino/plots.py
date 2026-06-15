# -*- coding: utf-8 -*-
"""Comparison plots for defect-correction exploration."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from residual import UCases, l2_rel

mpl.rcParams["figure.dpi"] = 800
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#FF5F05",
        "#13294B",
        "#009FD4",
        "#FCB316",
        "#006230",
        "#007E8E",
        "#5C0E41",
        "#7D3E13",
    ]
)
plt.rcParams["font.family"] = "Times New Roman"

# Match heterogeneous validation ticks (20–20–20–40 m zones).
XSTAR_TICKS = (0.0, 0.2, 0.4, 0.6, 1.0)
XSTAR_TICKS_NP = np.array(XSTAR_TICKS, dtype=np.float64)


def _grid_x_only(ax: plt.Axes, *, alpha: float = 0.3) -> None:
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=alpha)
    ax.yaxis.grid(False)


def _apply_xstar_ticks_x(ax: plt.Axes, *, pad: float = 0.0) -> None:
    ax.set_xlim(-pad, 1.0 + pad)
    ax.set_xticks(XSTAR_TICKS_NP)


def comsol_c_star_on_x_star(
    x_star: np.ndarray,
    x_m_comsol: np.ndarray,
    c_comsol: np.ndarray,
    *,
    l_m: float = 100.0,
    c_ref: float = 1.0,
) -> np.ndarray:
    x_star_comsol = x_m_comsol / l_m
    c_star = c_comsol / c_ref
    return np.interp(x_star, x_star_comsol, c_star)


def u_case_tag(u_case: UCases) -> str:
    return "_".join(f"{v:g}".replace(".", "p") for v in u_case)


def plot_correction_case(
    *,
    x_star: np.ndarray,
    t_days_list: list[float],
    t_star_list: list[float],
    c_tilde_slices: list[np.ndarray],
    c_corr_slices: list[np.ndarray],
    c_comsol_slices: list[np.ndarray],
    r_before_slices: list[np.ndarray],
    r_after_slices: list[np.ndarray],
    delta_latest: np.ndarray,
    u_case: UCases,
    mean_l2_before: float,
    mean_l2_after: float,
    out_png: Path,
) -> None:
    """Two-panel figure: concentration (corrected vs COMSOL) + residual before/after."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pde_lw = 1.0
    pde_alpha = 1.0
    pde_ylim = (-0.5, 0.5)

    fig = plt.figure(figsize=(8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 0.9], width_ratios=[1, 0.35], hspace=0.08, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_pde = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_delta = fig.add_subplot(gs[2, 0], sharex=ax)
    ax_inset = fig.add_subplot(gs[:, 1])

    fig.patch.set_facecolor("none")
    for a in (ax, ax_pde, ax_delta, ax_inset):
        a.set_facecolor("none")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    ax.plot([], [], linewidth=1.2, linestyle="-", color="gray", alpha=0.6, label="PINO")
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label=r"Corrected")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="COMSOL")
    ax_pde.plot([], [], linewidth=pde_lw, linestyle="-", color="gray", alpha=0.8, label="Before")
    ax_pde.plot([], [], linewidth=pde_lw, linestyle="-", color="#13294B", alpha=pde_alpha, label="After")

    for idx, (t_days, t_star, c0, cc, cref, rb, ra) in enumerate(
        zip(
            t_days_list,
            t_star_list,
            c_tilde_slices,
            c_corr_slices,
            c_comsol_slices,
            r_before_slices,
            r_after_slices,
            strict=True,
        )
    ):
        color = colors[idx % len(colors)]
        ax.plot(x_star, c0, linewidth=1.0, linestyle="-", color=color, alpha=0.35)
        ax.plot(x_star, cc, linewidth=2, linestyle="-", color=color)
        ax.plot(x_star, cref, linewidth=2, linestyle="--", color=color, alpha=0.75)
        ax.plot(
            [],
            [],
            marker="s",
            markersize=6,
            linestyle="None",
            color=color,
            label=rf"$t$={t_days:.0f} d",
        )
        ax_pde.plot(x_star, rb, linewidth=pde_lw, linestyle="-", color=color, alpha=0.35)
        ax_pde.plot(x_star, ra, linewidth=pde_lw, linestyle="-", color=color, alpha=pde_alpha)

    u1, u2, u3, u4 = u_case
    ax.set_title(
        rf"$u=({u1:g},{u2:g},{u3:g},{u4:g})$ m/d, "
        rf"rel. $L_2$: {mean_l2_before:.3e} $\rightarrow$ {mean_l2_after:.3e}"
    )
    ax.set_ylabel(r"$C^*$")
    ax_pde.set_ylabel(r"PDE residual")
    ax_pde.set_ylim(*pde_ylim)
    ax_delta.plot(x_star, delta_latest, linewidth=1.5, color="#5C0E41")
    ax_delta.set_ylabel(r"$\delta$")
    ax_delta.set_xlabel(r"$x^*$")

    t_star_arr = np.array(t_star_list, dtype=np.float64)
    # Simple 2D view of |delta| for latest times in inset
    ax_inset.imshow(
        np.abs(delta_latest).reshape(1, -1),
        aspect="auto",
        extent=[float(x_star[0]), float(x_star[-1]), 0, 1],
        origin="lower",
        cmap="magma",
    )
    ax_inset.set_title(r"$|\delta|$ at latest $t$")
    ax_inset.set_yticks([])
    ax_inset.set_xlabel(r"$x^*$")

    for a in (ax, ax_pde, ax_delta):
        _apply_xstar_ticks_x(a)
        _grid_x_only(a)
    ax.tick_params(labelbottom=False)
    ax_pde.tick_params(labelbottom=False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_pde.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", transparent=True)
    plt.close(fig)


def match_comsol_times(
    series: dict[float, np.ndarray],
    requested_days: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    out: list[tuple[float, np.ndarray]] = []
    for target in requested_days:
        hit = None
        for t_key, values in series.items():
            if np.isclose(float(t_key), float(target), rtol=0.0, atol=1e-10):
                hit = (float(t_key), values)
                break
        if hit is None:
            raise KeyError(f"No COMSOL slice at t={target} d.")
        out.append(hit)
    return out


def comsol_mean_rel_l2(
    x_star: np.ndarray,
    x_m: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    t_days: np.ndarray,
    c_slices: list[np.ndarray],
) -> float:
    matched = match_comsol_times(comsol_by_time, t_days)
    total = 0.0
    for (_, c_comsol), c_pred in zip(matched, c_slices, strict=True):
        c_ref = comsol_c_star_on_x_star(x_star, x_m, c_comsol)
        total += l2_rel(c_pred, c_ref)
    return total / len(matched)
