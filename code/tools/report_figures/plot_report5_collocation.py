#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report 5: training collocation in the (x*, t*) plane for one medium."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HETERO_ROOT = _REPO_ROOT / "code" / "heterogeneous"
_PINO_DIR = _HETERO_ROOT / "pino_heterogeneous"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "_src" / "report-5-heterogeneous" / "figs"

# Full two-column width; shallow height (not square — domain is stretched to axes box).
FIGSIZE = (7.0, 1.55)
FONT_SIZE = 8
MARKER_SIZE = 6
LEGEND_MARKER_EXTRA_PT = 2
T_SUBSAMPLE_STEP = 3
AXIS_PAD_X = 0.02
AXIS_PAD_Y = 0.04
COLLISION_MARKER_SIZE_SCALE = 1.8

U_REF = (0.03, 0.03, 0.03, 0.03)

_SET_STYLES: dict[str, dict] = {
    "PDE bulk": {
        "color": "#13294B",
        "marker": "o",
        "size_scale": 1.0,
        "alpha": 0.35,
        "zorder": 1,
    },
    "PDE interface": {
        "color": "#007E8E",
        "marker": "o",
        "size_scale": 0.55,
        "alpha": 0.85,
        "zorder": 2,
    },
    "IC": {"color": "#FF5F05", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 6},
    "Inlet BC": {"color": "#009FD4", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 4},
    "Outlet BC": {"color": "#8C1515", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 4},
}
_SET_ORDER = ["PDE bulk", "PDE interface", "IC", "Inlet BC", "Outlet BC"]


def _import_pino():
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


def _drop_manifold_overlap(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = ~(np.isclose(t, 0.0) | np.isclose(x, 0.0) | np.isclose(x, 1.0))
    return x[keep], t[keep]


def _unique_xy(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.unique(np.column_stack((x, t)), axis=0)
    return xy[:, 0], xy[:, 1]


def _subsample_by_t(
    x: np.ndarray, t: np.ndarray, *, step: int = T_SUBSAMPLE_STEP
) -> tuple[np.ndarray, np.ndarray]:
    t_levels = np.unique(t)
    t_keep = t_levels[::step]
    mask = np.isin(t, t_keep)
    return x[mask], t[mask]


def _subsample_1d(x: np.ndarray, t: np.ndarray, *, step: int = T_SUBSAMPLE_STEP) -> tuple[np.ndarray, np.ndarray]:
    return x[::step], t[::step]


def _build_xt_sets(pino) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    u_cases = np.array([list(U_REF)], dtype=np.float64)
    device = torch.device("cpu")
    dtype = torch.float32
    _, plot_data = pino.build_collocation_tensors(
        device, dtype, u_cases, include_interface=False
    )

    sm = plot_data["slice_mask"]
    is_interface = plot_data["is_interface_pde_np"][sm]
    x_pde = plot_data["x_star_pde_np"][sm]
    t_pde = plot_data["t_star_pde_np"][sm]

    x_bulk, t_bulk = _drop_manifold_overlap(x_pde[~is_interface], t_pde[~is_interface])
    x_if, t_if = _drop_manifold_overlap(x_pde[is_interface], t_pde[is_interface])

    # IC: bulk PDE x* grid only (matches training); no t* subsampling.
    x_ic = np.linspace(0.0, 1.0, pino.mesh_ic_nx)
    t_ic = np.zeros_like(x_ic)

    t_bc = np.linspace(0.0, 1.0, pino.mesh_bc_nt)
    x_in = np.zeros_like(t_bc)
    x_out = np.ones_like(t_bc)

    x_bulk, t_bulk = _subsample_by_t(x_bulk, t_bulk)
    x_if, t_if = _subsample_by_t(x_if, t_if)
    x_in, t_in = _subsample_1d(x_in, t_bc)
    x_out, t_out = _subsample_1d(x_out, t_bc)

    return {
        "PDE bulk": _unique_xy(x_bulk, t_bulk),
        "PDE interface": _unique_xy(x_if, t_if),
        "IC": (x_ic, t_ic),
        "Inlet BC": (x_in, t_in),
        "Outlet BC": (x_out, t_out),
    }


def _scatter_set(ax: plt.Axes, label: str, x_vals: np.ndarray, y_vals: np.ndarray) -> None:
    if x_vals.size == 0:
        return

    style = _SET_STYLES[label]
    size = MARKER_SIZE * style["size_scale"]
    marker = style["marker"]
    color = style["color"]

    if label == "IC":
        collide = np.isclose(y_vals, 0.0) & (np.isclose(x_vals, 0.0) | np.isclose(x_vals, 1.0))
        if np.any(~collide):
            ax.scatter(
                x_vals[~collide],
                y_vals[~collide],
                s=size,
                alpha=style["alpha"],
                marker=marker,
                color=color,
                edgecolors="none",
                linewidths=0.0,
                label=label,
                zorder=style["zorder"],
                rasterized=True,
            )
        if np.any(collide):
            ax.scatter(
                x_vals[collide],
                y_vals[collide],
                s=size,
                alpha=style["alpha"],
                marker="x",
                color=color,
                linewidths=0.9,
                label="_nolegend_",
                zorder=style["zorder"] + 1,
                rasterized=True,
            )
        return

    if label in {"Inlet BC", "Outlet BC"}:
        collide = np.isclose(y_vals, 0.0)
        if np.any(~collide):
            ax.scatter(
                x_vals[~collide],
                y_vals[~collide],
                s=size,
                alpha=style["alpha"],
                marker=marker,
                color=color,
                edgecolors="none",
                linewidths=0.0,
                label=label,
                zorder=style["zorder"],
                rasterized=True,
            )
        if np.any(collide):
            ax.scatter(
                x_vals[collide],
                y_vals[collide],
                s=size * COLLISION_MARKER_SIZE_SCALE,
                alpha=style["alpha"],
                marker=marker,
                color=color,
                edgecolors="none",
                linewidths=0.0,
                label="_nolegend_",
                zorder=style["zorder"],
                rasterized=True,
            )
        return

    ax.scatter(
        x_vals,
        y_vals,
        s=size,
        alpha=style["alpha"],
        marker=marker,
        color=color,
        edgecolors="none",
        linewidths=0.0,
        label=label,
        zorder=style["zorder"],
        rasterized=True,
    )


def plot_collocation_xt(*, out_pdf: Path) -> None:
    _apply_style()
    pino = _import_pino()
    dataset = _build_xt_sets(pino)
    xstar_ticks = np.array(pino.XSTAR_TICKS, dtype=np.float64)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.07, 0.20, 0.92, 0.62])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for label in _SET_ORDER:
        x_vals, y_vals = dataset[label]
        _scatter_set(ax, label, x_vals, y_vals)

    for xi in pino.ZONE_INTERFACE_XSTAR:
        ax.axvline(xi, color="#DDDDDD", linewidth=0.5, linestyle="--", zorder=0)

    ax.set_xlabel(r"$x^*$", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$t^*$", fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE, length=2.5, width=0.6)
    ax.set_xlim(-AXIS_PAD_X, 1.0 + AXIS_PAD_X)
    ax.set_ylim(-AXIS_PAD_Y, 1.0 + AXIS_PAD_Y)
    ax.set_xticks(xstar_ticks)
    ax.set_yticks([0.0, 0.5, 1.0])

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.53, 0.98),
            ncol=len(labels),
            frameon=False,
            fontsize=FONT_SIZE,
            handlelength=1.4,
            columnspacing=0.9,
            handletextpad=0.35,
            markerscale=1.0 + LEGEND_MARKER_EXTRA_PT / MARKER_SIZE,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    out_dir = Path(os.environ.get("REPORT5_FIGS_DIR", _FIGS_DIR)).resolve()
    out_pdf = Path(
        os.environ.get(
            "REPORT5_COLLOCATION_OUT_PDF",
            str(out_dir / "report5_collocation.pdf"),
        )
    ).resolve()
    plot_collocation_xt(out_pdf=out_pdf)


if __name__ == "__main__":
    main()
