#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate report-ready non-redundant collocation figures for Report 4."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "report 4 - CLF" / "figs"

# -----------------------------------------------------------------------------
# Style/config (editable)
# -----------------------------------------------------------------------------
mpl.rcParams["figure.dpi"] = 800
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
plt.rcParams["font.family"] = "Times New Roman"

FIGSIZE_DEFAULT = (4.5, 3.0)
FIGSIZE_3D = (4.5, 3.0)
MARKER_SIZE = 10
ALPHA = 1
TOP_LEGEND_COLS = 2
# Font sizes (editable)
# 2D
FONT_2D_AXIS_LABEL = 8
FONT_2D_TICK_LABEL = 8
FONT_2D_LEGEND = 8
# 3D
FONT_3D_AXIS_LABEL = 8
FONT_3D_TICK_LABEL = 8
FONT_3D_CFL_FALLBACK_LABEL = 7
LEGEND_MARKER_SCALE = 1.5
AXIS_PAD = 0.04
Z_LABEL_FALLBACK_X = 0.850
Z_LABEL_FALLBACK_Y = 0.52
COLLISION_MARKER_SIZE_SCALE = 2

# Set to False if you only want one type.
ENABLE_2D = True
ENABLE_3D = True

# Edit this list to limit generated pairwise plots.
PAIR_PLOTS = [("x", "t"), ("x", "cfl"), ("t", "cfl")]

# Mesh defaults copied from current CFL scripts.
MESH_NX_PDE = 20
MESH_NT_PDE = 20
MESH_NCFL_PDE = 20
MESH_N_IC = 20
MESH_N_BC = 20
MESH_IC_NX = 20
MESH_IC_NCFL = 20
MESH_BC_NT = 20
MESH_BC_NCFL = 20

# CFL range from U_VALUES=[0.01,...,0.05], L=100, T_MAX=1200 in current scripts.
CFL_MIN = 0.12
CFL_MAX = 0.60
T_FINAL_STAR = 1.0

_LABELS = {
    "x": "x*",
    "t": "t*",
    "cfl": "CFL",
}
_SET_STYLES = {
    "PDE": {"color": "#13294B", "marker": "o", "size_scale": 1.0, "alpha": 0.25, "zorder": 1},
    "IC": {"color": "#FF5F05", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 6},
    "Inlet BC": {"color": "#009FD4", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 3},
    "Outlet BC": {"color": "#8C1515", "marker": "o", "size_scale": 1.0, "alpha": 1.0, "zorder": 3},
}
_SET_ORDER = ["PDE", "IC", "Inlet BC", "Outlet BC"]
_SETS_BY_PAIR = {
    # Full set view is informative in x-t.
    ("x", "t"): ["PDE", "IC", "Inlet BC", "Outlet BC"],
    # Projecting away t makes IC collapse onto PDE support; keep PDE only.
    ("x", "cfl"): ["PDE"],
    # Projecting away x makes inlet/outlet collapse onto PDE support; keep PDE only.
    ("t", "cfl"): ["PDE"],
}

def _get_out_dir() -> Path:
    out_dir = Path(os.environ.get("CFL_COLLOCATION_REPORT_FIGS_DIR", _FIGS_DIR)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_baseline_sets() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_1d = np.linspace(0.0, 1.0, MESH_NX_PDE)
    t_1d = np.linspace(0.0, T_FINAL_STAR, MESH_NT_PDE)
    gx, gt = np.meshgrid(x_1d, t_1d, indexing="ij")

    x_pde = gx.reshape(-1)
    t_pde = gt.reshape(-1)
    cfl_pde = np.zeros_like(x_pde, dtype=np.float64)

    x_ic = np.linspace(0.0, 1.0, MESH_N_IC)
    t_ic = np.zeros_like(x_ic)
    cfl_ic = np.zeros_like(x_ic, dtype=np.float64)

    t_bc = np.linspace(0.0, T_FINAL_STAR, MESH_N_BC)
    x_in = np.zeros_like(t_bc)
    x_out = np.ones_like(t_bc)
    cfl_bc = np.zeros_like(t_bc, dtype=np.float64)

    return {
        "PDE": (x_pde, t_pde, cfl_pde),
        "IC": (x_ic, t_ic, cfl_ic),
        "Inlet BC": (x_in, t_bc, cfl_bc),
        "Outlet BC": (x_out, t_bc, cfl_bc),
    }


def _build_parametric_sets(
    *,
    mesh_nx_pde: int = MESH_NX_PDE,
    mesh_nt_pde: int = MESH_NT_PDE,
    mesh_ncfl_pde: int = MESH_NCFL_PDE,
    mesh_ic_nx: int = MESH_IC_NX,
    mesh_ic_ncfl: int = MESH_IC_NCFL,
    mesh_bc_nt: int = MESH_BC_NT,
    mesh_bc_ncfl: int = MESH_BC_NCFL,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cfl_1d = np.linspace(CFL_MIN, CFL_MAX, mesh_ncfl_pde)
    x_1d = np.linspace(0.0, 1.0, mesh_nx_pde)
    t_1d = np.linspace(0.0, T_FINAL_STAR, mesh_nt_pde)
    gx, gt, gcfl = np.meshgrid(x_1d, t_1d, cfl_1d, indexing="ij")

    x_pde = gx.reshape(-1)
    t_pde = gt.reshape(-1)
    cfl_pde = gcfl.reshape(-1)

    x_ic_1d = np.linspace(0.0, 1.0, mesh_ic_nx)
    cfl_ic_1d = np.linspace(CFL_MIN, CFL_MAX, mesh_ic_ncfl)
    gxi, gcfli = np.meshgrid(x_ic_1d, cfl_ic_1d, indexing="ij")
    x_ic = gxi.reshape(-1)
    t_ic = np.zeros_like(x_ic)
    cfl_ic = gcfli.reshape(-1)

    t_bc_1d = np.linspace(0.0, T_FINAL_STAR, mesh_bc_nt)
    cfl_bc_1d = np.linspace(CFL_MIN, CFL_MAX, mesh_bc_ncfl)
    gtb, gcfb = np.meshgrid(t_bc_1d, cfl_bc_1d, indexing="ij")
    t_bc = gtb.reshape(-1)
    cfl_bc = gcfb.reshape(-1)
    x_in = np.zeros_like(t_bc)
    x_out = np.ones_like(t_bc)

    return {
        "PDE": (x_pde, t_pde, cfl_pde),
        "IC": (x_ic, t_ic, cfl_ic),
        "Inlet BC": (x_in, t_bc, cfl_bc),
        "Outlet BC": (x_out, t_bc, cfl_bc),
    }


def _axis_data(
    xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_name: str,
) -> np.ndarray:
    x, t, cfl = xyz
    if axis_name == "x":
        return x
    if axis_name == "t":
        return t
    if axis_name == "cfl":
        return cfl
    raise ValueError(f"Unsupported axis: {axis_name}")


def _plot_2d_pair(
    *,
    dataset: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    method_key: str,
    method_title: str,
    axis_a: str,
    axis_b: str,
    out_dir: Path,
    with_top_legend: bool = False,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_DEFAULT, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove PDE points that lie on IC/inlet/outlet manifolds for clearer overlays.
    pde_x, pde_t, pde_cfl = dataset["PDE"]
    overlap_mask = np.isclose(pde_t, 0.0) | np.isclose(pde_x, 0.0) | np.isclose(pde_x, 1.0)
    dataset_for_plot = dict(dataset)
    dataset_for_plot["PDE"] = (pde_x[~overlap_mask], pde_t[~overlap_mask], pde_cfl[~overlap_mask])

    pair_key = (axis_a, axis_b)
    set_order_for_pair = _SETS_BY_PAIR.get(pair_key, _SET_ORDER)

    for label in set_order_for_pair:
        xyz = dataset_for_plot[label]
        x_vals = _axis_data(xyz, axis_a)
        y_vals = _axis_data(xyz, axis_b)
        # Avoid overdraw accumulation in 2D projections (which can visually
        # cancel intended alpha by repeatedly plotting identical coordinates).
        xy = np.column_stack((x_vals, y_vals))
        xy_unique = np.unique(xy, axis=0)
        x_vals = xy_unique[:, 0]
        y_vals = xy_unique[:, 1]
        style = _SET_STYLES[label]
        marker = style["marker"]
        size = MARKER_SIZE * style["size_scale"]
        # Mark only IC points that collide with BC edges (x*=0 or x*=1 at t*=0) as "x"
        # while keeping legend entries unchanged.
        if axis_a == "x" and axis_b == "t" and label == "IC":
            collide_mask = np.isclose(y_vals, 0.0) & (np.isclose(x_vals, 0.0) | np.isclose(x_vals, 1.0))
            if np.any(~collide_mask):
                ax.scatter(
                    x_vals[~collide_mask],
                    y_vals[~collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label=label,
                    zorder=style["zorder"],
                    rasterized=True,
                )
            if np.any(collide_mask):
                ax.scatter(
                    x_vals[collide_mask],
                    y_vals[collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker="x",
                    color=style["color"],
                    linewidths=0.9,
                    label="_nolegend_",
                    zorder=style["zorder"] + 1,
                    rasterized=True,
                )
        elif axis_a == "x" and axis_b == "t" and label in {"Inlet BC", "Outlet BC"}:
            collide_mask = np.isclose(y_vals, 0.0)
            if np.any(~collide_mask):
                ax.scatter(
                    x_vals[~collide_mask],
                    y_vals[~collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label=label,
                    zorder=style["zorder"],
                    rasterized=True,
                )
            if np.any(collide_mask):
                ax.scatter(
                    x_vals[collide_mask],
                    y_vals[collide_mask],
                    s=size * COLLISION_MARKER_SIZE_SCALE,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label="_nolegend_",
                    zorder=style["zorder"],
                    rasterized=True,
                )
        else:
            ax.scatter(
                x_vals,
                y_vals,
                s=size,
                alpha=style["alpha"],
                marker=marker,
                color=style["color"],
                edgecolors="none",
                linewidths=0.0,
                label=label,
                zorder=style["zorder"],
                rasterized=True,
            )

    ax.set_xlabel(_LABELS[axis_a], fontsize=FONT_2D_AXIS_LABEL)
    ax.set_ylabel(_LABELS[axis_b], fontsize=FONT_2D_AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_2D_TICK_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_2D_TICK_LABEL)
    ax.minorticks_off()
    if axis_a in {"x", "t"}:
        ax.set_xlim(-AXIS_PAD, 1.0 + AXIS_PAD)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.spines["bottom"].set_bounds(0.0, 1.0)
    if axis_b in {"x", "t"}:
        ax.set_ylim(-AXIS_PAD, 1.0 + AXIS_PAD)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.spines["left"].set_bounds(0.0, 1.0)
    if axis_b == "cfl":
        ax.set_ylabel(_LABELS[axis_b], fontsize=FONT_2D_AXIS_LABEL, labelpad=8)
        ax.set_ylim(CFL_MIN - AXIS_PAD, CFL_MAX + AXIS_PAD)
    if axis_a in {"x", "t"} and axis_b in {"x", "t"}:
        ax.set_aspect("equal", adjustable="box")
    if with_top_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.10),
                ncol=TOP_LEGEND_COLS,
                frameon=False,
                fontsize=FONT_2D_LEGEND,
                labelspacing=0.5,
                columnspacing=1.2,
                markerscale=LEGEND_MARKER_SCALE,
            )
            for text in legend.get_texts():
                text.set_color("black")
                text.set_alpha(1.0)

    out_pdf = out_dir / f"{method_key}_collocation_{axis_a}{axis_b}.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def _plot_3d(
    *,
    dataset: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    method_key: str,
    method_title: str,
    out_dir: Path,
) -> None:
    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    # Remove PDE points that overlap IC/inlet/outlet manifolds so BC/IC remain visible.
    pde_x, pde_t, pde_cfl = dataset["PDE"]
    overlap_mask = np.isclose(pde_t, 0.0) | np.isclose(pde_x, 0.0) | np.isclose(pde_x, 1.0)
    pde_filtered = (pde_x[~overlap_mask], pde_t[~overlap_mask], pde_cfl[~overlap_mask])
    dataset_for_plot = dict(dataset)
    dataset_for_plot["PDE"] = pde_filtered

    set_order_3d = ["PDE", "Inlet BC", "Outlet BC", "IC"]
    for label in set_order_3d:
        xyz = dataset_for_plot[label]
        x_vals, t_vals, cfl_vals = xyz
        style = _SET_STYLES[label]
        marker = style["marker"]
        size = MARKER_SIZE * style["size_scale"]
        # In 3D, mark IC points that collide with BC edges as "x" (same logic as 2D x-t).
        if label == "IC":
            collide_mask = np.isclose(t_vals, 0.0) & (np.isclose(x_vals, 0.0) | np.isclose(x_vals, 1.0))
            if np.any(~collide_mask):
                ax.scatter(
                    x_vals[~collide_mask],
                    t_vals[~collide_mask],
                    cfl_vals[~collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label=label,
                    zorder=style["zorder"],
                    depthshade=False,
                    rasterized=True,
                )
            if np.any(collide_mask):
                ax.scatter(
                    x_vals[collide_mask],
                    t_vals[collide_mask],
                    cfl_vals[collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker="x",
                    color=style["color"],
                    linewidths=0.9,
                    label="_nolegend_",
                    zorder=style["zorder"] + 1,
                    depthshade=False,
                    rasterized=True,
                )
        elif label in {"Inlet BC", "Outlet BC"}:
            collide_mask = np.isclose(t_vals, 0.0)
            if np.any(~collide_mask):
                ax.scatter(
                    x_vals[~collide_mask],
                    t_vals[~collide_mask],
                    cfl_vals[~collide_mask],
                    s=size,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label=label,
                    zorder=style["zorder"],
                    depthshade=False,
                    rasterized=True,
                )
            if np.any(collide_mask):
                ax.scatter(
                    x_vals[collide_mask],
                    t_vals[collide_mask],
                    cfl_vals[collide_mask],
                    s=size * COLLISION_MARKER_SIZE_SCALE,
                    alpha=style["alpha"],
                    marker=marker,
                    color=style["color"],
                    edgecolors="none",
                    linewidths=0.0,
                    label="_nolegend_",
                    zorder=style["zorder"],
                    depthshade=False,
                    rasterized=True,
                )
        else:
            ax.scatter(
                x_vals,
                t_vals,
                cfl_vals,
                s=size,
                alpha=style["alpha"],
                marker=marker,
                color=style["color"],
                edgecolors="none",
                linewidths=0.0,
                label=label,
                zorder=style["zorder"],
                depthshade=False,
                rasterized=True,
            )

    ax.set_xlabel(_LABELS["x"], fontsize=FONT_3D_AXIS_LABEL, labelpad=4)
    ax.set_ylabel(_LABELS["t"], fontsize=FONT_3D_AXIS_LABEL, labelpad=4)
    # Hide native 3D z-label to avoid renderer inconsistencies and duplicates.
    ax.set_zlabel("", fontsize=FONT_3D_AXIS_LABEL, labelpad=2)
    ax.set_xlim(-AXIS_PAD, 1.0 + AXIS_PAD)
    ax.set_ylim(-AXIS_PAD, 1.0 + AXIS_PAD)
    ax.set_zlim(CFL_MIN - 0.02, CFL_MAX + 0.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_zticks(np.arange(0.1, 0.61, 0.1))
    ax.tick_params(axis="x", labelsize=FONT_3D_TICK_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_3D_TICK_LABEL)
    ax.tick_params(axis="z", labelsize=FONT_3D_TICK_LABEL)
    ax.view_init(elev=25, azim=-55)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # Fallback label for PDF renderers that clip or omit 3D z-axis labels.
    fig.text(
        Z_LABEL_FALLBACK_X,
        Z_LABEL_FALLBACK_Y,
        _LABELS["cfl"],
        fontsize=FONT_3D_CFL_FALLBACK_LABEL,
        rotation=90,
        va="center",
        ha="center",
    )
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.95)

    out_pdf = out_dir / f"{method_key}_collocation_3d.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def _validate_pair(pair: tuple[str, str]) -> tuple[str, str] | None:
    if len(pair) != 2:
        return None
    a, b = pair
    if a not in _LABELS or b not in _LABELS or a == b:
        return None
    return a, b


def main() -> None:
    out_dir = _get_out_dir()

    shared_cfl_sets = _build_parametric_sets()
    shared_cfl_sets_3d = _build_parametric_sets(
        mesh_nx_pde=10,
        mesh_nt_pde=10,
        mesh_ncfl_pde=10,
        mesh_ic_nx=10,
        mesh_ic_ncfl=10,
        mesh_bc_nt=10,
        mesh_bc_ncfl=10,
    )

    valid_pairs: list[tuple[str, str]] = []
    for pair in PAIR_PLOTS:
        vpair = _validate_pair(pair)
        if vpair is not None:
            valid_pairs.append(vpair)

    # Parametric PINN and PINO use the same 50x50x50 (x,t,CFL) mesh.
    # Export one shared set to avoid redundant figures in the report.
    if ENABLE_2D:
        for axis_a, axis_b in valid_pairs:
            show_top_legend = axis_a == "x" and axis_b == "t"
            _plot_2d_pair(
                dataset=shared_cfl_sets,
                method_key="shared_cfl",
                method_title="Shared CFL mesh (Parametric PINN/PINO)",
                axis_a=axis_a,
                axis_b=axis_b,
                out_dir=out_dir,
                with_top_legend=show_top_legend,
            )
    if ENABLE_3D:
        _plot_3d(
            dataset=shared_cfl_sets_3d,
            method_key="shared_cfl",
            method_title="Shared CFL mesh (Parametric PINN/PINO)",
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
