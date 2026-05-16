# -*- coding: utf-8 -*-
"""Analytical vs numerical (COMSOL): one PDF per CFL/U value."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
from scipy.special import erfc, erfcx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_VALIDATION_DIR = _SCRIPT_DIR.parent
_DEFAULT_DATA_FILE = _VALIDATION_DIR / "data" / "comsol_1zone.txt"
_DEFAULT_RESULTS = _SCRIPT_DIR / "results"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "report 4 - CLF" / "figs"

U_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05]

L = 100.0
T_MAX = 1200.0
D_M2_S = 1e-6
SECONDS_PER_DAY = 86400.0
C0 = 1.0

TIMES_DAYS = np.array([100.0, 300.0, 600.0, 900.0, 1200.0], dtype=float)
NUM_POINTS = 5001
X_MAX = 100.0

D_M2_PER_DAY = D_M2_S * SECONDS_PER_DAY

# Shared height control for all five figures (first with legend + other four without legend).
FIGURE_WIDTH = 4.5
FIGURE_HEIGHT = 2
BASELINE_FIGSIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)

_ROW_META = re.compile(
    r"%\s*c\s*\([^)]+\)\s*@\s*t\s*=\s*([0-9.]+)\s*,\s*u1\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def _match_requested_times(
    available_by_time: dict[float, np.ndarray], requested_times: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    matched: list[tuple[float, np.ndarray]] = []
    available_items = list(available_by_time.items())
    for target in requested_times:
        hit: tuple[float, np.ndarray] | None = None
        for t_key, values in available_items:
            if np.isclose(float(t_key), float(target), rtol=0.0, atol=1e-10):
                hit = (float(t_key), values)
                break
        if hit is None:
            raise KeyError(f"No COMSOL slice at t={target} d for this U value.")
        matched.append(hit)
    return matched


def parse_comsol_export(path: Path, u1_target: float) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return (x_grid_m, {time_d: c_array}) for rows matching u1_target."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    x_grid: list[float] | None = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("% Grid"):
            if i + 1 >= len(lines):
                raise ValueError("Missing grid line after '% Grid'")
            parts = lines[i + 1].split()
            x_grid = [float(p) for p in parts]
            break
    if not x_grid:
        raise ValueError("Could not find '% Grid' section")

    x_arr = np.asarray(x_grid, dtype=np.float64)
    series: dict[float, np.ndarray] = {}

    j = 0
    while j < len(lines):
        m = _ROW_META.search(lines[j])
        if m:
            t_val = float(m.group(1))
            u_val = float(m.group(2))
            if abs(u_val - u1_target) < 1e-12 and j + 1 < len(lines):
                vals = np.array([float(s) for s in lines[j + 1].split()], dtype=np.float64)
                n = x_arr.size
                if vals.size == n + 1:
                    c = vals[1:]
                elif vals.size == n:
                    c = vals
                elif vals.size > n:
                    c = vals[-n:]
                else:
                    raise ValueError(
                        f"Data row at line {j + 2} has {vals.size} floats; expected at least {n}"
                    )
                series[t_val] = c
            j += 2
            continue
        j += 1

    if not series:
        raise ValueError(f"No data blocks found for u1={u1_target} in {path}")
    return x_arr, series


def ogata_banks_c(
    x_m: np.ndarray,
    t_days: float,
    u_m_per_d: float,
    d_m2_per_day: float,
    c0: float,
) -> np.ndarray:
    """Dimensional concentration (same units as c0)."""
    if t_days <= 0.0:
        return np.zeros_like(x_m, dtype=np.float64)
    sqrt_dt = np.sqrt(d_m2_per_day * t_days)
    term1 = erfc((x_m - u_m_per_d * t_days) / (2.0 * sqrt_dt))
    ux_over_d = u_m_per_d * x_m / d_m2_per_day
    b = (x_m + u_m_per_d * t_days) / (2.0 * sqrt_dt)
    exponent = ux_over_d - b**2
    c = (c0 / 2.0) * (term1 + np.exp(np.clip(exponent, -745.0, 700.0)) * erfcx(b))
    return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


def plot_one_cfl(
    *,
    cfl: float,
    u_val: float,
    x_num: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    x_line: np.ndarray,
    with_legend: bool,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig = plt.figure(figsize=BASELINE_FIGSIZE, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label="Numerical (COMSOL)")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

    matched_series = _match_requested_times(comsol_by_time, TIMES_DAYS)
    for idx, (ti, c_fd) in enumerate(matched_series):
        color = colors[idx % len(colors)]
        c_an = ogata_banks_c(x_line, float(ti), u_val, D_M2_PER_DAY, C0)
        ax.plot(x_num, c_fd, linewidth=2, linestyle="-", color=color)
        ax.plot(x_line, c_an, linewidth=2, linestyle="--", color=color, alpha=0.7)
        ax.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{ti:.1f}")

    ax.set_xlabel("Distance x (m)", fontsize=12)
    ax.set_ylabel("C (kg/m3)", fontsize=12)
    _pad_x, _pad_y = 0.05, 0.05
    ax.set_xlim(0 - _pad_x, L + _pad_x)
    ax.set_ylim(0 - _pad_y, C0 + _pad_y)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(NullLocator())

    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.35),
            ncol=4,
            frameon=False,
            fontsize=10,
            labelspacing=0.5,
            columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color("black")
            text.set_alpha(1.0)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_file = Path(os.environ.get("COMPARE_ANALYTICAL_NUMERICAL_DATA_FILE", _DEFAULT_DATA_FILE)).resolve()
    results_dir = Path(
        os.environ.get("COMPARE_ANALYTICAL_NUMERICAL_RESULTS_DIR", _DEFAULT_RESULTS)
    ).resolve()
    figs_dir = Path(
        os.environ.get("COMPARE_ANALYTICAL_NUMERICAL_REPORT_FIGS_DIR", _FIGS_DIR)
    ).resolve()

    mpl.rcParams["figure.dpi"] = 800
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "#FF5F05", "#13294B", "#009FD4",
            "#8C1515",
            "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13",
        ]
    )
    plt.rcParams["font.family"] = "Times New Roman"

    x_line = np.linspace(0.0, X_MAX, NUM_POINTS)

    print("Run configuration:")
    print(f"  Data: {data_file}")
    print(f"  L={L} m, D={D_M2_S} m^2/s, C0={C0}, T_MAX={T_MAX} d")

    for i, u_val in enumerate(U_VALUES):
        x_num, comsol_by_time = parse_comsol_export(data_file, float(u_val))
        cfl = float(u_val * T_MAX / L)
        cfl_token = f"{cfl:.2f}".replace(".", "p")
        out_pdf = figs_dir / f"compare_analytical_numerical_concentration_CFL{cfl_token}.pdf"
        out_png = results_dir / f"compare_analytical_numerical_concentration_CFL{cfl_token}.png"
        plot_one_cfl(
            cfl=cfl,
            u_val=float(u_val),
            x_num=x_num,
            comsol_by_time=comsol_by_time,
            x_line=x_line,
            with_legend=(i == 0),
            out_pdf=out_pdf,
            out_png=out_png,
        )
        print(f"Wrote {out_pdf}")
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
