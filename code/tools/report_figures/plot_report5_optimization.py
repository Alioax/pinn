#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bar chart of mean COMSOL validation error for Report 5 optimizer study."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "code" / "heterogeneous" / "pino_heterogeneous" / "results"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "_src" / "report-5-heterogeneous" / "figs"

# Single-column width for extarticle 9pt twocolumn on A4 (margin 1.5 cm, columnsep 0.5 cm).
FIGSIZE = (3.45, 2.05)
FONT_SIZE = 8
BAR_WIDTH = 0.62
Y_PAD_FRAC = 0.08
HOMOGENEOUS_TARGET_PCT = 1.0

# (run id, results folder, cycle color index, hatch pattern or None, x-axis subtitle)
RUNS: list[tuple[str, str, int, str | None, str]] = [
    ("A", "exp_A_capacity", 1, None, "in-sample"),
    ("B", "exp_B_lhc_N500", 1, None, "LHC"),
    ("C", "exp_C_maximin_N500", 1, None, "maximin"),
    ("I", "exp_I_maximin_N500_float64_lr01", 0, "///", "float64"),
    ("J", "exp_J_maximin_N500_float64_lr01_maxiter20", 0, None, "tuned"),
]


def _get_out_dir() -> Path:
    out_dir = Path(os.environ.get("REPORT5_FIGS_DIR", _FIGS_DIR)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _summary_csv(exp_folder: str) -> Path | None:
    base = _RESULTS_DIR / exp_folder
    for rel in ("comsol_validation_summary.csv", "comsol_validation/comsol_validation_summary.csv"):
        path = base / rel
        if path.is_file():
            return path
    return None


def _mean_rel_l2_percent(csv_path: Path) -> float:
    values: list[float] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            values.append(100.0 * float(row["mean_rel_l2"]))
    if not values:
        raise ValueError(f"No rows in {csv_path}")
    return float(np.mean(values))


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
    mpl.rcParams["hatch.linewidth"] = 0.55
    mpl.rcParams["hatch.color"] = "white"


def plot_optimization_bar(*, out_pdf: Path) -> None:
    _apply_style()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    labels: list[str] = []
    means: list[float] = []
    color_idxs: list[int] = []
    hatches: list[str | None] = []
    subtitles: list[str] = []

    for run_id, folder, color_idx, hatch, subtitle in RUNS:
        csv_path = _summary_csv(folder)
        if csv_path is None:
            raise FileNotFoundError(f"Missing validation summary for {folder}")
        labels.append(run_id)
        means.append(_mean_rel_l2_percent(csv_path))
        color_idxs.append(color_idx)
        hatches.append(hatch)
        subtitles.append(subtitle)

    fig = plt.figure(figsize=FIGSIZE, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x = np.arange(len(labels))
    ymax = max(means)
    y_top = ymax * (1.0 + Y_PAD_FRAC) + 1.0
    ax.set_ylim(0.0, y_top)

    target_color = cycle[5]

    bars = []
    for xi, mean, color_idx, hatch in zip(x, means, color_idxs, hatches, strict=True):
        color = cycle[color_idx]
        if hatch:
            bar = ax.bar(
                xi,
                mean,
                width=BAR_WIDTH,
                facecolor=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        else:
            bar = ax.bar(
                xi,
                mean,
                width=BAR_WIDTH,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                zorder=3,
            )
        bars.append(bar)

    # Homogeneous near-perfect target (<1%): C5 fill on top of bar bases; label just above band.
    ax.axhspan(
        0.0,
        HOMOGENEOUS_TARGET_PCT,
        facecolor=target_color,
        alpha=0.35,
        zorder=10,
        linewidth=0,
        edgecolor="none",
    )
    ax.text(
        (len(labels) - 1) / 2.0,
        HOMOGENEOUS_TARGET_PCT + 0.12,
        r"homogeneous target ($<1\,\%$)",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        color=target_color,
        zorder=11,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lab}\n{sub}" for lab, sub in zip(labels, subtitles, strict=True)],
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel(r"Mean relative $L_2$ error ($\%$)", fontsize=FONT_SIZE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE, length=2.5, width=0.6)
    ax.tick_params(axis="x", length=0)
    yticks = [t for t in (0.0, HOMOGENEOUS_TARGET_PCT, 5.0, 10.0, 15.0) if t <= y_top + 0.01]
    ax.set_yticks(yticks)
    for tick_label, tick_val in zip(ax.get_yticklabels(), yticks, strict=True):
        if tick_val == HOMOGENEOUS_TARGET_PCT:
            tick_label.set_color(target_color)
    for bar_container, val in zip(bars, means, strict=True):
        bar = bar_container[0]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.25,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def main() -> None:
    out_dir = _get_out_dir()
    out_pdf = Path(
        os.environ.get(
            "REPORT5_OPTIMIZATION_OUT_PDF",
            str(out_dir / "report5_optimization.pdf"),
        )
    ).resolve()
    plot_optimization_bar(out_pdf=out_pdf)


if __name__ == "__main__":
    main()
