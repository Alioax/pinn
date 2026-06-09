# -*- coding: utf-8 -*-
"""Parametric PINN (raw-CFL input): one PDF per CFL value."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erfc, erfcx

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RESULTS = _REPO_ROOT / "code" / "homogeneous_cfl" / "parametric_cfl_pinn" / "results"
_DEFAULT_MODEL = _DEFAULT_RESULTS / "parametric_cfl_pinn_model.pt"
_DEFAULT_CONFIG = _DEFAULT_RESULTS / "run_config.json"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "_src" / "report-4-cfl" / "figs"

CFL_LIST = [0.12, 0.36, 0.6]

_DT_STR_TO_TORCH = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
}

# Match plot_baseline_pinn_concentration.py: same figure and axes layout for every CFL.
BASELINE_FIGSIZE = (4.5, 3)


def build_mlp(architecture: list[int], activation_cls: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(architecture) - 1):
        in_f, out_f = architecture[i], architecture[i + 1]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(architecture) - 2:
            layers.append(activation_cls())
        else:
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class ParametricPINN(nn.Module):
    def __init__(self, architecture: list[int], activation_cls: type[nn.Module]):
        super().__init__()
        self.net = build_mlp(architecture, activation_cls)

    def forward(self, x_star: torch.Tensor, t_star: torch.Tensor, cfl: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x_star, t_star, cfl], dim=1)
        return self.net(inputs)


def _default_config_dict() -> dict:
    return {
        "dtype": "torch.float64",
        "pinn_architecture": [3, 16, 16, 16, 16, 1],
        "U_VALUES": [0.01, 0.02, 0.03, 0.04, 0.05],
        "L": 100.0,
        "T_MAX": 1200.0,
        "D_M2_S": 1e-6,
        "SECONDS_PER_DAY": 86400.0,
        "C0": 5.0,
        "times_days": [100.0, 300.0, 600.0, 900.0, 1200.0],
        "num_points": 5001,
        "x_max": 100.0,
    }


def load_run_config(path: Path) -> dict:
    if not path.is_file():
        return _default_config_dict()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_one_pe(
    *,
    cfl: float,
    u_val: float,
    d_m2_day: float,
    l_ref: float,
    t_phys: float,
    c0: float,
    model: ParametricPINN,
    device: torch.device,
    dtype: torch.dtype,
    times_days: np.ndarray,
    x_plot: np.ndarray,
    with_legend: bool,
    out_pdf: Path,
) -> None:
    xp = torch.tensor(x_plot.reshape(-1, 1) / l_ref, dtype=dtype, device=device)
    cfl_col = torch.full((len(x_plot), 1), float(cfl), dtype=dtype, device=device)

    fig = plt.figure(figsize=BASELINE_FIGSIZE, tight_layout=True)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

    for idx, ti in enumerate(times_days):
        color = colors[idx % len(colors)]
        if ti <= 0:
            c_ana = np.zeros_like(x_plot, dtype=np.float64)
        else:
            sqrt_dt = np.sqrt(d_m2_day * ti)
            term1 = erfc((x_plot - u_val * ti) / (2.0 * sqrt_dt))
            ux_over_d = u_val * x_plot / d_m2_day
            b = (x_plot + u_val * ti) / (2.0 * sqrt_dt)
            exponent = ux_over_d - b**2
            term2 = np.exp(np.clip(exponent, -745.0, 700.0)) * erfcx(b)
            c_ana = (c0 / 2.0) * (term1 + term2)
            c_ana = np.nan_to_num(c_ana, nan=0.0, posinf=0.0, neginf=0.0)
        t_star = torch.full((len(x_plot), 1), float(ti / t_phys), dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(xp, t_star, cfl_col).cpu().numpy().flatten()
        ax.plot(x_plot, c_pred * c0, linewidth=2, linestyle="-", color=color)
        ax.plot(x_plot, c_ana, linewidth=2, linestyle="--", color=color, alpha=0.7)
        ax.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{ti:.1f}")

    ax.set_xlabel("Distance x (m)", fontsize=12)
    ax.set_ylabel("Concentration C (kg/m3)", fontsize=12)
    _pad_x, _pad_y = 0.05, 0.05
    ax.set_xlim(0 - _pad_x, l_ref + _pad_x)
    ax.set_ylim(0 - _pad_y, c0 + _pad_y)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(NullLocator())

    if with_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=4,
            frameon=False,
            fontsize=10,
            labelspacing=0.5,
            columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color("black")
            text.set_alpha(1.0)

    figs_dir = out_pdf.parent
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model_path = Path(os.environ.get("PARAMETRIC_PINN_MODEL", _DEFAULT_MODEL)).resolve()
    config_path = Path(os.environ.get("PARAMETRIC_PINN_CONFIG", _DEFAULT_CONFIG)).resolve()
    figs_dir = Path(
        os.environ.get(
            "PARAMETRIC_PINN_REPORT_FIGS_DIR",
            os.environ.get("PINO_REPORT_FIGS_DIR", _FIGS_DIR),
        )
    ).resolve()

    cfg = load_run_config(config_path)
    d0 = _default_config_dict()
    merged_cfg = dict(d0)
    merged_cfg.update(cfg)
    dtype = _DT_STR_TO_TORCH.get(str(merged_cfg.get("dtype", "torch.float64")), torch.float64)
    torch.set_default_dtype(dtype)
    if "pinn_architecture" in merged_cfg:
        arch = [int(x) for x in merged_cfg["pinn_architecture"]]
    else:
        arch = [int(x) for x in d0["pinn_architecture"]]

    l_ref = float(merged_cfg["L"])
    t_phys = float(merged_cfg["T_MAX"])
    c0 = float(merged_cfg["C0"])
    d_m2_day = float(merged_cfg["D_M2_S"]) * float(merged_cfg.get("SECONDS_PER_DAY", 86400.0))

    mpl.rcParams["figure.dpi"] = 800
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#FF5F05", "#13294B", "#009FD4",
               "#8C1515",
               "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13",]
    )
    plt.rcParams["font.family"] = "Times New Roman"

    device = torch.device("cpu")
    model = ParametricPINN(arch, nn.Tanh).to(device=device, dtype=dtype)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    times_days = np.array(merged_cfg.get("times_days", d0["times_days"]), dtype=float)
    num_points = int(merged_cfg.get("num_points", d0["num_points"]))
    x_max = float(merged_cfg.get("x_max", d0["x_max"]))
    x_plot = np.linspace(0.0, x_max, num_points)

    for i, cfl in enumerate(CFL_LIST):
        u_val = float(cfl * l_ref / t_phys)
        cfl_token = f"{cfl:.2f}".replace(".", "p")
        out_pdf = figs_dir / f"parametric_pinn_concentration_CFL{cfl_token}.pdf"
        plot_one_pe(
            cfl=float(cfl),
            u_val=u_val,
            d_m2_day=d_m2_day,
            l_ref=l_ref,
            t_phys=t_phys,
            c0=c0,
            model=model,
            device=device,
            dtype=dtype,
            times_days=times_days,
            x_plot=x_plot,
            with_legend=(i == 0),
            out_pdf=out_pdf,
        )
        print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
