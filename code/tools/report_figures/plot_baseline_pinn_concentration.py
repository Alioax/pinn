# -*- coding: utf-8 -*-
"""Load a baseline PINN checkpoint and export a concentration PDF figure."""

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
_DEFAULT_RESULTS = _REPO_ROOT / "code" / "homogeneous_cfl" / "baseline_cfl_pinn" / "results"
_DEFAULT_MODEL = _DEFAULT_RESULTS / "baseline_cfl_pinn_model.pt"
_DEFAULT_CONFIG = _DEFAULT_RESULTS / "run_config.json"
_FIGS_DIR = _REPO_ROOT / "docs" / "reports" / "_src" / "report-4-cfl" / "figs"
_OUT_PDF = _FIGS_DIR / "baseline_pinn_concentration.pdf"

_DT_STR_TO_TORCH = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
}


class PINN_Transp(nn.Module):
    def __init__(self, num_layers: int, num_neurons: int, activation: type[nn.Module]):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, num_neurons), activation()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation())
        layers.append(nn.Linear(num_neurons, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x_star: torch.Tensor, tau_star: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x_star, tau_star], dim=1)
        return self.net(inputs)


def _default_config_dict() -> dict:
    return {
        "model_name": "baseline_cfl_pinn",
        "U": 0.05,
        "D_M2_S": 1e-6,
        "L": 100.0,
        "C0": 5.0,
        "T_MAX": 1200.0,
        "SECONDS_PER_DAY": 86400.0,
        "times_days": [100.0, 300.0, 600.0, 900.0, 1200.0],
        "num_points": 5001,
        "x_max": 100.0,
        "num_layers": None,
        "num_neurons": None,
        "dtype": "torch.float64",
    }


def load_run_config(path: Path) -> dict:
    if not path.is_file():
        return _default_config_dict()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_architecture_from_state_dict(state_dict: dict) -> tuple[int, int]:
    linear_weight_keys = []
    for key in state_dict:
        if key.startswith("net.") and key.endswith(".weight"):
            try:
                idx = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            linear_weight_keys.append((idx, key))
    linear_weight_keys.sort(key=lambda x: x[0])

    if len(linear_weight_keys) < 2:
        raise ValueError("Could not infer architecture from checkpoint weights.")

    first_weight = state_dict[linear_weight_keys[0][1]]
    num_neurons = int(first_weight.shape[0])
    num_layers = len(linear_weight_keys) - 1
    return num_layers, num_neurons


def apply_physics_constants(cfg: dict) -> tuple[float, float, float, float, float, float]:
    U = float(cfg["U"])
    L = float(cfg["L"])
    C0 = float(cfg["C0"])
    T_max = float(cfg["T_MAX"])
    seconds_per_day = float(cfg.get("SECONDS_PER_DAY", 86400.0))
    D_m2_s = float(cfg["D_M2_S"])
    D_m2_day = D_m2_s * seconds_per_day
    return U, L, C0, T_max, D_m2_s, D_m2_day


def main() -> None:
    model_path = Path(os.environ.get("BASELINE_PINN_MODEL", _DEFAULT_MODEL)).resolve()
    config_path = Path(os.environ.get("BASELINE_PINN_CONFIG", _DEFAULT_CONFIG)).resolve()
    figs_dir = Path(
        os.environ.get(
            "BASELINE_REPORT_FIGS_DIR",
            os.environ.get("PINO_REPORT_FIGS_DIR", _FIGS_DIR),
        )
    ).resolve()
    out_pdf = Path(
        os.environ.get(
            "BASELINE_PINN_OUT_PDF",
            str(figs_dir / _OUT_PDF.name),
        )
    ).resolve()

    cfg = load_run_config(config_path)
    dflt = _default_config_dict()
    merged_cfg = dict(dflt)
    merged_cfg.update(cfg)
    U, L, C0, T_max, _, D_m2_day = apply_physics_constants(merged_cfg)

    dtype = _DT_STR_TO_TORCH.get(str(merged_cfg.get("dtype", "torch.float64")), torch.float64)
    torch.set_default_dtype(dtype)

    mpl.rcParams["figure.dpi"] = 800
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#FF5F05", "#13294B", "#009FD4",
        "#8C1515",
        "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13",]
    )
    plt.rcParams["font.family"] = "Times New Roman"

    device = torch.device("cpu")
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    num_layers_cfg = merged_cfg.get("num_layers")
    num_neurons_cfg = merged_cfg.get("num_neurons")
    if num_layers_cfg is None or num_neurons_cfg is None:
        num_layers, num_neurons = infer_architecture_from_state_dict(state)
    else:
        num_layers = int(num_layers_cfg)
        num_neurons = int(num_neurons_cfg)

    model = PINN_Transp(num_layers, num_neurons, nn.Tanh).to(device=device, dtype=dtype)
    model.load_state_dict(state)
    model.eval()

    times_days = np.array(merged_cfg.get("times_days", dflt["times_days"]), dtype=float)
    num_points = int(merged_cfg.get("num_points", dflt["num_points"]))
    x_max = float(merged_cfg.get("x_max", dflt["x_max"]))
    X_PLOT = np.linspace(0, x_max, num_points)
    xp = torch.tensor(X_PLOT.reshape(-1, 1) / L, dtype=dtype, device=device)

    fig = plt.figure(figsize=(4.5, 3), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax.plot([], [], linewidth=2, linestyle="-", color="black", label="PINN")
    ax.plot([], [], linewidth=2, linestyle="--", color="black", label="Analytical")

    for idx, ti in enumerate(times_days):
        color = colors[idx % len(colors)]
        if ti <= 0:
            c1 = np.zeros_like(X_PLOT, dtype=np.float64)
        else:
            sqrt_dt = np.sqrt(D_m2_day * ti)
            term1 = erfc((X_PLOT - U * ti) / (2.0 * sqrt_dt))
            ux_over_d = U * X_PLOT / D_m2_day
            b = (X_PLOT + U * ti) / (2.0 * sqrt_dt)
            exponent = ux_over_d - b**2
            term2 = np.exp(np.clip(exponent, -745.0, 700.0)) * erfcx(b)
            c1 = (C0 / 2.0) * (term1 + term2)
            c1 = np.nan_to_num(c1, nan=0.0, posinf=0.0, neginf=0.0)
        tau_eval = torch.tensor((ti / T_max) * np.ones_like(X_PLOT).reshape(-1, 1), dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(xp, tau_eval).cpu().numpy()
        ax.plot(X_PLOT, c_pred * C0, linewidth=2, linestyle="-", color=color)
        ax.plot(X_PLOT, c1, linewidth=2, linestyle="--", color=color, alpha=0.7)
        ax.plot([], [], marker="s", markersize=8, linestyle="None", color=color, label=f"{ti:.1f}")

    ax.set_xlabel("Distance x (m)", fontsize=12)
    ax.set_ylabel("Concentration C (kg/m3)", fontsize=12)
    _pad_x, _pad_y = 0.05, 0.05
    ax.set_xlim(0 - _pad_x, x_max + _pad_x)
    ax.set_ylim(0 - _pad_y, C0 + _pad_y)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(NullLocator())
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

    figs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
