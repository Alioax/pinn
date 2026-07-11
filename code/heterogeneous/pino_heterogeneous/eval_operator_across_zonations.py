# -*- coding: utf-8 -*-
"""
eval_operator_across_zonations.py
=================================

Evaluate the best sensor-based PINO operators on EVERY zone count (1..6) against
the reference solutions, to measure how the operator generalises across zonation:

  * zones 2-6  -> Dr. Fahs's COMSOL solutions (all combos in each file)
  * zone 1     -> Ogata-Banks analytical solution (homogeneous, semi-infinite)

Both operators are sensor-based (branch = velocity field sampled at K fixed
sensors), so they take ANY zonation without retraining. We evaluate two:
  * GRF  = best random-field-trained  (G01: ell=0.05, K=50, SOAP)
  * Interface = best piecewise/zoned-trained (X3: 5-zone-only, K=100, SOAP)

For each (model, zonation, geometry, velocity combo):
  branch = CFL of the piecewise field sampled at the model's sensor x*;
  C*(x*,t*) = model(x*, t*, branch)  at the 5 reference times on the COMSOL grid;
  error = mean relative-L2 vs the reference over the 5 times.

Reuses the validated pieces (DeepONet forward from `deeponet.py`,
`piecewise_u_at_xstar` + `branch_cfl_from_grf_sensor_u` from `utils.grf_sampling`,
constants from `utils.zone_velocity`, Ogata-Banks from
`code/shared/analytical_solution`). Inference only - fast, runs on a laptop.

Run (from this folder):
    python eval_operator_across_zonations.py
    python eval_operator_across_zonations.py --exp-dir results/exp_G01_...   # single model

Outputs (under --out-dir, default results/zonation_eval):
    zonation_error_by_case.csv     one row per (model, zonation, geometry, combo)
    zonation_error_summary.csv     mean/max/min rel-L2 per (model, zonation)
    zonation_error.png             error vs zone count, both models
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve().parent            # pino_heterogeneous/
_HETERO = _THIS.parent                             # code/heterogeneous/
_SHARED = _HETERO.parent / "shared" / "analytical_solution"
for _p in (_THIS, _HETERO, str(_SHARED)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deeponet import build_deeponet                                    # noqa: E402
from utils.grf_sampling import (                                       # noqa: E402
    piecewise_u_at_xstar,
    branch_cfl_from_grf_sensor_u,
    default_sensor_xstar,
)
from utils.zone_velocity import L_DEFAULT as L, T_MAX_DEFAULT as T_MAX  # noqa: E402
from analytical_solution import analytical_solution                    # noqa: E402

# physical dispersion matching the PE default (D = 1e-6 m^2/s):
D_M2_PER_DAY = 1e-6 * 86400.0          # = 0.0864 m^2/day
COMSOL_TIMES = np.array([100.0, 300.0, 600.0, 900.0, 1200.0])

# best models (dir relative to this file) — both sensor-based, generalise to any n
DEFAULT_MODELS = {
    "GRF (G01, l=0.05/K=50)": "results/exp_G01_grf_ell005_K50_N500_default_float64_soap",
    "Interface (X3, 5-zone/K=100)": "results/exp_X3_soap_K100_N500_5zoneonly_default_float64",
}

# zonation -> list of (label, comsol_file | None, interfaces in x*)
# (velocities read from each file; zone-1 has no file -> Ogata-Banks)
ZONATIONS = {
    1: [("homogeneous", None, [])],
    2: [("2z_20m", "data/comsol/2zones/data_2zones_20m.txt", [0.20]),
        ("2z_30m", "data/comsol/2zones/data_2zones_30m.txt", [0.30]),
        ("2z_40m", "data/comsol/2zones/data_2zones_40m.txt", [0.40]),
        ("2z_50m", "data/comsol/2zones/data_2zones_50m.txt", [0.50])],
    3: [("3z_15_15_70", "data/comsol/3zones/data_3zones_15_15_70.txt", [0.15, 0.30]),
        ("3z_20_20_60", "data/comsol/3zones/data_3zones_20_20_60.txt", [0.20, 0.40])],
    4: [("4z", "data/comsol_4zones.txt", [0.20, 0.40, 0.60])],
    5: [("5z", "data/comsol/5zones/data_5_zones.txt", [0.12, 0.24, 0.36, 0.48])],
    6: [("6z", "data/comsol/6zones/data_6_zones.txt", [0.11, 0.22, 0.33, 0.44, 0.55])],
}
# zone-1 homogeneous velocities to test against Ogata-Banks
ZONE1_VELOCITIES = [0.01, 0.03, 0.05]


# --------------------------------------------------------------------------- #
# COMSOL reader (general n-zone)
# --------------------------------------------------------------------------- #
_HDR = re.compile(r"@\s*t\s*=\s*([0-9.]+)\s*,\s*(.*)$")
_UXX = re.compile(r"u\d+\s*=\s*([0-9.]+)")


def parse_comsol(path: Path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    grid = None
    data: dict[tuple, dict[float, np.ndarray]] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("% Grid"):
            grid = np.array([float(p) for p in lines[i + 1].split()], dtype=np.float64)
            i += 2
            continue
        m = _HDR.search(ln)
        if m and grid is not None:
            t = float(m.group(1))
            combo = tuple(float(x) for x in _UXX.findall(m.group(2)))
            vals = np.array([float(p) for p in lines[i + 1].split()], dtype=np.float64)
            if vals.size == grid.size + 1:
                vals = vals[1:]
            data.setdefault(combo, {})[t] = vals[-grid.size:]
            i += 2
            continue
        i += 1
    return grid, data


# --------------------------------------------------------------------------- #
# model loading + prediction
# --------------------------------------------------------------------------- #
def load_operator(exp_dir: Path, device, dtype):
    meta = json.loads((exp_dir / "run_meta.json").read_text(encoding="utf-8"))
    branch_arch = list(meta["branch_architecture"])
    trunk_arch = list(meta["trunk_architecture"])
    if meta.get("sensor_xstar") is not None:
        sensor_xstar = np.array(meta["sensor_xstar"], dtype=np.float64)
    else:
        sensor_xstar = default_sensor_xstar(branch_arch[0])
    model = build_deeponet(branch_arch, trunk_arch, nn.Tanh).to(device)
    state = torch.load(exp_dir / "pino_heterogeneous_model.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, sensor_xstar, branch_arch[0]


def predict_field(model, x_star, t_star_val, branch_vec, device, dtype):
    """C*(x*) at one time for one medium (branch_vec = CFL at K sensors)."""
    n = x_star.size
    x_t = torch.tensor(x_star.reshape(-1, 1), dtype=dtype, device=device)
    t_t = torch.full((n, 1), float(t_star_val), dtype=dtype, device=device)
    branch_unique = torch.tensor(branch_vec.reshape(1, -1), dtype=dtype, device=device)
    case_idx = torch.zeros(n, dtype=torch.long, device=device)
    with torch.no_grad():
        c = model(x_t, t_t, branch_unique, case_idx)
    return c.cpu().numpy().flatten()


def rel_l2(pred, ref):
    den = np.linalg.norm(ref)
    return float(np.linalg.norm(pred - ref) / den) if den > 0 else float(np.linalg.norm(pred))


def branch_for_medium(sensor_xstar, interfaces, zone_u):
    sensor_u = piecewise_u_at_xstar(sensor_xstar, np.array(interfaces, dtype=np.float64),
                                    np.array(zone_u, dtype=np.float64))
    return branch_cfl_from_grf_sensor_u(sensor_u, l_m=L, t_max_d=T_MAX)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None,
                    help="Evaluate a single model dir (default: both G01 and X3).")
    ap.add_argument("--out-dir", default="results/zonation_eval")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    models = ({Path(args.exp_dir).name: args.exp_dir} if args.exp_dir else DEFAULT_MODELS)
    out_dir = (_THIS / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_label, rel in models.items():
        exp_dir = (_THIS / rel).resolve()
        model, sensor_xstar, K = load_operator(exp_dir, device, dtype)
        print(f"\n=== {model_label}  (K={K}, dir={exp_dir.name}) ===")

        for nz in sorted(ZONATIONS):
            errs_zone = []
            for label, cfile, interfaces in ZONATIONS[nz]:
                if cfile is None:
                    # zone 1: homogeneous vs Ogata-Banks on the 101-node grid
                    x_m = np.linspace(0.0, 100.0, 101)
                    x_star = x_m / L
                    for u in ZONE1_VELOCITIES:
                        branch = branch_for_medium(sensor_xstar, [], [u])
                        l2s = []
                        for t_d in COMSOL_TIMES:
                            pred = predict_field(model, x_star, t_d / T_MAX, branch, device, dtype)
                            ref = analytical_solution(x_m, float(t_d), U_param=u,
                                                      D_param=D_M2_PER_DAY, C_0_param=1.0)
                            l2s.append(rel_l2(pred, ref))
                        e = float(np.mean(l2s))
                        errs_zone.append(e)
                        case_rows.append(dict(model=model_label, zones=nz, geometry=label,
                                              combo=f"{u:g}", mean_rel_l2=e))
                else:
                    grid, data = parse_comsol((_HETERO / cfile))
                    x_star = grid / L
                    for combo, by_t in data.items():
                        branch = branch_for_medium(sensor_xstar, interfaces, list(combo))
                        l2s = []
                        for t_d in COMSOL_TIMES:
                            ref = next((v for tk, v in by_t.items()
                                        if np.isclose(tk, t_d, atol=1e-9)), None)
                            if ref is None:
                                continue
                            pred = predict_field(model, x_star, t_d / T_MAX, branch, device, dtype)
                            l2s.append(rel_l2(pred, ref / 1.0))
                        e = float(np.mean(l2s))
                        errs_zone.append(e)
                        case_rows.append(dict(model=model_label, zones=nz, geometry=label,
                                              combo=";".join(f"{v:g}" for v in combo),
                                              mean_rel_l2=e))
            a = np.array(errs_zone)
            summary_rows.append(dict(model=model_label, zones=nz, n_cases=a.size,
                                     mean_pct=round(a.mean() * 100, 3),
                                     max_pct=round(a.max() * 100, 3),
                                     min_pct=round(a.min() * 100, 3)))
            print(f"  {nz}-zone: n={a.size:4d}  mean={a.mean()*100:6.3f}%  "
                  f"max={a.max()*100:6.3f}%  min={a.min()*100:6.3f}%")

    # write CSVs
    with open(out_dir / "zonation_error_by_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys())); w.writeheader(); w.writerows(case_rows)
    with open(out_dir / "zonation_error_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)

    # figure: error vs zone count, both models
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_label in models:
            rows = [r for r in summary_rows if r["model"] == model_label]
            rows.sort(key=lambda r: r["zones"])
            zs = [r["zones"] for r in rows]
            mean = [r["mean_pct"] for r in rows]
            mx = [r["max_pct"] for r in rows]
            line, = ax.plot(zs, mean, "-o", lw=2, label=f"{model_label} (mean)")
            ax.plot(zs, mx, ":", color=line.get_color(), lw=1.2, alpha=0.7,
                    label=f"{model_label} (max)")
        ax.axvline(5, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.text(4.98, ax.get_ylim()[1] * 0.96, "X3 trained on 5-zone", fontsize=8,
                color="gray", ha="right")
        ax.set_xlabel("number of zones"); ax.set_ylabel("relative-L2 vs reference (%)")
        ax.set_title("Operator generalisation across zonation\n"
                     "(G01 trained on GRF fields; X3 on 5-zone piecewise; 1=Ogata-Banks, 2-6=COMSOL)")
        ax.set_xticks([1, 2, 3, 4, 5, 6]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "zonation_error.png", dpi=150)
        print(f"\nWrote CSVs + zonation_error.png to {out_dir}")
    except Exception as exc:
        print(f"\nWrote CSVs to {out_dir} (figure skipped: {exc})")


if __name__ == "__main__":
    main()
