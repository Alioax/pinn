# -*- coding: utf-8 -*-
"""
plot_worst_case.py — overlay an operator's prediction on the COMSOL reference for
a single medium, to SEE how/where it diverges. Defaults to X3's worst 5-zone case
(the maximally-alternating 0.01,0.05,0.01,0.05,0.01 on the 12/24/36/48 geometry).

Reuses the loaders in eval_operator_across_zonations.py (model, branch, COMSOL
parser). Inference only.

Run (from this folder):
    python plot_worst_case.py
    python plot_worst_case.py --exp-dir results/exp_G01_grf_ell005_K50_N500_default_float64_soap
    python plot_worst_case.py --zones 6 --combo 0.01,0.05,0.01,0.02,0.02,0.02
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_operator_across_zonations as E  # reuse loaders (import-safe)

# geometry per zone count (interfaces in x*) + the COMSOL file
GEOM = {
    5: ("data/comsol/5zones/data_5_zones.txt", [0.12, 0.24, 0.36, 0.48]),
    6: ("data/comsol/6zones/data_6_zones.txt", [0.11, 0.22, 0.33, 0.44, 0.55]),
    4: ("data/comsol_4zones.txt", [0.20, 0.40, 0.60]),
    3: ("data/comsol/3zones/data_3zones_20_20_60.txt", [0.20, 0.40]),
    2: ("data/comsol/2zones/data_2zones_50m.txt", [0.50]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="results/exp_X3_soap_K100_N500_5zoneonly_default_float64")
    ap.add_argument("--zones", type=int, default=5)
    ap.add_argument("--combo", default="0.01,0.05,0.01,0.05,0.01",
                    help="Zone velocities m/d, comma-separated.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    cfile, interfaces = GEOM[args.zones]
    combo = tuple(float(x) for x in args.combo.split(","))
    model, sensor_xstar, K = E.load_operator((E._THIS / args.exp_dir).resolve(), device, dtype)
    grid, data = E.parse_comsol((E._HETERO / cfile))
    # match combo (exact grid values)
    key = next((c for c in data if len(c) == len(combo)
                and all(abs(c[i] - combo[i]) < 1e-9 for i in range(len(combo)))), None)
    if key is None:
        raise SystemExit(f"combo {combo} not found in {cfile}")
    x_m = grid
    x_star = grid / E.L
    branch = E.branch_for_medium(sensor_xstar, interfaces, list(combo))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    edges = [0] + [b * 100 for b in interfaces] + [100]
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        ax.axvspan(a, b, color=plt.cm.Blues(0.06 + 0.5 * (combo[k] / 0.05)), alpha=0.28, zorder=0)
    for xi in edges[1:-1]:
        ax.axvline(xi, color="k", ls="--", lw=0.8, alpha=0.5)

    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(E.COMSOL_TIMES)))
    errs = []
    for t_d, col in zip(E.COMSOL_TIMES, cmap):
        ref = data[key][float(t_d)]
        pred = E.predict_field(model, x_star, t_d / E.T_MAX, branch, device, dtype)
        e = E.rel_l2(pred, ref)
        errs.append(e)
        ax.plot(x_m, ref, "-", color=col, lw=2, label=f"COMSOL t={t_d:g}d")
        ax.plot(x_m, pred, "--", color=col, lw=1.8, label=f"operator t={t_d:g}d (L2={e*100:.1f}%)")

    ax.set_xlabel("x (m)"); ax.set_ylabel("C (dimensionless)")
    ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.15)
    tag = Path(args.exp_dir).name
    ax.set_title(f"{args.zones}-zone worst case  u={args.combo}\n"
                 f"{tag}  (solid=COMSOL, dashed=operator; mean L2={np.mean(errs)*100:.1f}%)")
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.25)
    out = args.out or f"results/zonation_eval/worst_{args.zones}z_{tag}.png"
    Path(E._THIS / out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(E._THIS / out, dpi=150)
    print(f"per-time L2: {[round(e*100,1) for e in errs]}  mean={np.mean(errs)*100:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
