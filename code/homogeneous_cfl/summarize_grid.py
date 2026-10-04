# -*- coding: utf-8 -*-
"""
Collect every grid_results/<run>/run_meta.json into one table (standard library only).

    python code/homogeneous_cfl/summarize_grid.py

Prints the table and writes grid_results/grid_summary.csv and grid_results/grid_summary.md.
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "grid_results")


def optimizer_label(m):
    if m.get("optimizer") == "soap":
        return "SOAP"
    lr, mi = m.get("lbfgs_lr"), m.get("lbfgs_max_iter")
    if lr == 1.0 and mi == 1:
        return "L-BFGS (Report 4)"
    if lr == 0.1 and mi == 20:
        return "L-BFGS (paper)"
    return f"L-BFGS (lr {lr}, max_iter {mi})"


def fmt_time(s):
    if s is None:
        return "n/a"
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.1f} s"


rows = []
for path in sorted(glob.glob(os.path.join(GRID, "*", "run_meta.json"))):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    rows.append({
        "run": m["run_name"],
        "model": m["model"],
        "samples": m.get("n_cfl_samples"),
        "optimizer": optimizer_label(m),
        "seed": m.get("seed", ""),
        "steps": m.get("steps_actual") if m.get("steps_actual") is not None else "n/a",
        "early_stopped": m.get("early_stopped", ""),
        "train_time": fmt_time(m.get("train_wall_clock_s")),
        "train_s": m.get("train_wall_clock_s"),
        "median_%": round(100 * m["rel_l2_median"], 3),
        "max_%": round(100 * m["rel_l2_max"], 3),
        "mean_%": round(100 * m["rel_l2_mean"], 3),
        "final_loss": (m.get("final_loss") or {}).get("total", ""),
        "device": (m.get("environment") or {}).get("gpu") or (m.get("environment") or {}).get("device", ""),
    })

if not rows:
    raise SystemExit(f"No run_meta.json found under {GRID}")

cols = list(rows[0].keys())
with open(os.path.join(GRID, "grid_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

show = ["run", "model", "samples", "optimizer", "steps", "train_time", "median_%", "max_%"]
md = ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
md += ["| " + " | ".join(str(r[c]) for c in show) + " |" for r in rows]
with open(os.path.join(GRID, "grid_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote {os.path.join(GRID, 'grid_summary.csv')} and grid_summary.md")
