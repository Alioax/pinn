# -*- coding: utf-8 -*-
"""
analyze_calibration.py  (numpy + matplotlib only - NO torch, runs on the laptop)
================================================================================

Reads the outputs of calibrate_single_instance_pinn.py and answers the step-1
question:  can the self-validation PROXY predict the TRUE COMSOL error, and what
CONSERVATIVE threshold tau lets us trust "proxy < tau => accurate"?

Deployment logic being calibrated:  during a run we STOP the first time the
proxy dips below tau and trust the model.  So for a candidate tau, the honest
error is, per case, the TRUE error at the FIRST eval where proxy < tau -- and the
guarantee is the WORST of those across all calibration cases.

Outputs (under --dir):
    calibration_report.txt      chosen tau, worst/mean true error, per-case table
    calibration_proxy_vs_true.png   scatter (proxy vs true, log-log, by contrast)
    calibration_threshold.png       worst-case true error vs tau, with target line

Usage:
    python analyze_calibration.py --dir results/calib_single --target 0.01
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def load_trajectories(d: Path) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(str(d / "calib_traj_*.csv"))):
        tag = Path(path).stem.replace("calib_traj_", "")
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
        rows.sort(key=lambda r: r["epoch"])
        cases[tag] = rows
    return cases


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def first_crossing_true(rows: list[dict], tau: float) -> float | None:
    """True error at the first eval where proxy_combined < tau (deployment stop)."""
    for r in rows:
        if r["proxy_combined"] < tau:
            return r["true_rel_l2"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/calib_single",
                    help="Folder with calib_traj_*.csv (relative to this script).")
    ap.add_argument("--target", type=float, default=0.01,
                    help="Target true rel-L2 the threshold must guarantee (default 0.01 = 1%).")
    ap.add_argument("--proxy-key", default="proxy_combined",
                    help="Which proxy column to calibrate (default proxy_combined).")
    args = ap.parse_args()

    d = (Path(__file__).resolve().parent / args.dir)
    cases = load_trajectories(d)
    if not cases:
        raise SystemExit(f"No calib_traj_*.csv found in {d}")

    # pooled points (all cases, all evals) for the correlation diagnostic
    proxy_all, true_all, contrast_all = [], [], []
    for tag, rows in cases.items():
        for r in rows:
            proxy_all.append(r[args.proxy_key])
            true_all.append(r["true_rel_l2"])
            contrast_all.append(r.get("contrast", np.nan))
    proxy_all = np.array(proxy_all)
    true_all = np.array(true_all)
    rho = spearman(proxy_all, true_all)

    # sweep tau: for each, worst first-crossing true error across cases + coverage
    taus = np.logspace(np.log10(max(proxy_all.min(), 1e-8)),
                       np.log10(proxy_all.max()), 60)
    sweep = []
    for tau in taus:
        crossings = {tag: first_crossing_true(rows, tau) for tag, rows in cases.items()}
        reached = {t: v for t, v in crossings.items() if v is not None}
        n_reached = len(reached)
        worst = max(reached.values()) if reached else float("nan")
        mean = float(np.mean(list(reached.values()))) if reached else float("nan")
        sweep.append((tau, n_reached, worst, mean))

    # chosen tau = largest tau that (a) all cases reached and (b) worst < target
    n_cases = len(cases)
    ok = [s for s in sweep if s[1] == n_cases and np.isfinite(s[2]) and s[2] < args.target]
    chosen = max(ok, key=lambda s: s[0]) if ok else None

    # ---- report ----
    lines = []
    lines.append("PINN self-validation proxy -> COMSOL true-error calibration")
    lines.append("=" * 62)
    lines.append(f"cases: {n_cases}   pooled evals: {proxy_all.size}")
    lines.append(f"proxy key: {args.proxy_key}")
    lines.append(f"Spearman(proxy, true) over pooled points: {rho:+.3f}  "
                 f"(closer to +1 = proxy tracks true error better)")
    lines.append(f"target true rel-L2 to guarantee: {args.target*100:.2f}%")
    lines.append("")
    if chosen:
        tau, nr, worst, mean = chosen
        lines.append(f">>> CONSERVATIVE THRESHOLD tau = {tau:.3e}")
        lines.append(f"    at this tau, all {nr}/{n_cases} cases reach it and the")
        lines.append(f"    WORST true error at first-crossing = {worst*100:.3f}%  "
                     f"(mean {mean*100:.3f}%)")
        lines.append(f"    -> use tau with margin, e.g. {tau/2:.3e}, in production.")
    else:
        lines.append(">>> No tau guarantees the target for ALL cases.")
        lines.append("    Either loosen --target, push --max-epochs higher, or improve")
        lines.append("    interface handling (the hard slow-fast-slow cases likely limit it).")
        # show best achievable: tau where all reached, min worst
        allr = [s for s in sweep if s[1] == n_cases and np.isfinite(s[2])]
        if allr:
            best = min(allr, key=lambda s: s[2])
            lines.append(f"    best achievable at full coverage: worst true "
                         f"{best[2]*100:.3f}% at tau={best[0]:.3e}")
    lines.append("")
    lines.append("per-case summary (final proxy, final/best true error):")
    lines.append(f"  {'tag':<26}{'contrast':>9}{'final_proxy':>13}{'final_true%':>12}{'best_true%':>11}")
    for tag, rows in sorted(cases.items(), key=lambda kv: kv[1][-1].get("contrast", 0)):
        last = rows[-1]
        best_true = min(r["true_rel_l2"] for r in rows)
        lines.append(f"  {tag:<26}{last.get('contrast', float('nan')):>9.2f}"
                     f"{last['proxy_combined']:>13.3e}{last['true_rel_l2']*100:>12.3f}"
                     f"{best_true*100:>11.3f}")

    report = "\n".join(lines)
    (d / "calibration_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)

    # ---- plots ----
    if HAVE_PLT:
        contrast_arr = np.array(contrast_all, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(proxy_all, true_all * 100, c=contrast_arr, cmap="viridis",
                        s=18, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"self-validation proxy ({args.proxy_key})")
        ax.set_ylabel("true rel-L2 vs COMSOL (%)")
        ax.set_title("Does the proxy predict the true error?")
        ax.axhline(args.target * 100, color="crimson", ls="--", lw=1,
                   label=f"target {args.target*100:.1f}%")
        if chosen:
            ax.axvline(chosen[0], color="k", ls=":", lw=1, label=f"tau={chosen[0]:.2e}")
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="adjacent-zone contrast")
        fig.tight_layout()
        fig.savefig(d / "calibration_proxy_vs_true.png", dpi=140)
        plt.close(fig)

        taus_a = np.array([s[0] for s in sweep])
        worst_a = np.array([s[2] for s in sweep]) * 100
        cover_a = np.array([s[1] for s in sweep]) / n_cases
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(taus_a, worst_a, "-o", ms=3, label="worst true error at first-crossing")
        ax.set_xscale("log")
        ax.set_xlabel("threshold tau (proxy)")
        ax.set_ylabel("worst-case true rel-L2 (%)", color="C0")
        ax.axhline(args.target * 100, color="crimson", ls="--", lw=1,
                   label=f"target {args.target*100:.1f}%")
        if chosen:
            ax.axvline(chosen[0], color="k", ls=":", lw=1, label=f"chosen tau={chosen[0]:.2e}")
        ax2 = ax.twinx()
        ax2.plot(taus_a, cover_a, "-s", ms=3, color="C1", alpha=0.6)
        ax2.set_ylabel("fraction of cases reaching tau", color="C1")
        ax2.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("Conservative threshold selection")
        fig.tight_layout()
        fig.savefig(d / "calibration_threshold.png", dpi=140)
        plt.close(fig)
        print(f"\nWrote calibration_report.txt + 2 PNGs to {d}")


if __name__ == "__main__":
    main()
