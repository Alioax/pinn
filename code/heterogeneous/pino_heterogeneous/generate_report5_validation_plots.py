# -*- coding: utf-8 -*-
"""
Generate COMSOL validation plots for Report 5 batch experiment checkpoints.

Loads each saved model and writes 81 comparison PNGs under
``results/exp_*/comsol_validation/`` (plus ``comsol_validation_summary.csv``).

Usage (from this directory):
  python generate_report5_validation_plots.py
  python generate_report5_validation_plots.py --experiments exp_B_lhc_N500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve()
_PINO_DIR = _SCRIPT.parent
_TRAIN_SCRIPT = _PINO_DIR / "pinn1d_heterogeneous_parametric_neural_operator.py"

REPORT5_EXPERIMENTS: list[tuple[str, str, list[str]]] = [
    (
        "exp_A_capacity",
        "capacity",
        ["--design", "capacity", "--out-dir", "results/exp_A_capacity"],
    ),
    (
        "exp_B_lhc_N500",
        "lhc",
        [
            "--design",
            "lhc",
            "--n-train",
            "500",
            "--out-dir",
            "results/exp_B_lhc_N500",
        ],
    ),
    (
        "exp_C_maximin_N500",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "500",
            "--out-dir",
            "results/exp_C_maximin_N500",
        ],
    ),
    (
        "exp_D_anchored_N500",
        "anchored",
        [
            "--design",
            "anchored",
            "--n-train",
            "500",
            "--out-dir",
            "results/exp_D_anchored_N500",
        ],
    ),
    (
        "exp_E_capacity_dense",
        "capacity",
        [
            "--design",
            "capacity",
            "--arch",
            "dense",
            "--out-dir",
            "results/exp_E_capacity_dense",
        ],
    ),
    (
        "exp_F_maximin_N500_float64",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "500",
            "--dtype",
            "float64",
            "--out-dir",
            "results/exp_F_maximin_N500_float64",
        ],
    ),
    (
        "exp_H_capacity_float64",
        "capacity",
        [
            "--design",
            "capacity",
            "--dtype",
            "float64",
            "--out-dir",
            "results/exp_H_capacity_float64",
        ],
    ),
    (
        "exp_I_maximin_N500_float64_lr01",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "500",
            "--dtype",
            "float64",
            "--lr-lbfgs",
            "0.1",
            "--out-dir",
            "results/exp_I_maximin_N500_float64_lr01",
        ],
    ),
    (
        "exp_J_maximin_N500_float64_lr01_maxiter20",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "500",
            "--dtype",
            "float64",
            "--lr-lbfgs",
            "0.1",
            "--lbfgs-max-iter",
            "20",
            "--out-dir",
            "results/exp_J_maximin_N500_float64_lr01_maxiter20",
        ],
    ),
    (
        "exp_K_maximin_N200_lr05_epochs5000",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "200",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.5",
            "--epochs",
            "5000",
            "--out-dir",
            "results/exp_K_maximin_N200_lr05_epochs5000",
        ],
    ),
    (
        "exp_L_maximin_N200_float64_lr05_epochs5000",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "200",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.5",
            "--epochs",
            "5000",
            "--out-dir",
            "results/exp_L_maximin_N200_float64_lr05_epochs5000",
        ],
    ),
    (
        "exp_M_maximin_N200_float64_lr01_epochs5000",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "200",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.1",
            "--epochs",
            "5000",
            "--out-dir",
            "results/exp_M_maximin_N200_float64_lr01_epochs5000",
        ],
    ),
    (
        "exp_N_maximin_N250_float64_lr01_maxiter25_epochs500",
        "maximin",
        [
            "--design",
            "maximin",
            "--n-train",
            "250",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.1",
            "--lbfgs-max-iter",
            "25",
            "--epochs",
            "500",
            "--out-dir",
            "results/exp_N_maximin_N250_float64_lr01_maxiter25_epochs500",
        ],
    ),
    (
        "exp_O_grf_K100_N500_mixed_ell005_01_w64_float64_lr01_maxiter5",
        "grf",
        [
            "--design",
            "grf",
            "--n-sensors",
            "100",
            "--n-grf",
            "300",
            "--n-piecewise-per-zones",
            "50",
            "--grf-corr-lengths",
            "0.05,0.1",
            "--arch",
            "w64",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.1",
            "--lbfgs-max-iter",
            "5",
            "--epochs",
            "1000",
            "--out-dir",
            "results/exp_O_grf_K100_N500_mixed_ell005_01_w64_float64_lr01_maxiter5",
        ],
    ),
    (
        "exp_P_piecewise_K100_N500_z1to5_min10pct_w64_float64_lr01_maxiter5",
        "piecewise",
        [
            "--design",
            "grf",
            "--n-sensors",
            "100",
            "--n-grf",
            "0",
            "--n-piecewise-per-zones",
            "100",
            "--piecewise-zone-counts",
            "1,2,3,4,5",
            "--min-zone-frac",
            "0.1",
            "--arch",
            "w64",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--lr-lbfgs",
            "0.1",
            "--lbfgs-max-iter",
            "5",
            "--epochs",
            "1000",
            "--out-dir",
            "results/exp_P_piecewise_K100_N500_z1to5_min10pct_w64_float64_lr01_maxiter5",
        ],
    ),
    (
        "exp_Ra_piecewise_K100_N500_adam1000_lr25e3_lbfgs250_maxiter5_float64",
        "piecewise",
        [
            "--design",
            "grf",
            "--n-sensors",
            "100",
            "--n-grf",
            "0",
            "--n-piecewise-per-zones",
            "100",
            "--piecewise-zone-counts",
            "1,2,3,4,5",
            "--min-zone-frac",
            "0.1",
            "--arch",
            "w64",
            "--dtype",
            "float64",
            "--early-stop-patience",
            "150",
            "--adam-epochs",
            "1000",
            "--lr-adam",
            "0.0025",
            "--lr-lbfgs",
            "0.1",
            "--lbfgs-max-iter",
            "5",
            "--epochs",
            "250",
            "--out-dir",
            "results/exp_Ra_piecewise_K100_N500_adam1000_lr25e3_lbfgs250_maxiter5_float64",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COMSOL validation plots for Report 5 checkpoints."
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Experiment folder names under results/ (default: all four Report 5 runs). "
            "Example: exp_B_lhc_N500"
        ),
    )
    return parser.parse_args()


def _resolve_out_dir(name: str, cli_args: list[str]) -> Path:
    if "--out-dir" in cli_args:
        idx = cli_args.index("--out-dir")
        return (_PINO_DIR / cli_args[idx + 1]).resolve()
    return _PINO_DIR / "results" / name


def main() -> int:
    args = parse_args()
    selected = set(args.experiments) if args.experiments else None

    failures: list[str] = []
    for name, _design, cli_args in REPORT5_EXPERIMENTS:
        if selected is not None and name not in selected:
            continue

        out_dir = _resolve_out_dir(name, cli_args)
        checkpoint = out_dir / "pino_heterogeneous_model.pt"
        if not checkpoint.is_file():
            print(f"SKIP {name}: no checkpoint at {checkpoint}")
            failures.append(name)
            continue

        plot_dir = out_dir / "comsol_validation"
        print(f"\n=== {name} -> {plot_dir} ===")
        cmd = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            *cli_args,
            "--validate-only",
        ]
        rc = subprocess.run(cmd, cwd=str(_PINO_DIR)).returncode
        if rc != 0:
            print(f"FAILED {name} (exit {rc})")
            failures.append(name)

    if failures:
        print(f"\nFinished with failures: {', '.join(failures)}")
        return 1
    print("\nAll validation plots generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
