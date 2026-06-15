# -*- coding: utf-8 -*-
"""
Exploratory Idea 1: post-training defect correction via FDM solve of L[delta] = -r.

Loads an imperfect PINO (default: exp_E_capacity_dense), computes autograd PDE
residuals, solves for delta, and reports physical + COMSOL metrics for C_tilde+delta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EXPLORE_DIR = Path(__file__).resolve().parent
_HETERO_ROOT = _EXPLORE_DIR.parent.parent / "heterogeneous"
for _p in (_HETERO_ROOT, _HETERO_ROOT / "pino_heterogeneous"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.comsol_4zones import load_case  # noqa: E402

from fdm_correction import solve_correction_fdm  # noqa: E402
from plots import (  # noqa: E402
    comsol_c_star_on_x_star,
    comsol_mean_rel_l2,
    match_comsol_times,
    plot_correction_case,
    u_case_tag,
)
from residual import (  # noqa: E402
    L,
    T_MAX,
    UCases,
    evaluate_model_on_grid,
    load_model_from_checkpoint,
    pde_residual_autograd_grid,
    pde_residual_autograd_slice,
    pde_residual_fd_grid,
    residual_interior_mask,
    residual_stats,
)

COMSOL_DATA_PATH = (_HETERO_ROOT / "data" / "comsol_4zones.txt").resolve()
COMSOL_TIMES_DAYS = np.array([100.0, 300.0, 600.0, 900.0, 1200.0], dtype=float)

DEFAULT_CHECKPOINT = (
    _HETERO_ROOT / "pino_heterogeneous" / "results" / "exp_E_capacity_dense"
)

PROBE_CASES: list[UCases] = [
    (0.01, 0.05, 0.01, 0.01),  # worst in exp_E validation
    (0.01, 0.01, 0.01, 0.01),  # uniform slow
    (0.01, 0.03, 0.01, 0.03),  # mid heterogeneous
]


def _nearest_time_index(t_star: np.ndarray, t_val: float) -> int:
    return int(np.argmin(np.abs(t_star - t_val)))


def run_one_case(
    *,
    model,
    device: torch.device,
    dtype: torch.dtype,
    u_case: UCases,
    x_star: np.ndarray,
    t_star: np.ndarray,
    x_m_comsol: np.ndarray,
    comsol_by_time: dict[float, np.ndarray],
    out_dir: Path,
    plot_x_star: np.ndarray,
) -> dict:
    """Full pipeline for a single velocity quadruple."""
    c_tilde = evaluate_model_on_grid(
        model, x_star, t_star, u_case, device=device, dtype=dtype
    )
    r = pde_residual_autograd_grid(
        model, x_star, t_star, u_case, device=device, dtype=dtype
    )

    delta, solve_info = solve_correction_fdm(r, x_star, t_star, u_case)
    c_corr = c_tilde + delta

    # Interpolate to plot grid for COMSOL comparison and FD residuals at COMSOL times.
    nt = t_star.size
    c_tilde_plot = np.zeros((plot_x_star.size, nt), dtype=np.float64)
    c_corr_plot = np.zeros((plot_x_star.size, nt), dtype=np.float64)
    for j in range(nt):
        c_tilde_plot[:, j] = np.interp(plot_x_star, x_star, c_tilde[:, j])
        c_corr_plot[:, j] = np.interp(plot_x_star, x_star, c_corr[:, j])

    mask = residual_interior_mask(c_tilde)
    r_before_ag = residual_stats(r, mask)
    r_after_fd = residual_stats(
        pde_residual_fd_grid(c_corr_plot, plot_x_star, t_star, u_case),
        residual_interior_mask(c_corr_plot),
    )

    # Autograd residual on corrected field: evaluate slice-by-slice via FD on sum
    # (autograd does not apply to numpy sum); report FD as primary "after" metric.
    r_after_ag_slices = []
    r_before_slices_plot = []
    r_after_slices_plot = []
    c_tilde_slices = []
    c_corr_slices = []
    c_comsol_slices = []
    t_days_list = []
    t_star_list = []

    matched = match_comsol_times(comsol_by_time, COMSOL_TIMES_DAYS)
    for t_days, c_comsol in matched:
        t_val = t_days / T_MAX
        j = _nearest_time_index(t_star, t_val)
        t_star_list.append(float(t_star[j]))
        t_days_list.append(t_days)

        with torch.enable_grad():
            r_b = pde_residual_autograd_slice(
                model, plot_x_star, float(t_star[j]), u_case, device=device, dtype=dtype
            )
        r_a = pde_residual_fd_grid(
            c_corr_plot,
            plot_x_star,
            np.array([t_star[max(j - 1, 1)], t_star[j]], dtype=np.float64),
            u_case,
        )[:, 1]

        r_before_slices_plot.append(r_b)
        r_after_slices_plot.append(r_a)
        c_tilde_slices.append(c_tilde_plot[:, j])
        c_corr_slices.append(c_corr_plot[:, j])
        c_comsol_slices.append(
            comsol_c_star_on_x_star(plot_x_star, x_m_comsol, c_comsol, l_m=L)
        )
        r_after_ag_slices.append(r_a)

    mean_l2_before = comsol_mean_rel_l2(
        plot_x_star, x_m_comsol, comsol_by_time, COMSOL_TIMES_DAYS, c_tilde_slices
    )
    mean_l2_after = comsol_mean_rel_l2(
        plot_x_star, x_m_comsol, comsol_by_time, COMSOL_TIMES_DAYS, c_corr_slices
    )

    j_last = _nearest_time_index(t_star, 1.0)
    delta_latest = np.interp(plot_x_star, x_star, delta[:, j_last])

    plot_path = out_dir / f"compare_correction_{u_case_tag(u_case)}.png"
    plot_correction_case(
        x_star=plot_x_star,
        t_days_list=t_days_list,
        t_star_list=t_star_list,
        c_tilde_slices=c_tilde_slices,
        c_corr_slices=c_corr_slices,
        c_comsol_slices=c_comsol_slices,
        r_before_slices=r_before_slices_plot,
        r_after_slices=r_after_slices_plot,
        delta_latest=delta_latest,
        u_case=u_case,
        mean_l2_before=mean_l2_before,
        mean_l2_after=mean_l2_after,
        out_png=plot_path,
    )

    return {
        "u_case": list(u_case),
        "grid": {"nx": int(x_star.size), "nt": int(t_star.size)},
        "pde_residual_autograd_before": r_before_ag,
        "pde_residual_fd_after": r_after_fd,
        "pde_residual_fd_after_slices_mean_abs": float(
            np.mean([np.mean(np.abs(s)) for s in r_after_ag_slices])
        ),
        "comsol_mean_rel_l2_before": mean_l2_before,
        "comsol_mean_rel_l2_after": mean_l2_after,
        "comsol_rel_l2_improvement": mean_l2_before - mean_l2_after,
        "solve": solve_info,
        "plot": str(plot_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-training defect correction (FDM)")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="PINO results folder with run_meta.json and checkpoint",
    )
    p.add_argument(
        "--u-case",
        type=float,
        nargs=4,
        metavar=("U1", "U2", "U3", "U4"),
        default=None,
        help="Single velocity case (m/d); default worst probe case",
    )
    p.add_argument(
        "--all-cases",
        action="store_true",
        help="Run three probe velocity cases",
    )
    p.add_argument("--nx", type=int, default=500, help="x* grid points")
    p.add_argument("--nt", type=int, default=200, help="t* time steps (nt+1 nodes)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<timestamp>)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    if args.all_cases:
        u_cases = PROBE_CASES
    elif args.u_case is not None:
        u_cases = [tuple(args.u_case)]  # type: ignore[misc]
    else:
        u_cases = [PROBE_CASES[0]]

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")
        out_dir = (_EXPLORE_DIR / "results" / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dtype, meta = load_model_from_checkpoint(args.checkpoint_dir, device=device)

    x_star = np.linspace(0.0, 1.0, args.nx, dtype=np.float64)
    t_star = np.linspace(0.0, 1.0, args.nt + 1, dtype=np.float64)
    plot_x_star = np.linspace(0.0, 1.0, 500, dtype=np.float64)

    case_results = []
    for u_case in u_cases:
        x_m, comsol_by_time = load_case(COMSOL_DATA_PATH, u_case)
        print(f"Running u={u_case} ...")
        case_metrics = run_one_case(
            model=model,
            device=device,
            dtype=dtype,
            u_case=u_case,
            x_star=x_star,
            t_star=t_star,
            x_m_comsol=x_m,
            comsol_by_time=comsol_by_time,
            out_dir=out_dir,
            plot_x_star=plot_x_star,
        )
        case_results.append(case_metrics)
        print(
            f"  COMSOL rel L2: {case_metrics['comsol_mean_rel_l2_before']:.4f} "
            f"-> {case_metrics['comsol_mean_rel_l2_after']:.4f}"
        )
        print(
            f"  PDE |r| mean: {case_metrics['pde_residual_autograd_before']['mean_abs']:.4e} "
            f"-> {case_metrics['pde_residual_fd_after']['mean_abs']:.4e}"
        )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "run_meta": meta,
        "comsol_data": str(COMSOL_DATA_PATH),
        "wall_clock_s": time.perf_counter() - t_start,
        "cases": case_results,
        "notes": (
            "Correction solves L[delta]=-r via implicit FDM (physical only). "
            "COMSOL used for evaluation plots/metrics only. "
            "Smooth delta may under-correct slope kinks at zone interfaces."
        ),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
