# -*- coding: utf-8 -*-
"""
calibrate_single_instance_pinn.py
=================================

STEP 1 of the "PINN-as-reference" plan: calibrate a *self-validation* proxy
against the COMSOL ground truth we already have for the 4-zone medium.

Idea
----
Train a **problem-specific** PINN (ONE fixed velocity medium, no parametric
generalisation) with dense collocation and many L-BFGS steps.  Every
`--eval-every` steps, measure two things:

  * PROXY  - a self-validation error that needs NO reference solution:
             the PDE residual + IC/BC violation evaluated on a DENSE, held-out
             mesh (denser than, and offset from, the training collocation).
             For a mesh-free PINN, "validate on a denser mesh" means checking
             the physics residual on an independent finer point set.
  * TRUTH  - the true relative-L2 error vs the matching COMSOL case (same metric
             the rest of the project reports: mean over the 5 COMSOL times on the
             standard x* grid).

Logging both, for cases spanning easy(homogeneous) -> hard(slow-fast-slow),
gives us the map  PROXY -> TRUTH.  From that map we pick a *conservative*
threshold tau such that (proxy < tau)  =>  (true error < target), and only then
do we trust PINN-generated references for geometries where COMSOL is missing.

This driver deliberately REUSES the proven numerics from the parametric PINN
module (model, collocation, PDE residual, COMSOL loader, physics loss); it only
adds the single-medium wiring and the periodic proxy/truth probe.  Nothing in
the original training scripts is modified.

Run (via the repo runner, cwd = this folder):
    python calibrate_single_instance_pinn.py --u 0.05,0.01,0.05,0.01 \
        --out-dir results/calib_single --max-epochs 3000 --eval-every 100

Outputs (under --out-dir):
    calib_traj_<tag>.csv        full per-eval trajectory (proxy pieces + truth)
    calibration_results.csv     one appended summary row per case (shared)
    ref_field_<tag>.npz         converged C* on (COMSOL times x dense x*) - the
                                candidate reference to reuse once trusted
    model_<tag>.pt              trained weights
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

# The parametric PINN module lives in this same folder and is import-safe
# (all execution is under its `if __name__ == "__main__"` guard).  Importing it
# also puts the heterogeneous root + utils on sys.path.
import pinn1d_heterogeneous_parametric_pinn as P


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Calibrate a self-validation proxy for a single-instance "
        "heterogeneous PINN against COMSOL."
    )
    ap.add_argument(
        "--u",
        required=True,
        help='Four zone velocities m/d, comma-separated, e.g. "0.05,0.01,0.05,0.01". '
        "Must be one of the 81 COMSOL grid tuples.",
    )
    ap.add_argument("--out-dir", default="results/calib_single",
                    help="Results folder relative to this script.")
    ap.add_argument("--tag", default=None, help="Override the auto case tag.")

    # training budget / recipe (defaults = the proven Exp-J recipe, but denser)
    ap.add_argument("--max-epochs", type=int, default=3000,
                    help="L-BFGS outer steps ceiling (default 3000).")
    ap.add_argument("--patience", type=int, default=300,
                    help="Early-stop patience on train loss (0=off, default 300).")
    ap.add_argument("--lr-lbfgs", type=float, default=0.1)
    ap.add_argument("--lbfgs-max-iter", type=int, default=20)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--seed", type=int, default=1234567)

    # dense TRAINING collocation (affordable: only 1 medium)
    ap.add_argument("--mesh-nx", type=int, default=120,
                    help="Training bulk x* points (default 120; parametric uses 50).")
    ap.add_argument("--mesh-nt", type=int, default=120,
                    help="Training t* points (default 120; parametric uses 50).")

    # DENSE, HELD-OUT proxy mesh (independent of, and finer than, training)
    ap.add_argument("--proxy-nx", type=int, default=401,
                    help="Held-out proxy x* points (default 401).")
    ap.add_argument("--proxy-nt", type=int, default=41,
                    help="Held-out proxy t* slices in (0,1] (default 41).")

    # inlet-corner exclusion for the residual proxy (v2). The x=0,t->0 corner
    # (IC=0 meets inlet C=1) is near-singular; its residual dwarfs the solution
    # error, so we drop a small box (x < corner-x AND t < corner-t) from the
    # interior-residual aggregate. Set both to 0 to disable (recover v1).
    ap.add_argument("--corner-x", type=float, default=0.05,
                    help="Inlet-corner exclusion half-width in x* (default 0.05).")
    ap.add_argument("--corner-t", type=float, default=0.05,
                    help="Inlet-corner exclusion half-width in t* (default 0.05).")

    # cadence + threshold
    ap.add_argument("--eval-every", type=int, default=100,
                    help="Compute proxy+truth every N outer steps (default 100).")
    ap.add_argument("--proxy-threshold", type=float, default=5e-3,
                    help="Declare 'converged' when combined proxy < this (default 5e-3). "
                    "The whole point of step 1 is to CALIBRATE this number, so it is "
                    "only used to record the crossing epoch; the run always continues "
                    "to --max-epochs so the full proxy->truth curve is captured.")
    return ap.parse_args()


def parse_u(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit(f"--u needs exactly 4 values, got {len(parts)}: {s}")
    return (parts[0], parts[1], parts[2], parts[3])


def case_tag(u: tuple[float, float, float, float]) -> str:
    return "u" + "_".join(f"{v:g}".replace(".", "p") for v in u)


# --------------------------------------------------------------------------- #
# Self-validation PROXY  (no reference solution needed)
# --------------------------------------------------------------------------- #
def compute_proxy(
    model,
    u_case: tuple[float, float, float, float],
    x_proxy: np.ndarray,
    t_proxy: np.ndarray,
    *,
    device,
    dtype,
    corner_x: float = 0.05,
    corner_t: float = 0.05,
) -> dict[str, float]:
    """Corner-robust self-validation proxy (v2) - no COMSOL used.

    v1 was dominated by the x=0,t->0 inlet-corner singularity (IC=0 meets inlet
    C=1), where the raw PDE residual blows up even though the solution is
    accurate elsewhere; it never tracked the true error.  v2:
      * excludes a small inlet-corner box (x < corner_x AND t < corner_t) from
        the interior-residual aggregate (`pde_rms_excl`, `pde_max_excl`);
      * also reports robust percentiles of |residual| over ALL points
        (`pde_p50`, `pde_p90`) - these ignore the few extreme corner values
        without needing a mask;
      * keeps IC/BC violations as separate, well-behaved terms.
    `proxy_combined` (v2) = sqrt(pde_rms_excl^2 + ic_rms^2 + inlet_rms^2 + outlet_rms^2).
    `pde_rms` (raw, all points) is still logged for diagnosis / v1 comparison.
    """
    # --- interior PDE residual, slice by slice over the dense time set ---
    raw_sq, raw_n = 0.0, 0                 # raw (all points) - v1 signal
    excl_sq, excl_n, excl_max = 0.0, 0, 0.0  # corner-excluded - v2 signal
    all_abs: list[np.ndarray] = []         # |residual| everywhere (for percentiles)
    for t_star in t_proxy:
        r = P._pde_residual_1d(
            model, x_proxy, float(t_star), u_case, device=device, dtype=dtype
        )
        all_abs.append(np.abs(r))
        raw_sq += float(np.sum(r * r))
        raw_n += r.size
        keep = (x_proxy >= corner_x) if float(t_star) < corner_t \
            else np.ones_like(x_proxy, dtype=bool)
        rk = r[keep]
        if rk.size:
            excl_sq += float(np.sum(rk * rk))
            excl_n += rk.size
            excl_max = max(excl_max, float(np.max(np.abs(rk))))
    pde_rms = float(np.sqrt(raw_sq / max(raw_n, 1)))
    pde_rms_excl = float(np.sqrt(excl_sq / max(excl_n, 1)))
    _allc = np.concatenate(all_abs)
    pde_p50 = float(np.percentile(_allc, 50))
    pde_p90 = float(np.percentile(_allc, 90))

    # --- IC / BC violation (targets: IC C=0 at t*=0, inlet C=1 at x*=0, outlet C=0 at x*=1) ---
    model.eval()
    with torch.no_grad():
        # IC: t*=0 across the dense x*
        n = x_proxy.size
        xt = torch.tensor(x_proxy.reshape(-1, 1), dtype=dtype, device=device)
        t0 = torch.zeros((n, 1), dtype=dtype, device=device)
        br = P.branch_tensor_from_u_case(u_case, n, device=device, dtype=dtype)
        ic = model(xt, t0, br).cpu().numpy().flatten()
        ic_rms = float(np.sqrt(np.mean(ic * ic)))

        # inlet x*=0 and outlet x*=1 across the dense time set
        m = t_proxy.size
        tt = torch.tensor(t_proxy.reshape(-1, 1), dtype=dtype, device=device)
        br_t = P.branch_tensor_from_u_case(u_case, m, device=device, dtype=dtype)
        x0 = torch.zeros((m, 1), dtype=dtype, device=device)
        x1 = torch.ones((m, 1), dtype=dtype, device=device)
        inlet = model(x0, tt, br_t).cpu().numpy().flatten() - 1.0
        outlet = model(x1, tt, br_t).cpu().numpy().flatten() - 0.0
        inlet_rms = float(np.sqrt(np.mean(inlet * inlet)))
        outlet_rms = float(np.sqrt(np.mean(outlet * outlet)))

    combined = float(
        np.sqrt(pde_rms_excl ** 2 + ic_rms ** 2 + inlet_rms ** 2 + outlet_rms ** 2)
    )
    return {
        "proxy_pde_rms": pde_rms,             # raw, all points (v1 - for diagnosis)
        "proxy_pde_rms_excl": pde_rms_excl,   # corner-excluded RMS  (v2)
        "proxy_pde_max_excl": excl_max,       # corner-excluded max
        "proxy_pde_p50": pde_p50,             # robust median |residual|
        "proxy_pde_p90": pde_p90,             # robust p90 |residual|
        "proxy_ic_rms": ic_rms,
        "proxy_inlet_rms": inlet_rms,
        "proxy_outlet_rms": outlet_rms,
        "proxy_combined": combined,           # v2 combined (used for threshold)
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    u_case = parse_u(args.u)
    tag = args.tag or case_tag(u_case)
    contrast = float(sum(abs(u_case[i + 1] - u_case[i]) for i in range(3)))

    # --- configure the reused module's globals to our single-instance recipe ---
    P.torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dtype = P.torch_dtype
    P.seed = args.seed
    P.lr_lbfgs = args.lr_lbfgs
    P.lbfgs_max_iter = args.lbfgs_max_iter
    P.num_epochs_lbfgs = args.max_epochs
    P.early_stop_patience = args.patience
    P.mesh_nx_pde = args.mesh_nx
    P.mesh_nt_pde = args.mesh_nt
    P.mesh_ic_nx = args.mesh_nx
    P.mesh_bc_nt = args.mesh_nt

    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[calib] u={u_case} tag={tag} contrast={contrast:.3f} "
          f"dtype={args.dtype} device={device}")
    print(f"[calib] PE={P.PE:.9g}  train mesh {args.mesh_nx}x{args.mesh_nt} "
          f"+ interface bands | proxy mesh {args.proxy_nx}x{args.proxy_nt}")

    out_dir = (Path(__file__).resolve().parent / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- COMSOL truth for this exact medium (must be one of the 81 grid tuples) ---
    combos = set(P.list_parameter_combos(P.COMSOL_DATA_PATH))
    if u_case not in combos:
        # tolerant match (float text vs literal)
        hit = None
        for c in combos:
            if all(abs(c[k] - u_case[k]) < 1e-9 for k in range(4)):
                hit = c
                break
        if hit is None:
            raise SystemExit(
                f"u={u_case} is not among the {len(combos)} COMSOL grid tuples; "
                "calibration needs a case with a COMSOL reference."
            )
        u_case = hit
    x_m, comsol_by_time = P.load_case(P.COMSOL_DATA_PATH, u_case)
    x_star_val = P.pde_collocation_x_star_1d()  # the standard truth grid

    # --- model + single-medium collocation (reused, proven) ---
    model = P.build_parametric_pinn(P.pinn_architecture, P.activation_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    u_cases = np.array([list(u_case)], dtype=np.float64)  # shape (1,4)
    tensors, _plot = P.build_collocation_tensors(device, dtype, u_cases)
    print(f"[calib] params={n_params:,}  collocation points={tensors['x_pde'].shape[0]:,}")

    # --- dense held-out proxy mesh (offset from training grid; excludes t*=0) ---
    x_proxy = np.linspace(0.0, 1.0, args.proxy_nx, dtype=np.float64)
    t_proxy = np.linspace(1.0 / args.proxy_nt, 1.0, args.proxy_nt, dtype=np.float64)

    # --- L-BFGS loop with periodic proxy+truth probe (mirrors train_model) ---
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=P.lr_lbfgs,
        max_iter=P.lbfgs_max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    stop_rtol = 1e-12 if dtype == torch.float64 else 1e-8

    def closure():
        opt.zero_grad(set_to_none=True)
        loss, metrics = P.compute_physics_loss(model, tensors)
        loss.backward()
        closure.metrics = metrics
        return loss

    traj: list[dict[str, float]] = []
    best_loss = float("inf")
    stale = 0
    crossed_epoch = None
    t0 = time.perf_counter()

    def probe(epoch: int) -> None:
        nonlocal crossed_epoch
        px = compute_proxy(model, u_case, x_proxy, t_proxy, device=device, dtype=dtype,
                           corner_x=args.corner_x, corner_t=args.corner_t)
        true_l2 = P.comsol_case_mean_l2(
            model=model, device=device, dtype=dtype,
            x_m=x_m, comsol_by_time=comsol_by_time, u_case=u_case, x_star=x_star_val,
        )
        row = {
            "epoch": epoch,
            "train_total": float(closure.metrics[0]) if hasattr(closure, "metrics") else float("nan"),
            "train_pde": float(closure.metrics[1]) if hasattr(closure, "metrics") else float("nan"),
            **px,
            "true_rel_l2": float(true_l2),
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        traj.append(row)
        if crossed_epoch is None and px["proxy_combined"] < args.proxy_threshold:
            crossed_epoch = epoch
        print(f"[calib] epoch {epoch:5d} | proxy_comb={px['proxy_combined']:.3e} "
              f"pde_rms={px['proxy_pde_rms']:.3e} | TRUE relL2={true_l2*100:.3f}%")

    probe(0)  # baseline at init
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        opt.step(closure)
        total = float(closure.metrics[0])
        if not np.isfinite(best_loss) or total < best_loss - (0.0 + stop_rtol * max(abs(best_loss), 1.0)):
            best_loss = total
            stale = 0
        else:
            stale += 1
        if epoch % args.eval_every == 0:
            probe(epoch)
        if args.patience > 0 and stale >= args.patience:
            print(f"[calib] early stop at epoch {epoch} (train loss flat {args.patience} steps).")
            if traj[-1]["epoch"] != epoch:
                probe(epoch)
            break
    else:
        if traj[-1]["epoch"] != args.max_epochs:
            probe(args.max_epochs)

    wall = time.perf_counter() - t0

    # --- write per-case trajectory ---
    traj_path = out_dir / f"calib_traj_{tag}.csv"
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader()
        w.writerows(traj)

    # --- save converged reference field on (COMSOL times x dense x*) ---
    model.eval()
    times_star = P.COMSOL_TIMES_DAYS / P.T_MAX
    n = x_proxy.size
    xt = torch.tensor(x_proxy.reshape(-1, 1), dtype=dtype, device=device)
    br = P.branch_tensor_from_u_case(u_case, n, device=device, dtype=dtype)
    fields = np.zeros((times_star.size, n), dtype=np.float64)
    with torch.no_grad():
        for k, ts in enumerate(times_star):
            tt = torch.full((n, 1), float(ts), dtype=dtype, device=device)
            fields[k] = model(xt, tt, br).cpu().numpy().flatten()
    np.savez(
        out_dir / f"ref_field_{tag}.npz",
        u=np.array(u_case), x_star=x_proxy, times_days=P.COMSOL_TIMES_DAYS,
        c_star=fields, contrast=contrast,
    )
    torch.save(model.state_dict(), out_dir / f"model_{tag}.pt")

    # --- append one summary row to the shared calibration table ---
    final = traj[-1]
    best_true = min(r["true_rel_l2"] for r in traj)
    at_cross = next((r for r in traj if r["epoch"] == crossed_epoch), None)
    summary = {
        "tag": tag, "u1": u_case[0], "u2": u_case[1], "u3": u_case[2], "u4": u_case[3],
        "contrast": contrast, "n_params": n_params,
        "mesh_nx": args.mesh_nx, "mesh_nt": args.mesh_nt,
        "proxy_threshold": args.proxy_threshold,
        "crossed_epoch": crossed_epoch if crossed_epoch is not None else -1,
        "proxy_at_cross": at_cross["proxy_combined"] if at_cross else float("nan"),
        "true_at_cross": at_cross["true_rel_l2"] if at_cross else float("nan"),
        "final_epoch": final["epoch"],
        "final_proxy": final["proxy_combined"],
        "final_true": final["true_rel_l2"],
        "best_true": best_true,
        "wall_s": round(wall, 1),
    }
    shared = out_dir / "calibration_results.csv"
    write_header = not shared.is_file()
    with open(shared, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print(f"[calib] DONE {tag}: final proxy={final['proxy_combined']:.3e} "
          f"final TRUE={final['true_rel_l2']*100:.3f}% best TRUE={best_true*100:.3f}% "
          f"cross@{summary['crossed_epoch']} in {wall/60:.1f} min")
    print(f"[calib] wrote {traj_path.name}, ref_field_{tag}.npz, model_{tag}.pt, "
          f"and appended to {shared.name}")


if __name__ == "__main__":
    main()
