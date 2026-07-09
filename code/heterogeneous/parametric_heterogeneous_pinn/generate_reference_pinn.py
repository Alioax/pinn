# -*- coding: utf-8 -*-
"""
generate_reference_pinn.py
==========================

Generate a PINN "reference solution" for an ARBITRARY n-zone medium (2, 3, ...
zones; any interface locations; continuous velocities) for which we have NO
COMSOL. This is the production step of the PINN-as-reference plan: train a
problem-specific PINN, self-certify it with the p90 residual proxy (calibrated
in step 1 against the 4-zone COMSOL), and save the field to reuse until COMSOL
arrives.

Why a separate script from calibrate_single_instance_pinn.py:
  * the parametric module's geometry is hardwired to 4 zones (interfaces
    0.2/0.4/0.6); this generator takes any zone count + interfaces;
  * it needs no COMSOL (there is none for these media) - trust comes from the
    calibrated p90 proxy instead.
It REUSES the validated pieces from the parametric module `P` (the same MLP
builder + activation, the autograd helper, and the physical constants L, T_MAX,
Pe), so the model family and PDE are identical to the calibrated 4-zone runs.

What "calibrated" means here (from results/calib_single_v2):
  * proxy signal = proxy_pde_p90 (90th-pct |PDE residual| on a dense held-out
    mesh); it tracks the true error, whereas RMS is poisoned by the inlet corner.
  * p90 < ~0.03 (margin below the tau=0.044 that guaranteed <2% on all 10
    four-zone calibration cases) => trust the solution to ~2%.
Two design choices baked in from the calibration findings:
  * the reported field is taken at the BEST-proxy epoch, not the final one
    (true error can drift up after its best);
  * BC/IC violations are tracked separately (they stay tiny).

Optionally, pass --comsol-check on a 4-zone grid medium (interfaces 0.2,0.4,0.6)
to confirm this generator reproduces the calibrated accuracy against COMSOL.

Run (cwd = this folder, via the repo runner):
    python generate_reference_pinn.py --zones 0.01,0.05,0.01 --out-dir results/gen_ref
    python generate_reference_pinn.py --zones 0.02,0.04 --interfaces 0.5 --out-dir results/gen_ref

Outputs (under --out-dir):
    ref_field_<tag>.npz     C* at the 5 standard times on a dense x* grid (+meta)
    gen_traj_<tag>.csv      per-eval proxy trajectory
    generated_references.csv  one appended summary row per medium (shared)
    model_<tag>.pt          best-proxy weights
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

import pinn1d_heterogeneous_parametric_pinn as P  # import-safe; reuse validated pieces


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a self-certified PINN "
                                 "reference for an arbitrary n-zone medium.")
    ap.add_argument("--zones", required=True,
                    help='Zone velocities m/d, comma-separated (n values -> n zones), '
                    'e.g. "0.01,0.05,0.01". Range should stay in [0.01, 0.05].')
    ap.add_argument("--interfaces", default=None,
                    help='n-1 interface x* in (0,1), comma-separated. Default = equal '
                    'zones (1/n, 2/n, ...). e.g. "0.5" for two equal zones.')
    ap.add_argument("--out-dir", default="results/gen_ref")
    ap.add_argument("--tag", default=None)

    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=400,
                    help="Early-stop patience on train loss (0=off, default 400).")
    ap.add_argument("--lr-lbfgs", type=float, default=0.1)
    ap.add_argument("--lbfgs-max-iter", type=int, default=20)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--seed", type=int, default=1234567)

    ap.add_argument("--mesh-nx", type=int, default=120)
    ap.add_argument("--mesh-nt", type=int, default=120)
    ap.add_argument("--proxy-nx", type=int, default=401)
    ap.add_argument("--proxy-nt", type=int, default=41)
    ap.add_argument("--corner-x", type=float, default=0.05)
    ap.add_argument("--corner-t", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--proxy-threshold", type=float, default=0.03,
                    help="Certify (trust to ~2%%) when best p90 proxy < this "
                    "(default 0.03; from the step-1 calibration).")

    ap.add_argument("--comsol-check", action="store_true",
                    help="If the medium is a 4-zone COMSOL grid case, also report "
                    "true rel-L2 vs COMSOL (generator self-validation).")
    return ap.parse_args()


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x != ""]


def default_interfaces(n: int) -> list[float]:
    return [i / n for i in range(1, n)]


def make_tag(u_vec: list[float], interfaces: list[float]) -> str:
    uz = "z" + "_".join(f"{v:g}".replace(".", "p") for v in u_vec)
    it = "i" + "_".join(f"{v:g}".replace(".", "p") for v in interfaces) if interfaces else "i"
    return f"{uz}__{it}"


# --------------------------------------------------------------------------- #
# General n-zone physics (mirrors the validated 4-zone recipe, any geometry)
# --------------------------------------------------------------------------- #
def cfl_local(x: torch.Tensor, u_vec: list[float], interfaces: list[float]) -> torch.Tensor:
    """Piecewise-constant local CFL(x*) = u_zone * T_MAX / L, any zone count."""
    scale = P.T_MAX / P.L
    out = torch.full_like(x, u_vec[-1] * scale)
    for k in range(len(interfaces) - 1, -1, -1):  # low interfaces overwrite last
        out = torch.where(x < interfaces[k],
                          torch.full_like(x, u_vec[k] * scale), out)
    return out


def branch_const(u_vec: list[float], m: int, device, dtype) -> torch.Tensor:
    """Constant branch input = per-zone CFL (matches the validated scaling)."""
    cfl = P.cfl_from_u_np(np.array(u_vec, dtype=np.float64)).reshape(1, -1)
    return torch.tensor(np.repeat(cfl, m, axis=0), dtype=dtype, device=device)


def build_colloc(u_vec, interfaces, device, dtype, mesh_nx, mesh_nt,
                 band_hw=0.025, band_n=12):
    x_bulk = np.linspace(0.0, 1.0, mesh_nx, dtype=np.float64)
    bands = [np.linspace(max(0.0, b - band_hw), min(1.0, b + band_hw), band_n,
                         dtype=np.float64) for b in interfaces]
    x1d = np.unique(np.concatenate([x_bulk, *bands])) if bands else np.unique(x_bulk)
    t1d = np.linspace(0.0, 1.0, mesh_nt, dtype=np.float64)
    gx, gt = np.meshgrid(x1d, t1d, indexing="ij")
    xp = gx.reshape(-1, 1)
    tp = gt.reshape(-1, 1)

    x_ic = np.linspace(0.0, 1.0, mesh_nx, dtype=np.float64).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    t_bc = np.linspace(0.0, 1.0, mesh_nt, dtype=np.float64).reshape(-1, 1)
    x_in = np.zeros_like(t_bc)
    x_out = np.ones_like(t_bc)

    def T(a, rg=False):
        return torch.tensor(a, dtype=dtype, device=device, requires_grad=rg)

    return {
        "x_pde": T(xp, True), "t_pde": T(tp, True), "branch_pde": branch_const(u_vec, xp.shape[0], device, dtype),
        "x_ic": T(x_ic), "t_ic": T(t_ic), "branch_ic": branch_const(u_vec, x_ic.shape[0], device, dtype),
        "x_in": T(x_in), "t_in": T(t_bc), "branch_in": branch_const(u_vec, t_bc.shape[0], device, dtype),
        "x_out": T(x_out), "t_out": T(t_bc.copy()), "branch_out": branch_const(u_vec, t_bc.shape[0], device, dtype),
        "n_colloc": xp.shape[0],
    }


def physics_loss(model, tensors, u_vec, interfaces):
    c = model(tensors["x_pde"], tensors["t_pde"], tensors["branch_pde"])
    dCdt = P.gradients(c, tensors["t_pde"])
    dCdx = P.gradients(c, tensors["x_pde"])
    d2 = P.gradients(dCdx, tensors["x_pde"])
    r = dCdt + cfl_local(tensors["x_pde"], u_vec, interfaces) * dCdx - P.PE * d2
    pde = torch.mean(r ** 2)
    ic = torch.mean(model(tensors["x_ic"], tensors["t_ic"], tensors["branch_ic"]) ** 2)
    inl = torch.mean((model(tensors["x_in"], tensors["t_in"], tensors["branch_in"]) - 1.0) ** 2)
    out = torch.mean(model(tensors["x_out"], tensors["t_out"], tensors["branch_out"]) ** 2)
    total = pde + ic + inl + out
    return total, (total.item(), pde.item(), ic.item(), inl.item(), out.item())


def residual_slice(model, x_np, t_star, u_vec, interfaces, device, dtype):
    n = x_np.size
    x = torch.tensor(x_np.reshape(-1, 1), dtype=dtype, device=device, requires_grad=True)
    t = torch.full((n, 1), float(t_star), dtype=dtype, device=device, requires_grad=True)
    br = branch_const(u_vec, n, device, dtype)
    c = model(x, t, br)
    dCdt = P.gradients(c, t)
    dCdx = P.gradients(c, x)
    d2 = P.gradients(dCdx, x)
    r = dCdt + cfl_local(x, u_vec, interfaces) * dCdx - P.PE * d2
    return r.detach().cpu().numpy().flatten()


def compute_proxy(model, u_vec, interfaces, x_proxy, t_proxy, *, device, dtype,
                  corner_x=0.05, corner_t=0.05) -> dict[str, float]:
    """Corner-robust p90 proxy (the calibrated self-check) + BC/IC terms."""
    excl_sq, excl_n = 0.0, 0
    all_abs: list[np.ndarray] = []
    for t_star in t_proxy:
        r = residual_slice(model, x_proxy, float(t_star), u_vec, interfaces, device, dtype)
        all_abs.append(np.abs(r))
        keep = (x_proxy >= corner_x) if float(t_star) < corner_t \
            else np.ones_like(x_proxy, dtype=bool)
        rk = r[keep]
        if rk.size:
            excl_sq += float(np.sum(rk * rk))
            excl_n += rk.size
    allc = np.concatenate(all_abs)
    pde_p50 = float(np.percentile(allc, 50))
    pde_p90 = float(np.percentile(allc, 90))
    pde_rms_excl = float(np.sqrt(excl_sq / max(excl_n, 1)))

    model.eval()
    with torch.no_grad():
        n = x_proxy.size
        xt = torch.tensor(x_proxy.reshape(-1, 1), dtype=dtype, device=device)
        t0 = torch.zeros((n, 1), dtype=dtype, device=device)
        ic = model(xt, t0, branch_const(u_vec, n, device, dtype)).cpu().numpy().flatten()
        m = t_proxy.size
        tt = torch.tensor(t_proxy.reshape(-1, 1), dtype=dtype, device=device)
        x0 = torch.zeros((m, 1), dtype=dtype, device=device)
        x1 = torch.ones((m, 1), dtype=dtype, device=device)
        br_t = branch_const(u_vec, m, device, dtype)
        inlet = model(x0, tt, br_t).cpu().numpy().flatten() - 1.0
        outlet = model(x1, tt, br_t).cpu().numpy().flatten()
    ic_rms = float(np.sqrt(np.mean(ic ** 2)))
    inlet_rms = float(np.sqrt(np.mean(inlet ** 2)))
    outlet_rms = float(np.sqrt(np.mean(outlet ** 2)))
    combined = float(np.sqrt(pde_p90 ** 2 + ic_rms ** 2 + inlet_rms ** 2 + outlet_rms ** 2))
    return {"proxy_pde_p90": pde_p90, "proxy_pde_p50": pde_p50,
            "proxy_pde_rms_excl": pde_rms_excl, "proxy_ic_rms": ic_rms,
            "proxy_inlet_rms": inlet_rms, "proxy_outlet_rms": outlet_rms,
            "proxy_combined": combined}


def comsol_true_l2(model, u_vec, interfaces, x_proxy, device, dtype):
    """Optional: true rel-L2 vs COMSOL, only for a 4-zone grid medium."""
    if len(u_vec) != 4 or not np.allclose(interfaces, [0.2, 0.4, 0.6], atol=1e-6):
        return None
    u_case = tuple(float(v) for v in u_vec)
    combos = set(P.list_parameter_combos(P.COMSOL_DATA_PATH))
    match = next((c for c in combos if all(abs(c[k] - u_case[k]) < 1e-9 for k in range(4))), None)
    if match is None:
        return None
    x_m, by_t = P.load_case(P.COMSOL_DATA_PATH, match)
    n = x_proxy.size
    xt = torch.tensor(x_proxy.reshape(-1, 1), dtype=dtype, device=device)
    br = branch_const(u_vec, n, device, dtype)
    l2, k = 0.0, 0
    model.eval()
    for t_days in P.COMSOL_TIMES_DAYS:
        c_ref_full = None
        for tk, vals in by_t.items():
            if np.isclose(float(tk), float(t_days), atol=1e-10):
                c_ref_full = vals
                break
        if c_ref_full is None:
            continue
        tt = torch.full((n, 1), float(t_days) / P.T_MAX, dtype=dtype, device=device)
        with torch.no_grad():
            c_pred = model(xt, tt, br).cpu().numpy().flatten()
        c_ref = P.comsol_c_star_on_x_star(x_proxy, x_m, c_ref_full)
        num = np.linalg.norm(c_pred - c_ref)
        den = np.linalg.norm(c_ref)
        l2 += float(num / den) if den > 0 else float(num)
        k += 1
    return l2 / k if k else None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    u_vec = parse_floats(args.zones)
    n = len(u_vec)
    if n < 2:
        raise SystemExit("--zones needs at least 2 velocities.")
    interfaces = parse_floats(args.interfaces) if args.interfaces else default_interfaces(n)
    if len(interfaces) != n - 1:
        raise SystemExit(f"{n} zones need {n-1} interfaces; got {len(interfaces)}.")
    if not all(0.0 < a < 1.0 for a in interfaces) or list(interfaces) != sorted(interfaces):
        raise SystemExit(f"interfaces must be sorted and in (0,1): {interfaces}")
    tag = args.tag or make_tag(u_vec, interfaces)
    contrast = float(sum(abs(u_vec[i + 1] - u_vec[i]) for i in range(n - 1)))

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[gen] {n}-zone u={u_vec} interfaces={interfaces} contrast={contrast:.3f} "
          f"dtype={args.dtype} device={device} tag={tag}")

    out_dir = (Path(__file__).resolve().parent / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # validated MLP family: input = [x*, t*, cfl_1..cfl_n]
    arch = [2 + n, 36, 36, 36, 36, 1]
    model = P.build_parametric_pinn(arch, P.activation_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    tensors = build_colloc(u_vec, interfaces, device, dtype, args.mesh_nx, args.mesh_nt)
    print(f"[gen] arch={arch} params={n_params:,} collocation={tensors['n_colloc']:,}")

    x_proxy = np.linspace(0.0, 1.0, args.proxy_nx, dtype=np.float64)
    t_proxy = np.linspace(1.0 / args.proxy_nt, 1.0, args.proxy_nt, dtype=np.float64)

    opt = torch.optim.LBFGS(model.parameters(), lr=args.lr_lbfgs,
                            max_iter=args.lbfgs_max_iter, history_size=50,
                            line_search_fn="strong_wolfe")
    stop_rtol = 1e-12 if dtype == torch.float64 else 1e-8

    def closure():
        opt.zero_grad(set_to_none=True)
        loss, metrics = physics_loss(model, tensors, u_vec, interfaces)
        loss.backward()
        closure.metrics = metrics
        return loss

    traj: list[dict[str, float]] = []
    best = {"proxy": float("inf"), "state": None, "epoch": -1}
    best_loss, stale = float("inf"), 0
    t0 = time.perf_counter()

    def probe(epoch: int) -> None:
        px = compute_proxy(model, u_vec, interfaces, x_proxy, t_proxy,
                           device=device, dtype=dtype,
                           corner_x=args.corner_x, corner_t=args.corner_t)
        row = {"epoch": epoch,
               "train_total": float(closure.metrics[0]) if hasattr(closure, "metrics") else float("nan"),
               "train_pde": float(closure.metrics[1]) if hasattr(closure, "metrics") else float("nan"),
               **px, "wall_s": round(time.perf_counter() - t0, 2)}
        traj.append(row)
        if px["proxy_combined"] < best["proxy"]:      # SAVE-BEST (not final)
            best["proxy"] = px["proxy_combined"]
            best["state"] = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best["epoch"] = epoch
        print(f"[gen] epoch {epoch:5d} | p90={px['proxy_pde_p90']:.3e} "
              f"combined={px['proxy_combined']:.3e} (best {best['proxy']:.3e}@{best['epoch']})")

    probe(0)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        opt.step(closure)
        total = float(closure.metrics[0])
        if not np.isfinite(best_loss) or total < best_loss - stop_rtol * max(abs(best_loss), 1.0):
            best_loss, stale = total, 0
        else:
            stale += 1
        if epoch % args.eval_every == 0:
            probe(epoch)
        if args.patience > 0 and stale >= args.patience:
            print(f"[gen] early stop at {epoch} (train loss flat {args.patience}).")
            if traj[-1]["epoch"] != epoch:
                probe(epoch)
            break
    else:
        if traj[-1]["epoch"] != args.max_epochs:
            probe(args.max_epochs)
    wall = time.perf_counter() - t0

    # restore BEST-proxy weights before exporting the reference
    if best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    certified = best["proxy"] < args.proxy_threshold

    # optional COMSOL cross-check (4-zone grid media only)
    true_l2 = comsol_true_l2(model, u_vec, interfaces, x_proxy, device, dtype) if args.comsol_check else None

    # export reference field at the 5 standard times on the dense x* grid
    times_star = P.COMSOL_TIMES_DAYS / P.T_MAX
    n_x = x_proxy.size
    xt = torch.tensor(x_proxy.reshape(-1, 1), dtype=dtype, device=device)
    br = branch_const(u_vec, n_x, device, dtype)
    fields = np.zeros((times_star.size, n_x), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i, ts in enumerate(times_star):
            tt = torch.full((n_x, 1), float(ts), dtype=dtype, device=device)
            fields[i] = model(xt, tt, br).cpu().numpy().flatten()
    np.savez(out_dir / f"ref_field_{tag}.npz",
             u=np.array(u_vec), interfaces=np.array(interfaces),
             x_star=x_proxy, times_days=P.COMSOL_TIMES_DAYS, c_star=fields,
             contrast=contrast, best_proxy=best["proxy"], best_epoch=best["epoch"],
             certified=certified)
    torch.save(model.state_dict(), out_dir / f"model_{tag}.pt")

    with open(out_dir / f"gen_traj_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader()
        w.writerows(traj)

    summary = {"tag": tag, "n_zones": n, "zones": ";".join(f"{v:g}" for v in u_vec),
               "interfaces": ";".join(f"{v:g}" for v in interfaces), "contrast": contrast,
               "best_proxy_p90combined": best["proxy"], "best_epoch": best["epoch"],
               "final_proxy": traj[-1]["proxy_combined"], "certified_2pct": certified,
               "comsol_true_rel_l2": true_l2 if true_l2 is not None else "",
               "n_params": n_params, "wall_s": round(wall, 1)}
    shared = out_dir / "generated_references.csv"
    write_header = not shared.is_file()
    with open(shared, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    verdict = "CERTIFIED (~2%)" if certified else "NOT certified (p90 above threshold)"
    extra = f" | COMSOL true rel-L2={true_l2*100:.3f}%" if true_l2 is not None else ""
    print(f"[gen] DONE {tag}: best p90-combined={best['proxy']:.3e}@{best['epoch']} -> {verdict}{extra}")
    print(f"[gen] wrote ref_field_{tag}.npz, model_{tag}.pt, gen_traj_{tag}.csv, "
          f"appended {shared.name} in {wall/60:.1f} min")


if __name__ == "__main__":
    main()
