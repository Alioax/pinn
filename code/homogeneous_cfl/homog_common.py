# -*- coding: utf-8 -*-
"""
Shared helpers for the homogeneous CFL models of Report 4 (parametric PINN and parametric PINO).

Used by parametric_cfl_pinn/, pino_cfl/ and eval_saved_models.py. It provides:
  * the two network classes, unchanged from Report 4;
  * the command-line flags of the sample x optimizer grid (defaults = Report 4 exactly);
  * the training loop: L-BFGS or SOAP, with the early stopping of the heterogeneous runs;
  * the held-out test: relative L2 error (paper Eq. 19) against Ogata-Banks (paper Eq. 18)
    at 101 positions and the five output times, averaged over the five times;
  * run_meta.json / test_errors.csv / loss_history.csv for every run.

Running a script with no flags reproduces Report 4 (50 CFL samples, L-BFGS lr 1.0,
max_iter 1, 1000 steps, no early stopping, outputs in its own results/ folder).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.special import erfc, erfcx
from tqdm import trange

HOMOG_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_DIR = os.path.join(HOMOG_DIR, "grid_results")
SOAP_PATH = os.path.join(HOMOG_DIR, "..", "heterogeneous", "utils", "soap.py")

# -----------------------------------------------------------------------------
# Physical setup (identical to Report 4)
# -----------------------------------------------------------------------------
L = 100.0            # m
T_MAX = 1200.0       # d
D_M2_S = 1e-6        # m^2/s
SECONDS_PER_DAY = 86400.0
D_M2_PER_DAY = D_M2_S * SECONDS_PER_DAY
PE = D_M2_S * T_MAX * SECONDS_PER_DAY / (L**2)       # = beta in the paper, 0.010368
U_MIN, U_MAX = 0.01, 0.05                            # m/d
CFL_MIN, CFL_MAX = U_MIN * T_MAX / L, U_MAX * T_MAX / L   # 0.12, 0.60

# -----------------------------------------------------------------------------
# Test protocol (same for every run, including the saved April models)
# -----------------------------------------------------------------------------
N_TEST = 40
TRAIN_GRIDS = (50, 100)   # the sample counts of the grid; test values must avoid both


def _held_out_cfl(n_test=N_TEST, grids=TRAIN_GRIDS, n_candidates=4001):
    """In each of n_test equal sub-intervals of [CFL_MIN, CFL_MAX], the CFL value farthest from every
    training sample of the uniform 50- and 100-sample grids. Every test value then lies at least
    0.0023 from any training value (about half the 100-sample spacing, 0.0048). Deterministic."""
    train = np.concatenate([np.linspace(CFL_MIN, CFL_MAX, n) for n in grids])
    cand = np.linspace(CFL_MIN, CFL_MAX, n_candidates)
    dist = np.min(np.abs(cand[:, None] - train[None, :]), axis=1)
    edges = np.linspace(CFL_MIN, CFL_MAX, n_test + 1)
    out = []
    for k in range(n_test):
        inside = (cand >= edges[k]) & (cand < edges[k + 1])
        out.append(cand[np.argmax(np.where(inside, dist, -1.0))])
    return np.array(out)


TEST_CFL = _held_out_cfl()
EVAL_TIMES_D = np.array([100.0, 300.0, 600.0, 900.0, 1200.0])   # COMSOL output times
EVAL_X_STAR = np.linspace(0.0, 1.0, 101)                          # 101 equally spaced positions


# -----------------------------------------------------------------------------
# Networks (unchanged from Report 4)
# -----------------------------------------------------------------------------
class ParametricPINN(nn.Module):
    def __init__(self, architecture, activation_cls):
        super().__init__()
        layers = []
        for i in range(len(architecture) - 1):
            in_f, out_f = architecture[i], architecture[i + 1]
            layers.append(nn.Linear(in_f, out_f))
            if i < len(architecture) - 2:
                layers.append(activation_cls())
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        gain = nn.init.calculate_gain("tanh")
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_star, t_star, cfl):
        return self.net(torch.cat([x_star, t_star, cfl], dim=1))


class DeepONetParametric(nn.Module):
    def __init__(self, branch_arch, trunk_arch, activation):
        super().__init__()
        branch_layers = []
        for i in range(len(branch_arch) - 1):
            in_f, out_f = branch_arch[i], branch_arch[i + 1]
            branch_layers.append(nn.Linear(in_f, out_f))
            if i < len(branch_arch) - 2:
                branch_layers.append(activation())
        self.branch = nn.Sequential(*branch_layers)

        trunk_layers = []
        for i in range(len(trunk_arch) - 1):
            in_f, out_f = trunk_arch[i], trunk_arch[i + 1]
            trunk_layers.append(nn.Linear(in_f, out_f))
            if i < len(trunk_arch) - 2:
                trunk_layers.append(activation())
        self.trunk = nn.Sequential(*trunk_layers)

        gain = nn.init.calculate_gain("tanh")
        for mod in (self.branch, self.trunk):
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x_star, t_star, branch_input):
        pts = torch.cat([x_star, t_star], dim=1)
        b_vec = self.branch(branch_input)
        t_vec = self.trunk(pts)
        return torch.sigmoid((b_vec * t_vec).sum(dim=-1, keepdim=True))


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def _floats(s: str):
    return tuple(float(v) for v in s.split(","))


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-name", default=None,
                   help="write outputs to homogeneous_cfl/grid_results/<run-name>/ "
                        "(default: the script's own results/ folder, as in Report 4)")
    p.add_argument("--n-cfl", type=int, default=50,
                   help="number of uniformly spaced CFL training samples in [0.12, 0.60] "
                        "(sets the CFL axis of the PDE, IC and BC meshes)")
    p.add_argument("--optimizer", choices=["lbfgs", "soap"], default="lbfgs")
    p.add_argument("--epochs", type=int, default=1000, help="L-BFGS outer steps (ceiling)")
    p.add_argument("--lr-lbfgs", type=float, default=1.0)
    p.add_argument("--lbfgs-max-iter", type=int, default=1)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--soap-epochs", type=int, default=25000, help="SOAP steps (ceiling)")
    p.add_argument("--soap-lr", type=float, default=3e-3)
    p.add_argument("--soap-betas", type=_floats, default=(0.95, 0.95))
    p.add_argument("--soap-weight-decay", type=float, default=0.01)
    p.add_argument("--soap-precond-freq", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop after N steps without total-loss improvement (0 = off)")
    p.add_argument("--seed", type=int, default=1234567)
    p.add_argument("--plot-dpi", type=int, default=800)
    p.add_argument("--t-levels", choices=["uniform", "sqrt"], default="uniform",
                   help="time levels of the fixed PDE and boundary meshes: uniform (Report 4) or "
                        "sqrt, t*_k = (k/(n-1))^2 (dense early, sparse late; see time_levels)")
    p.add_argument("--resample-every", type=int, default=0,
                   help="SOAP only: draw new uniform random (x*, t*) points for the PDE, IC and boundary "
                        "terms every N steps, same counts per CFL sample as the mesh (0 = fixed mesh)")
    p.add_argument("--dense-final-steps", type=int, default=0,
                   help="SOAP only: train the last N of --soap-epochs steps on a fixed uniform mesh with "
                        "--dense-n points per axis; early stopping, if on, applies only in this phase")
    p.add_argument("--dense-n", type=int, default=150, help="points per axis of the dense final mesh")
    p.add_argument("--eval-every", type=int, default=0,
                   help="also compute the held-out test error every N steps and at the last step "
                        "(written to test_error_history.csv; excluded from the training time; 0 = off)")
    return p.parse_args()


def output_dir(args, default_results_dir: str) -> str:
    out = default_results_dir if args.run_name is None else os.path.join(GRID_DIR, args.run_name)
    os.makedirs(out, exist_ok=True)
    return out


# -----------------------------------------------------------------------------
# Collocation: time levels, random resampling, dense final mesh
# -----------------------------------------------------------------------------
def time_levels(n: int, kind: str = "uniform") -> np.ndarray:
    """n time levels t* in [0, 1], both ends included.

    uniform: np.linspace(0, 1, n), as in Report 4 (24.5 d spacing for n = 50).
    sqrt:    t*_k = (k / (n - 1))^2, i.e. uniform in sqrt(t*). The front spreads as 2 sqrt(D t), so equal
             steps in sqrt(t) give each time interval the same growth of the front width: the levels are
             dense while the front is sharp and sparse once it is wide. For n = 50: 15 levels in the first
             100 d (spacing 0.5 d growing to 13.5 d), then 13.5 d growing to 48.5 d at the end."""
    if kind == "uniform":
        return np.linspace(0.0, 1.0, n)
    if kind == "sqrt":
        return (np.arange(n, dtype=np.float64) / (n - 1)) ** 2
    raise ValueError(kind)


class CollocationSchedule:
    """Called by train() before every step; swaps the training points through set_points(points), where
    points = (x_pde, t_pde, cfl_pde, x_ic, t_ic, cfl_ic, x_in, t_in, cfl_in, x_out, t_out, cfl_out),
    all (n, 1) tensors (x_pde, t_pde with requires_grad). Without --resample-every / --dense-final-steps
    it does nothing and the script's fixed mesh is used throughout."""

    def __init__(self, args, cfl_1d, n_x, n_t, device, dtype, set_points):
        self.resample = getattr(args, "resample_every", 0)
        self.dense = getattr(args, "dense_final_steps", 0)
        if (self.resample or self.dense) and args.optimizer != "soap":
            raise SystemExit("--resample-every / --dense-final-steps need --optimizer soap "
                             "(L-BFGS assumes a fixed objective)")
        self.n_steps = args.soap_epochs if args.optimizer == "soap" else args.epochs
        if self.dense >= self.n_steps:
            raise SystemExit("--dense-final-steps must be smaller than --soap-epochs")
        self.dense_start = self.n_steps - self.dense if self.dense else self.n_steps
        self.dense_n = getattr(args, "dense_n", 150)
        self.cfl = torch.tensor(np.asarray(cfl_1d), dtype=dtype, device=device)
        self.n_x, self.n_t, self.device, self.dtype = n_x, n_t, device, dtype
        self.set_points = set_points
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(args.seed) + 1)       # separate stream: network init is unchanged
        self.n_draws = 0

    def _col(self, a, grad=False):
        return a.reshape(-1, 1).to(self.dtype).requires_grad_(grad)

    def _random(self):
        nc, n_pde = self.cfl.numel(), self.n_x * self.n_t
        r = lambda n: torch.rand(n, generator=self.gen, dtype=self.dtype, device=self.device)
        x, t = r(nc * n_pde), r(nc * n_pde)
        c_pde = self.cfl.repeat_interleave(n_pde)
        x_ic, c_ic = r(nc * self.n_x), self.cfl.repeat_interleave(self.n_x)
        t_bc, c_bc = r(nc * self.n_t), self.cfl.repeat_interleave(self.n_t)
        z = torch.zeros_like
        return (self._col(x, True), self._col(t, True), self._col(c_pde),
                self._col(x_ic), self._col(z(x_ic)), self._col(c_ic),
                self._col(z(t_bc)), self._col(t_bc), self._col(c_bc),
                self._col(z(t_bc) + 1.0), self._col(t_bc.clone()), self._col(c_bc.clone()))

    def _mesh(self, n):
        """Uniform n x n mesh per CFL sample, ends included, same layout as the scripts' mesh."""
        g = torch.linspace(0.0, 1.0, n, dtype=self.dtype, device=self.device)
        gx, gt, gc = torch.meshgrid(g, g, self.cfl, indexing="ij")
        gxi, gci = torch.meshgrid(g, self.cfl, indexing="ij")
        gtb, gcb = torch.meshgrid(g, self.cfl, indexing="ij")
        z = torch.zeros_like
        return (self._col(gx, True), self._col(gt, True), self._col(gc),
                self._col(gxi), self._col(z(gxi)), self._col(gci),
                self._col(z(gtb)), self._col(gtb), self._col(gcb),
                self._col(z(gtb) + 1.0), self._col(gtb.clone()), self._col(gcb.clone()))

    def __call__(self, step):
        if self.dense and step == self.dense_start:
            self.set_points(self._mesh(self.dense_n))
            print(f"\nStep {step + 1}: dense final mesh {self.dense_n} x {self.dense_n} x {self.cfl.numel()}")
        elif self.resample and step < self.dense_start and step % self.resample == 0:
            self.set_points(self._random())
            self.n_draws += 1

    def describe(self):
        d = {"resample_every": self.resample, "random_draws": self.n_draws}
        if self.dense:
            d.update(dense_final_steps=self.dense, dense_mesh=f"{self.dense_n} x {self.dense_n} x {self.cfl.numel()}",
                     dense_from_step=self.dense_start + 1)
        return d


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def _load_soap():
    spec = importlib.util.spec_from_file_location("soap_vyas", os.path.abspath(SOAP_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SOAP


def _loss_improved(current: float, best: float, rtol: float) -> bool:
    # same rule as the heterogeneous scripts
    if not np.isfinite(best):
        return True
    return current < best - rtol * max(abs(best), 1.0)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(model, compute_loss, args, device, dtype, predict=None, on_step=None):
    """compute_loss() -> (total_loss tensor, metrics tuple (total, pde, ic, inlet, outlet)).
    predict(x, t, cfl) -> C*, used only for the optional test-error log (--eval-every).

    Returns (history, info). The trained model is the final state, as in the heterogeneous runs.
    With --eval-every, info["eval_history"] holds the test error every N steps; the time spent on
    these evaluations is measured and removed from the training time.
    on_step(step) is called before every step (CollocationSchedule). With --dense-final-steps, early
    stopping and the best-loss record start when the dense mesh does."""
    patience = args.early_stop_patience
    stop_rtol = (1e-12 if dtype == torch.float64 else 1e-8) if patience > 0 else 0.0
    history, best, stale, early, grad_evals = [], float("inf"), 0, False, 0
    fmt = "loss : %.3e  mse_pde %.3e  mse_ic %.3e  mse_in %.3e  mse_out %.3e"

    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr_lbfgs, max_iter=args.lbfgs_max_iter,
                                      history_size=args.lbfgs_history_size, line_search_fn="strong_wolfe")
        n_steps = args.epochs
    else:
        optimizer = _load_soap()(model.parameters(), lr=args.soap_lr, betas=tuple(args.soap_betas),
                                 weight_decay=args.soap_weight_decay,
                                 precondition_frequency=args.soap_precond_freq)
        n_steps = args.soap_epochs
    print(optimizer)
    dense = getattr(args, "dense_final_steps", 0)
    es_from = n_steps - dense if dense else 0
    if patience > 0:
        print(f"Early stopping: patience={patience} steps (no total-loss improvement, rtol={stop_rtol:g})")

    t_bar = trange(n_steps, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    def closure():
        nonlocal grad_evals
        grad_evals += 1
        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics = compute_loss()
        total_loss.backward()
        closure.latest = metrics
        t_bar.set_description(fmt % metrics)
        t_bar.refresh()
        return total_loss

    eval_every = getattr(args, "eval_every", 0) if predict is not None else 0
    eval_history, eval_s = [], 0.0

    def log_test_error(step_done):
        nonlocal eval_s
        _sync(device); te = time.perf_counter()
        model.eval()
        rows, summ = evaluate(predict, device, dtype)
        model.train()
        h = history[-1]
        rec = {"step": step_done, "total_loss": h[0], "pde_loss": h[1],
               "rel_l2_median": summ["rel_l2_median"], "rel_l2_max": summ["rel_l2_max"],
               "rel_l2_mean": summ["rel_l2_mean"]}
        for t_d in EVAL_TIMES_D:
            rec[f"median_t{int(t_d)}d"] = float(np.median([r[f"rel_l2_t{int(t_d)}d"] for r in rows]))
        eval_history.append(rec)
        _sync(device); eval_s += time.perf_counter() - te

    _sync(device)
    t0 = time.perf_counter()
    for step in t_bar:
        if on_step is not None:
            on_step(step)
        if es_from and step == es_from:
            best, stale = float("inf"), 0
        model.train()
        if args.optimizer == "lbfgs":
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        metrics = closure.latest
        history.append(list(metrics))
        if eval_every > 0 and (step + 1) % eval_every == 0:
            log_test_error(step + 1)
        if _loss_improved(metrics[0], best, stop_rtol):
            best, stale = metrics[0], 0
        else:
            stale += 1
            if patience > 0 and step >= es_from and stale >= patience:
                early = True
                print(f"\nEarly stopping at step {step + 1}/{n_steps}: total loss unchanged "
                      f"for {patience} steps (best={best:.6e}).")
                break
    t_bar.close()
    if eval_every > 0 and (not eval_history or eval_history[-1]["step"] != len(history)):
        log_test_error(len(history))
    _sync(device)
    train_s = time.perf_counter() - t0 - eval_s

    info = {
        "optimizer": args.optimizer,
        "steps_requested": n_steps,
        "steps_actual": len(history),
        "early_stop_patience": patience,
        "early_stopped": early,
        "gradient_evaluations": grad_evals,
        "train_wall_clock_s": round(train_s, 3),
        "train_s_per_step": round(train_s / max(len(history), 1), 6),
        "final_loss": dict(zip(["total", "pde", "ic", "inlet", "outlet"], history[-1])) if history else None,
        "best_total_loss": best,
    }
    if eval_every > 0:
        best_e = min(eval_history, key=lambda r: r["rel_l2_median"])
        info.update(eval_every=eval_every, eval_wall_clock_s=round(eval_s, 3),
                    best_logged_step=best_e["step"], best_logged_rel_l2_median=best_e["rel_l2_median"],
                    best_logged_rel_l2_max=best_e["rel_l2_max"], eval_history=eval_history)
    if args.optimizer == "lbfgs":
        info.update(lbfgs_lr=args.lr_lbfgs, lbfgs_max_iter=args.lbfgs_max_iter,
                    lbfgs_history_size=args.lbfgs_history_size, lbfgs_line_search="strong_wolfe")
    else:
        info.update(soap_lr=args.soap_lr, soap_betas=list(args.soap_betas),
                    soap_weight_decay=args.soap_weight_decay, soap_precond_freq=args.soap_precond_freq)
    return history, info


# -----------------------------------------------------------------------------
# Reference solution and test
# -----------------------------------------------------------------------------
def ogata_banks_star(x_star: np.ndarray, t_star: float, cfl: float) -> np.ndarray:
    """Ogata-Banks C* (Eq. 18), written in physical units as in the Report 4 plots (erfcx for stability)."""
    x_star = np.asarray(x_star, dtype=np.float64)
    if t_star <= 0:
        return np.zeros_like(x_star)
    u = cfl * L / T_MAX
    x, t = x_star * L, t_star * T_MAX
    sqrt_dt = np.sqrt(D_M2_PER_DAY * t)
    term1 = erfc((x - u * t) / (2.0 * sqrt_dt))
    b = (x + u * t) / (2.0 * sqrt_dt)
    term2 = np.exp(np.clip(u * x / D_M2_PER_DAY - b**2, -745.0, 700.0)) * erfcx(b)
    return np.nan_to_num(0.5 * (term1 + term2), nan=0.0, posinf=0.0, neginf=0.0)


def evaluate(predict, device, dtype):
    """predict(x_star (n,1), t_star (n,1), cfl float) -> C* tensor (n,1).

    Relative L2 error at the 101 positions per output time, averaged over the five times."""
    x_t = torch.tensor(EVAL_X_STAR.reshape(-1, 1), dtype=dtype, device=device)
    rows = []
    with torch.no_grad():
        for cfl in TEST_CFL:
            per_time = []
            for t_d in EVAL_TIMES_D:
                t_star = t_d / T_MAX
                t_t = torch.full_like(x_t, t_star)
                c_hat = predict(x_t, t_t, float(cfl)).detach().cpu().numpy().ravel()
                c_ref = ogata_banks_star(EVAL_X_STAR, t_star, float(cfl))
                per_time.append(float(np.linalg.norm(c_hat - c_ref) / np.linalg.norm(c_ref)))
            rows.append({"cfl": float(cfl), "u_m_per_d": float(cfl * L / T_MAX),
                         **{f"rel_l2_t{int(t)}d": e for t, e in zip(EVAL_TIMES_D, per_time)},
                         "rel_l2": float(np.mean(per_time))})
    e = np.array([r["rel_l2"] for r in rows])
    summary = {
        "test_n_cfl": N_TEST,
        "test_cfl_range": [float(TEST_CFL.min()), float(TEST_CFL.max())],
        "test_min_distance_to_training_cfl": float(np.min(np.abs(TEST_CFL[:, None] - np.concatenate(
            [np.linspace(CFL_MIN, CFL_MAX, n) for n in TRAIN_GRIDS])[None, :]))),
        "test_times_d": EVAL_TIMES_D.tolist(),
        "test_n_positions": int(EVAL_X_STAR.size),
        "rel_l2_median": float(np.median(e)),
        "rel_l2_max": float(e.max()),
        "rel_l2_mean": float(e.mean()),
        "rel_l2_min": float(e.min()),
        "cfl_at_max": float(TEST_CFL[int(e.argmax())]),
    }
    return rows, summary


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def environment(device) -> dict:
    env = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
           "host": platform.node(), "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def write_outputs(out_dir, meta, rows, history=None, eval_history=None):
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "test_errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if eval_history:
        with open(os.path.join(out_dir, "test_error_history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_history[0].keys()))
            w.writeheader()
            w.writerows(eval_history)
    if history:
        with open(os.path.join(out_dir, "loss_history.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "total", "pde", "ic", "inlet", "outlet"])
            for i, h in enumerate(history, 1):
                w.writerow([i, *h])


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_summary(name, summary, info=None):
    line = f"[{name}] test rel-L2 median {100 * summary['rel_l2_median']:.3f} %  max {100 * summary['rel_l2_max']:.3f} %"
    if info:
        line += f"  | {info['steps_actual']} steps, {info['train_wall_clock_s']:.1f} s"
    print(line)
    sys.stdout.flush()
