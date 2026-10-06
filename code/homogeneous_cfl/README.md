# Homogeneous medium — CFL parameterization (Report 4)

Fixed diffusion and time horizon; advection varies through **CFL**. Same three surrogate types as Report 3.

| Subfolder | Model |
|-----------|--------|
| `baseline_cfl_pinn/` | Single-CFL PINN |
| `parametric_cfl_pinn/` | Parametric PINN over CFL |
| `pino_cfl/` | Parametric PINO (DeepONet) |
| `validation/` | COMSOL vs analytical comparison plots |

## Sample x optimizer grid (paper Section 3.1, Oct 2026)

Both parametric scripts take command-line flags; with **no flags they reproduce Report 4 exactly**
(50 CFL samples, L-BFGS lr 1.0 / max_iter 1 / 1000 steps, outputs in their own `results/`).
Shared code (networks, flags, L-BFGS/SOAP loop with early stopping, held-out test, `run_meta.json`)
is in `homog_common.py`; SOAP is `../heterogeneous/utils/soap.py`.

| File | Purpose |
|------|---------|
| `homog_common.py` | networks (unchanged), flags, training loop, test against Ogata-Banks, outputs |
| `eval_saved_models.py` | E0: test the saved April models, no training |
| `summarize_grid.py` | collect `grid_results/*/run_meta.json` into `grid_summary.csv` / `.md` |
| `../../jobs_homog.txt` | E0 + H01-H12 (samples {50, 100} x optimizer {L-BFGS Report 4, L-BFGS paper, SOAP}) |
| `../../jobs_homog_long.txt` | H13-H14: PINN and PINO, SOAP, 50 samples, 25,000 steps without early stopping, test error logged every 1,000 steps |
| `../../jobs_homog_colloc.txt` | H15-H18: as H13-H14 but with the collocation changed: sqrt-spaced time levels (H15-H16); random points every step, then a dense 150 x 150 mesh with early stopping (H17-H18) |
| `../../jobs_homog_corner.txt` | H19-H20: as H13-H14 but with the point (x*=0, t*=0) removed from the initial-condition set (`--ic-skip-corner`) |

Run on the remote with `run.bat jobs_homog.txt`; after `git pull`, run `python code/homogeneous_cfl/summarize_grid.py`.
Each run writes `grid_results/<run>/`: `run_meta.json` (all settings, wall-clock, steps, device),
`test_errors.csv` (40 held-out CFL values vs Ogata-Banks at 101 positions x 5 output times),
`loss_history.csv`, the model and three PNGs.

`--eval-every N` logs the held-out test error every N steps (`test_error_history.csv`); it does not change training
and its time is excluded from the reported training time.

Collocation flags (defaults keep the fixed uniform Report 4 mesh): `--t-levels sqrt` places the time levels of the
PDE and boundary meshes at t*_k = (k/(n-1))^2; `--resample-every N` draws new uniform random (x*, t*) points for all
loss terms every N steps; `--dense-final-steps N --dense-n M` trains the last N steps on a fixed uniform M x M mesh
per CFL sample, and early stopping then applies only to that phase. The random and dense options need SOAP.
`--ic-skip-corner` removes the corner point (x*=0, t*=0) from the initial-condition set only; it stays in the PDE
and inlet sets, so the initial and inlet conditions no longer disagree at that point.
