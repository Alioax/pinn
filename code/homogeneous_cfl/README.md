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

Run on the remote with `run.bat jobs_homog.txt`; after `git pull`, run `python code/homogeneous_cfl/summarize_grid.py`.
Each run writes `grid_results/<run>/`: `run_meta.json` (all settings, wall-clock, steps, device),
`test_errors.csv` (40 held-out CFL values vs Ogata-Banks at 101 positions x 5 output times),
`loss_history.csv`, the model and three PNGs.
