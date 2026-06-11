# Heterogeneous medium — physics-informed neural operator (Report 5)

Four zones on \(x \in [0, L]\) with lengths **20 + 20 + 20 + 40 m** (COMSOL geometry) and zone velocities \(u_1,\ldots,u_4\) (m/d). The PINO learns

\[
(u_1, u_2, u_3, u_4) \mapsto C^*(x^*, t^*)
\]

from the dimensionless advection–diffusion equation with **piecewise** \(\mathrm{CFL}(x^*)\). Training is **physics-informed only**; COMSOL data are used for validation plots.

## Physics

| Quantity | Value |
|----------|--------|
| \(L\) | 100 m |
| \(T_{\max}\) | 1200 d |
| \(D\) | \(10^{-6}\) m²/s |
| \(Pe\) | \(D T_s / L^2\) (fixed) |
| Zones | z1: \([0,20)\) m, z2: \([20,40)\), z3: \([40,60)\), z4: \([60,100]\) → \(u_1,\ldots,u_4\); in \(x^*\): \([0,0.2), [0.2,0.4), [0.4,0.6), [0.6,1]\) |
| Interfaces | \(x^* = 0.2,\ 0.4,\ 0.6\) (20, 40, 60 m) |
| PDE | \(\partial C^*/\partial t^* + \mathrm{CFL}(x^*)\,\partial C^*/\partial x^* - Pe\,\partial^2 C^*/\partial x^{*2} = 0\) |

BCs: \(C^*(x^*,0)=0\), \(C^*(0,t^*)=1\), \(C^*(1,t^*)=0\).

### Zone geometry fix (important)

Earlier PINO runs (including every experiment below) used **incorrect equal-length zones** (\(25\) m each, interfaces at \(x^*=0.25, 0.5, 0.75\)). COMSOL and the intended problem use **20–20–20–40 m** zones. `utils.zone_velocity.py` and the training script now use the correct piecewise \(\mathrm{CFL}(x^*)\) and interface locations. **Re-train and re-validate** before comparing new numbers to the table below; prior validation \(L_2\) summaries are not meaningful for the true geometry.

## Model (current baseline)

- **Branch:** zone CFL \((\mathrm{CFL}_1,\ldots,\mathrm{CFL}_4)\) → MLP `[4, 12, 12, 12, 12]`
- **Trunk:** \((x^*, t^*)\) → MLP `[2, 12, 12, 12]`
- Output: \(\sigma(\mathbf{b}^\top \mathbf{t})\) (single trunk; globally \(C^\infty\) in \((x^*, t^*)\))
- **PDE:** piecewise \(\mathrm{CFL}(x^*)\) from branch via `utils.zone_velocity`

## Validation status (COMSOL, 81 media) — pre-geometry-fix baseline

*Recorded under the wrong 25–25–25–25 m zone assumption; re-run after the fix above.*

After training on the \(3^4\) grid (legacy) and validating against [`data/comsol_4zones.txt`](data/comsol_4zones.txt) (see `comsol_validation_summary.csv`):

| Metric | Rel. \(L_2\) (mean over 5 COMSOL times) |
|--------|----------------------------------------|
| All 81 cases | mean ≈ **8.8%**, max ≈ **15.9%**, min ≈ **3.6%** |
| Uniform media (\(u_1=u_2=u_3=u_4\), 3 cases) | mean ≈ **5.3%** |
| Heterogeneous (78 cases) | mean ≈ **8.9%**, max ≈ **15.9%** |

Target (homogeneous Report 4): near-perfection (\(\ll 1\%\)). Worst errors align with **slow–fast–slow** (and similar) profiles that need sharp slope changes at zone interfaces.

## What we tried (no meaningful gain on validation \(L_2\))

*All items through 7 were run with **equal 25 m zones** (wrong geometry vs COMSOL). Outcomes may change after the 20–20–20–40 m fix.*

1. **More iterations + wider nets** — L-BFGS 500 → 2000, hidden width 16 → 24. Training loss dropped; COMSOL validation unchanged.
2. **Kink-feature trunk inputs** — append \(\tanh(k\,(x^*-\xi_i))\) at the (then-wrong) interfaces \(\{0.25, 0.5, 0.75\}\), \(k=80\). Architecture can represent slope kinks; optimizer left them unused (lazy smooth minimum).
3. **Interface-band collocation densification** — extra PDE points in narrow bands around interfaces (now \(x^*=0.2, 0.4, 0.6\); was 0.25, 0.5, 0.75).
4. **Interface-band PDE-loss upweighting (mild)** — no effect on validation; reverted to uniform mean.
5. **Interface-band PDE-loss upweighting (100×)** — `mean(res_bulk²) + 100 × mean(res_interface²)` on interface-band points. **Worsened** COMSOL validation vs the simple uniform mean (ruined results). Removed from script.
6. **Revert to simple baseline** — same validation accuracy as the complex variant, confirming added complexity was not used by the optimizer.
7. **Adam pre-training before L-BFGS** — 2000 Adam steps (`lr=10^{-3}`) then 500 L-BFGS. Training loss improved; COMSOL validation \(L_2\) unchanged. Removed from script.

**Diagnosis:** The true solution has **slope discontinuities** at the three zone interfaces. One \(\sigma(\mathbf{b}^\top \mathbf{t})\) with smooth MLPs cannot represent that exactly; kink features were sufficient in principle but sat in a smooth fit that minimizes average PDE loss with less coordination. Worst cases are exactly those demanding piecewise advection structure.

## Zone trunks (`--trunk-mode zone`, experiment G)

Implemented in `deeponet.py` (`DeepONetZoneTrunks`):

- Shared branch on \((\mathrm{CFL}_1,\ldots,\mathrm{CFL}_4)\); **four zone-specific trunks** selected by `zone_index(x*)`.
- PDE residual uses **zone-constant** \(\mathrm{CFL}_z\) per collocation point.
- **Interface losses** at \(x^*=0.2, 0.4, 0.6\): \(C\) continuity and flux continuity
  \(\mathrm{CFL}_z C - Pe\,\partial_x C\) from both sides.

Enable with `--trunk-mode zone` (see `exp_G_maximin_N500_zone_trunks`).

## Collocation (defaults)

| Set | Size (example: 50×50 spatial–time mesh) |
|-----|------|
| PDE | \((n_{x,\mathrm{bulk}} + n_{x,\mathrm{if}}) \times n_t \times N_{\mathrm{LHC}}\) with \(N_{\mathrm{LHC}}=300\) |
| IC / inlet / outlet | \(n_x \times N_{\mathrm{LHC}}\) or \(n_t \times N_{\mathrm{LHC}}\) each |

**Training parameters:** 300 Latin hypercube samples in \([0.01, 0.05]^4\) m/d per zone (`scipy.stats.qmc.LatinHypercube`), excluding exact COMSOL grid tuples \(\{0.01,0.03,0.05\}^4\). Samples are written to `pino_heterogeneous/results/lhc_train_u300.csv` (reused on later runs; set `reload_lhc_train_cases=True` to regenerate).

**Validation:** fixed **81** COMSOL cases on the \(3^4\) velocity grid (unchanged).

Optimizer: L-BFGS (`num_epochs_lbfgs` in script), `float64`. PDE loss: uniform `mean(residual²)` over all collocation points.

## Quick run (train + validate all 81 COMSOL media)

```bash
cd code/heterogeneous/pino_heterogeneous
python pinn1d_heterogeneous_parametric_neural_operator.py
```

Edit the configuration block at the top of that script (`num_epochs_lbfgs`, `mesh_nx_pde`, `run_training`, `run_comsol_validation`, etc.), then run once.

### Outputs under `pino_heterogeneous/results/`

| File / folder | Content |
|---------------|---------|
| `pino_heterogeneous_model.pt` | Checkpoint |
| `pino_heterogeneous_loss.png` | Training loss |
| `pino_heterogeneous_collocation_points.png` | Collocation mesh |
| `pino_heterogeneous_concentration.png` | Sample operator profiles |
| `comsol_validation/` | **81** PINO vs COMSOL plots + `comsol_validation_summary.csv` |

Validation-only (no retrain): set `run_training = False` and `run_comsol_validation = True` in the script.

[`data/comsol_4zones.txt`](data/comsol_4zones.txt) is already dimensionless \(C^* \in [0,1]\) (do not rescale by Report-4 \(C_0=5\)).

## Report 5 batch runs (remote)

Four training-parameter designs, each writing to its own results subfolder:

| `--design` | Training tuples |
|------------|-----------------|
| `capacity` | 81 COMSOL grid \(\{0.01,0.03,0.05\}^4\) (in-sample velocities) |
| `lhc` | N plain Latin-hypercube samples (exclude COMSOL grid) |
| `maximin` | N maximin-optimized LHC samples |
| `anchored` | N LHC + boundary anchors (`--n-corner-anchors`, default 16 cube corners) |

| `--arch` | Branch / trunk MLP |
|----------|-------------------|
| `default` | `[4,16,16,32]` / `[2,32,32,32,32]` |
| `dense` | `[4,64,64,128]` / `[2,64,64,64,64,128]` |

| `--dtype` | `float32` (default) or `float64` |
| `--trunk-mode` | `single` (default) or `zone` (one trunk per zone + interface losses) |

Report 5 follow-on experiments (vs maximin C):

| Folder | Change vs `exp_C_maximin_N500` |
|--------|--------------------------------|
| `exp_F_maximin_N500_float64` | `--dtype float64` |
| `exp_G_maximin_N500_zone_trunks` | `--trunk-mode zone` (per-zone PDE CFL + interface C/flux losses) |

Example (single experiment):

```bash
cd code/heterogeneous/pino_heterogeneous
python pinn1d_heterogeneous_parametric_neural_operator.py \
  --design lhc --n-train 500 \
  --out-dir results/exp_B_lhc_N500 \
  --skip-validation-plots
```

**Remote workflow:** queue all four runs in [`jobs.txt`](../../jobs.txt) at the repo root, commit & push from your laptop, then run `run.bat` on the remote GPU. The runner pulls, executes jobs fail-fast, and pushes one commit with logs under `runs/<timestamp>/` plus artifacts under `pino_heterogeneous/results/exp_*`.

**Per-run outputs** (`results/exp_<name>/`):

| File | Content |
|------|---------|
| `pino_heterogeneous_model.pt` | Checkpoint (pull to laptop for local analysis) |
| `train_u_cases.csv` | Velocity tuples used for training |
| `comsol_validation_summary.csv` | L2 metrics over 81 COMSOL cases |
| `run_meta.json` | Design, N, wall-clock, mean/max L2 |
| `pino_heterogeneous_loss.png` | Training loss curve |

Use `--reload-train-cases` to regenerate `train_u_cases.csv`. Omit `--skip-validation-plots` to also write 81 COMSOL comparison PNGs.

**Validation plots after remote training** (local, from saved checkpoints):

```bash
cd code/heterogeneous/pino_heterogeneous
python generate_report5_validation_plots.py
```

Or one experiment:

```bash
python pinn1d_heterogeneous_parametric_neural_operator.py \
  --design lhc --n-train 500 --out-dir results/exp_B_lhc_N500 --validate-only
```

Plots land in `results/exp_*/comsol_validation/` (81 PNGs + `comsol_validation_summary.csv` per run).

## Layout

| Path | Role |
|------|------|
| `utils/zone_velocity.py` | Piecewise \(\mathrm{CFL}(x^*)\) |
| `utils/lhc_sampling.py` | LHC training design + CSV I/O |
| `utils/comsol_4zones.py` | COMSOL export parser |
| `pino_heterogeneous/pinn1d_heterogeneous_parametric_neural_operator.py` | **Train + COMSOL validation** |
| `pino_heterogeneous/deeponet.py` | DeepONet module |
| `data/comsol_4zones.txt` | Reference solutions (81 velocity tuples) |
