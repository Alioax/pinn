# PINO_2 (pure-GRF) experiment plan

Companion to `PINO_zoned_experiment_plan.md`. Covers the **GRF-only** variant of the
paper's heterogeneous section (Fahs outline: *"PINO_2: training with only GRF (500
samples) and testing for 4 zones"* + *"Impact of number of sensors should be discussed"*).

## Objective

Train the DeepONet operator on **smooth Gaussian-random-field velocity fields only**
(no piecewise/zoned media in training), then test on the sharp 4-zone cases. This is the
operator's out-of-distribution (smooth → sharp) generalization test. It reuses the
existing **81 four-zone COMSOL solutions** for validation, so **no new COMSOL is needed**.

## Grid — 12 runs

Correlation length **ℓ ∈ {0.05, 0.10, 0.20}** × sensor count **K ∈ {50, 100, 150, 200}**.

| | K=50 | K=100 | K=150 | K=200 |
|---|---|---|---|---|
| ℓ=0.05 | G01 | G02 | G03 | G04 |
| ℓ=0.10 | G05 | G06 | G07 | G08 |
| ℓ=0.20 | G09 | G10 | G11 | G12 |

## Fixed protocol (matches the validated X3 recipe)

- **N = 500** GRF fields (`--n-grf 500 --n-piecewise-per-zones 0`). Matches the outline literally and keeps the training budget comparable to the zoned X-series.
- **Velocity range** u ∈ [0.01, 0.05], squared-exponential kernel exp(−Δx²/2ℓ²), sigmoid-warped (unchanged from prior GRF code) — must match the zoned validation range.
- **float64**, default arch (branch [K,16,16,32], trunk [2,32,32,32,32]).
- **Optimizer SOAP**: lr 3e-3, betas 0.95/0.95, weight-decay 0.01, precond-freq 100.
- **Budget**: `--soap-epochs 25000` with `--early-stop-patience 150`. This reproduces X3, which early-stopped near 5.7k epochs, so 25k is a ceiling, not a target.
- **Seed** 1234567 (single seed for the grid; replicate the winner with 2 more seeds afterward, exactly as the X-series did).

## Why these values

**ℓ is the smoothness-vs-zone-transfer axis** — the whole point of PINO_2. It only matters
relative to the tested zone widths (~0.1–0.4; the finest 5-zone width ≈ 0.1):

- **ℓ = 0.20** — correlation spans a full zone; the field barely varies across one zone, so it can't represent independent per-zone velocities or interfaces. This is why the stale L-BFGS runs floored at 8–13%. Kept as the "smooth" anchor.
- **ℓ = 0.10** — features on the finest zone-width scale, cleanly sensor-resolved. Predicted sweet spot; the single ℓ to quote if one headline number is needed.
- **ℓ = 0.05** — roughest safe value. At the sparsest grid (K=50, spacing 0.020) it is 2.5 sensors per correlation length — at the Nyquist floor. Anything shorter aliases at K=50, so **do not go below 0.05**. The sigmoid warp turns short-ℓ fields into plateau+ramp shapes that drift toward the piecewise look — helpful for zone transfer.

**K answers the outline's mandated sensor study.** Prior evidence: K≈35 → 13.4%, K≈67 → 8.0%,
so more sensors clearly help; the sweep finds where accuracy saturates. Sensors-per-correlation-length
by (ℓ, K):

| | K=50 | K=100 | K=150 | K=200 |
|---|---|---|---|---|
| ℓ=0.05 | 2.5 | 5.0 | 7.5 | 10.0 |
| ℓ=0.10 | 4.9 | 9.9 | 14.9 | 19.9 |
| ℓ=0.20 | 9.8 | 19.8 | 29.8 | 39.8 |

**SOAP** was the decisive lever in the zoned series (X3 2.0% vs L-BFGS 3.7%; it also fixed the tail)
and has never been applied to GRF training — so this carries the one untested win into PINO_2.

## Expected outputs (per run)

`results/exp_G##_grf_ell###_K##_N500_default_float64_soap/` with `run_meta.json`
(mean/max/min rel-L2 over the 81 cases), `comsol_validation_summary.csv`, loss + concentration
+ collocation PNGs, and `pino_heterogeneous_model.pt`.

## Runtime

~50 min/run on the RTX 3090 → **~10 h wall** for all 12 (one overnight).

## After the grid

1. Read the accuracy-vs-(ℓ, K) table → pick the best config.
2. Re-run the winner with 2 extra seeds (2234567, 3234567) for a mean ± spread.
3. One L-BFGS run at the winner config → confirms SOAP's advantage holds for GRF (paper point).
4. Paper figure: accuracy heatmap over ℓ × K, plus best-model concentration plots.

*Note: mixed GRF + piecewise (outline's PINO_3) is intentionally excluded from this batch.*
