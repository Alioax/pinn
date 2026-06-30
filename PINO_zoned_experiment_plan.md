# PINO Zoned-Heterogeneous — Experiment Plan (for implementation)

Spec for the next round of experiments to wrap up the piecewise/zoned PINO study.
This document describes **what we want and why**. Code changes (new CLI flags, SOAP
optimizer, 9-vector branch) are to be implemented in
`code/heterogeneous/pino_heterogeneous/pinn1d_heterogeneous_parametric_neural_operator.py`
and the `code/heterogeneous/utils/` modules. Run them via `jobs.txt` + `run.bat`.

---

## Context (why this round)

- We are **optimizer-bound, not data-bound**. The largest accuracy gains so far came
  from a stronger L-BFGS budget (500×5 → 1000×20 dropped mean error 5.36% → 3.70%),
  not from more media. The N100→N250→N500 data curve shows diminishing returns
  (4.89% → 4.28% → 3.70%). A small architecture is competitive with w64. So the
  highest-value lever is the **optimizer (SOAP)**.
- **Validation is the binding constraint.** All errors are currently scored on the
  **81 fixed 4-zone COMSOL cases** (3⁴ grid of u ∈ {0.01, 0.03, 0.05}, interfaces at
  x* = 0.2/0.4/0.6). COMSOL references for 1/2/3/5-zone geometries are *pending from
  the supervisor*; any "generalize to other zone counts" claim is only partially
  testable (4-zone) until that data arrives.

## Reference config — **Va** (already run, do not re-run)

Va is the fast screening harness for this whole round (~1.9 h on an RTX 3090).
Every experiment below is compared against Va unless stated otherwise.

```
--design grf --n-sensors 100 --n-grf 0 \
--n-piecewise-per-zones 50 --piecewise-zone-counts 1,2,3,4,5 --min-zone-frac 0.1 \
--arch default --dtype float64 \
--early-stop-patience 150 --lr-lbfgs 0.1 --lbfgs-max-iter 20 --epochs 1000
```

- N = 250 (50 media per zone count × zones 1–5), branch `[100,16,16,32]`, trunk `[2,32,32,32,32]`.
- **Result: mean rel-L2 = 4.41%, max = 9.72%, min = 1.54%, ~1.9 h.**

**Comparison metric for all experiments:** mean and max relative-L2 on the 81 4-zone
COMSOL cases, vs Va. Keep arch = `default`, N = 250 unless noted, so each run differs
from Va on **one axis only**.

---

## Experiment 1 — SOAP optimizer

**Goal.** Decide whether to adopt SOAP, SOAP→L-BFGS, or neither. SOAP (ShampoO with
Adam in the Preconditioner's eigenbasis) targets the competing-gradient problem that
makes PINN optimization hard, which is exactly our bottleneck.

**Scope:** default arch, N = 250 only (i.e. the Va recipe). No w64. Two new runs,
compared against Va (which is pure L-BFGS).

- **1a — SOAP only.** Train with SOAP, no L-BFGS.
- **1b — SOAP → L-BFGS.** SOAP warm-up, then L-BFGS fine-tune (existing L-BFGS
  settings: `--epochs 1000 --lbfgs-max-iter 20 --lr-lbfgs 0.1`).

**SOAP hyperparameters** (from the supervisor's source, Vyas et al. 2025 / Wang et al. 2025):
betas = (0.95, 0.95), weight_decay = 0.01, precondition_frequency = 100. Learning rate
and step count to be tuned (start ~3e-3, see below).

**Fairness.** Choose the SOAP step count so each run's wall-clock lands near Va's
(~2 h). This makes it a "best model at equal compute" comparison rather than an
equal-step comparison.

**Code needed.**
- Add a SOAP optimizer implementation (drop in the reference single-file `soap.py`
  from the official repo or NVIDIA's emerging-optimizers, under `utils/` or `shared/`).
- New CLI flag `--optimizer {lbfgs, soap, soap_lbfgs}` (default `lbfgs`, preserves
  current behaviour).
- SOAP-specific flags: `--soap-epochs`, `--soap-lr`, `--soap-betas`,
  `--soap-weight-decay`, `--soap-precond-freq`.
- `soap_lbfgs` runs SOAP for `--soap-epochs`, then L-BFGS for `--epochs`.
- Record optimizer + SOAP settings in `run_meta.json`.

**Suggested out-dirs:**
`results/exp_W1a_soap_K100_N250_z1to5_default_float64`
`results/exp_W1b_soap_lbfgs_K100_N250_z1to5_default_float64`

**Decision rule.** If SOAP-only or SOAP→L-BFGS beats Va on mean and/or max at ~equal
compute, adopt it; otherwise stay with L-BFGS.

---

## Experiment 2 — Train on 5 zones only

**Goal.** The supervisor's core question: train on the richest class only, then test
whether it generalizes *down* to 2/3/4/5 zones. (Supervisor asked for N = 500; we run
**N = 250 first** as a fast, apples-to-apples comparison against Va — same protocol,
same arch, same total N, only the zone composition changes. N = 500 can follow once
this is informative.)

**Config:** Va recipe with `--piecewise-zone-counts 5 --n-piecewise-per-zones 250`
(250 media, all 5-zone). No code change required.

**Comparison:** vs Va (uniform zones 1–5, N = 250). Score on 4-zone COMSOL now;
on 2/3/5-zone when the supervisor's COMSOL data arrives.

**Suggested out-dir:**
`results/exp_W2_5zoneonly_K100_N250_default_float64_lbfgs1000_maxiter20`

---

## Experiment 3 — Per-zone allocation (ramped)

**Goal.** Test whether reallocating a fixed media budget toward higher zone counts
helps (higher-zone fields are harder / higher-dimensional). Earlier evidence (Ub:
z2–5, 125/zone) already suggested reallocation helps both mean and max.

**Allocation:** ramp proportional to zone count (weights 1:2:3:4:5, sum = 15);
per-zone-count count = round(k / 15 × N).

- **3a — N = 250, ramped:** `17, 33, 50, 67, 83` (sum 250).
  Compare vs **Va** (uniform `50,50,50,50,50`, N = 250).
- **3b — N = 500, ramped:** `33, 67, 100, 133, 167` (sum 500).
  Default arch, N = 500.

> **Baseline note for 3b:** there is no uniform **default-arch N = 500** run yet
> (Ua is uniform but w64, so comparing 3b to Ua confounds allocation with arch). For a
> clean N = 500 contrast, either add a uniform default-arch N = 500 run
> (`--n-piecewise-per-zones 100`) or interpret 3b mainly against 3a/Va across budgets.

**Code needed.**
- New CLI flag for an explicit per-zone-count list, e.g.
  `--n-piecewise-per-zone-list 17,33,50,67,83`, overriding the single
  `--n-piecewise-per-zones`. Length must equal `--piecewise-zone-counts`.
- Record the per-zone allocation in `run_meta.json`.

**Suggested out-dirs:**
`results/exp_W3a_ramp_K100_N250_z1to5_default_float64`
`results/exp_W3b_ramp_K100_N500_z1to5_default_float64`

**Caveat.** Until multi-zone COMSOL exists, "improvement" is measured on 4-zone only,
which slightly favors any allocation that loads zones 4–5. Judge fully once the other
COMSOL cases arrive.

---

## Experiment 4 — 9-vector parametric branch (aliasing probe)

**Goal.** Quantify how much the 100-sensor field encoding loses by *aliasing the
interface location* (a discontinuity between two sensors is misplaced). The 9-vector
encoding represents zone boundaries **exactly**, so the gap vs Va measures the cost of
sensor aliasing. This is a **probe / upper bound, not a replacement** — the sensor
encoding is kept because it is what the GRF / continuous-field endgame needs.

**Encoding.** Branch input = 10 numbers: `(u1, u2, u3, u4, u5, b1, b2, b3, b4, n_zones)`.

Padding rule (a k-zone case as a degenerate 5-zone case):
- velocities: fill `u_{k+1..5} = u_k` (repeat the last real zone velocity);
- boundaries: real internal interfaces in `b_1..b_{k-1}`; set `b_{k..4} = 1.0`
  (collapses the unused trailing zones to zero width);
- `n_zones = k`.

Example (2-zone): `(u1, u2, u2, u2, u2, b1, 1.0, 1.0, 1.0, 2)`.

COMSOL 4-zone validation case → `(u1, u2, u3, u4, u4, 0.2, 0.4, 0.6, 1.0, 4)`.

**Config:** same training media as Va (piecewise, maximin LHC, min_zone_frac 0.1,
zones 1–5, 50/zone, N = 250), default-size trunk; branch input dim = 10 instead of 100.
The PDE residual reconstructs the piecewise-constant field from `(u, b)` (reuse the
existing `piecewise_u_at_xstar` machinery).

**Comparison:** vs Va.

**Code needed.**
- New branch/media mode (e.g. `--branch-mode {sensor, zone9}` or a new `--design zone9`).
- Branch builder accepting a 10-dim input; encoder that turns each sampled piecewise
  medium into the padded `(u, b, n_zones)` vector.
- Validation path that encodes the fixed 4-zone COMSOL cases into the 9-vector form
  above.
- Record mode in `run_meta.json`.

**Suggested out-dir:**
`results/exp_W4_zone9_K0_N250_z1to5_default_float64_lbfgs1000_maxiter20`

---

## Not in scope this round

- **Sensor-interface alignment** (snapping training interfaces onto the sensor grid):
  dropped for now.
- **Adaptive / residual-based collocation:** out of scope for now.

---

## Run summary

| ID  | Name              | What changes vs Va        | N   | Arch    | Code change |
|-----|-------------------|---------------------------|-----|---------|-------------|
| W1a | SOAP only         | optimizer = SOAP          | 250 | default | SOAP + `--optimizer` |
| W1b | SOAP → L-BFGS     | optimizer = SOAP then LBFGS | 250 | default | SOAP + `--optimizer` |
| W2  | 5-zone only       | zones = {5}, 250/zone     | 250 | default | none (existing flags) |
| W3a | Ramp N250         | per-zone 17/33/50/67/83   | 250 | default | per-zone-list flag |
| W3b | Ramp N500         | per-zone 33/67/100/133/167 | 500 | default | per-zone-list flag |
| W4  | 9-vector probe    | branch = 10-dim (u,b,nz)  | 250 | default | branch mode + encoder + validation |

Reference: **Va** = mean 4.41% / max 9.72%, ~1.9 h (already run).

All runs: `--dtype float64`, K = 100 sensors (except W4), L-BFGS 1000 × maxiter 20,
lr-lbfgs 0.1, early-stop-patience 150 — i.e. the Va recipe, one axis changed at a time.

## Code-change checklist for the implementer

1. `--optimizer {lbfgs, soap, soap_lbfgs}` + SOAP optimizer module + SOAP hyperparam flags.
2. `--n-piecewise-per-zone-list` (explicit per-zone-count allocation; overrides the scalar).
3. 9-vector branch mode: encoder, branch arch (input 10), PDE field reconstruction, and
   COMSOL validation encoding.
4. Extend `run_meta.json` to log optimizer, per-zone allocation, and branch mode.
5. No changes to collocation sampling or sensor placement this round.
