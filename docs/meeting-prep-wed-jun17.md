# Meeting Prep — PINN vs PINO Heterogeneous Transport Paper

*Prepared Tue Jun 16, 2026 for the Wed Jun 17 meeting (10:00 Tehran / 08:30 Paris). Based on Report 5, the Experiment-J result files, the `defect_correction_pino` code/metrics, the email chain, the `paper_V0` outline (pasted by Ali), and a short literature scan.*

---

## Bottom line up front

Now that I've seen the outline, the picture is clearer and more favorable than my first read. The paper **is** a PAR_PINN-vs-PINO comparison, and the outline already expects the 4-zone operator to be *equal* to the parametric PINN ("PINO_1 is same as parametric"). So the result that worried me — PINN slightly beating PINO at 4 zones — is **the expected setup, not a problem.** The operator's actual payoff is **PINO_2: train on Gaussian random fields, test on the zoned cases, "better because we do not need refinement,"** plus generalizing 4→8 zones, which the parametric PINN structurally cannot do.

That reframes your 24-hour question:

- **GRF is no longer a risky side-quest — it is the paper's punchline, and it's feasible.** The de-risking insight: you **train PINO_2 on GRF draws with physics only (no reference data needed), then validate on the 81 four-zone COMSOL cases you already have.** You do **not** need to generate COMSOL solutions for GRF fields to get the headline result. So yes — **spend the window on PINO_2**, scoped tightly, with a fallback. Drop the vague "make the operator more efficient" option.
- **Two things in the comparison still need care** (below): the 4-zone PINN is actually a bit *better*, not identical (1.8% vs 2.6%), and the "PINN is 4–5× slower" timing claim is **not supported by any committed run** yet.
- **`defect_correction_pino`:** your instinct is right — for this linear PDE it's effectively an FDM solver. It isn't in the outline. Park it.
- **One novelty risk to fix before submission:** the outline says PINO "is not applied to transport in aquifers" (points 10, 14). That's contestable — there are 2025–2026 PI-DeepONet-for-porous-media papers. Narrow the claim and add a related-work paragraph.

---

## How your current results map onto the outline

| Outline section | What you have now | Read |
|---|---|---|
| Homogeneous: "same accuracy and same training time" | Reports 3–4 (both near-perfect <1%) | ✅ Supported |
| Het. 4 zones — PINO_1 "same as parametric" | PINO 2.61% vs PINN 1.80% (Exp J) | ⚠️ "Comparable," but PINN is meaningfully better — frame it, don't claim "same" |
| **Het. 4 zones — PINO_2: train GRF, test zones, no refinement** | **Not built yet** (code is 4-scalar only) | 🎯 The paper's key result — this is the 24 h target |
| 8 zones (generalization) | Not built; needs an 8-zone COMSOL/FE test set | ⏳ Stretch goal — the decisive "PINN can't do this" |
| "accuracy **and training time**" comparison | No clean same-hardware timing yet | ⚠️ Must measure cleanly (see below) |
| Results integrating data + physics | Not built | ⏳ Later section |

---

## Verified numbers (read from the result files, not the report text)

| Run — Exp J protocol (maximin N=500, float64, lr 0.1, max_iter 20) | Params | Mean rel-L2 | Max | Min | Wall-clock (as logged) |
|---|---|---|---|---|---|
| **PINO** `exp_J_maximin_N500_float64_lr01_maxiter20` | 4,160 | **2.61%** | 4.91% | 1.59% | 138.3 s |
| **Parametric PINN** `exp_J_..._pinn` | 4,285 | **1.80%** | 3.85% | 0.80% | 109.9 s |

Param counts verified by hand (PINN 4,285; PINO 896 branch + 3,264 trunk = 4,160). The PINN 1.80% is **new — not in Report 5**, which only covers PINO runs A→J. Report 5's A–J experiments use the **correct** 20-20-20-40 m geometry; the 8.8% table and the "interface up-weighting / kink features didn't help" notes in the README are from the **older wrong equal-25 m** geometry — don't quote those as current.

Also worth pre-empting: Report 5 is honest that the 10%→2.6% improvement came mostly from **float64 + L-BFGS line-search depth**, not architecture or sampling. That's fine for a report, but "we fixed our optimizer" isn't a contribution — the contribution has to be the comparison and PINO_2.

---

## Two things to get right in the comparison

The outline anticipates PINO_1 ≈ parametric, so this isn't a crisis — but two specifics need handling before Wednesday:

**1. The 4-zone PINN is a bit *better*, not identical (1.8% vs 2.6%).** Expected: a single fully-connected MLP is more expressive per parameter than a rank-32 branch·trunk inner product, so at matched params the PINN should edge it. Frame it as *"comparable; the operator is not meant to win on per-parameter accuracy at fixed zonation — its advantage is generalization across zonation (PINO_2, 8 zones), which we show next."* Don't claim "same."

**2. The "PINN is 4–5× slower (~40 h)" claim is currently unsupported.** The committed `exp_J_pinn` finished in **110 s and was faster** than the PINO (138 s). The 7,671 s in `exp_N` (a *smaller* N=250 PINO run) is almost certainly a different/CPU machine — i.e., **wall-clocks across your runs are not comparable.** Since the title promises "accuracy *and* training time," you need **one clean same-GPU, same-config timing comparison.** Two implementation notes that matter here:
   - In `deeponet.py`, `forward()` recomputes `b_vec = self.branch(branch_input)` for *every collocation point*. The operator's whole training-speed argument is that the branch is evaluated **once per medium** and only the trunk is auto-differentiated for ∂ₓ, ∂ₓₓ, ∂ₜ. As written you forfeit much of that. Implementing branch amortization is what makes the operator's *training-time* number actually win — especially for PINO_2, where the branch input is a high-dim sensor vector.
   - Matched parameter count (4,160 vs 4,285) is a fair accuracy control but slightly hides the operator's structural cost advantage; report both params *and* the amortized timing.

---

## PINO_2 (GRF) — the linchpin, and how to land it in 24 h

This is the result that makes the paper. The reason it's feasible despite the clock:

**You don't need reference solutions for GRF fields.** Training is physics-only (residual + IC/BC), so GRF draws need no COMSOL. You then **validate on the existing 81 four-zone COMSOL cases** — feed each zoned velocity field (sampled at the fixed sensor locations) into the GRF-trained branch and score against COMSOL exactly as you do now. The validation harness already exists.

**Concrete build (reuses your current PINO harness):**
1. **GRF sampler:** draw smooth velocity fields u(x) on [0,1] (e.g., exponentiated Gaussian field with a chosen correlation length), clipped to [0.01, 0.05] m/d. Sample each field at **K fixed sensor locations** → that K-vector is the new branch input.
2. **Architecture:** branch input dim 4 → **K**; trunk (x*, t*) unchanged; same σ(b·t). **No interface-refinement collocation** (that's the point — smooth fields don't need it).
3. **Residual:** CFL(x*) is now the continuous field value at each collocation point (from the sampled field), not a piecewise-zone lookup.
4. **Validate** on the 81 four-zone COMSOL cases by evaluating the zoned field at the K sensors. Report mean rel-L2 vs PINO_1 and the PINN.

**The real scientific question to flag (and test):** PINO_2 trains on **smooth** fields but is tested on **sharp** zoned fields — that's an out-of-distribution, smooth→sharp test. The outline's claim ("better, no refinement") is plausible if K is dense enough to resolve the jumps, but it could instead *hurt* at the interfaces. Either outcome is informative and exactly what Wednesday should discuss. Pick K with this in mind (denser sensors near where interfaces can fall, or just uniformly dense).

**Fallback discipline:** set a hard cutoff (e.g., T-6 h). If PINO_2 isn't converging by then, present the clean 4-zone comparison + amortized timing + the PINO_2 plan, and don't torch the night. The 8-zone test (needs a new COMSOL/FE set) is a **stretch goal**, not the headline — let it slip to next week if needed.

So, directly answering your question: **GRF is the better use of the 24 h than "efficiency," and it's time-efficient *if* you scope it to train-on-GRF / test-on-existing-COMSOL.** Fold the one genuinely useful efficiency item (branch amortization) into the timing measurement rather than treating it as a separate effort.

---

## `defect_correction_pino`: a solver, not an operator improvement — park it

It loads a trained PINO, computes its autograd residual `r`, and solves `L[δ] = −r` by backward-Euler **implicit FDM** over the whole space-time grid (`spsolve` per step), returning `C̃ + δ`. On the deliberately-weak `exp_E` model (worst case), COMSOL error went 19.1% → **12.8%** while the discrete residual collapsed to ~1e-15 — it nulls the residual but only partly fixes the solution (can't recover interface kinks).

Your instinct is correct: the PDE is **linear**, so a full implicit solve of the linearized operator is essentially just FDM-solving the transport equation, with the network as an initial guess the solve largely overwrites. If you'll do a full linear solve at inference, you could skip the net and FDM-solve to near-exact in similar time — the "why not just FDM?" critique is unanswerable here. It's a legitimate idea only for nonlinear/expensive problems. It's not in the outline; keep it as an appendix footnote at most.

---

## Is the paper competitive? Novelty & framing

**Honest read:** as a *forward-surrogate methods* paper the 4-zone PINO is not novel and is beaten by its own baseline — but that's not the paper. As a **comparison paper with the GRF/zonation-generalization punchline**, aimed at an *Advances in Water Resources*–type venue (where Fahs publishes PINN-for-heterogeneous-media work), it can be a solid contribution **if** the novelty claim is tightened:

- **Points 10 & 14 are risky as written.** "PINO not applied to transport in aquifers" is contestable: PI-DeepONet for porous-media transport already exists (PRT-DeepONet, *Comp. Geosciences* 2026; PI-DeepONet+FEM for convective transport in porous media, arXiv 2508.19847; "Porous-DeepONet"). Reviewers will cite these. **Do a focused related-work search now** and narrow to something defensible — e.g., *"a head-to-head physics-informed PINN-vs-operator comparison for **parametric** contaminant transport in **field-scale** aquifers, with generalization across zonation via GRF training."* The **comparison + zonation transfer** is your more defensible novelty than "first PINO for transport."
- **The interface floor is a known phenomenon you can cite, not a bug.** DeepONet smoothing of sharp fronts is well documented, with dedicated 2024–2026 architectures (φ-DeepONet — essentially your "zone trunks" idea; Cut-DeepONet; R-adaptive DeepONet). Cite these to explain the residual-at-interface behavior; don't try to out-engineer them.
- **Strongest extension if you want more than a comparison:** an **inverse application** — recover zone velocities (or the field) from sparse concentration observations. That's where an amortized operator genuinely beats FDM and a per-instance PINN, and it fits the outline's stated motivations (inverse problems, source identification). Natural follow-up paper, or a final section.

---

## Wednesday talking points

- **Lead with the comparison, framed per the outline:** homogeneous → equal; 4-zone → comparable (operator slightly behind at matched params, as expected); **operator's win is PINO_2 (GRF) and zonation transfer** — show progress or a concrete plan.
- **Report the corrected story on timing:** "the committed runs aren't same-hardware; here's the clean GPU comparison with branch amortization" (if you get it done).
- **Pre-empt "why an operator, not a PINN or FDM?":** generalization across zonation + inverse problems; not per-parameter accuracy.
- **Defect correction:** checked, it's ~FDM for a linear problem — parked.
- **Raise the novelty-claim risk** (points 10/14) and propose the narrowed framing.
- **Decide GRF scope for this paper:** PINO_2 on 4 zones (headline) now; 8-zone + data-integration sections next.

**Open items to confirm:** (1) a same-hardware PINN-vs-PINO timing number; (2) pick K and the GRF correlation length for PINO_2; (3) whether 8-zone COMSOL/FE references can be generated, and by whom.

---

### Sources (literature scan)
- Lu et al., DeepONet, *Nature Machine Intelligence* (2021): https://www.nature.com/articles/s42256-021-00302-5
- PRT-DeepONet, geometry-aware operator for pore-scale transport, *Computational Geosciences* (2026): https://www.sciencedirect.com/science/article/pii/S0098300425002481
- Physics-Informed DeepONet coupled with FEM for convective transport in porous media, arXiv 2508.19847: https://arxiv.org/pdf/2508.19847
- On the training of physics-informed neural operators for parametric PDEs, arXiv 2606.06164: https://arxiv.org/abs/2606.06164
- φ-DeepONet: a discontinuity-capturing neural operator, arXiv 2604.08076: https://arxiv.org/abs/2604.08076
- Cut-DeepONet (smooth piecewise cutting for discontinuities), arXiv 2605.19823: https://arxiv.org/html/2605.19823
- Separable physics-informed DeepONet, *CMAME* (2024): https://www.sciencedirect.com/science/article/abs/pii/S0045782524008405
- Fahs et al., PINNs for flow in heterogeneous porous media (mixed pressure-velocity), *Advances in Water Resources* (2023): https://www.sciencedirect.com/science/article/abs/pii/S0309170823001987
