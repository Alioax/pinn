# Archive: PINN-as-reference experiment (2026-07-08 / 09)

Single-instance ("problem-specific") PINN used as an interim numerical reference
generator for heterogeneous transport, **while Dr. Fahs's COMSOL was pending**.
COMSOL for 2-6 zones arrived 2026-07-09, so this stopgap is superseded and
archived. The *method* was validated (see below); the scripts remain live one
folder up (`calibrate_single_instance_pinn.py`, `generate_reference_pinn.py`,
`analyze_calibration.py`).

## What's here
- `calib_single/`, `calib_single_v2/` - step-1 proxy calibration vs the 4-zone
  COMSOL (v1 = raw residual proxy; v2 = corner-excluded + p90). 10 media each.
- `calib_smoke/`, `calib_smoke_v2/` - 1-minute wiring smoke tests.
- `gen_ref/` - the generated 2/3-zone references (+ 2 four-zone COMSOL validators).
- `gen_smoke/` - generation smoke test.

## Outcome (validated against real COMSOL)
Single-instance PINN accuracy is ~1%. Every reference that had a COMSOL
counterpart matched well:

| generated reference | geometry | rel-L2 vs COMSOL |
|---|---|---|
| 4-zone 0.01/0.01/0.01/0.01 | interfaces 0.2/0.4/0.6 | **0.68%** |
| 2-zone 0.05/0.01 | interface 0.5 (= data_2zones_50m) | **0.91%** |
| 4-zone 0.05/0.01/0.05/0.01 | interfaces 0.2/0.4/0.6 | **1.42%** |

The p90 residual proxy predicted these (p90 0.033->0.68%, 0.037->1.42%); the
calibrated threshold tau=0.044 guarantees <2%. So the self-certification works.

## Caveat / lesson
7 of the 10 generated 2/3-zone references could **not** be checked because they
were deliberately built OFF the COMSOL grid (velocities 0.02/0.04; 3-zone at
equal thirds 1/3,2/3). Fahs's COMSOL instead used the standard grid
{0.01,0.03,0.05} with specific geometries (2-zone at 0.2/0.3/0.4/0.5; 3-zone at
15/15/70 and 20/20/60). Lesson for any future PINN-reference generation: target
the COMSOL grid geometries + velocities so results are directly checkable.
