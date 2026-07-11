# COMSOL solutions - Dr. Fahs, delivered 2026-07-09

Reference solutions for 1D advection-dispersion in piecewise (multi-zone) media,
COMSOL 6.3, domain L=100 m, 101 nodes (x = 0..100 m). Same export format as the
original 4-zone file (`% c ... @ t=<days>, u1=..., u2=...` + a 101-value row).
All files: times **t = 0,100,200,...,1200 d** (13 slices). Concentrations are
dimensionless (inlet C=1). Parse with the general reader in
`parametric_heterogeneous_pinn/generate_reference_pinn.py` (comsol section) or
`utils/comsol_4zones.py` (extend the u1..uN regex).

| file | zones | interfaces (x*) | zone lengths (m) | velocities (m/d) | combos |
|---|---|---|---|---|---|
| `2zones/data_2zones_20m.txt` | 2 | 0.20 | 20 / 80 | {0.01,0.03,0.05} | 9 (3^2) |
| `2zones/data_2zones_30m.txt` | 2 | 0.30 | 30 / 70 | {0.01,0.03,0.05} | 9 |
| `2zones/data_2zones_40m.txt` | 2 | 0.40 | 40 / 60 | {0.01,0.03,0.05} | 9 |
| `2zones/data_2zones_50m.txt` | 2 | 0.50 | 50 / 50 | {0.01,0.03,0.05} | 9 |
| `3zones/data_3zones_15_15_70.txt` | 3 | 0.15, 0.30 | 15 / 15 / 70 | {0.01,0.03,0.05} | 27 (3^3) |
| `3zones/data_3zones_20_20_60.txt` | 3 | 0.20, 0.40 | 20 / 20 / 60 | {0.01,0.03,0.05} | 27 |
| `../comsol_4zones.txt` (canonical 4-zone) | 4 | 0.20, 0.40, 0.60 | 20/20/20/40 | {0.01,0.03,0.05} | 81 (3^4) |
| `5zones/data_5_zones.txt` | 5 | 0.12, 0.24, 0.36, 0.48 | 12/12/12/12/52 | {0.01,0.03,0.05} | 243 (3^5) |
| `6zones/data_6_zones.txt` | 6 | 0.11, 0.22, 0.33, 0.44, 0.55 | 11/11/11/11/11/45 | see note | 216 (designed subset) |

Notes:
- The **4-zone** file is byte-identical (md5 `b611bd5f...`) to the repo's existing
  `code/heterogeneous/data/comsol_4zones.txt`; it is NOT duplicated here - use the
  canonical path (that's what code's `COMSOL_DATA_PATH` points to).
- **Geometry (confirmed with Dr. Fahs, 2026-07-09):** 5-zone interfaces every 12 m
  (12/24/36/48; zones 12/12/12/12/52); 6-zone interfaces every 11 m
  (11/22/33/44/55; zones 11/11/11/11/11/45).
- **6-zone velocities** are a 216-combo designed subset: zones 1-3 drawn from
  {0.01,0.03,0.05} and zones 4-6 from {0.02,0.04} (27 x 8 = 216). A symmetric
  slow/fast alternation across all six zones is therefore NOT in the set.
- **6-zone: corrected 2026-07-10 (was invalid).** The original run had the final
  zone's velocity entered as **0.04 m/s instead of 0.04 m/d** (~86,400x too fast),
  so the last zone [55-100 m] plug-flowed instantly -> a nonphysical flat plateau
  (~0.59) ahead of the true front, then a single-node drop to 0 at x=100. Dr. Fahs
  re-ran it; the current file is fixed and verified physically consistent (the
  prior worst case now decays smoothly to ~0 near the outlet, front ~56 m; max
  C(x=99)@t=1200 across all 216 combos = 0.0007, on par with the 2-5 zone files).
- Geometry mismatch with our archived PINN experiment: our generated 2/3-zone
  references used interface 0.5 (2-zone, matches `data_2zones_50m`) and equal
  thirds 1/3,2/3 (3-zone, matches NEITHER 15/15/70 nor 20/20/60). See
  `../../parametric_heterogeneous_pinn/results/_archive_pinn_reference_2026-07/`.
