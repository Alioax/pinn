# Code layout

Scripts are grouped by **research phase** (aligned with reports), not by training algorithm alone.

| Folder | Report | Status | Contents |
|--------|--------|--------|----------|
| [`homogeneous_pe/`](homogeneous_pe/) | Report 3 (Mar 2026) | **Complete** | Baseline PINN, parametric PINN, PINO — fixed mesh, L-BFGS, **log Pe** parameter |
| [`homogeneous_cfl/`](homogeneous_cfl/) | Report 4 (Apr 2026) | **Complete** | Same three model classes — **CFL** parameter, fixed Pe group |
| [`heterogeneous/`](heterogeneous/) | Report 5 (next) | **Active** | Neural operator over heterogeneous medium |
| [`exploratory/`](exploratory/) | Reports 1–2 appendices | Archive | Grid search, adaptive collocation, sandboxes |
| [`tools/`](tools/) | — | Utility | Report figure replots, GIF builder |
| [`shared/`](shared/) | — | Library | Ogata–Banks analytical solution |
| [`archive/`](archive/) | Reports 1–2 | Legacy | Superseded `mainline_wip` / `review_ready` snapshots from git |

## Intended workflow

```
exploratory  →  homogeneous_pe / homogeneous_cfl  →  heterogeneous
```

When a phase is done, its scripts stay in the phase folder with `results/`. New work starts under `heterogeneous/`.

## Quick run

**Report 3 (Pe):**
```bash
cd code/homogeneous_pe/baseline_pinn && python pinn1d_transport_simple_baseline.py
```

**Report 4 (CFL):**
```bash
cd code/homogeneous_cfl/baseline_cfl_pinn && python pinn1d_cfl_pe_baseline.py
```

See [`../cfl_models_methodology.md`](../cfl_models_methodology.md) for Report 4 training details.
