# Repository Structure

This document describes the file and directory layout of the PINN (Physics-Informed Neural Networks) Code Base repository after the maturity-based reorganization.

---

## Root-Level Overview

| Item | Type | Description |
|------|------|-------------|
| `.gitignore` | file | Git ignore rules |
| `README.md` | file | Main project README |
| `requirements.txt` | file | Python dependencies |
| `REPO_STRUCTURE.md` | file | This document |
| `LICENSE` | file | License file |
| `code/` | directory | All source code (by maturity) |
| `docs/` | directory | Reports, slides, diagrams |
| `references/` | directory | Literature PDFs |

---

## Directory Tree

```
Code Base/
├── .gitignore
├── README.md
├── requirements.txt
├── REPO_STRUCTURE.md
├── LICENSE
│
├── code/
│   ├── review_ready/           # Supervisor-facing minimal scripts only
│   │   ├── pinn_baseline/
│   │   ├── parametric_pinn/
│   │   └── neural_operator_deeponet/
│   ├── mainline_wip/           # Main research path, active development
│   │   ├── pinn_baseline/
│   │   ├── parametric_pinn/
│   │   ├── neural_operator_supervised/
│   │   └── neural_operator_unsupervised/
│   ├── exploratory/           # Experiments and side ideas
│   │   ├── parametric_pinn_sandbox/
│   │   ├── pinn_grid_search/
│   │   ├── pinn_parameter_study/
│   │   ├── loss_distribution_learning/
│   │   ├── pinn_distribution_adaptive_adam_lbfgs/
│   │   └── volume_of_error_pinn/
│   └── shared/
│       └── analytical_solution/
│
├── docs/
│   ├── reports/
│   │   ├── report 1 - PINN Baseline/
│   │   └── report 2 - PINN Update/
│   ├── slides/
│   ├── diagrams/
│   └── neural_operator_architecture.md
│
└── references/
    └── *.pdf
```

---

## Code Folder Policy

- **review_ready**: Minimal, clean code prepared for supervisor/team review. Only the former baseline_scripts live here.
- **mainline_wip**: Ongoing code on the main research path (pinn_baseline, parametric_pinn, neural operator baselines).
- **exploratory**: Experimental and off-main-trajectory work (all former Experimental/ content).
- **shared**: Reusable pieces used by multiple projects (e.g. analytical_solution).

Every code project has its own `results/` directory. Multi-run experiments use subfolders (e.g. `results/exp_001/`).

---

## Promotion Workflow

```
exploratory → mainline_wip → review_ready
```

New ideas start in `exploratory/`. When aligned with the main research, move to `mainline_wip/`. When cleaned and ready for supervisor inspection, move to `review_ready/`.

---

## Maintenance Rules

1. Every new code idea goes in one of `review_ready`, `mainline_wip`, or `exploratory`.
2. Every code project owns its own `results/`.
3. Do not scatter output files loosely beside scripts.
4. Use `shared/` only for code reused across multiple projects.
5. Use consistent naming: lowercase, snake_case, no spaces, no numbered prefixes.
6. When a project matures, promote it (move) rather than copying.

---

*For run instructions and methodology, see the root `README.md` and per-folder `README.md` files.*
