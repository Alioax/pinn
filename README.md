# PINN for 1D Contaminant Transport in Aquifers

A Physics-Informed Neural Network (PINN) implementation for solving 1D advection-dispersion equation modeling contaminant transport in groundwater aquifers.

## Research Context

This project demonstrates the application of Physics-Informed Neural Networks to solve the 1D contaminant transport problem in aquifers. The work is part of Water Resources research focusing on efficient numerical methods for groundwater contamination modeling.

The PDE being solved:
```
∂C/∂t + U ∂C/∂x = D ∂²C/∂x²
```

with boundary conditions:
- Inlet: `C(0, t) = C₀` (Dirichlet)
- Outlet: `∂C/∂x(L, t) = 0` (Neumann)
- Initial: `C(x, 0) = 0` for x > 0

## Report

A comprehensive research report documenting the methodology, results, and analysis is available:
- **Report**: [`docs/reports/report-1-pinn-baseline.pdf`](docs/reports/report-1-pinn-baseline.pdf)

The report provides detailed theoretical background, implementation details, results analysis, and comparisons with analytical solutions.

## Repository layout

All source code lives under [`code/`](code/README.md), organized by **research phase** (aligned with reports):

| Folder | Phase |
|--------|--------|
| `code/homogeneous_pe/` | Report 3 — Pe-parametric PINN / PINO (complete) |
| `code/homogeneous_cfl/` | Report 4 — CFL-parametric surrogates (complete) |
| `code/heterogeneous/` | Report 5 — neural operator over heterogeneous medium (**active**) |
| `code/exploratory/` | Side experiments (grid search, adaptive collocation, …) |
| `code/archive/` | Superseded scripts from earlier reports |

See [`code/README.md`](code/README.md) for paths, quick-start commands, and the full map.

**Outputs** stay beside the script that produced them (`results/` per project), including trained model weights (`.pt`).

**Docs:** `docs/reports/` (PDF reports), `docs/methodology/` (architecture and training notes).

## Quick start (current canonical baseline)

Report 3 Pe-parametric baseline PINN:

```bash
cd code/homogeneous_pe/baseline_pinn
python pinn1d_transport_simple_baseline.py
```

Report 4 CFL-parametric baseline PINN:

```bash
cd code/homogeneous_cfl/baseline_cfl_pinn
python pinn1d_cfl_pe_baseline.py
```

Each run writes plots and checkpoints under that folder’s `results/`.

Implementation notes: dimensionless formulation, L-BFGS on regular collocation meshes, and Ogata–Banks validation are documented in the PDF reports and in [`docs/methodology/cfl_models_methodology.md`](docs/methodology/cfl_models_methodology.md) (Report 4).

## Reports

| # | PDF | Code folder |
|---|-----|-------------|
| 1 | [`docs/reports/report-1-pinn-baseline.pdf`](docs/reports/report-1-pinn-baseline.pdf) | `code/archive/` (superseded) |
| 2 | [`docs/reports/report-2-pinn-update.pdf`](docs/reports/report-2-pinn-update.pdf) | `code/exploratory/` + archive |
| 3 | [`docs/reports/report-3-pino.pdf`](docs/reports/report-3-pino.pdf) | `code/homogeneous_pe/` |
| 4 | [`docs/reports/report-4-cfl.pdf`](docs/reports/report-4-cfl.pdf) | `code/homogeneous_cfl/` |
| 5 | (planned) heterogeneous operator | `code/heterogeneous/` |

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch ≥2.0.0
- NumPy ≥1.24.0
- Matplotlib ≥3.7.0
- SciPy ≥1.10.0 (for analytical solution)
- imageio ≥2.31.0 (for GIF generation in experimental implementation)

## License

MIT License - see LICENSE file for details.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
- Ogata, A., & Banks, R. B. (1961). A solution of the differential equation of longitudinal dispersion in porous media. US Geological Survey Professional Paper, 411-A.
