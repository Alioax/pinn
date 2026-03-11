# Unsupervised Neural Operator Baseline (DeepONet + PINN)

Baseline demonstration script – a minimal, end-to-end DeepONet implementation for the parametric 1D advection-dispersion problem, trained with **physics-informed losses only** (no data).

## Overview

This is a baseline implementation: a minimal, clean neural operator (DeepONet) that learns the mapping from Peclet number to the dimensionless concentration field C*(x*, t*) using only PDE residual, initial condition, and boundary condition losses. No analytical or other data is used during training. The code is designed to be easy to understand and serves as a baseline for comparison with the supervised neural operator and PINN baselines. It follows the same style and structure as the other baselines in this repository.

## What it does

- Learns the operator **log Pe -> C*(x*, t*)** for the dimensionless advection-dispersion equation
- Uses **unsupervised (PINN-based) training only**: PDE residual loss, IC loss, inlet and outlet BC losses; no data
- **Branch net** encodes the parameter (log Pe); **trunk net** encodes query points (x*, t*); output is inner product + sigmoid
- Collocation points resampled each epoch over (x*, t*, log Pe); automatic differentiation for PDE derivatives
- Generates dimensionless concentration profiles at selected Pe values and time slices, with analytical overlay (for reference only)

## How to run

```bash
python unsupervised_neural_operator_baseline.py
```

## Output

Results are saved to `results/`:

- `unsupervised_neural_operator_baseline_model.pt` – trained model state
- `collocation_points.png`, `collocation_points.pdf` – x* vs t* collocation distribution
- `collocation_points_logpe.png`, `collocation_points_logpe.pdf` – parametric coverage (t* vs log Pe)
- `loss.png`, `loss.pdf` – total and component losses vs epoch
- `{pe} Cstar_profiles.png` and `.pdf` – C* vs x* at selected times for each plotted Pe (DeepONet vs Analytical)

## Configuration

All parameters are configurable at the top of `unsupervised_neural_operator_baseline.py`:

- Physics: Pe_min, Pe_max, t_final_star
- Model: branch_layers, branch_neurons, trunk_layers, trunk_neurons, latent_dim, activation
- Training: num_epochs, lr, num_collocation, num_ic, num_bc, weight_pde, weight_ic, weight_inlet_bc, weight_outlet_bc
- Plotting: times_tstar, pe_values_to_plot, num_spatial_points, plot_dpi

## Key Features

- **Single script**: All code in one file, config at the top
- **No data**: Training uses only PINN losses; analytical solution is used only for plot overlay
- **Dimensionless**: Same (x*, t*) domain and C* scaling as the parametric PINN
- **Resolution-invariant use**: At inference, branch is evaluated once per Pe and trunk at any (x*, t*) query points
- **Reproducible**: Seeds set for consistent results
- **Consistent style**: Matches the layout, plotting, and conventions of `pinn_baseline`, `parametric_pinn_baseline`, and `supervised_neural_operator_baseline`
