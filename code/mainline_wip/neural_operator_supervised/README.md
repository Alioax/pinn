# Supervised Neural Operator Baseline (DeepONet)

Baseline demonstration script – a minimal, end-to-end DeepONet implementation for the parametric 1D advection-dispersion problem, trained by supervised learning on analytical data.

## Overview

This is a baseline implementation: a minimal, clean neural operator (DeepONet) that learns the mapping from Peclet number to the dimensionless concentration field C*(x*, t*). The code is designed to be easy to understand and serves as a baseline for comparison with physics-informed and other operator approaches. It follows the same style and structure as the PINN baselines in this repository.

## What it does

- Learns the operator **log Pe -> C*(x*, t*)** for the dimensionless advection-dispersion equation
- Uses **supervised training** on analytical (Ogata–Banks) solution data over a fixed (x*, t*) grid
- **Branch net** encodes the parameter (log Pe); **trunk net** encodes query points (x*, t*); output is inner product + sigmoid
- Train/validation split over Peclet numbers; MSE loss on the grid
- Generates dimensionless concentration profiles at selected Pe values and time slices, with analytical overlay

## How to run

```bash
python supervised_neural_operator_baseline.py
```

## Output

Results are saved to `results/`:

- `supervised_neural_operator_baseline_model.pt` – trained model state
- `loss.png`, `loss.pdf` – train and validation MSE vs epoch
- `training_pe_coverage.png`, `training_pe_coverage.pdf` – distribution of training Peclet numbers
- `{pe} Cstar_profiles.png` and `.pdf` – C* vs x* at selected times for each plotted Pe (DeepONet vs Analytical)

## Configuration

All parameters are configurable at the top of `supervised_neural_operator_baseline.py`:

- Physics: Pe_min, Pe_max, t_final_star
- Data: num_pe_train, nx, nt, train_fraction
- Model: branch_layers, branch_neurons, trunk_layers, trunk_neurons, latent_dim, activation
- Training: num_epochs, batch_size, lr
- Plotting: times_tstar, pe_values_to_plot, num_spatial_points, plot_dpi

## Key Features

- **Single script**: All code in one file, config at the top
- **Dimensionless**: Same (x*, t*) domain and C* scaling as the parametric PINN
- **Resolution-invariant use**: At inference, branch is evaluated once per Pe and trunk at any (x*, t*) query points
- **Reproducible**: Seeds set for consistent results
- **Consistent style**: Matches the layout, plotting, and conventions of `pinn_baseline` and `parametric_pinn_baseline`
