# Baseline PINN Implementation

Baseline demonstration script - a minimal, end-to-end Physics-Informed Neural Network implementation for 1D contaminant transport.

## Overview

This is a baseline implementation: a minimal, clean PINN that demonstrates how to train a neural network to solve the advection-dispersion equation. The code is designed to be easy to understand and serves as a baseline for comparison with more advanced implementations.

## What it does

- Trains a neural network to solve: `∂C/∂t + U ∂C/∂x = D ∂²C/∂x²`
- Enforces physics through loss functions:
  - PDE residual loss
  - Initial condition loss (C(x,0) = 0)
  - Boundary condition losses (inlet and outlet)
- Uses dimensionless formulation for numerical stability
- Generates concentration profiles at different times

## How to run

```bash
python pinn_baseline.py
```

## Output

Results are saved to `results/pinn_baseline_concentration_profiles.png` showing concentration profiles at multiple time steps.

## Configuration

All parameters are configurable at the top of `pinn_baseline.py`:
- Physics parameters (U, D, C₀, L, T_phys)
- Model architecture (layers, neurons, activation)
- Training parameters (epochs, learning rate, sampling counts, loss weights)
- Plotting parameters (times to plot, spatial resolution)

## Key Features

- **Minimal**: ~330 lines of clean, readable code
- **Self-contained**: All code in one file
- **Dimensionless**: Trains in dimensionless form for stability
- **Reproducible**: Seeds set for consistent results
