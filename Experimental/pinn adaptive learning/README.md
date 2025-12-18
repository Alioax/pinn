# PINN vs Analytical Comparison

Extended analysis comparing PINN results with the analytical solution, including residual analysis.

## Overview

This script trains a PINN and provides detailed comparison with the analytical solution, including:
- Side-by-side concentration profile comparisons
- PDE residual analysis
- Initial condition residual
- Boundary condition residuals

## Features

- Trains PINN with same dimensionless formulation as simple demo
- Compares PINN predictions with analytical solution
- Visualizes residuals to identify where physics is violated
- More comprehensive than the simple demo

## How to run

```bash
python pinn_with_analytical_comparison.py
```

Note: This script imports `analytical_solution.py` from the root-level `analytical_solution/` folder.

## Output

Generates comparison plots saved to `results/plots/`:
- `pinn_analytical_comparison.png` - Concentration profiles comparison
- `pinn_analytical_comparison_with_pde_loss.png` - Multi-panel plot with residuals

## Configuration

Similar to simple demo, all parameters are configurable at the top:
- Physics parameters
- Model architecture
- Training parameters
- Plotting parameters

## Differences from Simple Demo

- Includes analytical solution comparison
- Shows residual plots
- More detailed analysis
- More complex (research-grade rather than minimal demo)
