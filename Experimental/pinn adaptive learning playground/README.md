# PINN Uniform Grid Collocation Points - Adaptive Learning Playground

A foundation implementation for Physics-Informed Neural Networks (PINN) with uniform grid collocation points, designed as a base for future adaptive learning experiments.

## Overview

This implementation provides a clean, simplified PINN for solving 1D contaminant transport in groundwater aquifers. The key feature is the use of **uniform linearly-spaced collocation points** organized in a grid structure, with separate specification of spatial (`collocation_points_x_star`) and temporal (`collocation_points_t_star`) point counts.

## Key Features

- **Uniform Grid Collocation Points**: Uses a structured grid of collocation points instead of random sampling
- **Separate x/t Specification**: Configure `collocation_points_x_star` and `collocation_points_t_star` independently
- **Fixed Points**: Collocation points are generated once and remain fixed throughout training (not resampled each epoch)
- **Simplified Output**: Only generates concentration comparison plots (PINN vs Analytical)
- **Foundation for Extension**: Clean structure ready for adding adaptive learning features later

## Problem Solved

The PDE being solved:
```
∂C/∂t + U ∂C/∂x = D ∂²C/∂x²
```

with boundary conditions:
- Inlet: `C(0, t) = C₀` (Dirichlet)
- Outlet: `∂C/∂x(L, t) = 0` (Neumann)
- Initial: `C(x, 0) = 0` for x > 0

## Configuration

### Collocation Points Configuration

The main difference from other implementations is the separate specification of collocation points:

```python
training_params = {
    'collocation_points_x_star': 50,  # Number of uniformly-spaced points in x* direction
    'collocation_points_t_star': 40,  # Number of uniformly-spaced points in t* direction
    # Total collocation points = 50 × 40 = 2000
}
```

This creates a uniform grid where:
- `x_star` values are uniformly spaced from 0 to 1: `[0, 1/49, 2/49, ..., 1]`
- `t_star` values are uniformly spaced from 0 to `t_final_star`: `[0, t_final_star/39, 2*t_final_star/39, ..., t_final_star]`
- All combinations of (x_star, t_star) are used as collocation points
- Total points = `collocation_points_x_star × collocation_points_t_star`

### Other Parameters

All other parameters follow the same structure as the adaptive learning implementation:

- **Physics parameters**: `U`, `D`, `C_0`, `L`
- **Model architecture**: `num_layers`, `num_neurons`, `activation`
- **Training parameters**: `num_epochs`, `lr`, `num_ic`, `num_bc`, loss weights
- **Plotting parameters**: `times_days`, `x_max`, `num_points`

## How to Run

```bash
cd "Experimental/pinn adaptive learning playground"
python pinn_adaptive_playground.py
```

## Output

Running the script generates:
- Concentration profiles at multiple time steps comparing PINN predictions vs. analytical solution
- Results saved to `results/pinn_adaptive_playground_profiles.png`

## Differences from Adaptive Learning Version

- **No adaptive learning**: All adaptive refinement code removed
- **Fixed uniform grid**: Collocation points generated once, not resampled
- **Separate x/t specification**: `collocation_points_x_star` and `collocation_points_t_star` instead of single `num_uniform`
- **Simplified output**: Only concentration plots, no residual analysis or GIFs
- **No GIF frames**: All GIF-related code removed
- **Cleaner structure**: Foundation ready for future enhancements

## Future Extensions

This implementation is designed as a foundation for:
- Adding adaptive learning features (RAR - Residual-based Adaptive Refinement)
- Experimenting with different collocation point distributions
- Comparing uniform grid vs. random vs. adaptive sampling strategies
- Parameter studies on collocation point density

## Code Structure

The implementation follows this organization:
1. Imports and setup
2. Configuration parameters (with separate x_star/t_star collocation specs)
3. Derived parameters
4. Dimensionless conversion functions
5. Neural network architecture
6. Loss functions
7. Uniform grid collocation point generation (`generate_uniform_grid_collocation_points`)
8. Training function (simplified, no adaptive logic)
9. Evaluation/prediction function
10. Plotting function (simplified, concentration profiles only)
11. Main execution

## Technical Details

### Collocation Point Generation

The `generate_uniform_grid_collocation_points()` function:
- Creates 1D arrays using `torch.linspace()` for x_star and t_star
- Uses `torch.meshgrid()` to create all combinations
- Flattens the grid to create pairs (x_star, t_star)
- Returns tensors with shape `(N, 1)` where `N = collocation_points_x_star × collocation_points_t_star`
- Points require gradients for automatic differentiation

### Training

- Collocation points are generated **once** before training starts
- Points remain fixed throughout all epochs (not resampled)
- Standard training loop with PDE residual, IC, and BC losses
- Loss history is stored for potential future analysis

## Requirements

Same as the main project:
- Python 3.8+
- PyTorch ≥2.0.0
- NumPy ≥1.24.0
- Matplotlib ≥3.7.0
- SciPy ≥1.10.0 (for analytical solution)

## See Also

- `pinn_baseline/` - Minimal baseline implementation with random collocation points
- `Experimental/pinn adaptive learning/` - Advanced implementation with adaptive learning
- `analytical_solution/` - Analytical solution for validation
