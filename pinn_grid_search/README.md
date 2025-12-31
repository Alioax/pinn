# PINN Grid Search System

A system for running multiple PINN (Physics-Informed Neural Network) configurations, saving results per experiment, and accumulating results across multiple runs without overwriting previous experiments.

## Overview

This grid search system allows you to:
- Define multiple PINN configurations in a single configuration file
- Automatically train each configuration
- Save results (plots, loss data, configurations) per experiment
- Resume interrupted runs (skips already completed experiments)
- Accumulate results across multiple runs in a master summary CSV

## Directory Structure

```
pinn_grid_search/
├── pinn_grid_search.py          # Main script with grid search logic
├── configs.py                    # Configuration definitions (list of config dicts)
├── README.md                     # This file
└── results/
    ├── experiments_summary.csv   # Master CSV with all experiments (config + final metrics)
    ├── exp_<hash>/
    │   ├── concentration_profiles.png
    │   ├── losses.csv            # Epoch-by-epoch losses
    │   └── config.json           # Saved configuration for reference
    └── ...
```

## Quick Start

1. **Edit configurations** in `configs.py`:
   ```python
   EXPERIMENT_CONFIGS = [
       {
           'num_layers': 1,
           'num_neurons': 64,
           'num_epochs': 4000,
           'lr': 0.001,
           'activation': 'Tanh',
           'num_collocation': 200000,
           'weight_pde': 1,
           'weight_ic': 1,
           'weight_inlet_bc': 1,
           'weight_outlet_bc': 1,
           'optimizer': 'Adam',
       },
       # ... more configs
   ]
   ```

2. **Run the grid search**:
   ```bash
   cd pinn_grid_search
   python pinn_grid_search.py
   ```

3. **View results**:
   - Individual experiment results: `results/exp_<hash>/`
   - Summary of all experiments: `results/experiments_summary.csv`

## Configuration Parameters

Each configuration dictionary can contain:

### Required Parameters

- **`num_layers`** (int): Number of hidden layers in the neural network
- **`num_neurons`** (int): Number of neurons per hidden layer
- **`num_epochs`** (int): Number of training epochs
- **`lr`** (float): Learning rate
- **`activation`** (str): Activation function name (see available options below)
- **`num_collocation`** (int): Number of collocation points for PDE residual
- **`weight_pde`** (float): Weight for PDE residual loss
- **`weight_ic`** (float): Weight for initial condition loss
- **`weight_inlet_bc`** (float): Weight for inlet boundary condition loss
- **`weight_outlet_bc`** (float): Weight for outlet boundary condition loss
- **`optimizer`** (str): Optimizer name (see available options below)

### Available Activation Functions

- `'Tanh'` - Hyperbolic tangent (default, smooth and bounded)
- `'ReLU'` - Rectified Linear Unit
- `'SiLU'` - Swish/Sigmoid Linear Unit (smooth, often works well for PINNs)
- `'GELU'` - Gaussian Error Linear Unit
- `'ELU'` - Exponential Linear Unit
- `'LeakyReLU'` - Leaky ReLU
- `'Sigmoid'` - Sigmoid
- `'Softplus'` - Smooth approximation of ReLU

### Available Optimizers

- `'Adam'` - Adaptive Moment Estimation (default)
- `'AdamW'` - Adam with weight decay
- `'SGD'` - Stochastic Gradient Descent with momentum
- `'Rprop'` - Resilient backpropagation
- `'LBFGS'` - Limited-memory BFGS

## Output Files

### Per-Experiment Directory (`results/exp_<hash>/`)

Each experiment gets its own directory identified by a unique hash:

- **`concentration_profiles.png`**: Plot comparing PINN predictions with analytical solution at multiple time steps
- **`losses.csv`**: Epoch-by-epoch loss data with columns:
  - `epoch`: Epoch number (1-indexed)
  - `total_loss`: Total weighted loss
  - `pde_loss`: PDE residual loss
  - `ic_loss`: Initial condition loss
  - `inlet_bc_loss`: Inlet boundary condition loss
  - `outlet_bc_loss`: Outlet boundary condition loss
- **`config.json`**: Saved configuration dictionary for reference

### Summary CSV (`results/experiments_summary.csv`)

Master CSV file that accumulates results across all experiments with columns:
- `hash`: Unique experiment identifier
- `num_layers`, `num_neurons`, `num_epochs`, `lr`, `activation`, `num_collocation`
- `weight_pde`, `weight_ic`, `weight_inlet_bc`, `weight_outlet_bc`
- `optimizer`
- `final_total_loss`, `final_pde_loss`, `final_ic_loss`, `final_inlet_bc_loss`, `final_outlet_bc_loss`
- `training_time_seconds`

## Features

### Hash-Based Identification

Each configuration is assigned a unique hash based on its parameters. This ensures:
- Consistent identification across runs
- Easy resume capability
- No overwriting of previous experiments

### Resume Capability

The system automatically checks if an experiment has already been completed (by checking for `losses.csv`). If found, it skips that configuration and moves to the next one. This allows you to:
- Interrupt and resume long-running grid searches
- Add new configurations without re-running existing ones
- Re-run the script safely without losing previous results

### Progress Tracking

The script provides:
- Overall progress: `[current/total]` for each configuration
- Per-experiment training progress (via tqdm if available)
- Final summary with counts of completed, skipped, and failed experiments

### Error Handling

If an experiment fails, the error is logged and the script continues with the next configuration. Failed experiments are counted in the final summary.

## Example Usage

### Basic Grid Search

```python
# In configs.py
EXPERIMENT_CONFIGS = [
    {
        'num_layers': 1,
        'num_neurons': 64,
        'num_epochs': 1000,  # Shorter for testing
        'lr': 0.001,
        'activation': 'Tanh',
        'num_collocation': 50000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },
    {
        'num_layers': 2,
        'num_neurons': 64,
        'num_epochs': 1000,
        'lr': 0.001,
        'activation': 'Tanh',
        'num_collocation': 50000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },
]
```

### Analyzing Results

After running the grid search, you can analyze results using pandas:

```python
import pandas as pd

# Load summary
df = pd.read_csv('results/experiments_summary.csv')

# Find best configuration by final total loss
best = df.loc[df['final_total_loss'].idxmin()]
print(f"Best configuration: {best['hash']}")
print(f"Final loss: {best['final_total_loss']}")

# Compare different architectures
arch_comparison = df.groupby(['num_layers', 'num_neurons'])['final_total_loss'].mean()
print(arch_comparison)

# Compare optimizers
optimizer_comparison = df.groupby('optimizer')['final_total_loss'].mean()
print(optimizer_comparison)
```

## Physics Parameters

The physics parameters (advection velocity, dispersion coefficient, etc.) are defined in `configs.py` as `DEFAULT_PHYSICS_PARAMS`:

```python
DEFAULT_PHYSICS_PARAMS = {
    'U': 0.1,                    # m/day (advection velocity)
    'D': 1e-7 * 86400,          # m²/day (dispersion coefficient)
    'C0': 5.0,                   # kg/m³ (inlet concentration)
    'L': 100.0,                  # m (domain length)
    'T_phys': 1000.0,            # days (physical time horizon)
}
```

These can be modified in `configs.py` if needed, or you can extend the system to allow per-configuration physics parameters.

## Dependencies

- Python 3.8+
- PyTorch ≥2.0.0
- NumPy ≥1.24.0
- Matplotlib ≥3.7.0
- pandas (for CSV handling)
- tqdm (for progress bars)

The system imports the analytical solution from `../analytical_solution/analytical_solution.py` (read-only).

## Notes

- **Baseline preservation**: The `pinn_baseline/` directory remains completely untouched. All code is self-contained in `pinn_grid_search/`.
- **Deterministic hashing**: Hash generation is deterministic, so the same configuration always gets the same hash.
- **Random seeds**: Each experiment uses a seed based on its configuration hash for reproducibility.
- **Memory usage**: Models are trained sequentially to avoid memory issues with large grid searches.

## Troubleshooting

### Import Errors

If you get import errors for the analytical solution, make sure you're running from the `pinn_grid_search/` directory and that the `analytical_solution/` directory exists at the parent level.

### Out of Memory

If you run out of memory:
- Reduce `num_collocation` in configurations
- Reduce `num_epochs` for testing
- Run fewer configurations at once

### Slow Training

Training time depends on:
- `num_epochs`: More epochs = longer training
- `num_collocation`: More points = longer per epoch
- Network size: Larger networks = longer per epoch

Consider starting with smaller configurations for testing.

