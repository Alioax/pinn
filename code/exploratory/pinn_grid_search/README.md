# PINN Grid Search

A system for automatically testing multiple PINN (Physics-Informed Neural Network) configurations to find the best hyperparameters.

## What It Does

The grid search system:
- Trains multiple PINN models with different configurations (network size, learning rate, optimizer, etc.)
- Saves results for each experiment (plots, loss data, configuration)
- Tracks computation time for each configuration
- Allows you to resume interrupted runs (skips already completed experiments)
- Accumulates all results in a master summary CSV for easy comparison

## How It Works

1. **Define configurations** in `configs.py` - add dictionaries with different hyperparameters
2. **Run the grid search**: `python pinn_grid_search.py` (from `Experimental/pinn_grid_search/`)
3. **View results**: 
   - Individual experiment results: `results/exp_<hash>/`
   - Summary of all experiments: `results/experiments_summary.csv`

## Key Features

- **Hash-based identification**: Each configuration gets a unique hash, so you can safely re-run without overwriting previous results
- **Resume capability**: Automatically skips experiments that are already completed
- **Time tracking**: Records both training time and total experiment time for computational cost analysis
- **Same methodology as baseline**: Uses identical PINN implementation as `pinn_baseline.py` for consistency

## Configuration Parameters

Each configuration dictionary includes:
- `num_layers`, `num_neurons`: Network architecture
- `num_epochs`, `lr`: Training parameters
- `activation`: Activation function (Tanh, ReLU, SiLU, etc.)
- `num_collocation`, `num_ic`, `num_bc`: Collocation point counts
- `weight_pde`, `weight_ic`, `weight_inlet_bc`, `weight_outlet_bc`: Loss weights
- `optimizer`: Optimizer type (Adam, AdamW, SGD, LBFGS, etc.)

## Output

- **Per experiment**: Concentration profile plots, epoch-by-epoch loss data, saved configuration
- **Summary CSV**: All configurations with final losses and computation times for easy analysis
