"""
Configuration file for PINN Grid Search

Define your experiment configurations here. Each configuration dictionary will be
used to train a PINN model with the specified hyperparameters.
"""

import torch.nn as nn
import torch.optim as optim

# ============================================================================
# Helper Functions for Activation and Optimizer Mapping
# ============================================================================

def get_activation(activation_name):
    """
    Convert activation function name (string) to PyTorch activation class.
    
    Args:
        activation_name: String name of activation function (e.g., 'Tanh', 'ReLU', 'SiLU')
    
    Returns:
        PyTorch activation class
    """
    activation_map = {
        'Tanh': nn.Tanh,
        'ReLU': nn.ReLU,
        'SiLU': nn.SiLU,
        'GELU': nn.GELU,
        'ELU': nn.ELU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Softplus': nn.Softplus,
    }
    
    if activation_name in activation_map:
        return activation_map[activation_name]
    else:
        raise ValueError(f"Unknown activation function: {activation_name}. "
                        f"Available options: {list(activation_map.keys())}")

def get_optimizer_class(optimizer_name):
    """
    Convert optimizer name (string) to PyTorch optimizer class.
    
    Args:
        optimizer_name: String name of optimizer (e.g., 'Adam', 'AdamW', 'SGD')
    
    Returns:
        PyTorch optimizer class
    """
    optimizer_map = {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'SGD': optim.SGD,
        'Rprop': optim.Rprop,
        'LBFGS': optim.LBFGS,
    }
    
    if optimizer_name in optimizer_map:
        return optimizer_map[optimizer_name]
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Available options: {list(optimizer_map.keys())}")

# ============================================================================
# Default Physics Parameters
# ============================================================================

# These can be overridden in individual configs if needed
DEFAULT_PHYSICS_PARAMS = {
    'U': 0.1,                    # m/day (advection velocity)
    'D': 1e-7 * 86400,          # m²/day (dispersion coefficient)
    'C0': 5.0,                   # kg/m³ (inlet concentration)
    'L': 100.0,                  # m (domain length)
    'T_phys': 1000.0,            # days (physical time horizon)
}

# ============================================================================
# Default Training Parameters (used if not specified in config)
# ============================================================================

DEFAULT_TRAINING_PARAMS = {
    'num_ic': 20000,             # number of points for initial condition
    'num_bc': 20000,             # number of points for boundary conditions
    'num_spatial_points': 500,   # number of spatial points for plotting
    'plot_dpi': 800,             # DPI for saved plots
    'times_days': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # times to plot
}

# ============================================================================
# Experiment Configurations
# ============================================================================

EXPERIMENT_CONFIGS = [
    # Baseline (time-efficient reference)
    {
        'num_layers': 1,
        'num_neurons': 64,
        'num_epochs': 20000,
        'lr': 0.01,
        'activation': 'Tanh',
        'num_collocation': 200000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    }

    # Compact depth vs width: 2×32 baseline LR
    {
        'num_layers': 2,
        'num_neurons': 32,
        'num_epochs': 20000,
        'lr': 0.001,
        'activation': 'Tanh',
        'num_collocation': 200000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },

    # Compact depth vs width: 2×32 lower LR
    {
        'num_layers': 2,
        'num_neurons': 32,
        'num_epochs': 20000,
        'lr': 0.0005,
        'activation': 'Tanh',
        'num_collocation': 200000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },

    # Simple architecture: 1×64 with mild PDE emphasis
    {
        'num_layers': 1,
        'num_neurons': 64,
        'num_epochs': 20000,
        'lr': 0.001,
        'activation': 'Tanh',
        'num_collocation': 200000,
        'weight_pde': 2,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },

    # Simple architecture: 1×64 with fewer collocation points
    {
        'num_layers': 1,
        'num_neurons': 64,
        'num_epochs': 20000,
        'lr': 0.001,
        'activation': 'Tanh',
        'num_collocation': 100000,
        'weight_pde': 1,
        'weight_ic': 1,
        'weight_inlet_bc': 1,
        'weight_outlet_bc': 1,
        'optimizer': 'Adam',
    },
]


