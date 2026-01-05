"""
PINN Grid Search System

Runs multiple PINN configurations, saves results per experiment, and accumulates
results across runs without overwriting previous experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.autograd import grad
import os
import sys
import json
import hashlib
import csv
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Import analytical solution from sibling folder
sys.path.append(str(Path(__file__).parent.parent / 'analytical_solution'))
from analytical_solution import analytical_solution

# Import configuration
from configs import (
    EXPERIMENT_CONFIGS,
    DEFAULT_PHYSICS_PARAMS,
    DEFAULT_TRAINING_PARAMS,
    get_activation,
    get_optimizer_class,
)

# Visualization settings
mpl.rcParams['figure.dpi'] = 800
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#FF5F05", "#13294B", "#009FD4", "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13"])

# ============================================================================
# Neural Network Architecture
# ============================================================================

class PINN(nn.Module):
    """Neural network that takes (x*, t*) and outputs dimensionless concentration C*."""
    
    def __init__(self, num_layers, num_neurons, activation):
        super(PINN, self).__init__()
        layer_sizes = [2] + [num_neurons] * num_layers + [1]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        
        # Apply Xavier normal initialization with gain to all linear layers
        gain = nn.init.calculate_gain('tanh')
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
    def forward(self, x_star, t_star):
        inputs = torch.cat([x_star, t_star], dim=1)
        return self.net(inputs)

# ============================================================================
# Helper Functions
# ============================================================================

def generate_config_hash(config):
    """
    Generate deterministic hash from configuration dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Hash string (first 8 characters of MD5 hash)
    """
    # Create a sorted, deterministic representation of the config
    # Exclude non-config keys like 'seed' if present
    config_keys = ['num_layers', 'num_neurons', 'num_epochs', 'lr', 'activation',
                   'num_collocation', 'weight_pde', 'weight_ic', 'weight_inlet_bc',
                   'weight_outlet_bc', 'optimizer', 'num_ic', 'num_bc']
    
    config_str = json.dumps({k: config.get(k) for k in config_keys if k in config}, sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()[:8]

def create_collocation_points(num_collocation, num_ic, num_bc, t_final_star):
    """
    Create collocation points for PDE, initial condition, and boundary conditions.
    
    Returns:
        Dictionary of tensors with collocation points
    """
    # PDE collocation points
    x_star_pde = np.random.rand(num_collocation, 1).astype(np.float32)
    t_star_pde = np.random.rand(num_collocation, 1).astype(np.float32) * t_final_star
    
    # Initial condition points (at t* = 0)
    x_star_ic = np.random.rand(num_ic, 1).astype(np.float32)
    t_star_ic = np.zeros((num_ic, 1), dtype=np.float32)
    
    # Inlet boundary points (at x* = 0)
    x_star_inlet = np.zeros((num_bc, 1), dtype=np.float32)
    t_star_inlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    
    # Outlet boundary points (at x* = 1)
    x_star_outlet = np.ones((num_bc, 1), dtype=np.float32)
    t_star_outlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    
    # Convert to tensors
    return {
        'x_star_pde': torch.tensor(x_star_pde, requires_grad=True),
        't_star_pde': torch.tensor(t_star_pde, requires_grad=True),
        'x_star_ic': torch.tensor(x_star_ic, requires_grad=True),
        't_star_ic': torch.tensor(t_star_ic, requires_grad=True),
        'x_star_inlet': torch.tensor(x_star_inlet, requires_grad=True),
        't_star_inlet': torch.tensor(t_star_inlet, requires_grad=True),
        'x_star_outlet': torch.tensor(x_star_outlet, requires_grad=True),
        't_star_outlet': torch.tensor(t_star_outlet, requires_grad=True),
    }

# ============================================================================
# Training Function
# ============================================================================

def train_pinn_model(config, physics_params, training_params, seed=None):
    """
    Train a PINN model with given configuration.
    
    Args:
        config: Configuration dictionary
        physics_params: Physics parameters dictionary
        training_params: Training parameters dictionary
        seed: Random seed (if None, uses hash of config)
    
    Returns:
        Dictionary with:
            - model: Trained model
            - losses: Dictionary of loss lists
            - training_time: Training time in seconds
    """
    # Set random seed
    if seed is None:
        seed = hash(generate_config_hash(config)) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Extract parameters
    U = physics_params['U']
    D = physics_params['D']
    C0 = physics_params['C0']
    L = physics_params['L']
    T_phys = physics_params['T_phys']
    
    num_layers = config['num_layers']
    num_neurons = config['num_neurons']
    num_epochs = config['num_epochs']
    lr = config['lr']
    activation = get_activation(config['activation'])
    num_collocation = config['num_collocation']
    weight_pde = config['weight_pde']
    weight_ic = config['weight_ic']
    weight_inlet_bc = config['weight_inlet_bc']
    weight_outlet_bc = config['weight_outlet_bc']
    optimizer_name = config['optimizer']
    
    # num_ic and num_bc can be in config (per-experiment) or fall back to training_params (default)
    num_ic = config.get('num_ic', training_params['num_ic'])
    num_bc = config.get('num_bc', training_params['num_bc'])
    
    # Derived parameters
    T = L / U
    Pe = (U * L) / D
    t_final_star = T_phys / T
    
    # Create model
    model = PINN(num_layers, num_neurons, activation)
    
    # Create collocation points
    colloc_points = create_collocation_points(num_collocation, num_ic, num_bc, t_final_star)
    
    # Create optimizer
    optimizer_class = get_optimizer_class(optimizer_name)
    if optimizer_name == 'LBFGS':
        optimizer = optimizer_class(model.parameters(), lr=lr, max_iter=20, max_eval=None,
                                    tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100,
                                    line_search_fn=None)
    elif optimizer_name == 'SGD':
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    
    # Initialize loss tracking
    losses = {
        'total': [],
        'pde': [],
        'ic': [],
        'inlet_bc': [],
        'outlet_bc': []
    }
    
    # Training closure
    def closure():
        optimizer.zero_grad()
        
        # PDE loss
        x_star_pde_grad = colloc_points['x_star_pde'].clone().detach().requires_grad_(True)
        t_star_pde_grad = colloc_points['t_star_pde'].clone().detach().requires_grad_(True)
        C_star_pde = model(x_star_pde_grad, t_star_pde_grad)
        dC_dt_star = grad(C_star_pde, t_star_pde_grad, grad_outputs=torch.ones_like(C_star_pde),
                          create_graph=True, retain_graph=True)[0]
        dC_dx_star = grad(C_star_pde, x_star_pde_grad, grad_outputs=torch.ones_like(C_star_pde),
                          create_graph=True, retain_graph=True)[0]
        d2C_dx2_star = grad(dC_dx_star, x_star_pde_grad, grad_outputs=torch.ones_like(dC_dx_star),
                            create_graph=True, retain_graph=True)[0]
        pde_residual = dC_dt_star + dC_dx_star - (1.0 / Pe) * d2C_dx2_star
        pde_loss = torch.mean(pde_residual**2)
        
        # Initial condition loss: C*(x*, 0) = 0
        C_star_ic = model(colloc_points['x_star_ic'], colloc_points['t_star_ic'])
        ic_loss = nn.MSELoss()(C_star_ic, torch.zeros_like(C_star_ic))
        
        # Inlet boundary condition loss: C*(0, t*) = 1
        C_star_inlet = model(colloc_points['x_star_inlet'], colloc_points['t_star_inlet'])
        inlet_loss = nn.MSELoss()(C_star_inlet, torch.ones_like(C_star_inlet))
        
        # Outlet boundary condition loss: C*(1, t*) = 0 (Dirichlet far-field approximation)
        C_star_outlet = model(colloc_points['x_star_outlet'], colloc_points['t_star_outlet'])
        outlet_loss = nn.MSELoss()(C_star_outlet, torch.zeros_like(C_star_outlet))
        
        # Total loss
        total_loss = (weight_pde * pde_loss + 
                     weight_ic * ic_loss + 
                     weight_inlet_bc * inlet_loss + 
                     weight_outlet_bc * outlet_loss)
        
        total_loss.backward()
        
        # Store losses
        losses['total'].append(total_loss.item())
        losses['pde'].append(pde_loss.item())
        losses['ic'].append(ic_loss.item())
        losses['inlet_bc'].append(inlet_loss.item())
        losses['outlet_bc'].append(outlet_loss.item())
        
        return total_loss
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.step(closure)
    training_time = time.time() - start_time
    
    return {
        'model': model,
        'losses': losses,
        'training_time': training_time,
    }

# ============================================================================
# Plotting Function
# ============================================================================

def plot_concentration_profiles(model, config, physics_params, training_params, output_path):
    """
    Plot concentration profiles comparing PINN predictions with analytical solution.
    
    Args:
        model: Trained PINN model
        config: Configuration dictionary
        physics_params: Physics parameters dictionary
        training_params: Training parameters dictionary
        output_path: Path to save the plot
    """
    U = physics_params['U']
    D = physics_params['D']
    C0 = physics_params['C0']
    L = physics_params['L']
    T_phys = physics_params['T_phys']
    
    T = L / U
    num_spatial_points = training_params['num_spatial_points']
    times_days = training_params['times_days']
    plot_dpi = training_params['plot_dpi']
    
    x_plot = np.linspace(0, L, num_spatial_points)
    plt.figure(figsize=(5, 3.5))
    
    # Add legend entries for PINN and Analytical at the beginning
    plt.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    plt.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')
    
    # Plot in reverse order so legend shows times in order
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time (cycling through colors)
        color = f'C{idx % 8}'
        
        # Convert to dimensionless
        x_star_plot = x_plot / L
        t_star_plot = t_days / T
        
        # PINN solution (solid line)
        x_star_plot_tensor = torch.tensor(x_star_plot, dtype=torch.float32).reshape(-1, 1)
        t_star_plot_tensor = torch.tensor(t_star_plot, dtype=torch.float32).reshape(-1, 1)
        if t_star_plot_tensor.shape[0] == 1 and x_star_plot_tensor.shape[0] > 1:
            t_star_plot_tensor = t_star_plot_tensor.expand(x_star_plot_tensor.shape[0], -1)
        
        model.eval()
        with torch.no_grad():
            C_star_pinn = model(x_star_plot_tensor, t_star_plot_tensor)
        C_pinn = (C_star_pinn * C0).cpu().numpy()
        plt.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Analytical solution (dashed line) with same color
        C_analytical = analytical_solution(x_plot, t_days, U_param=U, D_param=D, C_0_param=C0)
        plt.plot(x_plot, C_analytical, linewidth=2, linestyle='--', color=color, alpha=0.7)
        
        # Add marker-only entry for legend (square marker, no line)
        plt.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{t_days} days')
    
    plt.xlabel('Distance x (m)', fontsize=12)
    plt.ylabel('Concentration C (kg/m³)', fontsize=12)
    
    # Create legend at top
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                       ncol=4, frameon=False, fontsize=10,
                       labelspacing=0.5, columnspacing=1.2)
    # Style legend text items - make all text black
    for text in legend.get_texts():
        text.set_color('black')
        text.set_alpha(1.0)
    
    plt.xlim(0, L)
    plt.ylim(-0.25, C0 * 1.1)
    
    # Apply plot styling
    ax = plt.gca()
    grid_alpha = 0.3
    grid_color = 'black'
    grid_linewidth = 0.4
    
    # Apply grid - only show vertical (x-axis) grid lines, no horizontal (y-axis) grid lines
    ax.grid(True, axis='x', alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)
    
    # Remove vertical grid lines at x=0 and x=L
    x_ticks = ax.get_xticks()
    xgridlines = ax.get_xgridlines()
    indices_to_hide = []
    for i, tick_pos in enumerate(x_ticks):
        if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - L) < 1e-6:
            indices_to_hide.append(i)
    for i in indices_to_hide:
        if i < len(xgridlines):
            xgridlines[i].set_visible(False)
    
    # Hide top and right spines, show and style bottom and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['bottom'].set_alpha(grid_alpha)
    ax.spines['bottom'].set_linewidth(grid_linewidth)
    ax.spines['left'].set_color(grid_color)
    ax.spines['left'].set_alpha(grid_alpha)
    ax.spines['left'].set_linewidth(grid_linewidth)
    
    # Style tick marks - remove tick marks (length=0) but keep labels
    ax.tick_params(axis='x', which='major', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    ax.tick_params(axis='y', which='major', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    
    # Style tick labels - make all text black
    for label in ax.get_xticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    for label in ax.get_yticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    
    # Style axis labels - make all text black
    ax.xaxis.label.set_color('black')
    ax.xaxis.label.set_alpha(1.0)
    ax.yaxis.label.set_color('black')
    ax.yaxis.label.set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=plot_dpi, bbox_inches='tight')
    plt.close()

# ============================================================================
# CSV Logging Functions
# ============================================================================

def save_losses_csv(losses, output_path):
    """
    Save epoch-by-epoch losses to CSV.
    
    Args:
        losses: Dictionary with loss lists
        output_path: Path to save CSV file
    """
    num_epochs = len(losses['total'])
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'pde_loss', 'ic_loss', 'inlet_bc_loss', 'outlet_bc_loss'])
        for epoch in range(num_epochs):
            writer.writerow([
                epoch + 1,
                losses['total'][epoch],
                losses['pde'][epoch],
                losses['ic'][epoch],
                losses['inlet_bc'][epoch],
                losses['outlet_bc'][epoch],
            ])

def append_summary_csv(config, config_hash, losses, training_time, total_experiment_time, summary_path, training_params=None):
    """
    Append experiment summary to master CSV file.
    
    Args:
        config: Configuration dictionary
        config_hash: Hash string for this configuration
        losses: Dictionary with loss lists (final losses will be extracted)
        training_time: Training time in seconds
        total_experiment_time: Total experiment time in seconds (from start to finish)
        summary_path: Path to summary CSV file
        training_params: Training parameters dictionary (for fallback values)
    """
    # Extract final losses
    final_total_loss = losses['total'][-1] if losses['total'] else None
    final_pde_loss = losses['pde'][-1] if losses['pde'] else None
    final_ic_loss = losses['ic'][-1] if losses['ic'] else None
    final_inlet_bc_loss = losses['inlet_bc'][-1] if losses['inlet_bc'] else None
    final_outlet_bc_loss = losses['outlet_bc'][-1] if losses['outlet_bc'] else None
    
    # Prepare row data (order matters for CSV columns)
    row = {
        'hash': config_hash,
        'num_layers': config['num_layers'],
        'num_neurons': config['num_neurons'],
        'num_epochs': config['num_epochs'],
        'lr': config['lr'],
        'activation': config['activation'],
        'num_collocation': config['num_collocation'],
        'num_ic': config.get('num_ic', training_params.get('num_ic', None) if training_params else None),
        'num_bc': config.get('num_bc', training_params.get('num_bc', None) if training_params else None),
        'weight_pde': config['weight_pde'],
        'weight_ic': config['weight_ic'],
        'weight_inlet_bc': config['weight_inlet_bc'],
        'weight_outlet_bc': config['weight_outlet_bc'],
        'optimizer': config['optimizer'],
        'final_total_loss': final_total_loss,
        'final_pde_loss': final_pde_loss,
        'final_ic_loss': final_ic_loss,
        'final_inlet_bc_loss': final_inlet_bc_loss,
        'final_outlet_bc_loss': final_outlet_bc_loss,
        'training_time_seconds': training_time,
        'total_experiment_time_seconds': total_experiment_time,
    }
    
    # Check if file exists to determine if we need to write header
    file_exists = summary_path.exists()
    
    # Append to CSV
    df = pd.DataFrame([row])
    df.to_csv(summary_path, mode='a', header=not file_exists, index=False)

# ============================================================================
# Main Grid Search Loop
# ============================================================================

def main():
    """Main function to run grid search over all configurations."""
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    summary_csv_path = results_dir / 'experiments_summary.csv'
    
    # Get all configurations
    configs = EXPERIMENT_CONFIGS
    
    print(f"=" * 80)
    print(f"PINN Grid Search System")
    print(f"=" * 80)
    print(f"Total configurations: {len(configs)}")
    print(f"Results directory: {results_dir}")
    print(f"=" * 80)
    print()
    
    # Track progress
    completed = 0
    skipped = 0
    failed = 0
    
    # Loop through configurations
    for idx, config in enumerate(configs, 1):
        config_hash = generate_config_hash(config)
        exp_dir = results_dir / f'exp_{config_hash}'
        losses_csv_path = exp_dir / 'losses.csv'
        
        print(f"[{idx}/{len(configs)}] Configuration hash: {config_hash}")
        print(f"  Config: layers={config['num_layers']}, neurons={config['num_neurons']}, "
              f"epochs={config['num_epochs']}, lr={config['lr']}, "
              f"activation={config['activation']}, optimizer={config['optimizer']}")
        
        # Check if experiment already completed
        if losses_csv_path.exists():
            print(f"  ✓ Experiment already completed, skipping...")
            skipped += 1
            print()
            continue
        
        try:
            # Create experiment directory
            exp_dir.mkdir(exist_ok=True)
            
            # Start timing the entire experiment
            experiment_start_time = time.time()
            
            # Save configuration
            config_path = exp_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Train model
            print(f"  Training model...")
            result = train_pinn_model(config, DEFAULT_PHYSICS_PARAMS, DEFAULT_TRAINING_PARAMS)
            model = result['model']
            losses = result['losses']
            training_time = result['training_time']
            
            # Save losses CSV
            save_losses_csv(losses, losses_csv_path)
            print(f"  ✓ Losses saved to {losses_csv_path}")
            
            # Generate and save plot
            plot_path = exp_dir / 'concentration_profiles.png'
            plot_concentration_profiles(model, config, DEFAULT_PHYSICS_PARAMS, 
                                       DEFAULT_TRAINING_PARAMS, plot_path)
            print(f"  ✓ Plot saved to {plot_path}")
            
            # End timing the entire experiment
            experiment_end_time = time.time()
            total_experiment_time = experiment_end_time - experiment_start_time
            
            # Append to summary CSV
            append_summary_csv(config, config_hash, losses, training_time, total_experiment_time, summary_csv_path, DEFAULT_TRAINING_PARAMS)
            print(f"  ✓ Summary appended to {summary_csv_path}")
            
            print(f"  ✓ Experiment completed in {total_experiment_time:.2f} seconds (training: {training_time:.2f} seconds)")
            completed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()
        
        print()
    
    # Final summary
    print(f"=" * 80)
    print(f"Grid Search Complete")
    print(f"=" * 80)
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total: {len(configs)}")
    print(f"=" * 80)

if __name__ == "__main__":
    main()

