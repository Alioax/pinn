"""
Physics-Informed Neural Network (PINN) for 1D Contaminant Transport in an Aquifer
UNIFORM GRID COLLOCATION POINTS - Foundation for Adaptive Learning

This implementation:
- Trains the PINN entirely in dimensionless form
- Uses uniform linearly-spaced collocation points in a grid structure
- Accepts inputs in physical units (meters, days)
- Converts to dimensionless variables before evaluation
- Converts dimensionless output back to dimensional concentration
- Produces plots in physical units
- Compares PINN solution with analytical solution

Key Feature: Separate specification of collocation_points_x_star and collocation_points_t_star
for creating a uniform grid of collocation points.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from torch.autograd import grad
import os
import sys
from pathlib import Path

# Import analytical solution from root-level folder
sys.path.append(str(Path(__file__).parent.parent.parent / 'analytical_solution'))
from analytical_solution import analytical_solution

# Visualization settings
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#FF5F05", "#13294B", "#009FD4", "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13"])

# ============================================================================
# Configuration Parameters
# ============================================================================

# Physics parameters (dimensional)
physics_params = {
    'U': 0.1,                    # m/day (advection velocity)
    'D': 1e-7 * 86400,          # m²/day (dispersion coefficient, converted from m²/s)
    'C_0': 5.0,                 # kg/m³ (concentration scale)
    'L': 100.0,                 # m (length scale)
}

# Neural network model parameters
model_params = {
    'num_layers': 3,            # number of hidden layers
    'num_neurons': 10,          # number of neurons per hidden layer
    'activation': torch.nn.Tanh,  # activation function
}

# Training parameters
training_params = {
    'num_epochs': 10000,         # number of training epochs
    'lr': 0.005,                # learning rate
    # Collocation point configuration - UNIFORM GRID
    'collocation_points_x_star': 100,  # Number of uniformly-spaced points in x* direction
    'collocation_points_t_star': 25,  # Number of uniformly-spaced points in t* direction
    # Total collocation points = collocation_points_x_star × collocation_points_t_star = 2000
    'num_ic': 100,              # number of points for initial condition
    'num_bc': 100,              # number of points for boundary conditions
    't_final_star': 1.0,        # final dimensionless time
    'verbose': True,            # print training progress
    'export_interval': 100,     # export plot every N epochs (set to None to disable)
    # Loss weights (balance different loss components)
    'weight_pde': 1,            # weight for PDE residual loss
    'weight_ic': 1,             # weight for initial condition loss
    'weight_inlet_bc': 1,       # weight for inlet boundary condition loss
    'weight_outlet_bc': 1,      # weight for outlet boundary condition loss
}

# Plotting parameters
script_dir = Path(__file__).parent
plots_dir = script_dir / 'results'
plotting_params = {
    'times_days': [300, 500, 800],  # times for concentration profiles
    'x_max': 100.0,             # maximum spatial coordinate (m)
    'num_points': 500,          # number of spatial points for profiles
    'dpi': 300,                # resolution for saved figures
    'plots_dir': str(plots_dir),  # directory to save plots
}

# Create plots directory if it doesn't exist
os.makedirs(plotting_params['plots_dir'], exist_ok=True)

# ============================================================================
# Derived Parameters (computed from physics_params)
# ============================================================================

# Extract physics parameters
U = physics_params['U']
D = physics_params['D']
C_0 = physics_params['C_0']
L = physics_params['L']

# Compute derived parameters
T = L / U  # days (time scale, advective)
Pe = (U * L) / D  # Péclet number (dimensionless)

# Print configuration
print("="*60)
print("Physical Parameters")
print("="*60)
print(f"Advection velocity: u = {U} m/day")
print(f"Dispersion coefficient: D = {D:.6e} m²/day")
print(f"Concentration scale: C₀ = {C_0} kg/m³")
print(f"Length scale: L = {L} m")
print(f"Time scale: T = {T:.1f} days")
print(f"Péclet number: Pe = {Pe:.2f}")
print("="*60)

# ============================================================================
# Dimensionless Conversion Functions
# ============================================================================

def to_dimensionless(x, t):
    """
    Convert dimensional inputs to dimensionless variables.
    
    Parameters:
        x: spatial coordinate (m) - can be tensor or array
        t: time (days) - can be tensor or array
    
    Returns:
        x_star: dimensionless spatial coordinate
        t_star: dimensionless time
    """
    x_star = x / L
    t_star = t / T
    return x_star, t_star

def from_dimensionless(x_star, t_star):
    """
    Convert dimensionless variables back to dimensional.
    
    Parameters:
        x_star: dimensionless spatial coordinate
        t_star: dimensionless time
    
    Returns:
        x: spatial coordinate (m)
        t: time (days)
    """
    x = x_star * L
    t = t_star * T
    return x, t

def concentration_to_dimensionless(C):
    """
    Convert dimensional concentration to dimensionless.
    
    Parameters:
        C: concentration (kg/m³)
    
    Returns:
        C_star: dimensionless concentration
    """
    return C / C_0

def concentration_from_dimensionless(C_star):
    """
    Convert dimensionless concentration to dimensional.
    
    Parameters:
        C_star: dimensionless concentration
    
    Returns:
        C: concentration (kg/m³)
    """
    return C_star * C_0

# ============================================================================
# Neural Network Architecture
# ============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for dimensionless contaminant transport.
    
    The network takes dimensionless inputs (x*, t*) and outputs dimensionless
    concentration C*.
    """
    
    def __init__(self, num_layers=3, num_neurons=10, activation=nn.Tanh):
        """
        Initialize the PINN.
        
        Parameters:
            num_layers: number of hidden layers
            num_neurons: number of neurons per hidden layer
            activation: activation function
        """
        super(PINN, self).__init__()
        
        # Input layer: (x*, t*) -> hidden
        layers = [nn.Linear(2, num_neurons), activation()]
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation())
        
        # Output layer: hidden -> C*
        layers.append(nn.Linear(num_neurons, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x_star, t_star):
        """
        Forward pass: compute dimensionless concentration C*.
        
        Parameters:
            x_star: dimensionless spatial coordinate (tensor)
            t_star: dimensionless time (tensor)
        
        Returns:
            C_star: dimensionless concentration (tensor)
        """
        # Ensure inputs are tensors
        if not isinstance(x_star, torch.Tensor):
            x_star = torch.tensor(x_star, dtype=torch.float32)
        if not isinstance(t_star, torch.Tensor):
            t_star = torch.tensor(t_star, dtype=torch.float32)
        
        # Ensure they have the same shape for concatenation
        x_star = x_star.flatten()
        t_star = t_star.flatten()
        
        if x_star.shape != t_star.shape:
            # Broadcast to same shape if needed
            if x_star.numel() == 1:
                x_star = x_star.expand_as(t_star)
            elif t_star.numel() == 1:
                t_star = t_star.expand_as(x_star)
            else:
                raise ValueError(f"Shape mismatch: x_star.shape={x_star.shape}, t_star.shape={t_star.shape}")
        
        # Concatenate inputs
        inputs = torch.cat([x_star.reshape(-1, 1), t_star.reshape(-1, 1)], dim=1)
        
        # Network output
        C_star = self.net(inputs)
        
        return C_star

# ============================================================================
# Loss Functions (All in Dimensionless Form)
# ============================================================================

def compute_pde_residual(model, x_star, t_star):
    """
    Compute the PDE residual in dimensionless form.
    
    Dimensionless PDE: ∂C*/∂t* + ∂C*/∂x* - (1/Pe)*∂²C*/∂x*² = 0
    
    Parameters:
        model: PINN model
        x_star: dimensionless spatial coordinate (tensor, requires_grad=True)
        t_star: dimensionless time (tensor, requires_grad=True)
    
    Returns:
        residual: PDE residual (tensor)
    """
    # Ensure inputs require gradients
    x_star = x_star.clone().detach().requires_grad_(True)
    t_star = t_star.clone().detach().requires_grad_(True)
    
    # Forward pass
    C_star = model(x_star, t_star)
    
    # Compute gradients
    dC_dt_star = grad(C_star, t_star, grad_outputs=torch.ones_like(C_star),
                      create_graph=True, retain_graph=True)[0]
    
    dC_dx_star = grad(C_star, x_star, grad_outputs=torch.ones_like(C_star),
                      create_graph=True, retain_graph=True)[0]
    
    # Compute second derivative
    d2C_dx2_star = grad(dC_dx_star, x_star, grad_outputs=torch.ones_like(dC_dx_star),
                        create_graph=True, retain_graph=True)[0]
    
    # PDE residual: ∂C*/∂t* + ∂C*/∂x* - (1/Pe)*∂²C*/∂x*²
    residual = dC_dt_star + dC_dx_star - (1.0 / Pe) * d2C_dx2_star
    
    return residual

def compute_initial_condition_loss(model, x_star_init):
    """
    Compute initial condition loss: C*(x*, 0) = 0
    
    Parameters:
        model: PINN model
        x_star_init: dimensionless spatial coordinates at t*=0
    
    Returns:
        loss: MSE loss for initial condition
    """
    t_star_init = torch.zeros_like(x_star_init)
    C_star_pred = model(x_star_init, t_star_init)
    C_star_true = torch.zeros_like(C_star_pred)
    return nn.MSELoss()(C_star_pred, C_star_true)

def compute_boundary_condition_losses(model, t_star_bc, x_star_inlet=0.0, x_star_outlet=1.0):
    """
    Compute boundary condition losses.
    
    Boundary conditions:
    - Inlet (x*=0): C*(0, t*) = 1 (Dirichlet)
    - Outlet (x*=1): ∂C*/∂x*(1, t*) = 0 (zero-gradient)
    
    Parameters:
        model: PINN model
        t_star_bc: dimensionless time at boundaries
        x_star_inlet: dimensionless inlet position (default 0.0)
        x_star_outlet: dimensionless outlet position (default 1.0)
    
    Returns:
        inlet_loss: Dirichlet BC loss at inlet
        outlet_loss: Neumann BC loss at outlet
    """
    # Inlet boundary condition: C*(0, t*) = 1
    x_star_inlet_tensor = torch.full_like(t_star_bc, x_star_inlet)
    C_star_inlet_pred = model(x_star_inlet_tensor, t_star_bc)
    C_star_inlet_true = torch.ones_like(C_star_inlet_pred)
    inlet_loss = nn.MSELoss()(C_star_inlet_pred, C_star_inlet_true)
    
    # Outlet boundary condition: ∂C*/∂x*(1, t*) = 0
    x_star_outlet_tensor = torch.full_like(t_star_bc, x_star_outlet)
    x_star_outlet_tensor = x_star_outlet_tensor.clone().detach().requires_grad_(True)
    t_star_outlet_tensor = t_star_bc.clone().detach().requires_grad_(True)
    
    C_star_outlet = model(x_star_outlet_tensor, t_star_outlet_tensor)
    dC_dx_star_outlet = grad(C_star_outlet, x_star_outlet_tensor,
                             grad_outputs=torch.ones_like(C_star_outlet),
                             create_graph=True, retain_graph=True)[0]
    outlet_loss = nn.MSELoss()(dC_dx_star_outlet, torch.zeros_like(dC_dx_star_outlet))
    
    return inlet_loss, outlet_loss

# ============================================================================
# Uniform Grid Collocation Point Generation
# ============================================================================

def generate_uniform_grid_collocation_points(collocation_points_x_star, collocation_points_t_star, t_final_star):
    """
    Generate uniformly-spaced collocation points in a grid structure.
    
    Creates a 2D grid by combining all combinations of x_star and t_star values.
    Total number of points = collocation_points_x_star × collocation_points_t_star
    
    Parameters:
        collocation_points_x_star: number of uniformly-spaced points in x* direction
        collocation_points_t_star: number of uniformly-spaced points in t* direction
        t_final_star: final dimensionless time
    
    Returns:
        x_star_colloc: uniformly-spaced dimensionless spatial coordinates (tensor, shape: [N, 1])
        t_star_colloc: uniformly-spaced dimensionless time coordinates (tensor, shape: [N, 1])
        where N = collocation_points_x_star × collocation_points_t_star
    """
    # Create 1D arrays of uniformly-spaced points
    x_star_1d = torch.linspace(0, 1, collocation_points_x_star)
    t_star_1d = torch.linspace(0, t_final_star, collocation_points_t_star)
    
    # Create meshgrid to get all combinations
    x_star_grid, t_star_grid = torch.meshgrid(x_star_1d, t_star_1d, indexing='ij')
    
    # Flatten to create pairs: (x_star, t_star) for each grid point
    x_star_colloc = x_star_grid.flatten().reshape(-1, 1)
    t_star_colloc = t_star_grid.flatten().reshape(-1, 1)
    
    # Ensure points require gradients for training
    x_star_colloc = x_star_colloc.requires_grad_(True)
    t_star_colloc = t_star_colloc.requires_grad_(True)
    
    return x_star_colloc, t_star_colloc

# ============================================================================
# Training Function
# ============================================================================

def train_pinn(model, training_params=None, plot_callback=None):
    """
    Train the PINN model with uniform grid collocation points.
    
    Parameters:
        model: PINN model
        training_params: dictionary with training parameters
        plot_callback: optional function(model, epoch) called every 100 epochs for plotting
    
    Returns:
        losses: dictionary of loss history
    """
    if training_params is None:
        training_params = globals()['training_params']
    
    num_epochs = training_params['num_epochs']
    lr = training_params['lr']
    num_ic = training_params['num_ic']
    num_bc = training_params['num_bc']
    t_final_star = training_params['t_final_star']
    verbose = training_params['verbose']
    collocation_points_x_star = training_params['collocation_points_x_star']
    collocation_points_t_star = training_params['collocation_points_t_star']
    
    # Loss weights
    weight_pde = training_params.get('weight_pde', 1.0)
    weight_ic = training_params.get('weight_ic', 1.0)
    weight_inlet_bc = training_params.get('weight_inlet_bc', 1.0)
    weight_outlet_bc = training_params.get('weight_outlet_bc', 1.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss history
    losses = {
        'total': [],
        'pde': [],
        'ic': [],
        'inlet_bc': [],
        'outlet_bc': []
    }
    
    # Generate uniform grid collocation points ONCE (fixed throughout training)
    x_star_colloc, t_star_colloc = generate_uniform_grid_collocation_points(
        collocation_points_x_star, collocation_points_t_star, t_final_star
    )
    
    total_collocation_points = collocation_points_x_star * collocation_points_t_star
    
    if verbose:
        print(f"\nCollocation points configuration:")
        print(f"  x_star points: {collocation_points_x_star}")
        print(f"  t_star points: {collocation_points_t_star}")
        print(f"  Total collocation points: {total_collocation_points}")
        print(f"  Points are fixed (uniform grid, not resampled each epoch)")
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE residual loss (using fixed uniform grid collocation points)
        pde_residual = compute_pde_residual(model, x_star_colloc, t_star_colloc)
        pde_loss = torch.mean(pde_residual**2)
        
        # Initial condition loss
        x_star_ic = torch.rand(num_ic, 1) * 1.0  # [0, 1]
        ic_loss = compute_initial_condition_loss(model, x_star_ic)
        
        # Boundary condition losses
        t_star_bc = torch.rand(num_bc, 1) * t_final_star  # [0, t_final_star]
        inlet_bc_loss, outlet_bc_loss = compute_boundary_condition_losses(model, t_star_bc)
        
        # Total loss with weights
        total_loss = (weight_pde * pde_loss + 
                     weight_ic * ic_loss + 
                     weight_inlet_bc * inlet_bc_loss + 
                     weight_outlet_bc * outlet_bc_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        losses['total'].append(total_loss.item())
        losses['pde'].append(pde_loss.item())
        losses['ic'].append(ic_loss.item())
        losses['inlet_bc'].append(inlet_bc_loss.item())
        losses['outlet_bc'].append(outlet_bc_loss.item())
        
        # Print progress
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Total: {total_loss.item():.6e} | "
                  f"PDE: {pde_loss.item():.6e} | "
                  f"IC: {ic_loss.item():.6e} | "
                  f"Inlet BC: {inlet_bc_loss.item():.6e} | "
                  f"Outlet BC: {outlet_bc_loss.item():.6e}")
        
        # Export plot at specified interval
        export_interval = training_params.get('export_interval', 100)
        if plot_callback is not None and export_interval is not None and (epoch + 1) % export_interval == 0:
            plot_callback(model, epoch + 1)
    
    return losses

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_dimensional(model, x, t):
    """
    Evaluate the PINN model with dimensional inputs and return dimensional output.
    
    Parameters:
        model: trained PINN model
        x: spatial coordinate (m) - can be array or scalar
        t: time (days) - can be array or scalar
    
    Returns:
        C: concentration (kg/m³)
    """
    # Convert to dimensionless
    x_star, t_star = to_dimensionless(x, t)
    
    # Convert to numpy arrays first to handle broadcasting
    if isinstance(x_star, torch.Tensor):
        x_star = x_star.cpu().numpy()
    if isinstance(t_star, torch.Tensor):
        t_star = t_star.cpu().numpy()
    
    x_star = np.asarray(x_star)
    t_star = np.asarray(t_star)
    
    # Handle broadcasting: if one is scalar and the other is array, broadcast scalar
    if x_star.ndim == 0 and t_star.ndim > 0:
        x_star = np.broadcast_to(x_star, t_star.shape)
    elif t_star.ndim == 0 and x_star.ndim > 0:
        t_star = np.broadcast_to(t_star, x_star.shape)
    elif x_star.ndim == 0 and t_star.ndim == 0:
        # Both scalars - ensure they're 1D
        x_star = x_star.reshape(1)
        t_star = t_star.reshape(1)
    elif x_star.shape != t_star.shape:
        # If shapes don't match and neither is scalar, try to broadcast
        x_star, t_star = np.broadcast_arrays(x_star, t_star)
    
    # Convert to tensors
    x_star = torch.tensor(x_star, dtype=torch.float32)
    t_star = torch.tensor(t_star, dtype=torch.float32)
    
    # Ensure they're 1D for concatenation
    x_star = x_star.flatten()
    t_star = t_star.flatten()
    
    # Evaluate model (dimensionless)
    model.eval()
    with torch.no_grad():
        C_star = model(x_star, t_star)
    
    # Convert to dimensional
    C = concentration_from_dimensionless(C_star)
    
    # Convert to numpy if needed
    if isinstance(C, torch.Tensor):
        C = C.cpu().numpy()
    
    return C

# ============================================================================
# Plotting Function
# ============================================================================

def plot_concentration_profiles(model, times_days=None, x_max=None, num_points=None, 
                                training_params=None, model_params=None, physics_params=None,
                                epoch=None, output_dir=None, filename=None):
    """
    Plot concentration profiles at selected physical times.
    Shows both PINN solution (solid) and analytical solution (dashed).
    
    Parameters:
        model: trained PINN model
        times_days: list of times in days (default: from plotting_params)
        x_max: maximum spatial coordinate (m) (default: from plotting_params)
        num_points: number of spatial points (default: from plotting_params)
        training_params: dictionary with training parameters (for display)
        model_params: dictionary with model parameters (for display)
        physics_params: dictionary with physics parameters (for display)
        epoch: epoch number to display (optional)
        output_dir: directory to save plot (default: plotting_params['plots_dir'])
        filename: filename for plot (default: auto-generated)
    """
    if times_days is None:
        times_days = plotting_params['times_days']
    if x_max is None:
        x_max = plotting_params['x_max']
    if num_points is None:
        num_points = plotting_params['num_points']
    if training_params is None:
        training_params = globals()['training_params']
    if model_params is None:
        model_params = globals()['model_params']
    if physics_params is None:
        physics_params = globals()['physics_params']
    
    x_plot = np.linspace(0, x_max, num_points)
    
    # Create figure with wider width to accommodate parameter panel
    fig = plt.figure(figsize=(8, 3.5))
    # Create main axes for the plot (left side, taking ~70% of width)
    ax = fig.add_axes([0.1, 0.15, 0.55, 0.75])  # [left, bottom, width, height]
    
    # Get current color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Add legend entries for PINN and Analytical at the beginning
    ax.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    ax.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')
    
    # Plot in reverse order so legend shows times in order
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time (cycling through colors)
        color = colors[idx % len(colors)]
        
        # PINN solution (solid line)
        C_pinn = evaluate_dimensional(model, x_plot, t_days)
        ax.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Analytical solution (dashed line) with same color
        C_analytical = analytical_solution(x_plot, t_days)
        ax.plot(x_plot, C_analytical, linewidth=2, linestyle='--', color=color, alpha=0.7)
        
        # Add marker-only entry for legend (square marker, no line)
        ax.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{t_days} days')
    
    ax.set_xlabel('Distance x (m)', fontsize=12)
    ax.set_ylabel('Concentration C (kg/m³)', fontsize=12)
    
    # Create legend at top
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                       ncol=4, frameon=False, fontsize=10,
                       labelspacing=0.5, columnspacing=1.2)
    # Style legend text items - make all text black
    for text in legend.get_texts():
        text.set_color('black')
        text.set_alpha(1.0)
    
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.25, C_0 * 1.1)
    
    # Apply plot styling
    grid_alpha = 0.3
    grid_color = 'black'
    grid_linewidth = 0.4
    
    # Apply grid - only show vertical (x-axis) grid lines, no horizontal (y-axis) grid lines
    ax.grid(True, axis='x', alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)
    
    # Remove vertical grid lines at x=0 and x=x_max
    x_ticks = ax.get_xticks()
    xgridlines = ax.get_xgridlines()
    indices_to_hide = []
    for i, tick_pos in enumerate(x_ticks):
        if abs(tick_pos - 0.0) < 1e-6 or abs(tick_pos - x_max) < 1e-6:
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
    
    # Add parameter panel on the right side
    # Align top with legend: axes is at [0.1, 0.15, 0.55, 0.75], so top is at 0.9
    # Legend extends above axes, so adjust parameter panel to align top
    # Keep same height but adjust bottom to align top with legend
    param_ax = fig.add_axes([0.68, 0.15, 0.30, 0.75])  # Right side panel
    param_ax.axis('off')  # Hide axes
    
    # Prepare parameter text
    collocation_points_x_star = training_params['collocation_points_x_star']
    collocation_points_t_star = training_params['collocation_points_t_star']
    total_collocation = collocation_points_x_star * collocation_points_t_star
    
    # Count model parameters
    num_model_params = sum(p.numel() for p in model.parameters())
    
    # Format activation function name
    activation_name = model_params['activation'].__name__
    
    # Build parameter text with bold headings (no separator lines)
    param_text = []
    headings = []  # Track which lines are headings for bold formatting
    
    # Training Parameters (bold heading)
    param_text.append("Training Parameters")
    headings.append(0)
    param_text.append("")
    param_text.append(f"Epochs: {training_params['num_epochs']:,}")
    param_text.append(f"Learning Rate: {training_params['lr']}")
    param_text.append("")
    
    # Network Configuration (bold heading)
    param_text.append("Network Configuration")
    headings.append(len(param_text) - 1)
    param_text.append("")
    param_text.append(f"Hidden Layers: {model_params['num_layers']}")
    param_text.append(f"Neurons/Layer: {model_params['num_neurons']}")
    param_text.append(f"Activation: {activation_name}")
    param_text.append(f"Total Parameters: {num_model_params:,}")
    param_text.append("")
    
    # Collocation Points (bold heading)
    param_text.append("Collocation Points")
    headings.append(len(param_text) - 1)
    param_text.append("")
    param_text.append(f"x* Points: {collocation_points_x_star}")
    param_text.append(f"t* Points: {collocation_points_t_star}")
    param_text.append(f"Total: {total_collocation:,}")
    param_text.append("")
    
    # Boundary Conditions (bold heading)
    param_text.append("Boundary Conditions")
    headings.append(len(param_text) - 1)
    param_text.append("")
    param_text.append(f"IC Points: {training_params['num_ic']}")
    param_text.append(f"BC Points: {training_params['num_bc']}")
    param_text.append("")
    
    # Loss Weights (bold heading)
    param_text.append("Loss Weights")
    headings.append(len(param_text) - 1)
    param_text.append("")
    param_text.append(f"PDE: {training_params['weight_pde']}")
    param_text.append(f"IC: {training_params['weight_ic']}")
    param_text.append(f"Inlet BC: {training_params['weight_inlet_bc']}")
    param_text.append(f"Outlet BC: {training_params['weight_outlet_bc']}")
    
    # Display epoch text outside the box (above it) if provided
    epoch_fontsize = 14  # Larger font size for epoch display
    y_start = 0.98  # Start near top, aligned with legend top
    
    if epoch is not None:
        # Display epoch text above the parameter box
        epoch_y_pos = y_start + 0.2  # Position above the box
        param_ax.text(0.05, epoch_y_pos, f"Epoch: {epoch:,}", transform=param_ax.transAxes,
                     fontsize=epoch_fontsize, verticalalignment='top', fontfamily='monospace',
                     weight='bold', color='black')
    
    # Display parameters with bold headings
    # Use separate text calls for headings vs content to make headings bold
    fontsize = 8  # Smaller font size
    heading_line_height = 0.024  # Line height for headings (keep compact)
    content_line_height = 0.04125  # Larger line height for content lines (more spacing, ~10% more)
    
    y_pos = y_start
    for i, line in enumerate(param_text):
        if i in headings:
            # Bold heading
            param_ax.text(0.05, y_pos, line, transform=param_ax.transAxes,
                         fontsize=fontsize, verticalalignment='top', fontfamily='monospace',
                         weight='bold', color='black')
            y_pos -= heading_line_height
        elif line == "":
            # Empty line - use standard spacing
            y_pos -= content_line_height * 0.7  # Slightly less for empty lines
        else:
            # Regular text - use larger spacing
            param_ax.text(0.05, y_pos, line, transform=param_ax.transAxes,
                         fontsize=fontsize, verticalalignment='top', fontfamily='monospace',
                         color='black')
            y_pos -= content_line_height
    
    # Add background box around all text
    # Calculate approximate bounding box (use average line height)
    avg_line_height = (heading_line_height + content_line_height) / 2
    text_height = len(param_text) * avg_line_height
    param_ax.add_patch(FancyBboxPatch((0.02, y_start - text_height), 0.96, text_height,
                                     transform=param_ax.transAxes,
                                     facecolor='wheat', alpha=0.3, edgecolor='black', 
                                     linewidth=0.5, boxstyle='round'))
    
    # Save plot
    if output_dir is None:
        output_dir = plotting_params['plots_dir']
    
    if filename is None:
        if epoch is not None:
            filename = f'pinn_adaptive_playground_profiles_epoch_{epoch:05d}.png'
        else:
            filename = 'pinn_adaptive_playground_profiles.png'
    
    plot_path = Path(output_dir) / filename
    plt.savefig(str(plot_path), dpi=plotting_params['dpi'], bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("PINN for 1D Contaminant Transport - Uniform Grid Collocation Points")
    print("="*60)
    
    # Initialize model
    print("\nInitializing PINN model...")
    model = PINN(num_layers=model_params['num_layers'], 
                 num_neurons=model_params['num_neurons'],
                 activation=model_params['activation'])
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Print training parameters
    print(f"\nTraining parameters:")
    print(f"  Final dimensionless time: t*_final = {training_params['t_final_star']}")
    print(f"  Final physical time: t_final = {training_params['t_final_star'] * T:.1f} days")
    print(f"  Number of epochs: {training_params['num_epochs']}")
    print(f"  Learning rate: {training_params['lr']}")
    collocation_points_x_star = training_params['collocation_points_x_star']
    collocation_points_t_star = training_params['collocation_points_t_star']
    total_collocation = collocation_points_x_star * collocation_points_t_star
    print(f"  Collocation points: {total_collocation} total ({collocation_points_x_star} x* × {collocation_points_t_star} t*)")
    print(f"  IC points: {training_params['num_ic']}")
    print(f"  BC points: {training_params['num_bc']}")
    
    # Create subdirectory for epoch exports
    epoch_export_dir = Path(plotting_params['plots_dir']) / 'epoch_exports'
    epoch_export_dir.mkdir(exist_ok=True)
    print(f"Epoch exports will be saved to: {epoch_export_dir}")
    
    # Define callback function for epoch exports
    def plot_epoch_callback(model, epoch_num):
        """Callback function to plot and save at each epoch checkpoint."""
        plot_concentration_profiles(
            model, 
            times_days=plotting_params['times_days'],
            training_params=training_params, 
            model_params=model_params,
            physics_params=physics_params,
            epoch=epoch_num,
            output_dir=str(epoch_export_dir)
        )
    
    # Train model
    print("\n" + "="*60)
    print("Training PINN...")
    print("="*60)
    losses = train_pinn(model, training_params=training_params, plot_callback=plot_epoch_callback)
    
    # Generate final concentration profiles with analytical comparison
    print("\nGenerating final concentration profiles with analytical comparison...")
    plot_concentration_profiles(model, times_days=plotting_params['times_days'],
                                training_params=training_params, 
                                model_params=model_params,
                                physics_params=physics_params)
    
    print("\n" + "="*60)
    print("PINN training and comparison complete!")
    print("="*60)
