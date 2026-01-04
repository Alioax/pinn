"""
Physics-Informed Neural Network (PINN) for 1D Contaminant Transport in an Aquifer
LOSS DISTRIBUTION BASED ADAPTIVE COLLOCATION POINTS - ADAM + LBFGS OPTIMIZER SCHEDULE

This implementation:
- Trains the PINN entirely in dimensionless form
- Uses adaptive collocation points that redistribute based on PDE loss distribution
- Uses inverse CDF method to sample points according to loss probability distribution
- Maintains fixed total count of collocation points (collocation_points_x_star × collocation_points_t_star)
- Accepts inputs in physical units (meters, days)
- Converts to dimensionless variables before evaluation
- Converts dimensionless output back to dimensional concentration
- Produces plots in physical units
- Compares PINN solution with analytical solution

Key Features:
1. Optimizer Schedule: Uses Adam optimizer for first X epochs, then switches to LBFGS for Y epochs
   (X and Y are configurable via adam_epochs and lbfgs_epochs parameters)
2. Adaptive Collocation Points: Collocation points are periodically redistributed based on current PDE loss
   distribution using inverse CDF sampling, ensuring more points are placed in high-loss regions
   while maintaining a fixed total count.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, LogFormatterSciNotation, FixedLocator
from torch.autograd import grad
import os
import sys
from pathlib import Path
from tqdm import tqdm

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
    'num_layers': 1,            # number of hidden layers
    'num_neurons': 64,          # number of neurons per hidden layer
    'activation': torch.nn.Tanh,  # activation function
}

# Training parameters
training_params = {
    # Optimizer schedule: Adam for first X epochs, then LBFGS for Y epochs
    'adam_epochs': 10000,          # number of epochs with Adam optimizer
    'lbfgs_epochs': 500,        # number of epochs with LBFGS optimizer
    # Total epochs = adam_epochs + lbfgs_epochs (computed automatically)
    'num_epochs': None,           # will be set to adam_epochs + lbfgs_epochs
    'lr': 2.5e-3,                # learning rate (used for both optimizers)
    # Collocation point configuration - ADAPTIVE
    'collocation_points_x_star': 350,  # Number of points in x* direction (fixed count)
    'collocation_points_t_star': 350,  # Number of points in t* direction (fixed count)
    # Total collocation points = collocation_points_x_star × collocation_points_t_star = 2500
    'num_ic': 100,              # number of points for initial condition
    'num_bc': 100,              # number of points for boundary conditions
    't_final_star': 1.0,        # final dimensionless time
    'verbose': True,            # print training progress
    'export_interval': 100,     # export plot every N epochs (set to None to disable)
    'overwrite_gif_frames': True,  # if True, export gif frames with same name (overwriting)
    # Adaptive learning parameters
    'adaptive_update_interval': 11000,  # Number of epochs between collocation point updates
    'loss_evaluation_grid_x': 100,    # Resolution for loss evaluation grid in x* direction
    'loss_evaluation_grid_t': 100,    # Resolution for loss evaluation grid in t* direction
    'loss_smoothing_epsilon': 1e-8,   # Small value added to loss values to avoid zero probabilities
    # Loss weights (balance different loss components)
    'weight_pde': 1,            # weight for PDE residual loss
    'weight_ic': 1,             # weight for initial condition loss
    'weight_inlet_bc': 1,       # weight for inlet boundary condition loss
    'weight_outlet_bc': 1,      # weight for outlet boundary condition loss
    # Anchor collocation point parameters (prevents catastrophic forgetting)
    'anchor_ratio': 0.25,        # fraction of total points that are anchors (0.0 = all adaptive, 1.0 = all anchors)
    'anchor_distribution': 'uniform',  # distribution strategy for anchors ('uniform' only currently)
}

# Compute total epochs from Adam + LBFGS epochs
if training_params['num_epochs'] is None:
    training_params['num_epochs'] = training_params['adam_epochs'] + training_params['lbfgs_epochs']

# Plotting parameters
script_dir = Path(__file__).parent
plots_dir = script_dir / 'results_adam_lbfgs'
plotting_params = {
    'times_days': [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # times for concentration profiles
    'x_max': 100.0,             # maximum spatial coordinate (m)
    'num_points': 500,          # number of spatial points for profiles
    'dpi': 50,                # resolution for saved figures
    'plots_dir': str(plots_dir),  # directory to save plots
    'heatmap_resolution_x': 10,  # number of cells in x direction for collocation heatmap
    'heatmap_resolution_t': 10,  # number of cells in t direction for collocation heatmap
    'pde_loss_heatmap_resolution_x': 50,  # number of cells in x direction for PDE loss heatmap
    'pde_loss_heatmap_resolution_t': 50,  # number of cells in t direction for PDE loss heatmap
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
        
        # Apply Xavier normal initialization with gain to all linear layers
        gain = nn.init.calculate_gain('tanh')
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
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
# Loss Distribution Evaluation
# ============================================================================

def compute_loss_distribution_grid(model, loss_evaluation_grid_x, loss_evaluation_grid_t, 
                                   t_final_star, loss_smoothing_epsilon=1e-8):
    """
    Evaluate PDE loss distribution on a fine uniform grid.
    
    This function evaluates the PDE residual at a fine grid of points to estimate
    the loss distribution, which is then used to guide adaptive collocation point placement.
    
    Parameters:
        model: PINN model (should be in eval mode)
        loss_evaluation_grid_x: Resolution for loss evaluation grid in x* direction
        loss_evaluation_grid_t: Resolution for loss evaluation grid in t* direction
        t_final_star: Final dimensionless time
        loss_smoothing_epsilon: Small value added to loss values to avoid zero probabilities
    
    Returns:
        loss_grid: 2D numpy array of normalized loss values (probability distribution)
        x_star_grid: 1D numpy array of x* coordinates for grid
        t_star_grid: 1D numpy array of t* coordinates for grid
    """
    # Create fine uniform grid for loss evaluation
    x_star_eval = np.linspace(0, 1, loss_evaluation_grid_x)
    t_star_eval = np.linspace(0, t_final_star, loss_evaluation_grid_t)
    
    # Create meshgrid
    x_star_mesh, t_star_mesh = np.meshgrid(x_star_eval, t_star_eval, indexing='ij')
    
    # Initialize loss grid
    loss_grid = np.zeros((loss_evaluation_grid_x, loss_evaluation_grid_t))
    
    # Evaluate PDE residual at each grid point
    for i in range(loss_evaluation_grid_x):
        for j in range(loss_evaluation_grid_t):
            x_star_val = x_star_mesh[i, j]
            t_star_val = t_star_mesh[i, j]
            
            # Convert to physical dimensions for residual computation
            x_phys = x_star_val * L
            t_phys = t_star_val * T
            
            # Compute residual at this point
            residual = compute_pde_residual_at_points(model, np.array([x_phys]), t_phys)
            # Store squared residual (loss)
            loss_grid[i, j] = residual[0]**2
    
    # Add smoothing epsilon to avoid zero probabilities
    loss_grid = loss_grid + loss_smoothing_epsilon
    
    # Normalize to create probability distribution
    loss_grid = loss_grid / np.sum(loss_grid)
    
    return loss_grid, x_star_eval, t_star_eval

# ============================================================================
# Adaptive Collocation Point Generation (Inverse CDF Method)
# ============================================================================

def generate_adaptive_collocation_points_inverse_cdf(loss_grid, x_star_grid, t_star_grid,
                                                     num_adaptive_points, 
                                                     total_points_x_star=None,
                                                     total_points_t_star=None):
    """
    Generate adaptive collocation points using inverse CDF method based on loss distribution.
    
    This function uses the inverse cumulative distribution function (CDF) method to sample
    collocation points according to the loss probability distribution. The method ensures
    that more points are placed in regions with higher loss values.
    
    Algorithm:
    1. Compute marginal probability P(x*) = sum over t* of P(x*, t*)
    2. Compute conditional probability P(t*|x*) = P(x*, t*) / P(x*)
    3. Compute CDFs: CDF_x(x*) and CDF_t(t*|x*) for each x*
    4. Sample uniform values and use inverse CDF to get actual coordinates
    
    Parameters:
        loss_grid: 2D numpy array of normalized loss values (probability distribution)
        x_star_grid: 1D numpy array of x* coordinates for loss grid
        t_star_grid: 1D numpy array of t* coordinates for loss grid
        num_adaptive_points: Target number of adaptive points to generate
        total_points_x_star: Total number of points in x* direction (for aspect ratio, optional)
        total_points_t_star: Total number of points in t* direction (for aspect ratio, optional)
    
    Returns:
        x_star_colloc: Tensor of dimensionless spatial coordinates (shape: [N, 1])
        t_star_colloc: Tensor of dimensionless time coordinates (shape: [N, 1])
        where N = num_adaptive_points
    """
    # If no adaptive points requested, return empty tensors
    if num_adaptive_points == 0:
        x_star_colloc = torch.empty((0, 1), dtype=torch.float32)
        t_star_colloc = torch.empty((0, 1), dtype=torch.float32)
        return x_star_colloc, t_star_colloc
    
    # Compute adaptive grid dimensions
    # Try to maintain aspect ratio if total dimensions provided, otherwise use square-ish grid
    if total_points_x_star is not None and total_points_t_star is not None:
        aspect_ratio = total_points_t_star / total_points_x_star
    else:
        # Default to square-ish grid
        aspect_ratio = 1.0
    
    # Compute adaptive_x and adaptive_t such that adaptive_x × adaptive_t ≈ num_adaptive_points
    # and adaptive_t / adaptive_x ≈ aspect_ratio
    adaptive_x = int(np.sqrt(num_adaptive_points / aspect_ratio))
    adaptive_t = int(num_adaptive_points / adaptive_x) if adaptive_x > 0 else 0
    
    # Ensure we generate at least the requested number (may overshoot slightly)
    while adaptive_x * adaptive_t < num_adaptive_points:
        if adaptive_x <= adaptive_t:
            adaptive_x += 1
        else:
            adaptive_t += 1
    
    # Compute marginal probability distribution P(x*)
    # Sum over t* dimension (axis=1)
    p_x_star = np.sum(loss_grid, axis=1)
    
    # Normalize to ensure it's a proper probability distribution
    p_x_star = p_x_star / np.sum(p_x_star)
    
    # Compute cumulative distribution function for x*
    cdf_x_star = np.cumsum(p_x_star)
    # Ensure last value is exactly 1.0 (account for floating point errors)
    cdf_x_star[-1] = 1.0
    
    # Sample x* positions using inverse CDF
    # Generate uniform random samples
    u_x = np.linspace(0, 1, adaptive_x + 2)[1:-1]  # Avoid exact 0 and 1
    # Use inverse CDF to map uniform samples to x* positions
    x_star_samples = np.interp(u_x, cdf_x_star, x_star_grid)
    
    # For each x* position, sample t* positions using conditional distribution
    x_star_colloc_list = []
    t_star_colloc_list = []
    
    for x_star_val in x_star_samples:
        # Find closest x* index in grid
        x_idx = np.argmin(np.abs(x_star_grid - x_star_val))
        
        # Get conditional probability P(t*|x*)
        p_t_given_x = loss_grid[x_idx, :] / (p_x_star[x_idx] + 1e-12)  # Add small epsilon to avoid division by zero
        
        # Normalize conditional distribution
        p_t_given_x = p_t_given_x / np.sum(p_t_given_x)
        
        # Compute cumulative distribution function for t* given x*
        cdf_t_given_x = np.cumsum(p_t_given_x)
        cdf_t_given_x[-1] = 1.0  # Ensure last value is exactly 1.0
        
        # Sample t* positions using inverse CDF
        u_t = np.linspace(0, 1, adaptive_t + 2)[1:-1]  # Avoid exact 0 and 1
        t_star_samples = np.interp(u_t, cdf_t_given_x, t_star_grid)
        
        # Add all combinations of (x_star_val, t_star_samples)
        for t_star_val in t_star_samples:
            x_star_colloc_list.append(x_star_val)
            t_star_colloc_list.append(t_star_val)
    
    # Convert to tensors
    x_star_colloc = torch.tensor(x_star_colloc_list, dtype=torch.float32).reshape(-1, 1)
    t_star_colloc = torch.tensor(t_star_colloc_list, dtype=torch.float32).reshape(-1, 1)
    
    # If we generated more than requested (due to grid rounding), trim to exact count
    if x_star_colloc.shape[0] > num_adaptive_points:
        # Randomly sample to get exact count
        indices = torch.randperm(x_star_colloc.shape[0])[:num_adaptive_points]
        x_star_colloc = x_star_colloc[indices]
        t_star_colloc = t_star_colloc[indices]
    
    # Ensure points require gradients for training
    x_star_colloc = x_star_colloc.requires_grad_(True)
    t_star_colloc = t_star_colloc.requires_grad_(True)
    
    return x_star_colloc, t_star_colloc

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
# Anchor Collocation Point Generation
# ============================================================================

def generate_anchor_collocation_points(anchor_ratio, total_points_x_star, total_points_t_star, t_final_star):
    """
    Generate uniformly-distributed anchor collocation points.
    
    Anchor points are permanently present during training to prevent catastrophic
    forgetting. They ensure minimum physics enforcement coverage across the entire
    domain, even when adaptive resampling focuses on high-error regions.
    
    Parameters:
        anchor_ratio: fraction of total points that are anchors (0.0 to 1.0)
        total_points_x_star: total number of points in x* direction (for computing anchor count)
        total_points_t_star: total number of points in t* direction (for computing anchor count)
        t_final_star: final dimensionless time
    
    Returns:
        x_star_anchor: uniformly-spaced dimensionless spatial coordinates (tensor, shape: [N_anchor, 1])
        t_star_anchor: uniformly-spaced dimensionless time coordinates (tensor, shape: [N_anchor, 1])
        num_anchor: number of anchor points generated
    """
    # Compute total number of points
    total_points = total_points_x_star * total_points_t_star
    
    # Compute number of anchor points
    num_anchor = int(total_points * anchor_ratio)
    
    # If no anchor points requested, return empty tensors
    if num_anchor == 0:
        x_star_anchor = torch.empty((0, 1), dtype=torch.float32)
        t_star_anchor = torch.empty((0, 1), dtype=torch.float32)
        return x_star_anchor, t_star_anchor, num_anchor
    
    # Compute anchor grid dimensions to maintain aspect ratio
    # Try to maintain similar aspect ratio as total grid
    aspect_ratio = total_points_t_star / total_points_x_star
    
    # Compute anchor_x and anchor_t such that anchor_x × anchor_t ≈ num_anchor
    # and anchor_t / anchor_x ≈ aspect_ratio
    anchor_x = int(np.sqrt(num_anchor / aspect_ratio))
    anchor_t = int(num_anchor / anchor_x) if anchor_x > 0 else 0
    
    # Ensure we generate at least the requested number (may overshoot slightly)
    while anchor_x * anchor_t < num_anchor:
        if anchor_x <= anchor_t:
            anchor_x += 1
        else:
            anchor_t += 1
    
    # Generate uniformly-spaced anchor points
    x_star_anchor_1d = torch.linspace(0, 1, anchor_x)
    t_star_anchor_1d = torch.linspace(0, t_final_star, anchor_t)
    
    # Create meshgrid to get all combinations
    x_star_anchor_grid, t_star_anchor_grid = torch.meshgrid(x_star_anchor_1d, t_star_anchor_1d, indexing='ij')
    
    # Flatten to create pairs: (x_star, t_star) for each grid point
    x_star_anchor = x_star_anchor_grid.flatten().reshape(-1, 1)
    t_star_anchor = t_star_anchor_grid.flatten().reshape(-1, 1)
    
    # If we generated more than requested (due to grid rounding), trim to exact count
    if x_star_anchor.shape[0] > num_anchor:
        # Randomly sample to get exact count (or use first N)
        indices = torch.randperm(x_star_anchor.shape[0])[:num_anchor]
        x_star_anchor = x_star_anchor[indices]
        t_star_anchor = t_star_anchor[indices]
    
    # Ensure points require gradients for training
    x_star_anchor = x_star_anchor.requires_grad_(True)
    t_star_anchor = t_star_anchor.requires_grad_(True)
    
    return x_star_anchor, t_star_anchor, num_anchor

# ============================================================================
# Training Function
# ============================================================================

def train_pinn(model, training_params=None, plot_callback=None):
    """
    Train the PINN model with adaptive collocation points based on loss distribution.
    
    Collocation points are periodically redistributed based on the current PDE loss
    distribution using inverse CDF sampling. The total count of collocation points
    remains fixed throughout training.
    
    Parameters:
        model: PINN model
        training_params: dictionary with training parameters
        plot_callback: optional function(model, epoch, losses) called at export intervals for plotting
    
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
    
    # Adaptive learning parameters
    adaptive_update_interval = training_params.get('adaptive_update_interval', 100)
    loss_evaluation_grid_x = training_params.get('loss_evaluation_grid_x', 100)
    loss_evaluation_grid_t = training_params.get('loss_evaluation_grid_t', 100)
    loss_smoothing_epsilon = training_params.get('loss_smoothing_epsilon', 1e-8)
    
    # Anchor collocation point parameters
    anchor_ratio = training_params.get('anchor_ratio', 0.0)
    anchor_distribution = training_params.get('anchor_distribution', 'uniform')
    
    # Loss weights
    weight_pde = training_params.get('weight_pde', 1.0)
    weight_ic = training_params.get('weight_ic', 1.0)
    weight_inlet_bc = training_params.get('weight_inlet_bc', 1.0)
    weight_outlet_bc = training_params.get('weight_outlet_bc', 1.0)
    
    # Optimizer schedule: Adam for first X epochs, then LBFGS for Y epochs
    adam_epochs = training_params.get('adam_epochs', 5000)
    lbfgs_epochs = training_params.get('lbfgs_epochs', 15000)
    
    # Initialize with Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    current_optimizer_type = 'adam'
    
    # Loss history
    losses = {
        'total': [],
        'pde': [],
        'ic': [],
        'inlet_bc': [],
        'outlet_bc': []
    }
    
    # Compute total number of collocation points
    total_points = collocation_points_x_star * collocation_points_t_star
    
    # Generate anchor collocation points (permanently present during training)
    if anchor_ratio > 0.0:
        x_star_anchor, t_star_anchor, num_anchor = generate_anchor_collocation_points(
            anchor_ratio, collocation_points_x_star, collocation_points_t_star, t_final_star
        )
        
        # Compute number of adaptive points (remaining capacity)
        num_adaptive = total_points - num_anchor
        
        if verbose:
            print(f"\nAnchor collocation points initialized:")
            print(f"  Anchor ratio: {anchor_ratio:.2%}")
            print(f"  Anchor points: {num_anchor} ({num_anchor/total_points:.2%} of total)")
            print(f"  Adaptive points: {num_adaptive} ({num_adaptive/total_points:.2%} of total)")
    else:
        # No anchor points - all points are adaptive (backward compatibility)
        x_star_anchor = torch.empty((0, 1), dtype=torch.float32)
        t_star_anchor = torch.empty((0, 1), dtype=torch.float32)
        num_anchor = 0
        num_adaptive = total_points
        if verbose:
            print(f"\nNo anchor points (anchor_ratio=0.0) - all {num_adaptive} points are adaptive")
    
    # Initialize adaptive points as uniform grid (remaining capacity)
    # Compute adaptive grid dimensions
    if num_adaptive > 0:
        aspect_ratio = collocation_points_t_star / collocation_points_x_star
        adaptive_x = int(np.sqrt(num_adaptive / aspect_ratio))
        adaptive_t = int(num_adaptive / adaptive_x) if adaptive_x > 0 else 0
        
        # Ensure we generate at least the requested number
        while adaptive_x * adaptive_t < num_adaptive:
            if adaptive_x <= adaptive_t:
                adaptive_x += 1
            else:
                adaptive_t += 1
        
        # Generate uniform grid for adaptive points
        x_star_adaptive, t_star_adaptive = generate_uniform_grid_collocation_points(
            adaptive_x, adaptive_t, t_final_star
        )
        
        # Trim to exact count if needed
        if x_star_adaptive.shape[0] > num_adaptive:
            indices = torch.randperm(x_star_adaptive.shape[0])[:num_adaptive]
            x_star_adaptive = x_star_adaptive[indices]
            t_star_adaptive = t_star_adaptive[indices]
    else:
        # No adaptive points (all anchors)
        x_star_adaptive = torch.empty((0, 1), dtype=torch.float32)
        t_star_adaptive = torch.empty((0, 1), dtype=torch.float32)
    
    # Combine anchor and adaptive points for training
    if num_anchor > 0:
        x_star_colloc = torch.cat([x_star_anchor, x_star_adaptive], dim=0)
        t_star_colloc = torch.cat([t_star_anchor, t_star_adaptive], dim=0)
    else:
        # No anchors - use only adaptive points
        x_star_colloc = x_star_adaptive
        t_star_colloc = t_star_adaptive
    
    # Create progress bar
    pbar = tqdm(range(num_epochs), desc="Training PINN", disable=not verbose)
    
    # Training loop
    for epoch in pbar:
        # Switch to LBFGS optimizer after Adam epochs
        if epoch == adam_epochs and current_optimizer_type == 'adam':
            if verbose:
                print(f"\nEpoch {epoch + 1}: Switching from Adam to LBFGS optimizer")
            # Create LBFGS optimizer with same parameters
            optimizer = torch.optim.LBFGS(
                model.parameters(), 
                lr=lr, 
                max_iter=200, 
                max_eval=None, 
                tolerance_grad=1e-07, 
                tolerance_change=1e-09, 
                history_size=100, 
                line_search_fn=None
            )
            current_optimizer_type = 'lbfgs'
        
        # Check if optimizer is LBFGS (requires closure function)
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        # Generate random IC and BC points once per epoch (reused in closure if needed)
        # IC points don't need requires_grad (just evaluating C, not computing derivatives)
        x_star_ic = torch.rand(num_ic, 1) * 1.0  # [0, 1]
        # BC points don't need requires_grad (BC loss functions handle it internally if needed)
        t_star_bc = torch.rand(num_bc, 1) * t_final_star  # [0, t_final_star]
        
        # Define closure function for loss computation
        def closure():
            optimizer.zero_grad()
            
            # PDE residual loss (using adaptive collocation points)
            pde_residual = compute_pde_residual(model, x_star_colloc, t_star_colloc)
            pde_loss = torch.mean(pde_residual**2)
            
            # Initial condition loss
            ic_loss = compute_initial_condition_loss(model, x_star_ic)
            
            # Boundary condition losses
            inlet_bc_loss, outlet_bc_loss = compute_boundary_condition_losses(model, t_star_bc)
            
            # Total loss with weights
            total_loss = (weight_pde * pde_loss + 
                         weight_ic * ic_loss + 
                         weight_inlet_bc * inlet_bc_loss + 
                         weight_outlet_bc * outlet_bc_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Store losses (like baseline does - computed with gradients)
            losses['total'].append(total_loss.item())
            losses['pde'].append(pde_loss.item())
            losses['ic'].append(ic_loss.item())
            losses['inlet_bc'].append(inlet_bc_loss.item())
            losses['outlet_bc'].append(outlet_bc_loss.item())
            
            return total_loss
        
        # For LBFGS, pass closure to step(); for others, call closure then step()
        if is_lbfgs:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        
        # Update progress bar (update every 10 epochs for performance)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            pbar.set_postfix({
                'Loss': f"{losses['total'][-1]:.4e}",
                'PDE': f"{losses['pde'][-1]:.4e}"
            })
        
        # Update adaptive collocation points based on loss distribution
        # Anchor points remain unchanged throughout training
        if (epoch + 1) % adaptive_update_interval == 0 and epoch > 0 and num_adaptive > 0:
            # Set model to eval mode for loss evaluation
            model.eval()
            
            # Compute loss distribution grid
            loss_grid, x_star_grid, t_star_grid = compute_loss_distribution_grid(
                model, loss_evaluation_grid_x, loss_evaluation_grid_t,
                t_final_star, loss_smoothing_epsilon
            )
            
            # Generate new adaptive collocation points using inverse CDF method
            # Only generate adaptive points (anchor points are preserved)
            x_star_adaptive, t_star_adaptive = generate_adaptive_collocation_points_inverse_cdf(
                loss_grid, x_star_grid, t_star_grid,
                num_adaptive,
                collocation_points_x_star, collocation_points_t_star
            )
            
            # Combine anchor and adaptive points for training
            if num_anchor > 0:
                x_star_colloc = torch.cat([x_star_anchor, x_star_adaptive], dim=0)
                t_star_colloc = torch.cat([t_star_anchor, t_star_adaptive], dim=0)
            else:
                # No anchors - use only adaptive points
                x_star_colloc = x_star_adaptive
                t_star_colloc = t_star_adaptive
            
            # Set model back to training mode
            model.train()
            
            if verbose and (epoch + 1) % (adaptive_update_interval * 5) == 0:
                # Print update every 5 adaptive updates
                print(f"\nEpoch {epoch + 1}: Adaptive collocation points updated")
                print(f"  Anchor points: {num_anchor} (unchanged)")
                print(f"  Adaptive points: {num_adaptive} (redistributed)")
        
        # Export plot at specified interval
        export_interval = training_params.get('export_interval', 100)
        if plot_callback is not None and export_interval is not None and (epoch + 1) % export_interval == 0:
            # Create current losses dict (up to current epoch)
            current_losses = {
                'total': losses['total'][:epoch+1],
                'pde': losses['pde'][:epoch+1],
                'ic': losses['ic'][:epoch+1],
                'inlet_bc': losses['inlet_bc'][:epoch+1],
                'outlet_bc': losses['outlet_bc'][:epoch+1]
            }
            plot_callback(model, epoch + 1, current_losses, x_star_colloc, t_star_colloc)
    
    pbar.close()
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
# Residual Computation Functions (for plotting)
# ============================================================================

def compute_pde_residual_at_points(model, x, t):
    """
    Compute PDE residual at specific (x, t) points.
    
    Parameters:
        model: PINN model
        x: spatial coordinates (m) - array
        t: time (days) - scalar
    
    Returns:
        residual: PDE residual values (numpy array)
    """
    # Convert to dimensionless
    x_star, t_star = to_dimensionless(x, t)
    
    # Convert to numpy arrays first
    if isinstance(x_star, torch.Tensor):
        x_star = x_star.cpu().numpy()
    if isinstance(t_star, torch.Tensor):
        t_star = t_star.cpu().numpy()
    
    x_star = np.asarray(x_star)
    t_star = np.asarray(t_star)
    
    # Handle broadcasting: t is scalar, x is array
    if x_star.ndim == 0:
        x_star = x_star.reshape(1)
    if t_star.ndim == 0:
        t_star = np.full_like(x_star, t_star.item() if hasattr(t_star, 'item') else float(t_star))
    elif t_star.size == 1:
        t_star = np.full_like(x_star, float(t_star))
    
    # Convert to tensors
    x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True)
    t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True)
    
    model.eval()
    with torch.enable_grad():
        C_star = model(x_star, t_star)
        dC_dt_star = grad(C_star, t_star, grad_outputs=torch.ones_like(C_star),
                         create_graph=True, retain_graph=True)[0]
        dC_dx_star = grad(C_star, x_star, grad_outputs=torch.ones_like(C_star),
                         create_graph=True, retain_graph=True)[0]
        d2C_dx2_star = grad(dC_dx_star, x_star, grad_outputs=torch.ones_like(dC_dx_star),
                           create_graph=True, retain_graph=True)[0]
        
        residual = dC_dt_star + dC_dx_star - (1.0 / Pe) * d2C_dx2_star
    
    return residual.detach().cpu().numpy().flatten()

def compute_ic_residual_at_points(model, x):
    """
    Compute initial condition residual: C*(x*, 0) - 0 = C*(x*, 0)
    
    Parameters:
        model: PINN model
        x: spatial coordinates (m) - array
    
    Returns:
        residual: IC residual values (numpy array)
    """
    # Convert to dimensionless
    x_star, _ = to_dimensionless(x, 0.0)
    
    # Convert to tensor
    if isinstance(x_star, torch.Tensor):
        x_star = x_star.cpu().numpy()
    x_star = np.asarray(x_star)
    if x_star.ndim == 0:
        x_star = x_star.reshape(1)
    
    x_star = torch.tensor(x_star, dtype=torch.float32)
    t_star = torch.zeros_like(x_star)
    
    model.eval()
    with torch.no_grad():
        C_star = model(x_star, t_star)
        # IC residual: C*(x*, 0) - 0 = C*(x*, 0)
        residual = C_star
    
    return residual.detach().cpu().numpy().flatten()

def compute_bc_inlet_residual_at_points(model, t):
    """
    Compute inlet boundary condition residual: C*(0, t*) - 1
    
    Parameters:
        model: PINN model
        t: time (days) - scalar
    
    Returns:
        residual: BC inlet residual value (scalar as array)
    """
    # Convert to dimensionless
    _, t_star = to_dimensionless(0.0, t)
    
    # Convert to tensor
    if isinstance(t_star, torch.Tensor):
        t_star = t_star.cpu().numpy()
    t_star = np.asarray(t_star)
    if t_star.ndim == 0:
        t_star = t_star.reshape(1)
    
    t_star = torch.tensor(t_star, dtype=torch.float32)
    x_star = torch.zeros_like(t_star)
    
    model.eval()
    with torch.no_grad():
        C_star = model(x_star, t_star)
        # BC inlet residual: C*(0, t*) - 1
        residual = C_star - 1.0
    
    return residual.detach().cpu().numpy().flatten()

def compute_bc_outlet_residual_at_points(model, t):
    """
    Compute outlet boundary condition residual: ∂C*/∂x*(1, t*) - 0 = ∂C*/∂x*(1, t*)
    
    Parameters:
        model: PINN model
        t: time (days) - scalar
    
    Returns:
        residual: BC outlet residual value (scalar as array)
    """
    # Convert to dimensionless
    _, t_star = to_dimensionless(L, t)  # L is the outlet position
    
    # Convert to tensor
    if isinstance(t_star, torch.Tensor):
        t_star = t_star.cpu().numpy()
    t_star = np.asarray(t_star)
    if t_star.ndim == 0:
        t_star = t_star.reshape(1)
    
    t_star = torch.tensor(t_star, dtype=torch.float32, requires_grad=True)
    x_star = torch.ones_like(t_star)  # x* = 1 at outlet
    x_star = x_star.requires_grad_(True)
    
    model.eval()
    with torch.enable_grad():
        C_star = model(x_star, t_star)
        dC_dx_star = grad(C_star, x_star, grad_outputs=torch.ones_like(C_star),
                        create_graph=True, retain_graph=True)[0]
        # BC outlet residual: ∂C*/∂x*(1, t*) - 0 = ∂C*/∂x*(1, t*)
        residual = dC_dx_star
    
    return residual.detach().cpu().numpy().flatten()

# ============================================================================
# Plotting Function
# ============================================================================

def plot_concentration_profiles(model, times_days=None, x_max=None, num_points=None, 
                                training_params=None, model_params=None, physics_params=None,
                                epoch=None, output_dir=None, filename=None, losses=None,
                                x_star_colloc=None, t_star_colloc=None):
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
        losses: dictionary of loss history with keys 'pde', 'ic', etc. (optional)
        x_star_colloc: actual collocation points x* coordinates (tensor or array, optional)
        t_star_colloc: actual collocation points t* coordinates (tensor or array, optional)
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
    
    # Create figure with two-column layout
    # Left column: main plot, PDE residual, IC residual (all sharing x-axis)
    # Right column: empty for main, PDE loss, IC loss
    fig = plt.figure(figsize=(10, 6.5))
    
    # Left column axes (all sharing x-axis)
    # Main plot axes (top left)
    ax_main = fig.add_axes([0.08, 0.55, 0.40, 0.35])  # [left, bottom, width, height]
    
    # Residual subplots (below main plot, left column)
    # PDE residual (bigger)
    ax_pde = fig.add_axes([0.08, 0.35, 0.40, 0.15])
    # IC residual
    ax_ic = fig.add_axes([0.08, 0.15, 0.40, 0.15])
    
    # Right column axes
    # Top right: two heatmaps (collocation distribution and PDE loss)
    # Collocation distribution heatmap (left side of top right)
    ax_colloc_heatmap = fig.add_axes([0.55, 0.60, 0.18, 0.30])
    # PDE loss heatmap (right side of top right)
    ax_pde_loss_heatmap = fig.add_axes([0.74, 0.60, 0.21, 0.30])
    
    # Loss plots (below heatmaps)
    # PDE loss (aligned with PDE residual)
    ax_pde_loss = fig.add_axes([0.55, 0.35, 0.40, 0.15])
    # IC loss (aligned with IC residual)
    ax_ic_loss = fig.add_axes([0.55, 0.15, 0.40, 0.15])
    
    # Get current color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Create color mapping: each time gets a consistent color across all plots
    time_to_color = {t_days: colors[idx % len(colors)] for idx, t_days in enumerate(times_days)}
    
    # ===== Main Plot: Concentration Profiles =====
    # Add legend entries for PINN and Analytical at the beginning
    ax_main.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    ax_main.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')
    
    # Plot in reverse order so legend shows times in order
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time from mapping
        color = time_to_color[t_days]
        
        # PINN solution (solid line)
        C_pinn = evaluate_dimensional(model, x_plot, t_days)
        ax_main.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Analytical solution (dashed line) with same color
        C_analytical = analytical_solution(x_plot, t_days)
        ax_main.plot(x_plot, C_analytical, linewidth=2, linestyle='--', color=color, alpha=0.7)
        
        # Add marker-only entry for legend (square marker, no line)
        ax_main.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{t_days} days')
    
    ax_main.set_ylabel('Concentration C (kg/m³)', fontsize=12)
    
    ax_main.set_xlim(0, x_max)
    ax_main.set_ylim(-0.25, C_0 * 1.1)
    
    # Set consistent tick formatting for main plot
    # X-axis: always include 0 and x_max (L), plus intermediate ticks
    # Get intermediate ticks first
    intermediate_locator = MaxNLocator(nbins=5, prune='both')
    intermediate_ticks = intermediate_locator.tick_values(0, x_max)
    # Ensure 0 and x_max are included
    x_ticks = sorted(set([0.0, x_max] + list(intermediate_ticks)))
    ax_main.set_xticks(x_ticks)
    ax_main.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # Y-axis: fixed ticks at 0, 1, 2, 3, 4, 5
    ax_main.set_yticks([0, 1, 2, 3, 4, 5])
    ax_main.yaxis.set_major_formatter(FormatStrFormatter('%d'))  # Integer format
    
    # ===== Residual Plots =====
    # Plot residuals for each time
    for t_days in times_days:
        # Get color for this time from mapping (same as main plot)
        color = time_to_color[t_days]
        
        # PDE residual
        pde_residual = compute_pde_residual_at_points(model, x_plot, t_days)
        ax_pde.plot(x_plot, pde_residual, linewidth=1.5, color=color, alpha=0.8, label=f'{t_days} days')
    
    # IC residual (only at t=0)
    ic_residual = compute_ic_residual_at_points(model, x_plot)
    ax_ic.plot(x_plot, ic_residual, linewidth=1.5, color=colors[0], alpha=0.8)
    
    # Set labels and limits for residual plots
    ax_pde.set_ylabel('PDE Residual', fontsize=10)
    ax_pde.set_xlim(0, x_max)
    ax_pde.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    # Set consistent tick formatting for PDE residual
    # X-axis: always include 0 and x_max (L), plus intermediate ticks
    intermediate_locator_pde = MaxNLocator(nbins=5, prune='both')
    intermediate_ticks_pde = intermediate_locator_pde.tick_values(0, x_max)
    x_ticks_pde = sorted(set([0.0, x_max] + list(intermediate_ticks_pde)))
    ax_pde.set_xticks(x_ticks_pde)
    ax_pde.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_pde.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax_pde.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))  # Scientific notation for residuals
    
    ax_ic.set_ylabel('IC Residual', fontsize=10)
    ax_ic.set_xlim(0, x_max)
    ax_ic.set_xlabel('Distance x (m)', fontsize=12)
    ax_ic.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    # Set consistent tick formatting for IC residual
    # X-axis: always include 0 and x_max (L), plus intermediate ticks
    intermediate_locator_ic = MaxNLocator(nbins=5, prune='both')
    intermediate_ticks_ic = intermediate_locator_ic.tick_values(0, x_max)
    x_ticks_ic = sorted(set([0.0, x_max] + list(intermediate_ticks_ic)))
    ax_ic.set_xticks(x_ticks_ic)
    ax_ic.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_ic.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    ax_ic.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))  # Scientific notation for residuals
    
    # Hide x-axis labels for all plots except bottom one in left column
    ax_main.set_xticklabels([])
    ax_pde.set_xticklabels([])
    
    # Define styling parameters (used for all plots including heatmaps)
    grid_alpha = 0.3
    grid_color = 'black'
    grid_linewidth = 0.4
    
    # ===== Heatmaps (Top Right) =====
    # Get collocation point configuration
    collocation_points_x_star = training_params['collocation_points_x_star']
    collocation_points_t_star = training_params['collocation_points_t_star']
    t_final_star = training_params['t_final_star']
    
    # Get heatmap resolutions from plotting params
    heatmap_res_x = plotting_params.get('heatmap_resolution_x', 10)
    heatmap_res_t = plotting_params.get('heatmap_resolution_t', 10)
    pde_loss_res_x = plotting_params.get('pde_loss_heatmap_resolution_x', 50)
    pde_loss_res_t = plotting_params.get('pde_loss_heatmap_resolution_t', 50)
    
    # Create grid for collocation heatmap (using configurable resolution)
    x_star_heatmap = np.linspace(0, 1, heatmap_res_x + 1)  # +1 for bin edges
    t_star_heatmap = np.linspace(0, t_final_star, heatmap_res_t + 1)  # +1 for bin edges
    
    # Bin centers for collocation heatmap plotting
    x_star_centers_colloc = (x_star_heatmap[:-1] + x_star_heatmap[1:]) / 2
    t_star_centers_colloc = (t_star_heatmap[:-1] + t_star_heatmap[1:]) / 2
    
    # 1. Collocation Distribution Heatmap
    # Count how many collocation points fall in each bin
    # Use actual collocation points if provided, otherwise reconstruct uniform grid
    if x_star_colloc is not None and t_star_colloc is not None:
        # Convert tensors to numpy arrays if needed
        if isinstance(x_star_colloc, torch.Tensor):
            x_star_colloc_flat = x_star_colloc.detach().cpu().numpy().flatten()
        else:
            x_star_colloc_flat = np.asarray(x_star_colloc).flatten()
        
        if isinstance(t_star_colloc, torch.Tensor):
            t_star_colloc_flat = t_star_colloc.detach().cpu().numpy().flatten()
        else:
            t_star_colloc_flat = np.asarray(t_star_colloc).flatten()
    else:
        # Fallback: reconstruct uniform grid from training params
        x_star_colloc_1d = np.linspace(0, 1, collocation_points_x_star)
        t_star_colloc_1d = np.linspace(0, t_final_star, collocation_points_t_star)
        x_star_colloc_grid, t_star_colloc_grid = np.meshgrid(x_star_colloc_1d, t_star_colloc_1d, indexing='ij')
        x_star_colloc_flat = x_star_colloc_grid.flatten()
        t_star_colloc_flat = t_star_colloc_grid.flatten()
    
    # Count points in each bin
    colloc_distribution = np.zeros((heatmap_res_x, heatmap_res_t))
    for i in range(heatmap_res_x):
        for j in range(heatmap_res_t):
            # Count points in this bin
            x_mask = (x_star_colloc_flat >= x_star_heatmap[i]) & (x_star_colloc_flat < x_star_heatmap[i+1])
            t_mask = (t_star_colloc_flat >= t_star_heatmap[j]) & (t_star_colloc_flat < t_star_heatmap[j+1])
            # Handle edge case for last bin (include right edge)
            if i == heatmap_res_x - 1:
                x_mask = (x_star_colloc_flat >= x_star_heatmap[i]) & (x_star_colloc_flat <= x_star_heatmap[i+1])
            if j == heatmap_res_t - 1:
                t_mask = (t_star_colloc_flat >= t_star_heatmap[j]) & (t_star_colloc_flat <= t_star_heatmap[j+1])
            colloc_distribution[i, j] = np.sum(x_mask & t_mask)
    
    # Calculate interval sizes in real dimensions
    deltax = x_max / heatmap_res_x
    deltat = (t_final_star * T) / heatmap_res_t
    
    # Create custom colormap: white to 70% black
    from matplotlib.colors import LinearSegmentedColormap
    colors_colloc = ['white', (0.3, 0.3, 0.3)]  # white to 70% black (0.3 = 30% of 255)
    cmap_colloc = LinearSegmentedColormap.from_list('colloc_cmap', colors_colloc, N=256)
    
    # Use real dimensions for extent
    im_colloc = ax_colloc_heatmap.imshow(colloc_distribution, aspect='auto', origin='lower',
                                        extent=[0, x_max, 0, t_final_star * T],
                                        cmap=cmap_colloc, interpolation='nearest')
    
    # Add text in each cell showing the count (using real dimensions)
    for i in range(heatmap_res_x):
        for j in range(heatmap_res_t):
            x_center = (i + 0.5) * deltax
            t_center = (j + 0.5) * deltat
            count = int(colloc_distribution[i, j])
            ax_colloc_heatmap.text(x_center, t_center, str(count), 
                                  ha='center', va='center', 
                                  color='black', fontsize=6, weight='bold')
    
    ax_colloc_heatmap.set_xlabel('Distance x (m)', fontsize=9)
    ax_colloc_heatmap.set_ylabel('Time t (days)', fontsize=9)
    ax_colloc_heatmap.set_title(f'Collocation Points\n(Δx={deltax:.1f} m, Δt={deltat:.1f} days)', fontsize=10, pad=5)
    # Set tick marks to only show 0, 0.5*x_max, and x_max for x, and 0, 0.5*t_final, t_final for t
    ax_colloc_heatmap.set_xticks([0, 0.5 * x_max, x_max])
    t_final_days = t_final_star * T
    ax_colloc_heatmap.set_yticks([0, 0.5 * t_final_days, t_final_days])
    ax_colloc_heatmap.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax_colloc_heatmap.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    # 2. PDE Loss Heatmap
    # Create grid for PDE loss heatmap (using finer, separately configurable resolution)
    x_star_heatmap_pde = np.linspace(0, 1, pde_loss_res_x + 1)  # +1 for bin edges
    t_star_heatmap_pde = np.linspace(0, t_final_star, pde_loss_res_t + 1)  # +1 for bin edges
    
    # Bin centers for PDE loss heatmap
    x_star_centers_pde = (x_star_heatmap_pde[:-1] + x_star_heatmap_pde[1:]) / 2
    t_star_centers_pde = (t_star_heatmap_pde[:-1] + t_star_heatmap_pde[1:]) / 2
    
    # Compute PDE residual at each grid point (using finer resolution)
    pde_residuals_grid = np.zeros((pde_loss_res_x, pde_loss_res_t))
    
    for i, x_star_val in enumerate(x_star_centers_pde):
        for j, t_star_val in enumerate(t_star_centers_pde):
            x_days = x_star_val * L
            t_days = t_star_val * T
            # Compute residual at this single point
            residual = compute_pde_residual_at_points(model, np.array([x_days]), t_days)
            pde_residuals_grid[i, j] = residual[0]
    
    # Compute squared residuals (loss) for heatmap
    pde_loss_grid = pde_residuals_grid**2
    
    # Calculate interval sizes in real dimensions for PDE loss heatmap
    deltax_pde = x_max / pde_loss_res_x
    deltat_pde = (t_final_star * T) / pde_loss_res_t
    
    # Create grayscale colormap (full black to white)
    cmap_pde_loss = plt.cm.gray_r  # reversed: white (low) to black (high)
    
    # Use real dimensions for extent (same as collocation heatmap)
    im_pde_loss = ax_pde_loss_heatmap.imshow(pde_loss_grid, aspect='auto', origin='lower',
                                             extent=[0, x_max, 0, t_final_star * T],
                                             cmap=cmap_pde_loss, interpolation='nearest')
    
    # Keep x label (shared with left heatmap), remove y label and tick labels (shares y-axis)
    ax_pde_loss_heatmap.set_xlabel('Distance x (m)', fontsize=9)
    ax_pde_loss_heatmap.set_ylabel('')  # Remove y label
    ax_pde_loss_heatmap.set_yticklabels([])  # Remove y tick labels
    ax_pde_loss_heatmap.set_title(f'PDE Loss\n(Δx={deltax_pde:.1f} m, Δt={deltat_pde:.1f} days)', fontsize=10, pad=5)
    # Set tick marks to only show 0, 0.5*x_max, and x_max (same pattern as collocation heatmap)
    ax_pde_loss_heatmap.set_xticks([0, 0.5 * x_max, x_max])
    ax_pde_loss_heatmap.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # Set y-axis ticks to match collocation heatmap (but no labels)
    ax_pde_loss_heatmap.set_yticks([0, 0.5 * t_final_days, t_final_days])
    
    # ===== Loss Plots (Right Column) =====
    if losses is not None:
        epochs = np.arange(1, len(losses['pde']) + 1)
        
        # PDE loss plot
        ax_pde_loss.plot(epochs, losses['pde'], linewidth=1.5, color='black', alpha=0.8)
        ax_pde_loss.set_ylabel('PDE Loss', fontsize=10)
        ax_pde_loss.set_xticklabels([])  # Remove x tick labels (shared with IC loss below)
        ax_pde_loss.set_yscale('log')
        # Set consistent tick formatting for loss plots
        ax_pde_loss.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax_pde_loss.yaxis.set_major_formatter(LogFormatterSciNotation())
        
        # IC loss plot
        ax_ic_loss.plot(epochs, losses['ic'], linewidth=1.5, color='black', alpha=0.8)
        ax_ic_loss.set_ylabel('IC Loss', fontsize=10)
        ax_ic_loss.set_xlabel('Epoch', fontsize=12)  # Only bottom plot has x label
        ax_ic_loss.set_yscale('log')
        # Set consistent tick formatting for loss plots
        ax_ic_loss.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
        ax_ic_loss.xaxis.set_major_formatter(FormatStrFormatter('%d'))  # Integer format for epochs
        ax_ic_loss.yaxis.set_major_formatter(LogFormatterSciNotation())
    else:
        # Hide loss plots if no losses provided
        ax_pde_loss.axis('off')
        ax_ic_loss.axis('off')
    
    # Apply plot styling to all axes
    axes_list = [ax_main, ax_pde, ax_ic]
    heatmap_axes_list = [ax_colloc_heatmap, ax_pde_loss_heatmap]
    if losses is not None:
        axes_list.extend([ax_pde_loss, ax_ic_loss])
    
    # Style heatmap axes
    for ax in heatmap_axes_list:
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
        
        # Style tick marks
        ax.tick_params(axis='x', which='major', 
                      colors=grid_color, 
                      width=grid_linewidth,
                      length=0)
        ax.tick_params(axis='y', which='major', 
                      colors=grid_color, 
                      width=grid_linewidth,
                      length=0)
        
        # Style tick labels and axis labels
        for label in ax.get_xticklabels():
            label.set_color('black')
            label.set_alpha(1.0)
        for label in ax.get_yticklabels():
            label.set_color('black')
            label.set_alpha(1.0)
        ax.xaxis.label.set_color('black')
        ax.xaxis.label.set_alpha(1.0)
        ax.yaxis.label.set_color('black')
        ax.yaxis.label.set_alpha(1.0)
        ax.title.set_color('black')
        ax.title.set_alpha(1.0)
    
    for ax in axes_list:
        # Apply grid - only show vertical (x-axis) grid lines for left column plots
        # Loss plots (right column) get full grid later
        is_loss_plot = losses is not None and ax in [ax_pde_loss, ax_ic_loss]
        if not is_loss_plot:
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
        
        # For loss plots, apply grid with both major and minor ticks (full grid, not just x-axis)
        if is_loss_plot:
            ax.grid(True, which='major', alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)
            ax.grid(True, which='minor', alpha=grid_alpha*0.5, color=grid_color, linewidth=grid_linewidth*0.5)
    
    # Create figure-level legend (centered, using whole plot area)
    # Get handles and labels from main plot
    handles, labels = ax_main.get_legend_handles_labels()
    # Create legend at figure level, centered, with more columns
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), 
                       ncol=6, frameon=False, fontsize=10,
                       labelspacing=0.5, columnspacing=1.2)
    # Style legend text items - make all text black
    for text in legend.get_texts():
        text.set_color('black')
        text.set_alpha(1.0)
    
    # Add epoch number in top right corner if provided
    if epoch is not None:
        fig.text(0.98, 0.98, f'Epoch: {epoch:,}', 
                transform=fig.transFigure,
                fontsize=14, 
                verticalalignment='top', 
                horizontalalignment='right',
                fontfamily='monospace',
                weight='bold', 
                color='black')
    
    # Save plot
    if output_dir is None:
        output_dir = plotting_params['plots_dir']
    
    if filename is None:
        if epoch is not None:
            # Check if we should overwrite frames with same name
            overwrite_frames = training_params.get('overwrite_gif_frames', False)
            if overwrite_frames:
                filename = 'pinn_simple_profiles_epoch.png'
            else:
                filename = f'pinn_simple_profiles_epoch_{epoch:05d}.png'
        else:
            filename = 'pinn_simple_profiles.png'
    
    plot_path = Path(output_dir) / filename
    plt.savefig(str(plot_path), dpi=plotting_params['dpi'], bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("PINN for 1D Contaminant Transport - Loss Distribution Adaptive Collocation Points")
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
    print(f"  Total epochs: {training_params['num_epochs']} (Adam: {training_params['adam_epochs']}, LBFGS: {training_params['lbfgs_epochs']})")
    print(f"  Learning rate: {training_params['lr']}")
    collocation_points_x_star = training_params['collocation_points_x_star']
    collocation_points_t_star = training_params['collocation_points_t_star']
    total_collocation = collocation_points_x_star * collocation_points_t_star
    print(f"  Collocation points: {total_collocation} total ({collocation_points_x_star} x* × {collocation_points_t_star} t*)")
    anchor_ratio = training_params.get('anchor_ratio', 0.0)
    if anchor_ratio > 0.0:
        num_anchor = int(total_collocation * anchor_ratio)
        num_adaptive = total_collocation - num_anchor
        print(f"  Anchor points: {num_anchor} ({anchor_ratio:.2%} of total) - permanently fixed")
        print(f"  Adaptive points: {num_adaptive} ({1.0 - anchor_ratio:.2%} of total) - redistributed every {training_params.get('adaptive_update_interval', 100)} epochs")
    else:
        print(f"  Anchor points: 0 (all points are adaptive)")
    print(f"  Adaptive update interval: {training_params.get('adaptive_update_interval', 100)} epochs")
    print(f"  Loss evaluation grid: {training_params.get('loss_evaluation_grid_x', 100)} × {training_params.get('loss_evaluation_grid_t', 100)}")
    print(f"  IC points: {training_params['num_ic']}")
    print(f"  BC points: {training_params['num_bc']}")
    
    # Create subdirectory for epoch exports
    epoch_export_dir = Path(plotting_params['plots_dir']) / 'epoch_exports'
    epoch_export_dir.mkdir(exist_ok=True)
    export_interval = training_params.get('export_interval', 100)
    if export_interval is not None:
        print(f"  Epoch export interval: {export_interval}")
        print(f"  Epoch exports will be saved to: {epoch_export_dir}")
    
    # Define callback function for epoch exports
    def plot_epoch_callback(model, epoch_num, current_losses, x_star_colloc=None, t_star_colloc=None):
        """Callback function to plot and save at each epoch checkpoint."""
        plot_concentration_profiles(
            model, 
            times_days=plotting_params['times_days'],
            training_params=training_params, 
            model_params=model_params,
            physics_params=physics_params,
            epoch=epoch_num,
            output_dir=str(epoch_export_dir),
            losses=current_losses,  # Pass current losses up to this epoch
            x_star_colloc=x_star_colloc,  # Pass actual collocation points
            t_star_colloc=t_star_colloc
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
                                physics_params=physics_params,
                                losses=losses)
    
    print("\n" + "="*60)
    print("PINN training and comparison complete!")
    print("="*60)
