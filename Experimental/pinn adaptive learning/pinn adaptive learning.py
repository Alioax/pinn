"""
Physics-Informed Neural Network (PINN) for 1D Contaminant Transport in an Aquifer
WITH ANALYTICAL SOLUTION COMPARISON

This implementation:
- Trains the PINN entirely in dimensionless form
- Accepts inputs in physical units (meters, days)
- Converts to dimensionless variables before evaluation
- Converts dimensionless output back to dimensional concentration
- Produces plots in physical units
- Compares PINN solution with analytical solution
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from torch.autograd import grad
import os
import sys
from pathlib import Path

# Import analytical solution from root-level folder
sys.path.append(str(Path(__file__).parent.parent.parent / 'analytical_solution'))
from analytical_solution import analytical_solution

# Import GIF creation function
from create_training_gif import create_training_gif

# ============================================================================
# Configuration Parameters
# ============================================================================
# 
# To modify the simulation, simply edit the dictionaries below:
# - physics_params: physical properties of the system
# - model_params: neural network architecture
# - training_params: training hyperparameters
# - plotting_params: visualization settings
#

# Physics parameters (dimensional)
physics_params = {
    'U': 0.1,                    # m/day (advection velocity)
    'D': 1e-7 * 86400,          # m²/day (dispersion coefficient, converted from m²/s)
    'C_0': 5.0,                 # kg/m³ (concentration scale)
    'L': 100.0,                 # m (length scale)
}

#### TO LATER LOOK INTO:
    # 'num_layers': 3,            # number of hidden layers
    # 'num_neurons': 20,          # number of neurons per hidden layer
    # 'lr': 0.1,                # learning rate


# Neural network model parameters
model_params = {
    'num_layers': 3,            # number of hidden layers
    'num_neurons': 10,          # number of neurons per hidden layer
    'activation': torch.nn.Tanh,  # activation function
}

# Training parameters
training_params = {
    'num_epochs': 2000,         # number of training epochs
    'lr': 0.001,                # learning rate
    # Collocation point configuration (hybrid uniform-adaptive)
    'num_uniform': 2000,        # number of uniformly-spaced points (fixed, never change)
    'num_adaptive': 0,          # number of adaptive points (redistributed around high-residual regions)
    'num_ic': 100,              # number of points for initial condition
    'num_bc': 100,              # number of points for boundary conditions
    't_final_star': 1.0,        # final dimensionless time
    'verbose': True,            # print training progress
    # Loss weights (balance different loss components)
    'weight_pde': 1,            # weight for PDE residual loss
    'weight_ic': 1,             # weight for initial condition loss
    'weight_inlet_bc': 1,       # weight for inlet boundary condition loss
    'weight_outlet_bc': 1,      # weight for outlet boundary condition loss
    'weight_monotonicity': 0,   # weight for monotonicity loss (dC/dx ≤ 0, C never rises)
    # Adaptive learning parameters
    'adaptive_refinement': False,  # enable/disable adaptive collocation point refinement
    'refinement_interval': 2000,   # number of epochs between refinements
    'residual_threshold_percentile': 5,  # top percentile threshold for high-residual regions (e.g., 5 = top 5%)
    'sigma_spatial': 0.05,      # standard deviation for spatial Gaussian distribution (dimensionless)
    'sigma_temporal': None,     # standard deviation for temporal Gaussian (None = auto-calculate)
    # Plotting parameters
    'plot_interval': 2001,       # number of epochs between diagnostic plots (for GIF animation)
}

# Plotting parameters
# Make plots_dir relative to script location
script_dir = Path(__file__).parent
plots_dir = script_dir / 'results'
gif_frames_dir = plots_dir / 'gif_frames'
plotting_params = {
    'times_days': [300, 700],  # times for concentration profiles
    'x_max': 100.0,             # maximum spatial coordinate (m)
    'num_points': 500,          # number of spatial points for profiles
    'num_x': 200,               # number of spatial points for contour
    'num_t': 200,               # number of time points for contour
    'dpi': 800,                 # resolution for saved figures
    'plots_dir': str(plots_dir),       # directory to save plots (absolute path)
    'gif_frames_dir': str(gif_frames_dir),  # directory to save GIF frame images
}

# Custom color palette (C0 to C5)
custom_colors = ['#212c68', '#e62f25', '#ff6900', '#f6c800', '#009148', '#2b2b2b']

# Set matplotlib default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

# Create plots directory if it doesn't exist
os.makedirs(plotting_params['plots_dir'], exist_ok=True)
os.makedirs(plotting_params['gif_frames_dir'], exist_ok=True)

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
    
    def __init__(self, num_layers=4, num_neurons=50, activation=nn.Tanh):
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

def compute_monotonicity_loss(model, x_star, t_star):
    """
    Compute loss to ensure C never rises as x increases (dC/dx ≤ 0).
    
    This enforces that the concentration is non-increasing in space,
    which is physically reasonable for contaminant transport away from source.
    
    Parameters:
        model: PINN model
        x_star: dimensionless spatial coordinate (tensor, requires_grad=True)
        t_star: dimensionless time (tensor, requires_grad=True)
    
    Returns:
        loss: penalty for positive dC/dx (tensor)
    """
    # Ensure inputs require gradients
    x_star = x_star.clone().detach().requires_grad_(True)
    t_star = t_star.clone().detach().requires_grad_(True)
    
    # Forward pass
    C_star = model(x_star, t_star)
    
    # Compute first derivative
    dC_dx_star = grad(C_star, x_star, grad_outputs=torch.ones_like(C_star),
                      create_graph=True, retain_graph=True)[0]
    
    # Penalize positive values of dC/dx (i.e., when C is increasing with x)
    # Use ReLU to only penalize positive values: max(0, dC/dx)
    violation = torch.relu(dC_dx_star)
    loss = torch.mean(violation**2)
    
    return loss

# ============================================================================
# Adaptive Collocation Point Selection
# ============================================================================

def select_adaptive_collocation_points(model, num_points, candidate_points, t_final_star, 
                                      residual_threshold_percentile=90):
    """
    Select collocation points adaptively based on PDE residual (RAR - Residual-based Adaptive Refinement).
    
    This function:
    1. Generates a large uniform grid of candidate points
    2. Evaluates PDE residual at all candidate points
    3. Identifies points above residual threshold (high-error regions)
    4. Samples new points preferentially from high-residual regions
    
    Parameters:
        model: PINN model (in eval mode for evaluation)
        num_points: number of collocation points to select
        candidate_points: number of candidate points to evaluate
        t_final_star: final dimensionless time
        residual_threshold_percentile: percentile threshold for high-residual regions (default 90)
    
    Returns:
        x_star_colloc: selected dimensionless spatial coordinates (tensor, requires_grad=True)
        t_star_colloc: selected dimensionless time coordinates (tensor, requires_grad=True)
    """
    model.eval()
    
    # Generate candidate grid uniformly in (x*, t*) space
    x_star_candidates = torch.rand(candidate_points, 1) * 1.0  # [0, 1]
    t_star_candidates = torch.rand(candidate_points, 1) * t_final_star  # [0, t_final_star]
    
    # Ensure candidates require gradients for residual computation
    x_star_temp = x_star_candidates.requires_grad_(True)
    t_star_temp = t_star_candidates.requires_grad_(True)
    
    # Evaluate PDE residual at all candidate points
    # Note: compute_pde_residual needs gradients enabled, but we detach after for selection
    residuals = compute_pde_residual(model, x_star_temp, t_star_temp)
    residual_magnitudes = torch.abs(residuals).detach().squeeze()
    
    # Find threshold for high-residual regions
    threshold_value = torch.quantile(residual_magnitudes, residual_threshold_percentile / 100.0)
    
    # Identify high-residual points
    high_residual_mask = residual_magnitudes >= threshold_value
    high_residual_indices = torch.where(high_residual_mask)[0]
    
    # If we have enough high-residual points, sample from them preferentially
    # Otherwise, use weighted sampling based on residual magnitudes
    if len(high_residual_indices) >= num_points:
        # Sample uniformly from high-residual points
        selected_indices = torch.randperm(len(high_residual_indices))[:num_points]
        selected_indices = high_residual_indices[selected_indices]
    else:
        # Use weighted sampling: probability proportional to residual magnitude
        # Normalize residuals to probabilities
        probabilities = residual_magnitudes / (residual_magnitudes.sum() + 1e-10)
        probabilities = probabilities / probabilities.sum()  # Ensure normalization
        
        # Sample indices according to probabilities
        selected_indices = torch.multinomial(probabilities, num_points, replacement=True)
    
    # Extract selected points
    x_star_selected = x_star_candidates[selected_indices].clone().detach().requires_grad_(True)
    t_star_selected = t_star_candidates[selected_indices].clone().detach().requires_grad_(True)
    
    # Verify we have the correct number of points
    assert x_star_selected.shape[0] == num_points, f"Expected {num_points} points, got {x_star_selected.shape[0]}"
    assert t_star_selected.shape[0] == num_points, f"Expected {num_points} points, got {t_star_selected.shape[0]}"
    
    # Return statistics for logging
    stats = {
        'mean_residual': residual_magnitudes.mean().item(),
        'max_residual': residual_magnitudes.max().item(),
        'threshold': threshold_value.item(),
        'high_residual_count': len(high_residual_indices),
        'selected_from_high_residual': len(high_residual_indices) >= num_points
    }
    
    return x_star_selected, t_star_selected, stats

# ============================================================================
# Uniform Collocation Point Generation
# ============================================================================

def generate_uniform_collocation_points(num_points, t_final_star):
    """
    Generate uniformly-spaced collocation points.
    
    Parameters:
        num_points: number of uniformly-spaced points to generate
        t_final_star: final dimensionless time
    
    Returns:
        x_star_uniform: uniformly-spaced dimensionless spatial coordinates (tensor)
        t_star_uniform: uniformly-distributed dimensionless time coordinates (tensor)
    """
    # Uniformly-spaced spatial points from 0 to 1 (dimensionless)
    x_star_uniform = torch.linspace(0, 1, num_points).reshape(-1, 1)
    
    # Uniformly-distributed temporal points from 0 to t_final_star
    # Use linspace for uniform distribution across time
    t_star_uniform = torch.linspace(0, t_final_star, num_points).reshape(-1, 1)
    
    return x_star_uniform, t_star_uniform

# ============================================================================
# Gaussian-Based Adaptive Collocation Point Selection
# ============================================================================

def select_adaptive_collocation_points_gaussian(model, num_points, x_star_uniform, t_star_uniform,
                                               residual_threshold_percentile=5,
                                               sigma_spatial=0.05, sigma_temporal=None):
    """
    Select adaptive collocation points using Gaussian distribution around high-residual regions.
    
    This function:
    1. Uses the provided uniform collocation points as candidates
    2. Evaluates PDE residual at all candidate points
    3. Identifies points above residual threshold (top percentile)
    4. Samples new points from normal distributions centered at high-residual locations
    
    Parameters:
        model: PINN model (in eval mode for evaluation)
        num_points: number of adaptive collocation points to select
        x_star_uniform: uniform dimensionless spatial coordinates to use as candidates (tensor)
        t_star_uniform: uniform dimensionless time coordinates to use as candidates (tensor)
        residual_threshold_percentile: top percentile threshold for high-residual regions (default 5 = top 5%)
        sigma_spatial: standard deviation for spatial Gaussian (dimensionless, default 0.05)
        sigma_temporal: standard deviation for temporal Gaussian (None = auto-calculate as sigma_spatial * t_final_star)
    
    Returns:
        x_star_selected: selected dimensionless spatial coordinates (tensor, requires_grad=True)
        t_star_selected: selected dimensionless time coordinates (tensor, requires_grad=True)
        stats: dictionary with selection statistics
    """
    model.eval()
    
    # Auto-calculate temporal sigma if not provided
    if sigma_temporal is None:
        t_final_star = t_star_uniform.max().item()
        sigma_temporal = sigma_spatial * t_final_star
    
    # Use uniform points as candidates (clone to avoid modifying originals)
    x_star_candidates = x_star_uniform.clone()
    t_star_candidates = t_star_uniform.clone()
    
    # Ensure candidates require gradients for residual computation
    x_star_temp = x_star_candidates.requires_grad_(True)
    t_star_temp = t_star_candidates.requires_grad_(True)
    
    # Evaluate PDE residual at all candidate points
    residuals = compute_pde_residual(model, x_star_temp, t_star_temp)
    residual_magnitudes = torch.abs(residuals).detach().squeeze()
    
    # Find threshold for high-residual regions (top percentile)
    # Convert top percentile to quantile: e.g., top 5% = 95th percentile = 0.95 quantile
    threshold_value = torch.quantile(residual_magnitudes, (100 - residual_threshold_percentile) / 100.0)
    
    # Identify high-residual points
    high_residual_mask = residual_magnitudes >= threshold_value
    high_residual_indices = torch.where(high_residual_mask)[0]
    high_residual_x = x_star_candidates[high_residual_indices]
    high_residual_t = t_star_candidates[high_residual_indices]
    
    # Collect selected points
    x_star_selected_list = []
    t_star_selected_list = []
    
    if len(high_residual_indices) > 0:
        # Distribute points around high-residual locations using Gaussian distribution
        # Calculate how many points per high-residual location
        points_per_location = max(1, num_points // len(high_residual_indices))
        remaining_points = num_points
        
        for i, (x_center, t_center) in enumerate(zip(high_residual_x, high_residual_t)):
            if remaining_points <= 0:
                break
            
            # Determine how many points to sample around this location
            if i == len(high_residual_indices) - 1:
                # Last location gets all remaining points
                n_samples = remaining_points
            else:
                n_samples = min(points_per_location, remaining_points)
            
            # Sample from normal distribution centered at this high-residual point
            x_samples = torch.normal(x_center.item(), sigma_spatial, size=(n_samples, 1))
            t_samples = torch.normal(t_center.item(), sigma_temporal, size=(n_samples, 1))
            
            # Clip to domain bounds
            x_samples = torch.clamp(x_samples, 0.0, 1.0)
            t_samples = torch.clamp(t_samples, 0.0, t_final_star)
            
            x_star_selected_list.append(x_samples)
            t_star_selected_list.append(t_samples)
            
            remaining_points -= n_samples
        
        # Concatenate all samples
        if len(x_star_selected_list) > 0:
            x_star_selected = torch.cat(x_star_selected_list, dim=0)
            t_star_selected = torch.cat(t_star_selected_list, dim=0)
        else:
            # Fallback: uniform random if no high-residual points
            x_star_selected = torch.rand(num_points, 1) * 1.0
            t_star_selected = torch.rand(num_points, 1) * t_final_star
    else:
        # Fallback: if no high-residual points found, use weighted sampling
        probabilities = residual_magnitudes / (residual_magnitudes.sum() + 1e-10)
        probabilities = probabilities / probabilities.sum()
        selected_indices = torch.multinomial(probabilities, num_points, replacement=True)
        x_star_selected = x_star_candidates[selected_indices]
        t_star_selected = t_star_candidates[selected_indices]
    
    # Ensure we have exactly num_points (handle rounding issues)
    if x_star_selected.shape[0] != num_points:
        if x_star_selected.shape[0] < num_points:
            # Add more points if needed
            n_needed = num_points - x_star_selected.shape[0]
            if len(high_residual_indices) > 0:
                # Sample more from high-residual locations
                idx = torch.randint(0, len(high_residual_indices), (n_needed,))
                x_centers = high_residual_x[idx]
                t_centers = high_residual_t[idx]
                x_add = torch.normal(x_centers.squeeze(), sigma_spatial, size=(n_needed, 1))
                t_add = torch.normal(t_centers.squeeze(), sigma_temporal, size=(n_needed, 1))
                x_add = torch.clamp(x_add, 0.0, 1.0)
                t_add = torch.clamp(t_add, 0.0, t_final_star)
                x_star_selected = torch.cat([x_star_selected, x_add], dim=0)
                t_star_selected = torch.cat([t_star_selected, t_add], dim=0)
            else:
                # Fallback to uniform random
                x_add = torch.rand(n_needed, 1) * 1.0
                t_add = torch.rand(n_needed, 1) * t_final_star
                x_star_selected = torch.cat([x_star_selected, x_add], dim=0)
                t_star_selected = torch.cat([t_star_selected, t_add], dim=0)
        else:
            # Remove excess points
            x_star_selected = x_star_selected[:num_points]
            t_star_selected = t_star_selected[:num_points]
    
    # Ensure points require gradients
    x_star_selected = x_star_selected.clone().detach().requires_grad_(True)
    t_star_selected = t_star_selected.clone().detach().requires_grad_(True)
    
    # Verify we have the correct number of points
    assert x_star_selected.shape[0] == num_points, f"Expected {num_points} points, got {x_star_selected.shape[0]}"
    assert t_star_selected.shape[0] == num_points, f"Expected {num_points} points, got {t_star_selected.shape[0]}"
    
    # Return statistics for logging
    stats = {
        'mean_residual': residual_magnitudes.mean().item(),
        'max_residual': residual_magnitudes.max().item(),
        'threshold': threshold_value.item(),
        'high_residual_count': len(high_residual_indices),
        'points_sampled_from_gaussian': len(high_residual_indices) > 0
    }
    
    return x_star_selected, t_star_selected, stats

# ============================================================================
# Training Function
# ============================================================================

def train_pinn(model, training_params=None):
    """
    Train the PINN model.
    
    Parameters:
        model: PINN model
        training_params: dictionary with training parameters
            - num_epochs: number of training epochs
            - lr: learning rate
            - num_uniform: number of uniformly-spaced points (default: 1000)
            - num_adaptive: number of adaptive points (default: 1000)
            - num_collocation: total collocation points (computed as num_uniform + num_adaptive if None)
            - num_ic: number of points for initial condition
            - num_bc: number of points for boundary conditions
            - t_final_star: final dimensionless time
            - verbose: print training progress
            - adaptive_refinement: enable/disable adaptive collocation point refinement
            - refinement_interval: number of epochs between refinements
            - residual_threshold_percentile: percentile threshold for high-residual regions (default: 5)
            - sigma_spatial: standard deviation for spatial Gaussian distribution (default: 0.05)
            - sigma_temporal: standard deviation for temporal Gaussian (None = auto-calculate)
    
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
    
    # Collocation point configuration (hybrid uniform-adaptive)
    num_uniform = training_params.get('num_uniform', 1000)
    num_adaptive = training_params.get('num_adaptive', 1000)
    num_collocation = training_params.get('num_collocation')
    if num_collocation is None:
        num_collocation = num_uniform + num_adaptive
    
    # Adaptive learning parameters
    adaptive_refinement = training_params.get('adaptive_refinement', False)
    refinement_interval = training_params.get('refinement_interval', 200)
    residual_threshold_percentile = training_params.get('residual_threshold_percentile', 5)
    sigma_spatial = training_params.get('sigma_spatial', 0.05)
    sigma_temporal = training_params.get('sigma_temporal', None)
    
    # Plotting parameters
    plot_interval = training_params.get('plot_interval', 100)
    
    # Loss weights
    weight_pde = training_params.get('weight_pde', 1.0)
    weight_ic = training_params.get('weight_ic', 1.0)
    weight_inlet_bc = training_params.get('weight_inlet_bc', 1.0)
    weight_outlet_bc = training_params.get('weight_outlet_bc', 1.0)
    weight_monotonicity = training_params.get('weight_monotonicity', 1.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss history
    losses = {
        'total': [],
        'pde': [],
        'ic': [],
        'inlet_bc': [],
        'outlet_bc': [],
        'monotonicity': []
    }
    
    # Initialize collocation points (hybrid uniform-adaptive approach)
    # For baseline mode (num_adaptive=0, adaptive_refinement=False), use random sampling each epoch
    # Otherwise, use fixed uniform points
    baseline_mode = (num_adaptive == 0 and not adaptive_refinement)
    
    if baseline_mode:
        # Baseline mode: will resample random points each epoch (matching baseline implementation)
        x_star_uniform = None  # Will be resampled each epoch
        t_star_uniform = None
        x_star_adaptive = None
        t_star_adaptive = None
        if verbose:
            print(f"Baseline mode: Random collocation points ({num_uniform} points, resampled each epoch)")
    else:
        # Uniform points: fixed, uniformly-spaced, never change
        x_star_uniform, t_star_uniform = generate_uniform_collocation_points(num_uniform, t_final_star)
        x_star_uniform = x_star_uniform.requires_grad_(True)
        t_star_uniform = t_star_uniform.requires_grad_(True)
        
        if adaptive_refinement:
            # Initialize adaptive points with same uniform distribution as uniform points (epoch 0)
            # They will be redistributed adaptively starting from epoch 1
            x_star_adaptive, t_star_adaptive = generate_uniform_collocation_points(num_adaptive, t_final_star)
            x_star_adaptive = x_star_adaptive.requires_grad_(True)
            t_star_adaptive = t_star_adaptive.requires_grad_(True)
            if verbose:
                print(f"Initial hybrid collocation points:")
                print(f"  Uniform points: {num_uniform} (fixed, uniformly-spaced)")
                print(f"  Adaptive points: {num_adaptive} (uniformly-spaced, will adapt during training)")
            
            # Export initial plot (losses will be empty at start, so pass None)
            plot_filename = 'pinn_analytical_comparison_with_pde_loss_epoch_0000.png'
            plot_concentration_with_pde_loss(
                model,
                times_days=plotting_params['times_days'],
                num_collocation=100,
                filename=plot_filename,
                collocation_points=((x_star_uniform, t_star_uniform), (x_star_adaptive, t_star_adaptive)),
                losses=None,  # No losses yet at initialization
                output_dir=plotting_params['gif_frames_dir']  # Save to gif_frames directory
            )
            model.train()  # Return to training mode
        else:
            # Use uniform random sampling for adaptive points (non-adaptive mode)
            x_star_adaptive = torch.rand(num_adaptive, 1, requires_grad=True) * 1.0  # [0, 1]
            t_star_adaptive = torch.rand(num_adaptive, 1, requires_grad=True) * t_final_star  # [0, t_final_star]
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Baseline mode: resample random collocation points each epoch (matching baseline implementation)
        if baseline_mode:
            x_star_all = torch.rand(num_uniform, 1)
            t_star_all = torch.rand(num_uniform, 1) * t_final_star
            x_star_all = x_star_all.requires_grad_(True)
            t_star_all = t_star_all.requires_grad_(True)
        # Adaptive refinement: replace only adaptive points periodically (uniform points stay fixed)
        elif adaptive_refinement and (epoch + 1) % refinement_interval == 0 and epoch > 0:
            # Store current residual for comparison (on combined points)
            model.eval()
            temp_x_all = torch.cat([x_star_uniform, x_star_adaptive]).clone().detach().requires_grad_(True)
            temp_t_all = torch.cat([t_star_uniform, t_star_adaptive]).clone().detach().requires_grad_(True)
            old_residuals = compute_pde_residual(model, temp_x_all, temp_t_all)
            old_mean_residual = torch.abs(old_residuals).detach().mean().item()
            
            # Select new adaptive points (keep uniform points unchanged)
            x_star_adaptive, t_star_adaptive, stats = select_adaptive_collocation_points_gaussian(
                model, num_adaptive, x_star_uniform, t_star_uniform,
                residual_threshold_percentile, sigma_spatial, sigma_temporal
            )
            
            # Log refinement statistics
            if verbose:
                print(f"\nEpoch {epoch+1}: Adaptive refinement performed")
                print(f"  Uniform points: {num_uniform} (unchanged)")
                print(f"  Adaptive points: {num_adaptive} (redistributed)")
                print(f"  Previous mean residual: {old_mean_residual:.6e}")
                print(f"  New mean residual: {stats['mean_residual']:.6e}")
                print(f"  Max residual: {stats['max_residual']:.6e}")
                print(f"  Threshold: {stats['threshold']:.6e}")
                print(f"  High-residual points: {stats['high_residual_count']}/{num_uniform}")
                print(f"  Points sampled from Gaussian: {stats['points_sampled_from_gaussian']}")
            
            # Export plot at this refinement epoch
            model.eval()  # Ensure model is in eval mode for plotting
            plot_filename = f'pinn_analytical_comparison_with_pde_loss_epoch_{epoch+1:04d}.png'
            plot_concentration_with_pde_loss(
                model, 
                times_days=plotting_params['times_days'],
                num_collocation=100,
                filename=plot_filename,
                collocation_points=((x_star_uniform, t_star_uniform), (x_star_adaptive, t_star_adaptive)),
                losses=losses,  # Pass current loss history
                output_dir=plotting_params['gif_frames_dir']  # Save to gif_frames directory
            )
            model.train()  # Return to training mode
        
        # Generate plot every plot_interval epochs (for GIF animation)
        if (epoch + 1) % plot_interval == 0:
            model.eval()  # Ensure model is in eval mode for plotting
            plot_filename = f'pinn_analytical_comparison_with_pde_loss_epoch_{epoch+1:04d}.png'
            plot_concentration_with_pde_loss(
                model,
                times_days=plotting_params['times_days'],
                num_collocation=100,
                filename=plot_filename,
                collocation_points=((x_star_uniform, t_star_uniform), (x_star_adaptive, t_star_adaptive)),
                losses=losses,  # Pass current loss history
                output_dir=plotting_params['gif_frames_dir']  # Save to gif_frames directory
            )
            model.train()  # Return to training mode
        
        # Combine uniform and adaptive points for training (uniform points never change)
        # Skip if baseline_mode (already set above)
        if not baseline_mode:
            x_star_all = torch.cat([x_star_uniform, x_star_adaptive])
            t_star_all = torch.cat([t_star_uniform, t_star_adaptive])
            
            # Ensure points require gradients for training
            x_star_all = x_star_all.clone().detach().requires_grad_(True)
            t_star_all = t_star_all.clone().detach().requires_grad_(True)
        
        model.train()  # Ensure model is in training mode
        
        # PDE residual loss (using combined uniform + adaptive points)
        pde_residual = compute_pde_residual(model, x_star_all, t_star_all)
        pde_loss = torch.mean(pde_residual**2)
        
        # Initial condition loss
        x_star_ic = torch.rand(num_ic, 1) * 1.0  # [0, 1]
        ic_loss = compute_initial_condition_loss(model, x_star_ic)
        
        # Boundary condition losses
        t_star_bc = torch.rand(num_bc, 1) * t_final_star  # [0, t_final_star]
        inlet_bc_loss, outlet_bc_loss = compute_boundary_condition_losses(model, t_star_bc)
        
        # Monotonicity loss (d²C/dx² ≤ 0) - use combined points
        monotonicity_loss = compute_monotonicity_loss(model, x_star_all, t_star_all)
        
        # Total loss with weights
        total_loss = (weight_pde * pde_loss + 
                     weight_ic * ic_loss + 
                     weight_inlet_bc * inlet_bc_loss + 
                     weight_outlet_bc * outlet_bc_loss +
                     weight_monotonicity * monotonicity_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        losses['total'].append(total_loss.item())
        losses['pde'].append(pde_loss.item())
        losses['ic'].append(ic_loss.item())
        losses['inlet_bc'].append(inlet_bc_loss.item())
        losses['outlet_bc'].append(outlet_bc_loss.item())
        losses['monotonicity'].append(monotonicity_loss.item())
        
        # Print progress
        if verbose and (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Total: {total_loss.item():.6e} | "
                  f"PDE: {pde_loss.item():.6e} | "
                  f"IC: {ic_loss.item():.6e} | "
                  f"Inlet BC: {inlet_bc_loss.item():.6e} | "
                  f"Outlet BC: {outlet_bc_loss.item():.6e} | "
                  f"Monotonicity: {monotonicity_loss.item():.6e}")
    
    return losses

# ============================================================================
# Plot Styling Helper Function
# ============================================================================

def apply_plot_style(ax=None, x_min=None, x_max=None):
    """
    Apply consistent styling to plots:
    - Remove all spines
    - Match tick marks, labels, and spines to grid style
    - Set grid opacity to 0.3
    - Show only vertical (x-axis) grid lines, no horizontal (y-axis) grid lines
    - Remove grid lines at x=0 and x=x_max boundaries
    - Match axis labels opacity to tick labels
    """
    if ax is None:
        ax = plt.gca()
    
    # Grid styling
    grid_alpha = 0.3
    grid_color = 'black'
    grid_linewidth = 0.4
    
    # Get x-axis limits if not provided
    if x_min is None or x_max is None:
        x_min, x_max = ax.get_xlim()
    
    # Apply grid - only show vertical (x-axis) grid lines, no horizontal (y-axis) grid lines
    ax.grid(True, axis='x', alpha=grid_alpha, color=grid_color, linewidth=grid_linewidth)
    
    # Remove vertical grid lines at x=x_min (0) and x=x_max (L)
    # Match grid lines to tick positions by index
    x_ticks = ax.get_xticks()
    xgridlines = ax.get_xgridlines()
    
    # Find indices of ticks at x=x_min (0) and x=x_max (L)
    x_min_val = x_min if x_min is not None else 0.0
    indices_to_hide = []
    for i, tick_pos in enumerate(x_ticks):
        if abs(tick_pos - x_min_val) < 1e-6 or abs(tick_pos - x_max) < 1e-6:
            indices_to_hide.append(i)
    
    # Hide only those specific grid lines
    for i in indices_to_hide:
        if i < len(xgridlines):
            xgridlines[i].set_visible(False)
    
    # Show all spines (for layout visibility)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # Style spines to match grid
    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_alpha(grid_alpha)
        spine.set_linewidth(grid_linewidth)
    
    # Style tick marks to match grid exactly
    # X-axis: remove tick marks (length=0) but keep labels
    ax.tick_params(axis='x', which='major', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    ax.tick_params(axis='x', which='minor', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    # Y-axis: remove tick marks (length=0) but keep labels
    ax.tick_params(axis='y', which='major', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    ax.tick_params(axis='y', which='minor', 
                   colors=grid_color, 
                   width=grid_linewidth,
                   length=0)
    
    # Explicitly set tick mark properties (x-axis markers are hidden, but set properties anyway)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
    
    # Style tick labels to match grid
    for label in ax.get_xticklabels():
        label.set_color(grid_color)
        label.set_alpha(grid_alpha)
    for label in ax.get_yticklabels():
        label.set_color(grid_color)
        label.set_alpha(grid_alpha)
    
    # Style axis labels to match tick labels
    ax.xaxis.label.set_color(grid_color)
    ax.xaxis.label.set_alpha(grid_alpha)
    ax.yaxis.label.set_color(grid_color)
    ax.yaxis.label.set_alpha(grid_alpha)

# ============================================================================
# Evaluation and Plotting Functions (Dimensional Output)
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

def plot_concentration_profiles(model, times_days=None, x_max=None, num_points=None, num_collocation=100):
    """
    Plot concentration profiles at selected physical times.
    Shows both PINN solution (solid) and analytical solution (dashed).
    Includes PDE residual subplot below.
    
    Parameters:
        model: trained PINN model
        times_days: list of times in days (default: from plotting_params)
        x_max: maximum spatial coordinate (m) (default: from plotting_params)
        num_points: number of spatial points (default: from plotting_params)
        num_collocation: number of collocation points for residual evaluation
    """
    # Set font family
    plt.rcParams["font.family"] = "Times New Roman"
    
    if times_days is None:
        times_days = plotting_params['times_days']
    if x_max is None:
        x_max = plotting_params['x_max']
    if num_points is None:
        num_points = plotting_params['num_points']
    x_plot = np.linspace(0, x_max, num_points)
    
    # Create figure with two subplots: concentration (top) and PDE residual (bottom)
    fig = plt.figure(figsize=(5, 4.3))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Concentration plot
    ax2 = fig.add_subplot(gs[1, 0])  # PDE residual plot
    
    # Get current color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # ========================================================================
    # Top subplot: Concentration profiles
    # ========================================================================
    plt.sca(ax1)
    
    # Add legend entries for PINN and Analytical at the beginning
    # Use invisible plots to create legend entries
    plt.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    plt.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')
    
    # Plot in reverse order so legend shows 0 at bottom, 1000 at top
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time (cycling through colors)
        color = colors[idx % len(colors)]
        
        # Plot PINN solution (solid line)
        C_pinn = evaluate_dimensional(model, x_plot, t_days)
        plt.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Plot analytical solution (dashed line) with same color
        C_analytical = analytical_solution(x_plot, t_days)
        plt.plot(x_plot, C_analytical, linewidth=2, linestyle='--', color=color, alpha=0.7)
        
        # Add marker-only entry for legend (square marker, no line)
        plt.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{t_days} days')
    
    plt.ylabel('Concentration C (kg/m³)', fontsize=12)
    
    # Create legend at top
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), 
                       ncol=4, frameon=False, fontsize=10,
                       labelspacing=0.5, columnspacing=1.2)
    # Style legend text items - make all text black
    for text in legend.get_texts():
        text.set_color('black')
        text.set_alpha(1.0)
    
    plt.xlim(0, x_max)
    plt.ylim(-0.25, 5.25)
    
    # Set x-axis ticks to include 0, L (x_max), and advective distances (U * time) for each time in legend
    advective_distances = [U * t for t in times_days]
    # Filter out distances that are beyond x_max
    advective_distances = [d for d in advective_distances if d <= x_max]
    # Combine: 0, advective distances, and L (x_max)
    x_ticks = [0.0] + advective_distances + [x_max]
    # Remove duplicates and sort
    x_ticks = sorted(list(set(x_ticks)))
    ax1.set_xticks(x_ticks)
    ax1.set_xlabel('')  # Remove xlabel (shared with bottom subplot)
    
    # Filter out unwanted y-ticks
    ticks = ax1.get_yticks()
    # Remove ticks below -0.1 and above 5.25
    ticks = [t for t in ticks if -0.1 <= t <= 5.25]
    ax1.set_yticks(ticks)
    
    apply_plot_style(ax=ax1, x_min=0, x_max=x_max)
    
    # Remove top and right spines from top subplot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Make all text black in top subplot
    for label in ax1.get_xticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    for label in ax1.get_yticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    ax1.xaxis.label.set_color('black')
    ax1.xaxis.label.set_alpha(1.0)
    ax1.yaxis.label.set_color('black')
    ax1.yaxis.label.set_alpha(1.0)
    
    # ========================================================================
    # Bottom subplot: PDE residual
    # ========================================================================
    plt.sca(ax2)
    
    # Sample collocation points along x-axis for residual evaluation
    x_colloc = np.linspace(0, x_max, num_collocation)
    
    # Convert to dimensionless
    x_star_colloc, _ = to_dimensionless(x_colloc, np.zeros_like(x_colloc))
    
    model.eval()  # Set to eval mode (but we still need gradients for residuals)
    
    # Plot PDE residual separately for each time
    for idx, t_days in enumerate(reversed(times_days)):
        color = colors[idx % len(colors)]
        _, t_star = to_dimensionless(np.zeros_like(x_colloc), np.full_like(x_colloc, t_days))
        x_star_tensor = torch.tensor(x_star_colloc, dtype=torch.float32, requires_grad=True)
        t_star_tensor = torch.tensor(t_star, dtype=torch.float32, requires_grad=True)
        pde_residual = compute_pde_residual(model, x_star_tensor, t_star_tensor)
        pde_residual = pde_residual.detach().cpu().numpy()
        plt.plot(x_colloc, pde_residual, linewidth=2, linestyle='--', color=color)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.4, alpha=0.3)
    plt.ylabel('PDE Residual', fontsize=12)
    plt.xlabel('Distance x (m)', fontsize=12)
    ax2.set_xticks(x_ticks)
    apply_plot_style(ax=ax2, x_min=0, x_max=x_max)
    
    # Remove top and right spines from bottom subplot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Make all text black in bottom subplot
    for label in ax2.get_xticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    for label in ax2.get_yticklabels():
        label.set_color('black')
        label.set_alpha(1.0)
    ax2.xaxis.label.set_color('black')
    ax2.xaxis.label.set_alpha(1.0)
    ax2.yaxis.label.set_color('black')
    ax2.yaxis.label.set_alpha(1.0)
    
    plt.tight_layout()
    plot_path = os.path.join(plotting_params['plots_dir'], 'pinn_analytical_comparison.png')
    plt.savefig(plot_path, dpi=plotting_params['dpi'], bbox_inches='tight')
    print(f"Comparison plot saved to '{plot_path}'")
    
    # Export PDF copy to report/figs/
    script_dir = Path(__file__).parent
    report_figs_dir = script_dir.parent.parent / 'report' / 'figs'
    report_figs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_figs_dir / 'pinn_analytical_comparison.pdf'
    plt.savefig(str(pdf_path), format='pdf', bbox_inches='tight')
    print(f"PDF copy saved to '{pdf_path}'")
    
    plt.close()

def plot_concentration_with_pde_loss(model, times_days=None, x_max=None, num_points=None, num_collocation=100, filename=None, collocation_points=None, losses=None, output_dir=None):
    """
    Plot concentration profiles with analytical comparison (top subplot) and 
    multiple residual subplots (PDE, IC, Inlet BC, Outlet BC) along x-axis, 
    sharing the same x-axis. Residual subplots are smaller in height.
    Right column shows error vs epochs for each residual component.
    
    Parameters:
        model: trained PINN model
        times_days: list of times in days (default: from plotting_params)
        x_max: maximum spatial coordinate (m) (default: from plotting_params)
        num_points: number of spatial points (default: from plotting_params)
        num_collocation: number of collocation points for residual evaluation
        filename: custom filename for the plot (default: 'pinn_analytical_comparison_with_pde_loss.png')
        collocation_points: tuple for collocation points visualization (optional)
            - New format: ((x_uniform, t_uniform), (x_adaptive, t_adaptive)) for hybrid approach
            - Old format: (x_star, t_star) for backward compatibility
        losses: dictionary of loss history with keys ['pde', 'ic', 'inlet_bc', 'outlet_bc'] (optional)
        output_dir: directory to save the plot (default: plotting_params['plots_dir'])
    """
    if times_days is None:
        times_days = plotting_params['times_days']
    if x_max is None:
        x_max = plotting_params['x_max']
    if num_points is None:
        num_points = plotting_params['num_points']
    
    # Create figure with two columns: left (spatial) and right (temporal/epochs)
    # Height ratios: concentration plot is larger, heatmap is very small, residual plots are smaller
    # Calculate heatmap height to make cells square: if x_max = 100m and 20 cells, each cell width = 5m
    # For square cells, height should be 5m * (figure_width_in_inches / x_max_in_plot_units)
    # Approximate: figure width is ~14 inches, left column is ~half, so ~7 inches
    # x_max = 100m, so 1m = 7/100 = 0.07 inches
    # For 20 cells of width 5m each, height should be 5m = 0.35 inches
    # But we'll use a fixed small height and adjust aspect ratio
    fig = plt.figure(figsize=(14, 10.3))
    gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[4, 0.5, 2, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # Left column: spatial plots
    ax1 = fig.add_subplot(gs[0, 0])  # Concentration plot (larger)
    ax_heatmap = fig.add_subplot(gs[1, 0], sharex=ax1)  # Collocation points distribution heatmap (very small)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)  # PDE residual vs x
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)  # IC residual vs x
    ax4 = fig.add_subplot(gs[4, 0], sharex=ax1)  # Inlet BC residual vs x
    ax5 = fig.add_subplot(gs[5, 0], sharex=ax1)  # Outlet BC residual vs x
    
    # Right column: temporal plots (error vs epochs)
    ax2_right = fig.add_subplot(gs[2, 1])  # PDE error vs epochs
    ax3_right = fig.add_subplot(gs[3, 1])  # IC error vs epochs
    ax4_right = fig.add_subplot(gs[4, 1])  # Inlet BC error vs epochs
    ax5_right = fig.add_subplot(gs[5, 1])  # Outlet BC error vs epochs
    
    # Get current color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    x_plot = np.linspace(0, x_max, num_points)
    
    # ========================================================================
    # Top subplot: Concentration profiles (same as original plot)
    # ========================================================================
    plt.sca(ax1)
    
    # Add legend entries for PINN, Analytical, and Residual
    plt.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    plt.plot([], [], linewidth=2, linestyle=':', color='black', label='Analytical')
    plt.plot([], [], linewidth=2, linestyle='--', color='black', label='Residual')
    
    # Plot in reverse order so legend shows 0 at bottom, 1000 at top
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time (cycling through colors)
        color = colors[idx % len(colors)]
        
        # Plot PINN solution (solid line)
        C_pinn = evaluate_dimensional(model, x_plot, t_days)
        plt.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Plot analytical solution (dotted line) with same color
        C_analytical = analytical_solution(x_plot, t_days)
        plt.plot(x_plot, C_analytical, linewidth=2, linestyle=':', color=color, alpha=0.7)
        
        # Add marker-only entry for legend (square marker, no line)
        plt.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{t_days} days')
    
    plt.ylabel('Concentration C (kg/m³)', fontsize=12)
    
    # Collect handles and labels for figure-level legend
    handles, labels = ax1.get_legend_handles_labels()
    
    plt.xlim(0, x_max)
    plt.ylim(-0.25, 5.25)
    
    # Set x-axis ticks to include 0, L (x_max), and advective distances (U * time) for each time in legend
    advective_distances = [U * t for t in times_days]
    # Filter out distances that are beyond x_max
    advective_distances = [d for d in advective_distances if d <= x_max]
    # Combine: 0, advective distances, and L (x_max)
    x_ticks = [0.0] + advective_distances + [x_max]
    # Remove duplicates and sort
    x_ticks = sorted(list(set(x_ticks)))
    ax1.set_xticks(x_ticks)
    
    # Filter out unwanted y-ticks
    ticks = ax1.get_yticks()
    # Remove ticks below -0.1 and above 5.25
    ticks = [t for t in ticks if -0.1 <= t <= 5.25]
    ax1.set_yticks(ticks)
    
    apply_plot_style(ax=ax1, x_min=0, x_max=x_max)
    
    # ========================================================================
    # Subplot: Collocation points distribution heatmap
    # ========================================================================
    plt.sca(ax_heatmap)
    
    if collocation_points is not None:
        # Handle both old format (single tuple) and new format (tuple of uniform and adaptive)
        if isinstance(collocation_points[0], tuple):
            # New format: ((x_uniform, t_uniform), (x_adaptive, t_adaptive))
            (x_star_uniform, t_star_uniform), (x_star_adaptive, t_star_adaptive) = collocation_points
            # Combine uniform and adaptive points for heatmap
            if isinstance(x_star_uniform, torch.Tensor):
                x_star_uniform_np = x_star_uniform.detach().cpu().numpy().flatten()
                x_star_adaptive_np = x_star_adaptive.detach().cpu().numpy().flatten()
            else:
                x_star_uniform_np = np.asarray(x_star_uniform).flatten()
                x_star_adaptive_np = np.asarray(x_star_adaptive).flatten()
            x_colloc_dimensional = np.concatenate([x_star_uniform_np * L, x_star_adaptive_np * L])
        else:
            # Old format: (x_star_colloc, t_star_colloc) - backward compatibility
            x_star_colloc, t_star_colloc = collocation_points
            if isinstance(x_star_colloc, torch.Tensor):
                x_star_colloc_np = x_star_colloc.detach().cpu().numpy().flatten()
            else:
                x_star_colloc_np = np.asarray(x_star_colloc).flatten()
            x_colloc_dimensional = x_star_colloc_np * L
        
        # Divide domain into 20 segments (for square cells)
        num_segments = 20
        segment_edges = np.linspace(0, x_max, num_segments + 1)
        segment_width = x_max / num_segments
        
        # Count points in each segment
        counts, _ = np.histogram(x_colloc_dimensional, bins=segment_edges)
        
        # Normalize counts for visualization (0 to 1)
        if counts.max() > 0:
            counts_normalized = counts / counts.max()
        else:
            counts_normalized = counts
        
        # Create heatmap data (1 row, 20 columns)
        heatmap_data = counts_normalized.reshape(1, -1)
        
        # Set y-axis limits to match segment width for square cells
        plt.ylim(0, segment_width)
        plt.xlim(0, x_max)
        
        # Calculate aspect ratio to make cells square
        # For square cells: each cell should have equal width and height in data coordinates
        # width_per_cell = x_max / num_segments
        # height_per_cell = segment_width = x_max / num_segments
        # So data aspect ratio is already 1:1 per cell
        # But we need to account for physical axes dimensions
        # Get axes bbox (this needs to be done after figure is laid out)
        fig.canvas.draw()  # Force layout calculation
        bbox = ax_heatmap.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width_inches = bbox.width
        height_inches = bbox.height
        
        # Calculate aspect: (data_x_range / data_y_range) * (physical_height / physical_width)
        # For square cells in data space: data_x_range / num_segments = data_y_range
        # So: aspect = (x_max / segment_width) * (height_inches / width_inches)
        # Since segment_width = x_max / num_segments: aspect = num_segments * (height_inches / width_inches)
        aspect_ratio = num_segments * (height_inches / width_inches)
        
        # Create custom colormap: white to 50% gray (0 to 0.5 in gray scale)
        colors_list = [(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]  # White to 50% gray
        custom_cmap = LinearSegmentedColormap.from_list('white_to_gray', colors_list, N=256)
        
        # Create simple heatmap using imshow with custom colormap
        im = plt.imshow(heatmap_data, aspect=aspect_ratio, cmap=custom_cmap, vmin=0, vmax=1,
                       extent=[0, x_max, 0, segment_width], interpolation='nearest')
        
        # Add text labels showing point counts in each cell (all black)
        for i in range(num_segments):
            cell_center_x = (segment_edges[i] + segment_edges[i+1]) / 2
            cell_center_y = segment_width / 2
            plt.text(cell_center_x, cell_center_y, str(int(counts[i])), 
                    ha='center', va='center', fontsize=7, 
                    color='black', weight='bold')
    
    # Remove all axis elements
    ax_heatmap.set_yticks([])
    ax_heatmap.set_ylabel('')
    ax_heatmap.set_xticks([])
    ax_heatmap.set_xlabel('')
    ax_heatmap.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_heatmap.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_heatmap.grid(False)
    # Show spines with styling
    grid_alpha = 0.3
    grid_color = 'black'
    grid_linewidth = 0.4
    ax_heatmap.spines['top'].set_visible(True)
    ax_heatmap.spines['right'].set_visible(True)
    ax_heatmap.spines['left'].set_visible(True)
    ax_heatmap.spines['bottom'].set_visible(True)
    for spine in ax_heatmap.spines.values():
        spine.set_color(grid_color)
        spine.set_alpha(grid_alpha)
        spine.set_linewidth(grid_linewidth)
    plt.xlim(0, x_max)
    # ylim is set above in heatmap section if collocation_points exist, otherwise set default
    if collocation_points is None:
        plt.ylim(0, 1)
    
    # Sample collocation points along x-axis for residual evaluation
    x_colloc = np.linspace(0, x_max, num_collocation)
    
    # Convert to dimensionless
    x_star_colloc, _ = to_dimensionless(x_colloc, np.zeros_like(x_colloc))
    
    model.eval()  # Set to eval mode (but we still need gradients for residuals)
    
    # ========================================================================
    # Subplot 2: PDE residual along x-axis
    # ========================================================================
    plt.sca(ax2)
    
    # Plot PDE residual separately for each time
    for idx, t_days in enumerate(reversed(times_days)):
        color = colors[idx % len(colors)]
        _, t_star = to_dimensionless(np.zeros_like(x_colloc), np.full_like(x_colloc, t_days))
        x_star_tensor = torch.tensor(x_star_colloc, dtype=torch.float32, requires_grad=True)
        t_star_tensor = torch.tensor(t_star, dtype=torch.float32, requires_grad=True)
        pde_residual = compute_pde_residual(model, x_star_tensor, t_star_tensor)
        pde_residual = pde_residual.detach().cpu().numpy()
        plt.plot(x_colloc, pde_residual, linewidth=2, linestyle='--', color=color)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.4, alpha=0.3)
    plt.ylabel('PDE Residual', fontsize=12)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel('')  # Remove xlabel (shared with bottom row)
    apply_plot_style(ax=ax2, x_min=0, x_max=x_max)
    
    # ========================================================================
    # Subplot 3: Initial condition residual along x-axis
    # ========================================================================
    plt.sca(ax3)
    
    # IC residual: C*(x*, 0) - 0 = C*(x*, 0) at t=0
    # This is time-independent, so plot once with first color
    x_star_ic = torch.tensor(x_star_colloc, dtype=torch.float32)
    t_star_ic = torch.zeros_like(x_star_ic)
    C_star_ic = model(x_star_ic, t_star_ic)
    ic_residual = C_star_ic.detach().cpu().numpy().flatten()
    
    # Use first color from reversed times (600 days color)
    color = colors[0]
    plt.plot(x_colloc, ic_residual, linewidth=2, linestyle='--', color=color)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.4, alpha=0.3)
    plt.ylabel('IC', fontsize=12)
    ax3.set_xticks(x_ticks)
    ax3.set_xlabel('')  # Remove xlabel (shared with bottom row)
    apply_plot_style(ax=ax3, x_min=0, x_max=x_max)
    
    # ========================================================================
    # Subplot 4: Inlet boundary condition residual
    # ========================================================================
    plt.sca(ax4)
    
    # Inlet BC residual: C*(0, t*) - 1, evaluated at x=0 for each time
    for idx, t_days in enumerate(reversed(times_days)):
        color = colors[idx % len(colors)]
        _, t_star = to_dimensionless(np.array([0.0]), np.array([t_days]))
        x_star_inlet = torch.tensor([0.0], dtype=torch.float32)
        t_star_inlet = torch.tensor(t_star, dtype=torch.float32)
        C_star_inlet = model(x_star_inlet, t_star_inlet)
        inlet_residual = (C_star_inlet - 1.0).detach().cpu().numpy().item()
        # Plot as constant value along x-axis
        plt.plot(x_colloc, np.full_like(x_colloc, inlet_residual), linewidth=2, linestyle='--', color=color)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.4, alpha=0.3)
    plt.ylabel('Inlet BC', fontsize=12)
    ax4.set_xticks(x_ticks)
    ax4.set_xlabel('')  # Remove xlabel (shared with bottom row)
    apply_plot_style(ax=ax4, x_min=0, x_max=x_max)
    
    # ========================================================================
    # Subplot 5: Outlet boundary condition residual
    # ========================================================================
    plt.sca(ax5)
    
    # Outlet BC residual: ∂C*/∂x*(1, t*) - 0, evaluated at x*=1 for each time
    for idx, t_days in enumerate(reversed(times_days)):
        color = colors[idx % len(colors)]
        _, t_star = to_dimensionless(np.array([x_max]), np.array([t_days]))
        x_star_outlet = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        t_star_outlet = torch.tensor(t_star, dtype=torch.float32, requires_grad=True)
        C_star_outlet = model(x_star_outlet, t_star_outlet)
        dC_dx_star_outlet = grad(C_star_outlet, x_star_outlet,
                                 grad_outputs=torch.ones_like(C_star_outlet),
                                 create_graph=True, retain_graph=True)[0]
        outlet_residual = dC_dx_star_outlet.detach().cpu().numpy().item()
        # Plot as constant value along x-axis
        plt.plot(x_colloc, np.full_like(x_colloc, outlet_residual), linewidth=2, linestyle='--', color=color)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.4, alpha=0.3)
    plt.ylabel('Outlet BC', fontsize=12)
    plt.xlabel('Distance x (m)', fontsize=12)
    ax5.set_xticks(x_ticks)
    apply_plot_style(ax=ax5, x_min=0, x_max=x_max)
    
    # ========================================================================
    # Right Column: Error vs Epochs
    # ========================================================================
    
    if losses is not None:
        epochs = np.arange(1, len(losses['pde']) + 1)
        
        # PDE Error vs Epochs
        plt.sca(ax2_right)
        if len(losses['pde']) > 0:
            plt.semilogy(epochs, losses['pde'], linewidth=2, color='black', alpha=0.7)
        ax2_right.set_ylabel('')  # Remove ylabel (same as left column)
        ax2_right.set_xlabel('')  # Remove xlabel (shared with bottom row)
        ax2_right.set_xticklabels([])  # Remove xticklabels (shared with bottom row)
        apply_plot_style(ax=ax2_right)
        ax2_right.grid(True, axis='y', alpha=0.3)
        
        # IC Error vs Epochs
        plt.sca(ax3_right)
        if len(losses['ic']) > 0:
            plt.semilogy(epochs, losses['ic'], linewidth=2, color='black', alpha=0.7)
        ax3_right.set_ylabel('')  # Remove ylabel (same as left column)
        ax3_right.set_xlabel('')  # Remove xlabel (shared with bottom row)
        ax3_right.set_xticklabels([])  # Remove xticklabels (shared with bottom row)
        apply_plot_style(ax=ax3_right)
        ax3_right.grid(True, axis='y', alpha=0.3)
        
        # Inlet BC Error vs Epochs
        plt.sca(ax4_right)
        if len(losses['inlet_bc']) > 0:
            plt.semilogy(epochs, losses['inlet_bc'], linewidth=2, color='black', alpha=0.7)
        ax4_right.set_ylabel('')  # Remove ylabel (same as left column)
        ax4_right.set_xlabel('')  # Remove xlabel (shared with bottom row)
        ax4_right.set_xticklabels([])  # Remove xticklabels (shared with bottom row)
        apply_plot_style(ax=ax4_right)
        ax4_right.grid(True, axis='y', alpha=0.3)
        
        # Outlet BC Error vs Epochs
        plt.sca(ax5_right)
        if len(losses['outlet_bc']) > 0:
            plt.semilogy(epochs, losses['outlet_bc'], linewidth=2, color='black', alpha=0.7)
        ax5_right.set_ylabel('')  # Remove ylabel (same as left column)
        plt.xlabel('Epoch', fontsize=12)  # Keep xlabel (bottom row)
        # Keep xticklabels visible for bottom row
        apply_plot_style(ax=ax5_right)
        ax5_right.grid(True, axis='y', alpha=0.3)
    
    # Create figure-level legend at the top with multiple columns
    # Handles and labels were collected earlier from ax1
    if len(handles) > 0:
        # Use more columns for better horizontal layout (up to number of items)
        ncols = min(len(handles), 6)  # Allow up to 6 columns
        fig_legend = fig.legend(handles, labels, loc='upper center', 
                               ncol=ncols, frameon=False, 
                               fontsize=10, bbox_to_anchor=(0.5, 0.925),
                               labelspacing=0.5, columnspacing=1.2)
        # Style legend text items to match tick labels
        for text in fig_legend.get_texts():
            text.set_color('black')
            text.set_alpha(0.3)
    
    # Adjust spacing between subplots (add top padding for legend)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save plot
    if filename is None:
        filename = 'pinn_analytical_comparison_with_pde_loss.png'
    # Use output_dir if provided, otherwise use default plots_dir
    save_dir = output_dir if output_dir is not None else plotting_params['plots_dir']
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=plotting_params['dpi'], bbox_inches='tight')
    print(f"Comparison plot with residuals saved to '{plot_path}'")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Set seeds for reproducibility (matching baseline)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "="*60)
    print("PINN for 1D Contaminant Transport - With Analytical Comparison")
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
    num_uniform = training_params.get('num_uniform', 1000)
    num_adaptive = training_params.get('num_adaptive', 1000)
    num_collocation_total = training_params.get('num_collocation')
    if num_collocation_total is None:
        num_collocation_total = num_uniform + num_adaptive
    print(f"  Collocation points: {num_collocation_total} total ({num_uniform} uniform + {num_adaptive} adaptive)")
    print(f"  IC points: {training_params['num_ic']}")
    print(f"  BC points: {training_params['num_bc']}")
    
    # Train model
    print("\n" + "="*60)
    print("Training PINN...")
    print("="*60)
    losses = train_pinn(model, training_params=training_params)
    
    # Generate concentration profiles with analytical comparison
    print("\nGenerating concentration profiles with analytical comparison...")
    plot_concentration_profiles(model, times_days=plotting_params['times_days'])
    
    # Generate concentration profiles with PDE residual subplot
    print("\nGenerating concentration profiles with PDE residual...")
    plot_concentration_with_pde_loss(
        model, 
        times_days=plotting_params['times_days'], 
        num_collocation=100,
        losses=losses  # Pass final loss history
    )
    
    # Generate GIF animation from training epoch plots
    print("\n" + "="*60)
    print("Generating GIF animation from training plots...")
    print("="*60)
    create_training_gif()
    
    print("\n" + "="*60)
    print("PINN training and comparison complete!")
    print("="*60)
