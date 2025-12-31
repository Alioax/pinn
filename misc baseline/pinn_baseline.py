"""
Baseline PINN Implementation for 1D Contaminant Transport

Minimal script that trains a Physics-Informed Neural Network to solve:
    ∂C/∂t + U ∂C/∂x = D ∂²C/∂x²

with boundary conditions:
    C(0, t) = C0  (inlet)
    ∂C/∂x(L, t) = 0  (outlet)
    C(x, 0) = 0  (initial condition)

All parameters are configurable at the top of the file.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.autograd import grad
import os
import sys
from pathlib import Path

# Import analytical solution from sibling folder
sys.path.append(str(Path(__file__).parent.parent / 'analytical_solution'))
from analytical_solution import analytical_solution

# Visualization settings
mpl.rcParams['figure.dpi']= 800
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#FF5F05", "#13294B", "#009FD4", "#FCB316", "#006230", "#007E8E", "#5C0E41", "#7D3E13"])

# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics parameters
U = 0.1                    # m/day (advection velocity)
D = 1e-7 * 86400          # m²/day (dispersion coefficient)
C0 = 5.0                   # kg/m³ (inlet concentration)
L = 100.0                  # m (domain length)
T_phys = 1000.0            # days (physical time horizon)

# Model architecture
num_layers = 0             # number of hidden layers
num_neurons = 64           # number of neurons per hidden layer
activation = torch.nn.Tanh # activation function

# Training parameters
num_epochs = 20000          # number of training epochs
lr = 0.0001                 # learning rate
num_collocation = 200000     # number of collocation points for PDE
num_ic = 20000               # number of points for initial condition
num_bc = 20000               # number of points for boundary conditions
weight_pde = 1           # weight for PDE residual loss
weight_ic = 1            # weight for initial condition loss
weight_inlet_bc = 1      # weight for inlet boundary condition loss
weight_outlet_bc = 1     # weight for outlet boundary condition loss

# Plotting parameters
times_days = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # times to plot (days)
num_points = 500            # spatial resolution for plots

# Derived parameters
T = L / U                  # days (time scale, advective)
Pe = (U * L) / D           # Péclet number (dimensionless)
t_final_star = T_phys / T  # dimensionless final time

# ============================================================================
# Dimensionless Conversion Functions
# ============================================================================

def to_dimensionless(x, t):
    """Convert dimensional (x, t) to dimensionless (x*, t*)."""
    return x / L, t / T

def concentration_from_dimensionless(C_star):
    """Convert dimensionless concentration C* to dimensional C."""
    return C_star * C0

# ============================================================================
# Neural Network Architecture
# ============================================================================

class PINN(nn.Module):
    """Neural network that takes (x*, t*) and outputs dimensionless concentration C*."""
    
    def __init__(self, num_layers, num_neurons, activation):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, num_neurons), activation()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation())
        layers.append(nn.Linear(num_neurons, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x_star, t_star):
        inputs = torch.cat([x_star, t_star], dim=1)
        return self.net(inputs)

# ============================================================================
# Loss Functions (All in Dimensionless Form)
# ============================================================================

def compute_pde_residual(model, x_star, t_star, Pe):
    """Compute PDE residual: ∂C*/∂t* + ∂C*/∂x* - (1/Pe)*∂²C*/∂x*²."""
    x_star = x_star.clone().detach().requires_grad_(True)
    t_star = t_star.clone().detach().requires_grad_(True)
    C_star = model(x_star, t_star)
    dC_dt_star = grad(C_star, t_star, grad_outputs=torch.ones_like(C_star),
                      create_graph=True, retain_graph=True)[0] ### Create_graph not needed really but 
    dC_dx_star = grad(C_star, x_star, grad_outputs=torch.ones_like(C_star),
                      create_graph=True, retain_graph=True)[0]
    d2C_dx2_star = grad(dC_dx_star, x_star, grad_outputs=torch.ones_like(dC_dx_star),
                        create_graph=True, retain_graph=True)[0]
    return dC_dt_star + dC_dx_star - (1.0 / Pe) * d2C_dx2_star

def compute_ic_loss(model, x_star_init):
    """Compute initial condition loss: C*(x*, 0) = 0."""
    t_star_init = torch.zeros_like(x_star_init)
    C_star_pred = model(x_star_init, t_star_init)
    return nn.MSELoss()(C_star_pred, torch.zeros_like(C_star_pred))

def compute_bc_losses(model, t_star_bc):
    """Compute boundary condition losses: C*(0, t*) = 1 and ∂C*/∂x*(1, t*) = 0."""
    # Inlet: C*(0, t*) = 1
    x_inlet = torch.zeros_like(t_star_bc)
    C_inlet = model(x_inlet, t_star_bc)
    inlet_loss = nn.MSELoss()(C_inlet, torch.ones_like(C_inlet))
    
    # Outlet: ∂C*/∂x*(1, t*) = 0
    x_outlet = torch.ones_like(t_star_bc).requires_grad_(True)
    t_outlet = t_star_bc.clone().detach().requires_grad_(True)
    C_outlet = model(x_outlet, t_outlet)
    dC_dx = grad(C_outlet, x_outlet, grad_outputs=torch.ones_like(C_outlet),
                 create_graph=True, retain_graph=True)[0]
    outlet_loss = nn.MSELoss()(dC_dx, torch.zeros_like(dC_dx))
    
    return inlet_loss, outlet_loss

# ============================================================================
# Training Function
# ============================================================================

def train_pinn(model):
    """Train the PINN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Sample collocation points
        x_star = torch.rand(num_collocation, 1)
        t_star = torch.rand(num_collocation, 1) * t_final_star
        
        # PDE loss
        pde_residual = compute_pde_residual(model, x_star, t_star, Pe)
        pde_loss = torch.mean(pde_residual**2)
        
        # Initial condition loss
        x_ic = torch.rand(num_ic, 1)
        ic_loss = compute_ic_loss(model, x_ic)
        
        # Boundary condition losses
        t_bc = torch.rand(num_bc, 1) * t_final_star
        inlet_loss, outlet_loss = compute_bc_losses(model, t_bc)
        
        # Total loss
        total_loss = (weight_pde * pde_loss + 
                     weight_ic * ic_loss + 
                     weight_inlet_bc * inlet_loss + 
                     weight_outlet_bc * outlet_loss)
        
        total_loss.backward()
        optimizer.step()

# ============================================================================
# Evaluation Function
# ============================================================================

def predict_concentration(model, x, t):
    """Predict concentration C(x,t) given dimensional inputs (x in m, t in days)."""
    x_star, t_star = to_dimensionless(x, t)
    x_star = torch.tensor(x_star, dtype=torch.float32).reshape(-1, 1)
    t_star = torch.tensor(t_star, dtype=torch.float32).reshape(-1, 1)
    if t_star.shape[0] == 1 and x_star.shape[0] > 1:
        t_star = t_star.expand(x_star.shape[0], -1)
    
    model.eval()
    with torch.no_grad():
        C_star = model(x_star, t_star)
    
    C = concentration_from_dimensionless(C_star)
    return C.cpu().numpy()

# ============================================================================
# Plotting Function
# ============================================================================

def plot_concentration_profiles(model):
    """Plot concentration profiles at selected times."""
    # Get script directory and create results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    x_plot = np.linspace(0, L, num_points)
    plt.figure(figsize=(5, 3.5))
    
    # Get current color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Add legend entries for PINN and Analytical at the beginning
    plt.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
    plt.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')
    
    # Plot in reverse order so legend shows times in order
    for idx, t_days in enumerate(reversed(times_days)):
        # Get color for this time (cycling through colors)
        color = colors[idx % len(colors)]
        
        # PINN solution (solid line)
        C_pinn = predict_concentration(model, x_plot, t_days)
        plt.plot(x_plot, C_pinn, linewidth=2, linestyle='-', color=color)
        
        # Analytical solution (dashed line) with same color
        C_analytical = analytical_solution(x_plot, t_days)
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
    
    # Save PNG plot to results directory
    plot_path = results_dir / 'pinn_baseline_concentration_profiles.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Save PDF copy to report/figs directory
    report_figs_dir = script_dir.parent / 'report' / 'figs'
    report_figs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = report_figs_dir / 'pinn_baseline_concentration_profiles.pdf'
    plt.savefig(str(pdf_path), format='pdf', bbox_inches='tight')
    print(f"PDF copy saved to: {pdf_path}")
    
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    model = PINN(num_layers, num_neurons, activation)
    train_pinn(model)
    plot_concentration_profiles(model)
