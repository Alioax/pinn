"""
Baseline PINN Implementation for 1D Contaminant Transport

Minimal script that trains a Physics-Informed Neural Network to solve:
    ∂C/∂t + U ∂C/∂x = D ∂²C/∂x²

with boundary conditions:
    C(0, t) = C0  (inlet)
    C(L, t) = 0  (outlet, far-field approximation)
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
from tqdm import tqdm

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
num_layers = 3             # number of hidden layers
num_neurons = 16           # number of neurons per hidden layer
activation = torch.nn.Tanh # activation function
# Alternative activation functions (uncomment to use):
# activation = torch.nn.ReLU              # Rectified Linear Unit - simple but may have vanishing gradients
# activation = torch.nn.SiLU              # Swish/Sigmoid Linear Unit - smooth, often works well for PINNs
# activation = torch.nn.GELU               # Gaussian Error Linear Unit - smooth activation, good for deep networks
# activation = torch.nn.ELU                # Exponential Linear Unit - smooth with negative values
# activation = torch.nn.LeakyReLU          # Leaky ReLU - variant of ReLU that allows small negative values
# activation = torch.nn.Sigmoid            # Sigmoid - smooth and bounded, but can saturate
# activation = torch.nn.Softplus           # Smooth approximation of ReLU
# activation = torch.sin                   # Sinusoidal - periodic, good for oscillatory solutions (note: no parentheses)

# Training parameters
num_epochs = 20000          # number of training epochs
lr = 0.001                  # learning rate
num_collocation = 150*150     # number of collocation points for PDE
num_ic = 150               # number of points for initial condition
num_bc = 150               # number of points for boundary conditions
weight_pde = 1           # weight for PDE residual loss
weight_ic = 1            # weight for initial condition loss
weight_inlet_bc = 1      # weight for inlet boundary condition loss
weight_outlet_bc = 1     # weight for outlet boundary condition loss

# Plotting parameters
times_days = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # times to plot (days)
num_spatial_points = 500            # number of spatial points for plotting
plot_dpi = 800                      # DPI (dots per inch) for saved plots

# Derived parameters
T = L / U                  # days (time scale, advective)
Pe = (U * L) / D           # Péclet number (dimensionless)
t_final_star = T_phys / T  # dimensionless final time

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
# Initialize Model and Set Random Seeds
# ============================================================================

torch.manual_seed(123456789)
np.random.seed(123456789)
model = PINN(num_layers, num_neurons, activation)

# ============================================================================
# Create Collocation Points in NumPy
# ============================================================================

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
x_star_pde_tensor = torch.tensor(x_star_pde, requires_grad=True)
t_star_pde_tensor = torch.tensor(t_star_pde, requires_grad=True)
x_star_ic_tensor = torch.tensor(x_star_ic, requires_grad=True)
t_star_ic_tensor = torch.tensor(t_star_ic, requires_grad=True)
x_star_inlet_tensor = torch.tensor(x_star_inlet, requires_grad=True)
t_star_inlet_tensor = torch.tensor(t_star_inlet, requires_grad=True)
x_star_outlet_tensor = torch.tensor(x_star_outlet, requires_grad=True)
t_star_outlet_tensor = torch.tensor(t_star_outlet, requires_grad=True)

# ============================================================================
# Plot Collocation Points (Pretraining)
# ============================================================================

script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
results_dir.mkdir(exist_ok=True)

plt.figure(figsize=(6.5, 4))
plt.scatter(x_star_pde, t_star_pde, s=1, alpha=1, color='C1', label='PDE', edgecolors='none')  # Dark blue
plt.scatter(x_star_ic, t_star_ic, s=1, alpha=1, color='C0', label='Initial Condition', edgecolors='none')  # Orange
plt.scatter(x_star_inlet, t_star_inlet, s=1, alpha=1, color='C2', label='Inlet BC', edgecolors='none')  # Light blue
plt.scatter(x_star_outlet, t_star_outlet, s=1, alpha=1, color='C3', label='Outlet BC', edgecolors='none')  # Yellow
plt.xlabel('x* (dimensionless)', fontsize=12)
plt.ylabel('t* (dimensionless)', fontsize=12)
plt.title('Collocation Points Distribution', fontsize=14, pad=20)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False, 
           fontsize=10, markerscale=5, handletextpad=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

collocation_plot_path = results_dir / 'collocation_points.png'
plt.savefig(str(collocation_plot_path), dpi=plot_dpi, bbox_inches='tight')
print(f"Collocation points plot saved to: {collocation_plot_path}")
plt.close()

# ============================================================================
# Training Loop with Closure Function
# ============================================================================

# Optimizer options - uncomment the one you want to use
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Initialize loss tracking lists
losses = {
    'total': [],
    'pde': [],
    'ic': [],
    'inlet_bc': [],
    'outlet_bc': []
}

def closure():
    optimizer.zero_grad()
    
    # PDE loss
    x_star_pde_grad = x_star_pde_tensor.clone().detach().requires_grad_(True)
    t_star_pde_grad = t_star_pde_tensor.clone().detach().requires_grad_(True)
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
    C_star_ic = model(x_star_ic_tensor, t_star_ic_tensor)
    ic_loss = nn.MSELoss()(C_star_ic, torch.zeros_like(C_star_ic))
    
    # Inlet boundary condition loss: C*(0, t*) = 1
    C_star_inlet = model(x_star_inlet_tensor, t_star_inlet_tensor)
    inlet_loss = nn.MSELoss()(C_star_inlet, torch.ones_like(C_star_inlet))
    
    # Outlet boundary condition loss: C*(1, t*) = 0 (Dirichlet far-field approximation)
    C_star_outlet = model(x_star_outlet_tensor, t_star_outlet_tensor)
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

# Training loop with progress bar
pbar = tqdm(range(num_epochs), desc="Training PINN")
for epoch in pbar:
    optimizer.step(closure)
    
    # Update progress bar with current losses (update every 10 epochs for performance)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        pbar.set_postfix({
            'Loss': f"{losses['total'][-1]:.4e}",
            'PDE': f"{losses['pde'][-1]:.4e}",
            'IC': f"{losses['ic'][-1]:.4e}",
            'Inlet': f"{losses['inlet_bc'][-1]:.4e}",
            'Outlet': f"{losses['outlet_bc'][-1]:.4e}"
        })
pbar.close()

# ============================================================================
# Plot Concentration Profiles
# ============================================================================

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
plt.savefig(str(plot_path), dpi=plot_dpi, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# Save PDF copy to report/figs directory
report_figs_dir = script_dir.parent / 'report' / 'figs'
report_figs_dir.mkdir(parents=True, exist_ok=True)
pdf_path = report_figs_dir / 'pinn_baseline_concentration_profiles.pdf'
plt.savefig(str(pdf_path), format='pdf', bbox_inches='tight')
print(f"PDF copy saved to: {pdf_path}")

plt.close()
