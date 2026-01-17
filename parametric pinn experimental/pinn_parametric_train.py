"""
Parametric PINN training for 1D Advection-Dispersion.

Trains a Physics-Informed Neural Network on the dimensionless PDE:
    C*_t* + C*_x* - (1/Pe) C*_{x*x*} = 0

The network learns C* = f(x*, t*, log Pe) across a range of Peclet numbers.
Saves a checkpoint that can be loaded later for plotting.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics parameters (dimensionless)
Pe_min = 1                # minimum Peclet number for training
Pe_max = 1e5               # maximum Peclet number for training
t_final_star = 1.0         # dimensionless final time

# Model architecture
num_layers = 3             # number of hidden layers
num_neurons = 16           # number of neurons per hidden layer
activation_name = "Tanh"   # activation function (string name)

# Training parameters
num_epochs = 30000         # number of training epochs
lr = 0.001                 # learning rate
num_collocation = 200 * 200  # number of collocation points for PDE
num_ic = 200               # number of points for initial condition
num_bc = 200               # number of points for boundary conditions
weight_pde = 1             # weight for PDE residual loss
weight_ic = 1              # weight for initial condition loss
weight_inlet_bc = 1        # weight for inlet boundary condition loss
weight_outlet_bc = 1       # weight for outlet boundary condition loss

# Checkpoint settings
model_save_name = "pinn_parametric_30000_200_1e-0_3x16.pt"

# ============================================================================
# Derived parameters
# ============================================================================

logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)

# ============================================================================
# Neural Network Architecture
# ============================================================================

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def get_activation(name):
    activations = {
        "Tanh": torch.nn.Tanh,
        "ReLU": torch.nn.ReLU,
        "SiLU": torch.nn.SiLU,
        "GELU": torch.nn.GELU,
        "ELU": torch.nn.ELU,
        "LeakyReLU": torch.nn.LeakyReLU,
        "Sigmoid": torch.nn.Sigmoid,
        "Softplus": torch.nn.Softplus,
        "Sin": Sine,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation_name: {name}")
    return activations[name]

class PINN(nn.Module):
    """Neural network that takes (x*, t*, log Pe) and outputs dimensionless concentration C*."""

    def __init__(self, num_layers, num_neurons, activation_cls):
        super().__init__()
        layer_sizes = [3] + [num_neurons] * num_layers + [1]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation_cls())
        self.net = nn.Sequential(*layers)

        gain = nn.init.calculate_gain("tanh")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x_star, t_star, log_pe):
        inputs = torch.cat([x_star, t_star, log_pe], dim=1)
        return self.net(inputs)

# ============================================================================
# Initialize Model and Set Random Seeds
# ============================================================================

torch.manual_seed(123456789)
np.random.seed(123456789)

activation_cls = get_activation(activation_name)
model = PINN(num_layers, num_neurons, activation_cls)

# ============================================================================
# Collocation Point Sampling
# ============================================================================

def sample_collocation_points():
    x_star_pde = np.random.rand(num_collocation, 1).astype(np.float32)
    t_star_pde = np.random.rand(num_collocation, 1).astype(np.float32) * t_final_star
    log_pe_pde = (np.random.rand(num_collocation, 1).astype(np.float32) *
                  (logPe_max - logPe_min) + logPe_min)

    x_star_ic = np.random.rand(num_ic, 1).astype(np.float32)
    t_star_ic = np.zeros((num_ic, 1), dtype=np.float32)
    log_pe_ic = (np.random.rand(num_ic, 1).astype(np.float32) *
                 (logPe_max - logPe_min) + logPe_min)

    x_star_inlet = np.zeros((num_bc, 1), dtype=np.float32)
    t_star_inlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    log_pe_inlet = (np.random.rand(num_bc, 1).astype(np.float32) *
                    (logPe_max - logPe_min) + logPe_min)

    x_star_outlet = np.ones((num_bc, 1), dtype=np.float32)
    t_star_outlet = np.random.rand(num_bc, 1).astype(np.float32) * t_final_star
    log_pe_outlet = (np.random.rand(num_bc, 1).astype(np.float32) *
                     (logPe_max - logPe_min) + logPe_min)

    return {
        "x_star_pde": x_star_pde,
        "t_star_pde": t_star_pde,
        "log_pe_pde": log_pe_pde,
        "x_star_ic": x_star_ic,
        "t_star_ic": t_star_ic,
        "log_pe_ic": log_pe_ic,
        "x_star_inlet": x_star_inlet,
        "t_star_inlet": t_star_inlet,
        "log_pe_inlet": log_pe_inlet,
        "x_star_outlet": x_star_outlet,
        "t_star_outlet": t_star_outlet,
        "log_pe_outlet": log_pe_outlet,
    }

# ============================================================================
# Training Loop
# ============================================================================

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lr / 10)

losses = {
    "total": [],
    "pde": [],
    "ic": [],
    "inlet_bc": [],
    "outlet_bc": [],
}

def compute_losses(samples):
    x_star_pde_tensor = torch.tensor(samples["x_star_pde"], requires_grad=True)
    t_star_pde_tensor = torch.tensor(samples["t_star_pde"], requires_grad=True)
    log_pe_pde_tensor = torch.tensor(samples["log_pe_pde"])
    C_star_pde = model(x_star_pde_tensor, t_star_pde_tensor, log_pe_pde_tensor)
    dC_dt_star = grad(C_star_pde, t_star_pde_tensor, grad_outputs=torch.ones_like(C_star_pde),
                      create_graph=True, retain_graph=True)[0]
    dC_dx_star = grad(C_star_pde, x_star_pde_tensor, grad_outputs=torch.ones_like(C_star_pde),
                      create_graph=True, retain_graph=True)[0]
    d2C_dx2_star = grad(dC_dx_star, x_star_pde_tensor, grad_outputs=torch.ones_like(dC_dx_star),
                        create_graph=True, retain_graph=True)[0]

    pe_pde = torch.exp(log_pe_pde_tensor)
    pde_residual = dC_dt_star + dC_dx_star - (1.0 / pe_pde) * d2C_dx2_star
    pde_loss = torch.mean(pde_residual ** 2)

    x_star_ic_tensor = torch.tensor(samples["x_star_ic"])
    t_star_ic_tensor = torch.tensor(samples["t_star_ic"])
    log_pe_ic_tensor = torch.tensor(samples["log_pe_ic"])
    C_star_ic = model(x_star_ic_tensor, t_star_ic_tensor, log_pe_ic_tensor)
    ic_loss = nn.MSELoss()(C_star_ic, torch.zeros_like(C_star_ic))

    x_star_inlet_tensor = torch.tensor(samples["x_star_inlet"])
    t_star_inlet_tensor = torch.tensor(samples["t_star_inlet"])
    log_pe_inlet_tensor = torch.tensor(samples["log_pe_inlet"])
    C_star_inlet = model(x_star_inlet_tensor, t_star_inlet_tensor, log_pe_inlet_tensor)
    inlet_loss = nn.MSELoss()(C_star_inlet, torch.ones_like(C_star_inlet))

    x_star_outlet_tensor = torch.tensor(samples["x_star_outlet"])
    t_star_outlet_tensor = torch.tensor(samples["t_star_outlet"])
    log_pe_outlet_tensor = torch.tensor(samples["log_pe_outlet"])
    C_star_outlet = model(x_star_outlet_tensor, t_star_outlet_tensor, log_pe_outlet_tensor)
    outlet_loss = nn.MSELoss()(C_star_outlet, torch.zeros_like(C_star_outlet))

    total_loss = (weight_pde * pde_loss +
                  weight_ic * ic_loss +
                  weight_inlet_bc * inlet_loss +
                  weight_outlet_bc * outlet_loss)

    return total_loss, pde_loss, ic_loss, inlet_loss, outlet_loss

def train():
    pbar = tqdm(range(num_epochs), desc="Training PINN")
    for epoch in pbar:
        samples = sample_collocation_points()
        optimizer.zero_grad()
        total_loss, pde_loss, ic_loss, inlet_loss, outlet_loss = compute_losses(samples)
        total_loss.backward()
        optimizer.step()

        losses["total"].append(total_loss.item())
        losses["pde"].append(pde_loss.item())
        losses["ic"].append(ic_loss.item())
        losses["inlet_bc"].append(inlet_loss.item())
        losses["outlet_bc"].append(outlet_loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            pbar.set_postfix({
                "Loss": f"{losses['total'][-1]:.4e}",
                "PDE": f"{losses['pde'][-1]:.4e}",
                "IC": f"{losses['ic'][-1]:.4e}",
                "Inlet": f"{losses['inlet_bc'][-1]:.4e}",
                "Outlet": f"{losses['outlet_bc'][-1]:.4e}",
            })
    pbar.close()

def save_checkpoint():
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / model_save_name

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "num_layers": num_layers,
            "num_neurons": num_neurons,
            "activation_name": activation_name,
            "Pe_min": Pe_min,
            "Pe_max": Pe_max,
            "t_final_star": t_final_star,
        },
        "losses": losses,
    }

    torch.save(checkpoint, model_save_path)
    print(f"Model checkpoint saved to: {model_save_path}")

if __name__ == "__main__":
    train()
    save_checkpoint()
