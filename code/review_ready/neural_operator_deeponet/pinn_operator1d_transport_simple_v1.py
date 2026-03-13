# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:28:11 2025

@author: Ali Haghighi
Supervised by: Afshin Ashrafzadeh, François Lehmann, Marwan Fahs

"""

"""
Unsupervised Neural Operator (DeepONet) for 1D Contaminant Transport (dimensionless).

Same problem as the parametric PINN: learn C* = f(x*, t*, log Pe) over a range of
Péclet numbers using only physics (no labeled solution data). PDE:
    ∂C*/∂t* + ∂C*/∂x* - (1/Pe) ∂²C*/∂x*² = 0

Boundary conditions:
    C*(0, t*) = 1   (inlet)
    C*(1, t*) = 0   (outlet)
    C*(x*, 0) = 0   (initial condition)

The operator uses a DeepONet: branch net encodes log Pe, trunk net encodes (x*, t*);
output C* = sigmoid(branch · trunk). Trained with PDE residual, IC, and BC losses only.

All parameters are configurable at the top of the file.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 800
from scipy.special import erfc
from scipy.stats import qmc
from tqdm import trange

# Directory where this script lives (figures saved here)
script_dir = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Init Torch for cpu or GPU and seed
# =============================================================================
torch.set_default_dtype(torch.float32)
torch.manual_seed(1234567)
np.random.seed(1234567)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    ngpus = torch.cuda.device_count()
    print("Using {} GPU(s)...".format(ngpus))
    print(torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")


# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics (dimensionless): Pe range and time
Pe_min = 10.0
Pe_max = 1e5
t_final_star = 1.0

# DeepONet: branch encodes log Pe, trunk encodes (x*, t*), latent_dim = output size of both
branch_layers = 2
branch_neurons = 16
trunk_layers = 3
trunk_neurons = 32
latent_dim = 32
activation = torch.nn.Tanh

# Training parameters (aligned with baseline/parametric for consistency)
num_epochs = 50
lr = 0.001
num_collocation = 40000
num_ic = 1000
num_bc = 5000
weight_pde = 1
weight_ic = 1
weight_inlet_bc = 1
weight_outlet_bc = 1

# Plotting
times_tstar = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
pe_values_to_plot = np.array([10, 50, 100, 500, 1000, 10000])
num_points = 5001

# Derived
logPe_min = np.log(Pe_min)
logPe_max = np.log(Pe_max)


# ============================================================================
# DeepONet: branch(log Pe) and trunk(x*, t*) -> C* = sigmoid(branch · trunk)
# ============================================================================

class DeepONet_Transp(nn.Module):
    """Neural operator: (x*, t*, log Pe) -> dimensionless concentration C*."""

    def __init__(self, branch_layers, branch_neurons, trunk_layers, trunk_neurons,
                 latent_dim, activation):
        super(DeepONet_Transp, self).__init__()
        # Branch: log_pe (1) -> ... -> latent_dim
        bl = [nn.Linear(1, branch_neurons), activation()]
        for _ in range(branch_layers - 1):
            bl.append(nn.Linear(branch_neurons, branch_neurons))
            bl.append(activation())
        bl.append(nn.Linear(branch_neurons, latent_dim))
        self.branch = nn.Sequential(*bl)
        # Trunk: (x*, t*) (2) -> ... -> latent_dim
        tl = [nn.Linear(2, trunk_neurons), activation()]
        for _ in range(trunk_layers - 1):
            tl.append(nn.Linear(trunk_neurons, trunk_neurons))
            tl.append(activation())
        tl.append(nn.Linear(trunk_neurons, latent_dim))
        self.trunk = nn.Sequential(*tl)

    def init_weights(self):
        gain = nn.init.calculate_gain('tanh')
        for m in self.branch:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x_star, t_star, log_pe):
        points = torch.cat([x_star, t_star], dim=1)
        b = self.branch(log_pe)
        t = self.trunk(points)
        return torch.sigmoid((b * t).sum(dim=-1, keepdim=True))


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)[0]


# =============================================================================
# Collocation points: (x*, t*) with LHC, log_pe in [logPe_min, logPe_max]
# =============================================================================

sampler2d = qmc.LatinHypercube(d=2, seed=1234567)
sampler1d = qmc.LatinHypercube(d=1, seed=1234568)

# PDE
pde_samples = sampler2d.random(n=num_collocation)
x_star = pde_samples[:, 0]
t_star = pde_samples[:, 1] * t_final_star
log_pe = sampler1d.random(n=num_collocation).flatten() * (logPe_max - logPe_min) + logPe_min

# Initial condition
x_star_init = sampler1d.random(n=num_ic).flatten()
t_star_init = np.zeros_like(x_star_init)
log_pe_init = sampler1d.random(n=num_ic).flatten() * (logPe_max - logPe_min) + logPe_min

# Inlet BC (x* = 0)
x_star_in = np.zeros(num_bc)
t_star_in = sampler1d.random(n=num_bc).flatten() * t_final_star
log_pe_in = sampler1d.random(n=num_bc).flatten() * (logPe_max - logPe_min) + logPe_min

# Outlet BC (x* = 1)
x_star_out = np.ones(num_bc)
t_star_out = sampler1d.random(n=num_bc).flatten() * t_final_star
log_pe_out = sampler1d.random(n=num_bc).flatten() * (logPe_max - logPe_min) + logPe_min

# Tensors on device (PDE inputs need requires_grad for derivatives)
train_x = torch.tensor(x_star.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
train_t = torch.tensor(t_star.reshape(-1, 1), requires_grad=True, dtype=torch.float32).to(device)
train_log_pe = torch.tensor(log_pe.reshape(-1, 1), dtype=torch.float32).to(device)

train_x_init = torch.tensor(x_star_init.reshape(-1, 1), dtype=torch.float32).to(device)
train_t_init = torch.tensor(t_star_init.reshape(-1, 1), dtype=torch.float32).to(device)
train_log_pe_init = torch.tensor(log_pe_init.reshape(-1, 1), dtype=torch.float32).to(device)
C_init = torch.tensor(np.zeros_like(x_star_init).reshape(-1, 1), dtype=torch.float32).to(device)

train_x_in = torch.tensor(x_star_in.reshape(-1, 1), dtype=torch.float32).to(device)
train_t_in = torch.tensor(t_star_in.reshape(-1, 1), dtype=torch.float32).to(device)
train_log_pe_in = torch.tensor(log_pe_in.reshape(-1, 1), dtype=torch.float32).to(device)

train_x_out = torch.tensor(x_star_out.reshape(-1, 1), dtype=torch.float32).to(device)
train_t_out = torch.tensor(t_star_out.reshape(-1, 1), dtype=torch.float32).to(device)
train_log_pe_out = torch.tensor(log_pe_out.reshape(-1, 1), dtype=torch.float32).to(device)


# =============================================================================
# Load the network
# =============================================================================

model = DeepONet_Transp(branch_layers, branch_neurons, trunk_layers, trunk_neurons,
                       latent_dim, activation).to(device)
model.init_weights()


# =============================================================================
# Loss function with closure (PDE uses Pe = exp(log_pe))
# =============================================================================

obj = []

def closure():
    optimizer.zero_grad(set_to_none=True)
    C = model(train_x, train_t, train_log_pe)

    dC_dt = gradients(C, train_t)
    dC_dx = gradients(C, train_x)
    d2C_dx2 = gradients(dC_dx, train_x)

    Pe = torch.exp(train_log_pe)
    f1 = dC_dt + dC_dx - (1.0 / Pe) * d2C_dx2
    pde_loss = (f1**2).mean()

    C_ini_pred = model(train_x_init, train_t_init, train_log_pe_init)
    ic_loss = ((C_ini_pred - C_init)**2).mean()

    C_BL_pred = model(train_x_in, train_t_in, train_log_pe_in)
    in_loss = ((C_BL_pred - 1.0)**2).mean()

    C_BR_pred = model(train_x_out, train_t_out, train_log_pe_out)
    out_loss = ((C_BR_pred - 0.0)**2).mean()

    total_loss = (weight_pde * pde_loss +
                  weight_ic * ic_loss +
                  weight_inlet_bc * in_loss +
                  weight_outlet_bc * out_loss)

    total_loss.backward()
    obj.append([total_loss.item(), pde_loss.item(),
                ic_loss.item(), in_loss.item(), out_loss.item()])
    t_bar.set_description("loss : %.8f mse_pde %.8f mse_ic %.8f mse_bc_l %.8f mse_bc_r %.8f" %
                          (total_loss.item(), pde_loss.item(),
                           ic_loss.item(), in_loss.item(), out_loss.item()))
    t_bar.refresh()
    return total_loss


# =============================================================================
# Training (LBFGS, same as baseline and parametric)
# =============================================================================

params = list(model.parameters())
print("Network info")
print(model.state_dict)
print("Total number of parameters :", sum(p.numel() for p in params))

optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                              max_iter=100, max_eval=None, tolerance_grad=1e-10,
                              tolerance_change=1e-12, history_size=100,
                              line_search_fn=None)
print(optimizer)

EPOCHS = num_epochs
t_bar = trange(EPOCHS)
for epoch in t_bar:
    model.train()
    optimizer.step(closure)
t_bar.close()


# ============================================================================
# Evaluation and plots (model and analytical solution)
# ============================================================================

model.eval().to("cpu")

x_star_plot = np.linspace(0, 1, num_points)
xp = torch.tensor(x_star_plot.reshape(-1, 1), dtype=torch.float32)

# Analytical C* for 1D advection-diffusion (dimensionless: x*, t*, Pe)
def analytical_Cstar(x_star, t_star, Pe):
    arg = (x_star - t_star) / (2.0 * np.sqrt(t_star / Pe))
    C_ana = 0.5 * erfc(arg)
    mask = (Pe * x_star) < 700
    C_ana[mask] = 0.5 * (
        erfc((x_star[mask] - t_star) / (2.0 * np.sqrt(t_star / Pe)))
        + np.exp(Pe * x_star[mask])
        * erfc((x_star[mask] + t_star) / (2.0 * np.sqrt(t_star / Pe)))
    )
    return C_ana

# Concentration profiles: one figure per Pe (model and analytical)
for pe_val in pe_values_to_plot:
    log_pe_plot = np.log(pe_val)
    log_pe_plot_t = torch.tensor(log_pe_plot * np.ones((num_points, 1)), dtype=torch.float32)

    fig = plt.figure(figsize=(10, 6), tight_layout=True)
    ax = fig.add_subplot(111)
    fig.suptitle('DeepONet (unsupervised): C* vs x* (Pe = %g)' % pe_val, fontsize=14)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax.plot([], [], linewidth=2, linestyle='-', color='black', label='Model')
    ax.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')

    for idx, ti in enumerate(times_tstar):
        color = colors[idx % len(colors)]
        tp = torch.tensor(ti * np.ones((num_points, 1)), dtype=torch.float32)
        with torch.no_grad():
            C = model(xp, tp, log_pe_plot_t)
        C = C.numpy()
        C_ana = analytical_Cstar(x_star_plot, ti, pe_val)
        ax.plot(x_star_plot, C, linewidth=2, linestyle='-', color=color)
        ax.plot(x_star_plot, C_ana, linewidth=2, linestyle='--', color=color, alpha=0.7)
        ax.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label='t* = %.2f' % ti)

    ax.set_xlabel(r'Dimensionless distance $x^*$ (-)', fontsize=12)
    ax.set_ylabel(r'Dimensionless concentration $C^*$ (-)', fontsize=12)
    ax.grid()
    ax.minorticks_on()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=10,
              labelspacing=0.5, columnspacing=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    plt.savefig(os.path.join(script_dir, "Fig_Pe%g_Cstar.jpg" % pe_val))
    plt.show()
    plt.close(fig)

# Collocation points (x*, t*)
fig2 = plt.figure(figsize=(10, 6), tight_layout=True)
ax1 = fig2.add_subplot(111)
ax1.set_title('Collocation Points (x*, t*)', fontsize=14)
ax1.scatter(x_star, t_star, label="PDE", s=1, alpha=0.6)
ax1.scatter(x_star_init, t_star_init, label="I.C.", s=5)
ax1.scatter(x_star_in, t_star_in, label="B.C. In", s=5)
ax1.scatter(x_star_out, t_star_out, label="B.C. Out", s=5)
ax1.set_xlabel(r'$x^*$ (-)', fontsize=12)
ax1.set_ylabel(r'$t^*$ (-)', fontsize=12)
ax1.grid()
ax1.minorticks_on()
ax1.legend(loc='best', fontsize=10)
plt.savefig(os.path.join(script_dir, "Fig_coll_points.jpg"))

# Parametric coverage (t* vs log Pe)
fig2b = plt.figure(figsize=(10, 6), tight_layout=True)
ax1b = fig2b.add_subplot(111)
ax1b.set_title('Parametric coverage (t* vs log Pe)', fontsize=14)
ax1b.scatter(t_star, log_pe, label="PDE", s=1, alpha=0.6)
ax1b.set_xlabel(r'$t^*$ (-)', fontsize=12)
ax1b.set_ylabel(r'log(Pe) (-)', fontsize=12)
ax1b.grid()
ax1b.minorticks_on()
plt.savefig(os.path.join(script_dir, "Fig_coll_points_logpe.jpg"))

# Loss vs iterations
fig3 = plt.figure(figsize=(10, 6), tight_layout=True)
obj = np.array(obj)
ax3 = fig3.add_subplot(111)
ax3.plot(obj[:, 0], '-b', label="total Loss")
ax3.plot(obj[:, 1], 'y', label="MSE PDE")
ax3.plot(obj[:, 2], 'k', label="MSE IC", alpha=0.2)
ax3.plot(obj[:, 3], 'm', label="MSE BC Left")
ax3.plot(obj[:, 4], 'r', label="MSE BC Right")
ax3.set_yscale('log', base=10)
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Loss")
ax3.minorticks_on()
ax3.grid()
ax3.legend(loc='best', fontsize=8)
plt.savefig(os.path.join(script_dir, "Fig3_loss.jpg"))
