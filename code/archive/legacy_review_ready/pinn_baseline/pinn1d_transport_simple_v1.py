# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:28:11 2025

@author: Ali Haghighi
Supervised by: Afshin Ashrafzadeh, François Lehmann, Marwan Fahs

"""

"""
Baseline PINN Implementation for 1D Contaminant Transport

Minimal script that trains a Physics-Informed Neural Network to solve:
    ∂C/∂t + U ∂C/∂x = D ∂²C/∂x²
    
    ∂C*/∂t* + ∂C*/∂x* -1/Pe  ∂²C*/∂x*²=0
    
with boundary conditions:
    C(0, t) = C0  (inlet)
    C(L, t) = 0   (outlet)
    C(x, 0) = 0  (initial condition)

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

class Soft_Tanh(nn.Module):
    def __init__(self):
        super(Soft_Tanh, self).__init__()
        self.a = torch.nn.parameter.Parameter(torch.tensor(0.1))
        self.a.requires_grad = True
        
    def forward(self, x):
        return self.a*torch.tensor(10.0)*torch.tanh(x)
       
      
# ============================================================================
# Configuration - Edit parameters here
# ============================================================================

# Physics parameters
U = 0.1                    # m/day (advection velocity)
D = 1e-8 * 86400          # m²/day (dispersion coefficient)
C0 = 5.0                   # kg/m³ (inlet concentration)
C_0 = 5.0
L = 100.0                  # m (domain length)
T_phys = 800.0            # days (physical time horizon)

# Model architecture
num_layers = 2             # number of hidden layers
num_neurons = 16           # number of neurons per hidden layer
activation = torch.nn.Tanh  #Soft_Tanh     # activation function  torch.nn.Tanh

# Training parameters
num_epochs = 50          # number of training epochs
lr = 0.001                 # learning rate
num_collocation = 40000     # number of collocation points for PDE
num_ic = 1000               # number of points for initial condition
num_bc = 5000               # number of points for boundary conditions
weight_pde = 1           # weight for PDE residual loss
weight_ic = 1            # weight for initial condition loss
weight_inlet_bc = 1      # weight for inlet boundary condition loss
weight_outlet_bc = 1     # weight for outlet boundary condition loss

# Plotting parameters
times_days = np.array( [10, 50, 100, 200, 300, 400, 500, 600, T_phys])  # times to plot (days)
num_points = 5001            # spatial resolution for plots

# Derived parameters
T = L / U                  # days (time scale, advective)
Pe = (U * L) / D           # Péclet number (dimensionless)
t_final_star = T_phys / T  # dimensionless final time


# ============================================================================
# Neural Network Architecture
# ============================================================================


        
class PINN_Transp(nn.Module):
    """Neural network that takes (x*, t*) and outputs dimensionless concentration C*."""
    
    def __init__(self, num_layers, num_neurons, activation):
        super(PINN_Transp, self).__init__()
        layers = [nn.Linear(2, num_neurons), activation()]
        for _ in range(num_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(activation())
        layers.append(nn.Linear(num_neurons, 1))
        layers.append(torch.nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def init_weights(self, ) -> None:
         """
         Initializes the weights and 
         biases of all the layers in the model            
         """
         for param in self.parameters():
             if len(param.shape) >= 2: 
                 torch.nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('tanh') )
             elif len(param.shape) == 1: torch.nn.init.zeros_(param)
           #  torch.nn.init.xavier_uniform_()  torch.nn.init.xavier_normal_    
           
    def forward(self, x_star, t_star):
        inputs = torch.cat([x_star, t_star], dim=1)
        return self.net(inputs)


def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True
                               )[0]
# Sample collocation points
# 2D Latin Hypercube for (x*, t*) PDE points in the unit square, then scale t*
sampler2d = qmc.LatinHypercube(d=2, seed=1234567)
pde_samples = sampler2d.random(n=num_collocation)
x_star = pde_samples[:, 0]
t_star = pde_samples[:, 1] * t_final_star

# 1D Latin Hypercube for initial condition (x* only; t* = 0)
sampler1d = qmc.LatinHypercube(d=1, seed=1234568)
x_star_init = sampler1d.random(n=num_ic).flatten()
t_star_init = np.zeros_like(x_star_init)

# Boundary conditions: x* fixed, 1D LHC for t*
x_star_in = np.zeros(num_bc)
t_star_in = sampler1d.random(n=num_bc).flatten() * t_final_star

x_star_out = np.ones(num_bc)
t_star_out = sampler1d.random(n=num_bc).flatten() * t_final_star

# Torch tensor on device gpu or cpu

train_x = torch.tensor(x_star.reshape(-1,1), requires_grad=True, dtype=torch.float32).to(device)
train_t = torch.tensor(t_star.reshape(-1,1), requires_grad=True,  dtype=torch.float32).to(device)

train_x_init = torch.tensor(x_star_init.reshape(-1,1), dtype=torch.float32).to(device)
train_t_init = torch.tensor(t_star_init.reshape(-1,1),  dtype=torch.float32).to(device)
C_init = torch.tensor(np.zeros_like(x_star_init).reshape(-1,1),  dtype=torch.float32).to(device)

train_x_in = torch.tensor(x_star_in.reshape(-1,1), dtype=torch.float32).to(device)
train_t_in = torch.tensor(t_star_in.reshape(-1,1),  dtype=torch.float32).to(device)

train_x_out = torch.tensor(x_star_out.reshape(-1,1), dtype=torch.float32).to(device)
train_t_out = torch.tensor(t_star_out.reshape(-1,1),  dtype=torch.float32).to(device)

# =============================================================================
# Load the network
# =============================================================================
   
model = PINN_Transp(num_layers,num_neurons,activation).to(device) 
model.init_weights()


# =============================================================================
# Def loss function with closure
# =============================================================================
obj = []
def closure():
    optimizer.zero_grad(set_to_none=True)
    #x = torch.cat(( train_x, train_t), dim=1)  # x[:,0].reshape(-1,1)==train_t :-) ok True
    C = model(train_x, train_t)

    dC_dt = gradients(C, train_t)
    dC_dx = gradients(C, train_x)
    d2C_dx2 = gradients(dC_dx, train_x)
    

            
    f1 =  dC_dt + dC_dx - 1.0/Pe*d2C_dx2

    pde_loss = (f1**2).mean()   #torch.mean(torch.square(f))
    

    C_ini_pred = model(train_x_init, train_t_init)     
    ic_loss = ((C_ini_pred - C_init)**2).mean()  #torch.mean(torch.square(u_ini_pred - u_ini))
    
    
    C_BL_pred = model(train_x_in, train_t_in)
   # C_BL_f = 1-torch.exp(-10*train_t_in)
    in_loss = ((C_BL_pred -  1.0)**2).mean() # 
    
    
    C_BR_pred = model(train_x_out, train_t_out)
    out_loss = ((C_BR_pred - 0.0)**2).mean()  # C* = 0 at outlet 
    # Total loss
    total_loss = (weight_pde * pde_loss + 
                 weight_ic * ic_loss + 
                 weight_inlet_bc * in_loss + 
                 weight_outlet_bc * out_loss)
    
  #  print(mse_f,mse_u_ini,mse_u_bc)
    total_loss.backward()   # retain_graph=True
    obj.append([total_loss.item(), pde_loss.item(),
                        ic_loss.item(),in_loss.item(),out_loss.item()])
    t_bar.set_description("loss : %.8f \
                           mse_pde  %.8f \
                           mse_ic  %.8f \
                           mse_bc_l = %.8f \
                           mse_bc_r = %.8f \
                           " % (total_loss.item(), pde_loss.item(),
                           ic_loss.item(),in_loss.item(),out_loss.item()))
    t_bar.refresh() # to show immediately the update

    return total_loss   

# =============================================================================
# ADAM train
# =============================================================================

params = list(model.parameters()) 

# Print model info
print("Network info")
print(model.state_dict)
# Total number of parameters
nb_param = sum(p.numel() for p in params)
print("Total number of parameters :",nb_param)


# optimizer = torch.optim.Adam(params, lr=lr)
# #optimizer = torch.optim.RMSprop(params, lr=lr, centered=True)
# #optimizer = torch.optim.Rprop(params, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-08, 50.0))
# print(optimizer)
# t_bar = trange(num_epochs)  # use tqdm as progress bar

# for epoch in t_bar:
#     model.train()   
#     optimizer.step(closure)
# t_bar.close()          # close the progress bar


# =============================================================================
# LBFGS ,lr=1.0,tolerance_grad=1e-08, tolerance_change=1e-09
# =============================================================================

optimizer = torch.optim.LBFGS(model.parameters(),lr=0.1,
                    max_iter=100, max_eval=None, tolerance_grad=1e-10,
                    tolerance_change=1e-12, history_size=100,
                    line_search_fn=None )  # 'strong_wolfe'
print(optimizer)
EPOCHS = 50
t_bar = trange(EPOCHS) # open a new progress bar for lbfgs
# in order to adapt the learning rate use a scheduler
# if needed
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
for epoch in t_bar:
    model.train()
     # to show immediately the update
    optimizer.step(closure)
  #  scheduler.step()
    
t_bar.close()
# ============================================================================
# Evaluation of the model
# ============================================================================

model.eval().to("cpu")

# Spatial domain (semi-infinite, but we plot up to 100 m)
X_MAX = 100.0  # meters
X_PLOT = np.linspace(0, X_MAX, num_points)
xp = torch.tensor(X_PLOT.reshape(-1,1)/X_MAX,dtype=torch.float32)




fig = plt.figure(figsize=(10,6),tight_layout=True)
ax = fig.add_subplot(111) # fig.add_subplot(111)
fig.suptitle('Concentration Profiles at Different Times in (days)', fontsize=14)

# Get current color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

    # Add legend entries for PINN and Analytical at the beginning
ax.plot([], [], linewidth=2, linestyle='-', color='black', label='PINN')
ax.plot([], [], linewidth=2, linestyle='--', color='black', label='Analytical')

mask = (U*X_PLOT)/D<700
for idx, ti in enumerate(times_days):
    # Get color for this time (cycling through colors)
    color = colors[idx % len(colors)]
    C1 = (C_0/2.)*(erfc((X_PLOT-U*ti)/(2*np.sqrt(D*ti))))
    C1[mask] = (C_0/2.)*(erfc((X_PLOT[mask]-U*ti)/(2*np.sqrt(D*ti)))+
                np.exp(U*X_PLOT[mask]/D)*erfc((X_PLOT[mask]+U*ti)/(2*np.sqrt(D*ti))) )
   
    
    tp = torch.tensor(ti*U/L*np.ones_like(X_PLOT).reshape(-1,1),dtype=torch.float32)
    C = model(xp, tp)
    C = C.detach().numpy()
    ax.plot(X_PLOT,C*C_0,linewidth=2, linestyle='-', color=color)
    ax.plot(X_PLOT,C1,linewidth=2, linestyle='--', color=color, alpha=0.7)
    # Add marker-only entry for legend (square marker, no line)
    ax.plot([], [], marker='s', markersize=8, linestyle='None', color=color, label=f'{ti:.1f}')
ax.set_xlabel('Distance x (m)', fontsize=12)
ax.set_ylabel('Concentration C (kg/m³)', fontsize=12)


ax.grid()    
ax.minorticks_on()
# Create legend at top
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), 
                   ncol=4, frameon=False, fontsize=10,
                   labelspacing=0.5, columnspacing=1.2)
# Style legend text items - make all text black
for text in legend.get_texts():
    text.set_color('black')
    text.set_alpha(1.0)

plt.savefig(os.path.join(script_dir, "Fig_Ana&Pinns_Transp.jpg"))
plt.show()
# ============================================================================
# Plot Collocation points
# ============================================================================
 
fig2 = plt.figure(figsize=(10,6),tight_layout=True)
ax1 = fig2.add_subplot(111) # fig.add_subplot(111)
ax1.set_title('Collocation Points', fontsize=14)
ax1.scatter(x_star,t_star,label="Training (x,t)")
ax1.scatter(x_star_init,t_star_init,label="I.C.")
ax1.scatter(x_star_in,t_star_in,label="B.C. In")
ax1.scatter(x_star_out,t_star_out,label="B.C. Out")
ax1.set_xlabel(r'Dimensionless Distance $x^* = x/L$ (-)', fontsize=12)
ax1.set_ylabel(r'Dimensionless Time $t^* = t U/L$ (-)', fontsize=12)
ax1.grid()    
ax1.minorticks_on()
ax1.legend(loc='best', fontsize=10) 
plt.savefig(os.path.join(script_dir, "Fig_coll_points.jpg"))

# =============================================================================
# Plot loss function vs iterations
# =============================================================================

fig3 = plt.figure(figsize=(10,6),tight_layout=True)
obj = np.array(obj)
ax3 = fig3.add_subplot(111) # fig.add_subplot(111)
ax3.plot(obj[:,0],'-b',label="total Loss")
ax3.plot(obj[:,1],'y', label = "MSE PDE")
ax3.plot(obj[:,2],'k', label = "MSE BI ",alpha=0.2)
ax3.plot(obj[:,3],'m', label = "MSE BC Left")
ax3.plot(obj[:,4],'r', label = "MSE BC Right")

ax3.set_yscale('log', base=10)
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Loss function")
ax3.minorticks_on()
#ax3.tick_params(axis='y', which='minor', length=10, width=1.2, color='k')
ax3.grid()
ax3.legend(loc='best',fontsize=8)
plt.savefig(os.path.join(script_dir, "Fig3_loss.jpg"))
