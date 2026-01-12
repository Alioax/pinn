# pinn_parametric_baseline.py overview

## Purpose
Trains a parametric Physics-Informed Neural Network (PINN) to solve 1D advection-dispersion (contaminant transport) and compare against an analytical solution across a range of Peclet numbers. The network learns a dimensionless concentration C* as a function of x*, t*, and log(Pe).

## Governing equation and conditions
Dimensional PDE:
- dC/dt + U dC/dx = D d2C/dx2

Boundary/initial conditions:
- C(0, t) = C0 (inlet)
- C(L, t) = 0 (outlet, far-field)
- C(x, 0) = 0 (initial)

Dimensionless form (learned by the network):
- dC*/dt* + dC*/dx* - (1/Pe) d2C*/dx* = 0

## Parameterization
- Trains on a range of Peclet numbers by sampling log(Pe) uniformly between log(Pe_min) and log(Pe_max).
- Inputs to the model are (x*, t*, log(Pe)).

## Model architecture
- Feed-forward MLP with 3 inputs and 1 output (C*).
- Configurable hidden layers, neurons, and activation (default: 3 layers, 16 neurons, Tanh).
- Xavier normal initialization with gain for Tanh.

## Sampling strategy
Generates random collocation points:
- PDE residual points: x* in [0, 1], t* in [0, t_final_star], log(Pe) in [logPe_min, logPe_max]
- Initial condition points: t* = 0
- Inlet boundary points: x* = 0
- Outlet boundary points: x* = 1

## Losses and training
Loss components:
- PDE residual loss
- Initial condition loss
- Inlet boundary condition loss
- Outlet boundary condition loss

Total loss:
- weighted sum of the four components

Training:
- Optimizer: AdamW (others commented for quick swapping)
- Training loop with tqdm progress bar

## Outputs
Plots saved relative to the script directory:
- Collocation point distribution: `parametric pinn_baseline/results/collocation_points.png`
- Concentration profiles (PINN vs analytical): `parametric pinn_baseline/results/pinn_baseline_concentration_profiles.png`
- PDF copy for report: `report/figs/pinn_baseline_concentration_profiles.pdf`

## Notes
- Uses `analytical_solution` from `analytical_solution/analytical_solution.py` for comparison.
- Dimensionless scaling uses T = L/U and t_final_star = T_phys / T.
- For plotting, the script evaluates a single Pe at the geometric mean of Pe_min and Pe_max.
- Predicted C* is converted back to C via C = C* * C0.
