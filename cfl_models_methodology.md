# Methodology Notes for CFL-Based Surrogates

This document summarizes the implementation methodology used in:

- `code/homogeneous_cfl/baseline_cfl_pinn/pinn1d_cfl_pe_baseline.py`
- `code/homogeneous_cfl/parametric_cfl_pinn/pinn1d_cfl_pe_parametric_pinn.py`
- `code/homogeneous_cfl/pino_cfl/pinn1d_cfl_pe_parametric_neural_operator.py`

It is intentionally focused on model architecture, collocation strategy, training, and evaluation. The governing problem definition and non-dimensional equations are assumed to be already covered in the LaTeX report.

---

## 1) Shared Experimental Setup

## 1.1 Physical-to-dimensionless parameter handling

All three scripts use a common physical reference setup:

- Domain length: `L = 100 m`
- Time horizon: `T_MAX = 1200 d`
- Dispersion: `D = 1e-6 m^2/s`
- Reference concentration scale: `C0 = 5.0` (used for dimensional plotting in baseline)

Derived quantities:

- `T_SECONDS = T_MAX * 86400`
- `Pe = D * T_SECONDS / L^2` (fixed across runs)
- `CFL = U * T_MAX / L`

For parametric models, velocity set:

- `U_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05] m/d`
- Corresponding `CFL_VALUES = U_VALUES * T_MAX / L = [0.12, 0.24, 0.36, 0.48, 0.60]`

So the parametric surrogate learns a continuous mapping over a CFL interval, while `Pe` remains fixed.

## 1.2 Numerical and training defaults

Common defaults in all scripts:

- Framework: PyTorch
- Precision: `float64` (`torch.float64`)
- Random seed: `1234567` for both NumPy and PyTorch
- Device: GPU (`cuda:0`) if available, otherwise CPU
- Optimizer: `torch.optim.LBFGS` with:
  - learning rate `lr = 1.0`
  - `history_size = 50`
  - `line_search_fn = "strong_wolfe"`
  - `max_iter = 1` per outer step
- Outer optimization steps (epochs): `1000`
- Loss components are equally weighted:
  - PDE residual: `1.0`
  - Initial condition: `1.0`
  - Inlet BC: `1.0`
  - Outlet BC: `1.0`

All output heads use a sigmoid constraint to keep predicted concentration in `[0, 1]` (dimensionless scale).

## 1.3 Automatic differentiation and residual construction

All PDE residual terms are computed via autograd:

- First derivatives `dC/dt*`, `dC/dx*`
- Second derivative `d2C/dx*2`
- Residual form:
  - Baseline: `r = dC/dt* + CFL * dC/dx* - Pe * d2C/dx*2` (single fixed CFL)
  - Parametric models: `r = dC/dt* + CFL_input * dC/dx* - Pe * d2C/dx*2`

Mean-squared error is used for each term and summed into the total objective.

---

## 2) Collocation and Tensor Representation

## 2.1 Baseline PINN collocation (2D)

Regular fixed meshes in `(x*, t*)`:

- PDE mesh: `mesh_nx_pde = 50`, `mesh_nt_pde = 50`
  - Total PDE points: `50 x 50 = 2500`
- IC points at `t*=0`: `mesh_n_ic = 50`
- Inlet BC points at `x*=0`: `mesh_n_bc = 50`
- Outlet BC points at `x*=1`: `mesh_n_bc = 50`

Tensorization:

- PDE coordinates are flattened from meshgrid and converted into tensors.
- `x*` and `t*` PDE tensors have `requires_grad=True`.

## 2.2 Parametric PINN and PINO collocation (3D)

Regular fixed meshes in `(x*, t*, CFL)`:

- PDE mesh:
  - `mesh_nx_pde = 50`
  - `mesh_nt_pde = 50`
  - `mesh_ncfl_pde = 50`
  - Total PDE points: `50 x 50 x 50 = 125000`
- IC mesh (`t*=0`):
  - `mesh_ic_nx = 50`, `mesh_ic_ncfl = 50`
  - Total IC points: `2500`
- BC meshes:
  - `mesh_bc_nt = 50`, `mesh_bc_ncfl = 50`
  - Inlet and outlet each: `2500` points

This creates a deterministic tensor-product collocation set over space, time, and parameter.

---

## 3) Baseline PINN (Single-Case Surrogate)

## 3.1 Input-output mapping

- Inputs: `(x*, t*)`
- Output: scalar `C*(x*, t*)` in `[0,1]`

## 3.2 Architecture

`PINN_Transp`:

- Hidden depth parameterized by:
  - `num_layers = 4`
  - `num_neurons = 16`
- Construction:
  - Input layer: `Linear(2 -> 16)` + `Tanh`
  - Then 4 repeated hidden blocks: `Linear(16 -> 16)` + `Tanh`
  - Output: `Linear(16 -> 1)` + `Sigmoid`
- Initialization:
  - Xavier uniform with tanh gain for weights
  - Zero bias

## 3.3 Objective function

Total loss:

`L = L_pde + L_ic + L_inlet + L_outlet`

where:

- `L_pde = mean(r^2)` with fixed run-specific `CFL`
- `L_ic = mean((C*(x*,0) - 0)^2)`
- `L_inlet = mean((C*(0,t*) - 1)^2)`
- `L_outlet = mean((C*(1,t*) - 0)^2)`

## 3.4 Training protocol

- Pure L-BFGS training for 1000 steps
- A closure computes the full-batch loss and gradients at each step
- Loss history records total and each component separately for diagnostics

---

## 4) Parametric PINN (Operator-like PINN with Direct Parameter Injection)

## 4.1 Input-output mapping

- Inputs: `(x*, t*, CFL)`
- Output: scalar `C*(x*, t*, CFL)` in `[0,1]`

The model learns a continuous parameterized solution manifold over the CFL interval.

## 4.2 Architecture

`ParametricPINN` with architecture list:

- `pinn_architecture = [3, 16, 16, 16, 16, 1]`
- Layer sequence:
  - `Linear(3 -> 16) + Tanh`
  - `Linear(16 -> 16) + Tanh` (repeated)
  - Final `Linear(16 -> 1) + Sigmoid`
- Initialization:
  - Xavier normal with tanh gain
  - Zero biases

## 4.3 Residual and constraints

Residual uses the pointwise CFL input:

`r = dC/dt* + CFL * dC/dx* - Pe * d2C/dx*2`

Conditions are enforced over the expanded `(x*, CFL)` or `(t*, CFL)` sets:

- IC: `C*(x*,0,CFL)=0`
- Inlet BC: `C*(0,t*,CFL)=1`
- Outlet BC: `C*(1,t*,CFL)=0`

## 4.4 Training protocol

- Full-batch L-BFGS for 1000 steps
- Same equal weighting and closure strategy as baseline
- Deterministic regular 3D collocation grid

---

## 5) Parametric PINO (DeepONet-Style Physics-Informed Neural Operator)

## 5.1 Operator interpretation

This model approximates an operator from parameter input (`CFL`) to field output over coordinates `(x*, t*)`.

- Branch net encodes parameter signal
- Trunk net encodes coordinates
- Their latent interaction produces concentration output

## 5.2 Architecture

`DeepONetParametric` with:

- `sensor_count = 1` (single scalar sensor carrying raw CFL)
- Branch architecture: `[1, 16, 16, 16]`
- Trunk architecture: `[2, 16, 16, 16]`
- Activations: `Tanh` in hidden layers
- Output fusion:
  - `b = branch(CFL)`
  - `t = trunk([x*, t*])`
  - `C* = sigmoid(sum(b * t))`

Initialization:

- Xavier normal with tanh gain on both branch and trunk linear layers
- Zero biases

## 5.3 Physics-informed training objective

Same PDE residual and boundary/initial penalties as parametric PINN, but with DeepONet forward pass.

In implementation:

- Branch input at each collocation point is the CFL tensor expanded to shape `(N, sensor_count)`.
- Trunk input is the coordinate pair `(x*, t*)`.

## 5.4 Training protocol

- Same L-BFGS setup and 1000-step schedule
- Same deterministic 3D collocation strategy
- Same loss decomposition and logging

---

## 6) Evaluation and Post-Processing Pipeline

## 6.1 Reference comparison

All scripts compare learned predictions against Ogata-Banks analytical profiles for selected times and CFL values.

- Baseline:
  - Uses dimensional concentration (`C = C* * C0`) and dimensional distance for plots.
- Parametric PINN / PINO:
  - Plots in normalized coordinates (`x*`, `C*`), one panel per CFL.

## 6.2 Reported diagnostics

Each script saves:

- Collocation figure (sampling layout)
- Concentration profile comparison figure
- Loss-history figure (total + component losses on log scale)
- Trained model weights (`.pt`)

---

## 7) Cross-Model Methodological Comparison

- **Baseline PINN**
  - Learns one fixed-CFL solution map.
  - Lowest input dimensionality and smallest PDE collocation set.
  - Appropriate as a single-scenario reference model.

- **Parametric PINN**
  - Learns one joint map over `(x*, t*, CFL)`.
  - Directly injects CFL into a single MLP.
  - Retains conventional PINN structure while becoming parameter-aware.

- **Parametric PINO (DeepONet)**
  - Explicitly separates parameter encoding (branch) from coordinate encoding (trunk).
  - Represents a neural-operator style factorization for parametric transport.
  - Uses the same physics-informed residual framework and collocation logic as parametric PINN.

---

## 8) Suggested Methodology Text Integration Points (for LaTeX)

If you want to map this into your `Methods` section quickly:

- Put Sections 1 and 2 under:
  - Shared optimization and loss
  - Collocation sets and tensor representation
- Put Sections 3, 4, and 5 under:
  - Baseline PINN
  - Parametric PINN
  - Parametric PINO (DeepONet)
- Use Sections 6 and 7 for:
  - Training/evaluation protocol paragraph
  - A concise comparison table between model classes

This avoids repeating the PDE/problem statement while preserving reproducibility details.
