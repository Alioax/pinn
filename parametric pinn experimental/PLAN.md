# Revised end-to-end plan: remove `U` and keep analytical overlay (dimensionless-only)

## Objective
Make both scripts purely dimensionless:
- PINN trains on: \((x^*, t^*, \log Pe)\rightarrow C^*\)
- Plots show: \(C^*(x^*, t^*)\) for selected \(Pe\)
- No dependence on \(U\), \(L\), \(T_{\text{phys}}\), or “days”
- Keep **analytical overlay** by calling the existing analytical solver in a dimensionless “fake-physical” mode (no edits to the analytical module)

---

## 1) Dimensionless problem statement (docs/header)
### PDE
\[
C^*_{t^*} + C^*_{x^*} - \frac{1}{Pe} C^*_{x^*x^*} = 0
\]

### Domain
- \(x^* \in [0,1]\)
- \(t^* \in [0,t^*_{\text{final}}]\)
- \(Pe \in [Pe_{\min}, Pe_{\max}]\)

### IC/BC (as implemented)
- IC: \(C^*(x^*,0)=0\)
- Inlet: \(C^*(0,t^*)=1\)
- Outlet: \(C^*(1,t^*)=0\)

---

## 2) Training script revisions (`pinn_parametric_train.py`)
### 2.1 Remove dimensional parameters
Delete from config:
- `U`, `L`, `T_phys`, `C0`

Keep:
- `Pe_min`, `Pe_max`
- network/training hyperparameters
- `t_final_star`

### 2.2 Use direct dimensionless time horizon
Add/configure:
- `t_final_star = <value>` (dimensionless)

Delete derived scaling:
- `T = L / U`
- `t_final_star = T_phys / T`

Keep:
- `logPe_min`, `logPe_max`

### 2.3 Sampling (already dimensionless)
- Sample \(x^*\in[0,1]\)
- Sample \(t^*\in[0,t^*_{\text{final}}]\)
- Sample \(\log Pe\) uniformly over \([\log Pe_{\min}, \log Pe_{\max}]\)

### 2.4 Checkpoint becomes dimensionless
Store in checkpoint `config`:
- `num_layers`, `num_neurons`, `activation_name`
- `t_final_star`
- `Pe_min`, `Pe_max`
- (optional) loss history

Do not store:
- `U`, `L`, `T_phys`, `C0`

---

## 3) Plot script revisions (`pinn_parametric_plot.py`)
### 3.1 Convert plotting to dimensionless
Replace:
- `times_days` → `times_tstar`
- `x_plot = linspace(0, L, ...)` → `x_star_plot = linspace(0, 1, ...)`
Remove:
- `U`, `L`, `T = L/U`, `D = U*L/Pe`
Update labels:
- x-axis: `x*`
- y-axis: `C*`

### 3.2 Keep analytical overlay without changing analytical module
Keep:
- `from analytical_solution import analytical_solution`

Call analytical solver using dimensionless “fake-physical” mapping:
- set `U_fake = 1`
- set `C0_fake = 1`
- set `D_fake = 1/Pe`
- pass `x = x_star_plot` and `t = t_star` (t treated as “days” numerically, but is dimensionless)

Overlay:
- PINN curve: \(C^*_{\text{PINN}}(x^*,t^*,Pe)\)
- Analytical curve: \(C^*_{\text{ana}}(x^*,t^*,Pe)\) returned by the solver under fake-physical mapping

### 3.3 Read `t_final_star` from checkpoint and validate plot times
- Load `t_final_star` from checkpoint config
- Ensure `max(times_tstar) <= t_final_star`

### 3.4 Output naming
- Save: `plots/{Pe}_Cstar_profiles.png`
- Keep loss plot as-is

---

## 4) Consistency checks after edits
- \(t^*=0\): profile near 0
- \(x^*=0\): profile near 1 for all \(t^*\)
- \(x^*=1\): profile near 0 for all \(t^*\)
- Increasing \(Pe\): sharper fronts; decreasing \(Pe\): smoother diffusion

---

## 5) Suggested minimal configs (example)
### Training
- `t_final_star = 2.0`
- `Pe_min = 10`, `Pe_max = 1e5`

### Plotting
- `times_tstar = [0, 0.2, 0.4, ..., 2.0]`
- `pe_values_to_plot = [10, 50, 1e2, 5e2, 1e3, 1e4, 1e5]`

---

## 6) Deliverables
- Checkpoint contains only dimensionless config
- Plots show \(C^*(x^*)\) at \(t^*\) for multiple \(Pe\)
- Analytical solution remains overlaid (dimensionless via fake-physical mapping)
- No explicit \(U\), \(L\), \(T_{\text{phys}}\), or day-based plotting
