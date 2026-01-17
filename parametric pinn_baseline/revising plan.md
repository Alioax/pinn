# Final Plan: Sandbox-Style Rewrite of `pinn_parametric_baseline.py` (Linear, Minimal Abstractions)

This plan produces a **linear, top-to-bottom** script that keeps the **same functionality** as the current parametric PINN:
- parametric input over a **range of Peclet numbers** via `log(Pe)`
- **resample collocation points every epoch**
- same PDE residual + IC/BC losses, weights, optimizer choice, and training loop behavior
- same outputs: collocation plots, loss plot, concentration-profile plots

**Reference files**
- Parametric code to simplify: fileciteturn0file0  
- Sandbox style example: fileciteturn1file0  

---

## 0) Decisions locked in (based on your answers)

- Keep **one** minimal `PINN(nn.Module)` class (allowed).  
- **No** `if __name__ == "__main__":` block (script runs linearly).  
- **No** extra activation classes (no `Sine` / `torch.sin` option); only built-in `torch.nn.*` activations.  
- Collocation resampling code is written **explicitly inside the epoch loop** (**Option 4A**).  
- Plotting can be cleaned up while preserving **same outputs and filenames** (not required to preserve every cosmetic quirk).

---

## 1) Target script layout (exact order)

1. Imports  
2. Visualization settings (`mpl.rcParams`, fonts, colors)  
3. Configuration (edit parameters here)  
4. Derived parameters (`logPe_min`, `logPe_max`, etc.)  
5. Minimal `PINN` class  
6. Random seeds + model init  
7. Results directory creation  
8. **One-time** collocation sampling + collocation plots  
9. Optimizer selection + loss lists  
10. Training loop (resample each epoch; compute PDE/IC/BC losses inline)  
11. Loss plots  
12. Concentration profile plots (loop over Pe and times)  
13. End of file (no main guard)

---

## 2) Configuration block (sandbox style)

### 2.1 Physics parameters (dimensionless)
Keep:
- `Pe_min`, `Pe_max`
- `t_final_star`

### 2.2 Model architecture (no activation factory)
Use one active activation line + commented alternatives (only built-in `torch.nn` activations):

```python
# Model architecture
num_layers = 3
num_neurons = 16
activation = torch.nn.Tanh  # activation function

# Alternative activation functions (uncomment to use):
# activation = torch.nn.ReLU
# activation = torch.nn.SiLU
# activation = torch.nn.GELU
# activation = torch.nn.ELU
# activation = torch.nn.LeakyReLU
# activation = torch.nn.Sigmoid
# activation = torch.nn.Softplus
```

### 2.3 Training parameters
Keep as-is:
- `num_epochs`, `lr`
- `num_collocation`, `num_ic`, `num_bc`
- `weight_pde`, `weight_ic`, `weight_inlet_bc`, `weight_outlet_bc`

### 2.4 Plotting parameters
Keep as-is:
- `times_tstar`, `pe_values_to_plot`
- `num_spatial_points`, `plot_dpi`

---

## 3) Derived parameters (explicit, near the config)

Compute immediately after config:

- `logPe_min = np.log(Pe_min)`
- `logPe_max = np.log(Pe_max)`

(Everything else remains identical to the parametric script’s meaning.)

---

## 4) Model definition (minimal, single class)

Keep a single `PINN(nn.Module)` class, similar to the sandbox baseline, but with **3 inputs**:

- Inputs: `(x_star, t_star, log_pe)`  
- Forward: `torch.cat([x_star, t_star, log_pe], dim=1)` then `self.net(...)`  
- Network layout:
  - `layer_sizes = [3] + [num_neurons]*num_layers + [1]`
  - build `nn.Sequential` with `Linear` + `activation()` between layers
- Initialization:
  - Xavier normal init for each `Linear`
  - bias zeros
  - gain consistent with tanh (or keep same gain as current parametric code)

No extra methods, no helper functions.

---

## 5) Seeds + model initialization (explicit)

Immediately after model class:

```python
torch.manual_seed(123456789)
np.random.seed(123456789)

model = PINN(num_layers, num_neurons, activation)
```

---

## 6) Results directory (explicit)

Inline:

```python
script_dir = Path(__file__).parent
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
```

---

## 7) One-time collocation sampling + plots (pretraining)

### 7.1 Sample one batch (NumPy)
Write explicit blocks:

- **PDE collocation**
  - `x_star_pde ~ Uniform(0,1)`
  - `t_star_pde ~ Uniform(0,t_final_star)`
  - `log_pe_pde ~ Uniform(logPe_min, logPe_max)`

- **IC**
  - `x_star_ic ~ Uniform(0,1)`
  - `t_star_ic = 0`
  - `log_pe_ic ~ Uniform(logPe_min, logPe_max)`

- **Inlet BC**
  - `x_star_inlet = 0`
  - `t_star_inlet ~ Uniform(0,t_final_star)`
  - `log_pe_inlet ~ Uniform(logPe_min, logPe_max)`

- **Outlet BC**
  - `x_star_outlet = 1`
  - `t_star_outlet ~ Uniform(0,t_final_star)`
  - `log_pe_outlet ~ Uniform(logPe_min, logPe_max)`

### 7.2 Plot collocation distributions
Create and save (same filenames as parametric script):

1. Scatter `(x*, t*)` with four groups  
   - Save: `results/collocation_points.png`

2. Scatter `(t*, log Pe)` using PDE points  
   - Save: `results/collocation_points_logpe.png`

Keep the plot styling consistent with your current aesthetics as desired.

---

## 8) Optimizer + loss tracking (explicit)

### 8.1 Optimizer options (sandbox style)
Keep the same default optimizer as the parametric script:

```python
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lr/10)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
```

### 8.2 Loss lists (explicit)
Use lists:

```python
loss_total = []
loss_pde = []
loss_ic = []
loss_inlet = []
loss_outlet = []
```

---

## 9) Training loop (inline everything; resample each epoch)

### 9.1 Loop skeleton
Use tqdm (optional but matches current workflow):

```python
pbar = tqdm(range(num_epochs), desc="Training PINN")
for epoch in pbar:
    model.train()
    optimizer.zero_grad()
    ...
    total_loss.backward()
    optimizer.step()
    ...
pbar.close()
```

### 9.2 Resampling code INSIDE the loop (Option A)
At the top of each epoch, explicitly repeat the sampling code blocks from Section 7.1 to generate:
- `x_star_pde, t_star_pde, log_pe_pde`
- `x_star_ic, t_star_ic, log_pe_ic`
- `x_star_inlet, t_star_inlet, log_pe_inlet`
- `x_star_outlet, t_star_outlet, log_pe_outlet`

### 9.3 Convert to tensors (inside epoch)
- PDE tensors:
  - `x_star_pde_tensor = torch.tensor(..., requires_grad=True)`
  - `t_star_pde_tensor = torch.tensor(..., requires_grad=True)`
  - `log_pe_pde_tensor = torch.tensor(...)`
- IC/BC tensors:
  - gradients not required

### 9.4 Compute PDE residual loss (inline)
1. Forward:
   - `C_pde = model(x_pde, t_pde, log_pe_pde)`
2. Derivatives:
   - `dC_dt = grad(C_pde, t_pde, ...)`
   - `dC_dx = grad(C_pde, x_pde, ...)`
   - `d2C_dx2 = grad(dC_dx, x_pde, ...)`
3. Peclet:
   - `Pe = torch.exp(log_pe_pde)`
4. Residual:
   - `r = dC_dt + dC_dx - (1.0/Pe)*d2C_dx2`
5. Loss:
   - `pde_loss = mean(r**2)`

### 9.5 Compute IC/BC losses (inline)
- IC: `C(x*,0)=0`  
- Inlet: `C(0,t*)=1`  
- Outlet: `C(1,t*)=0`  

Use `nn.MSELoss()` directly (either instantiate once before loop or call inline).

### 9.6 Total loss (same weights)
```python
total_loss = (
    weight_pde * pde_loss +
    weight_ic * ic_loss +
    weight_inlet_bc * inlet_loss +
    weight_outlet_bc * outlet_loss
)
```

### 9.7 Track losses + tqdm postfix
Append `.item()` values to lists.  
Update progress display every ~10 epochs (or same cadence as current script).

---

## 10) Loss plotting (post-training)

After training, produce and save:

- `results/loss.png`

Minimum requirements:
- total loss curve
- component losses (PDE, IC, inlet, outlet)
- log y-scale where appropriate

You may simplify plotting code relative to the parametric script, but keep the saved filename.

---

## 11) Concentration profile plotting (post-training)

### 11.1 Setup
- `x_star_plot = np.linspace(0, 1, num_spatial_points)`

### 11.2 Loop over Peclet values
For each `pe_plot` in `pe_values_to_plot`:
- `log_pe_plot = np.log(pe_plot)`

### 11.3 Loop over times
For each `t_star` in `times_tstar` (keep same ordering intention):
- Convert to tensors:
  - `x = torch.tensor(x_star_plot).reshape(-1,1)`
  - `t = torch.full_like(x, t_star)`
  - `logpe = torch.full_like(x, log_pe_plot)`
- Evaluate:
  - `model.eval()`, `torch.no_grad()`
- Plot `C*(x*, t*)`

### 11.4 Save figure per Pe
Save exactly as before:
- `results/{pe_plot} Cstar_profiles.png`

---

## 12) What to delete from the original parametric file

Remove these abstractions and replace with inline blocks:
- `get_activation()`
- `sample_collocation_points()`
- `compute_losses()`
- `train()`
- `plot_collocation_points()`
- `plot_loss_curves()`
- `plot_concentration_profiles()`
- `if __name__ == "__main__":` block

Keep only the minimal `PINN` class.

---

## 13) Final reviewer-friendly summary

The rewritten script should read as:

> Parameters → Model → Sample points → Plot sampling → Train (resample each epoch) → Plot losses → Plot profiles

No hidden logic, no indirection, no helper functions, no main guard—just a clear, linear experiment.
