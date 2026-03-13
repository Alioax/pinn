# PINN for 1D Contaminant Transport in Aquifers

A Physics-Informed Neural Network (PINN) implementation for solving 1D advection-dispersion equation modeling contaminant transport in groundwater aquifers.

## Research Context

This project demonstrates the application of Physics-Informed Neural Networks to solve the 1D contaminant transport problem in aquifers. The work is part of Water Resources research focusing on efficient numerical methods for groundwater contamination modeling.

The PDE being solved:
```
∂C/∂t + U ∂C/∂x = D ∂²C/∂x²
```

with boundary conditions:
- Inlet: `C(0, t) = C₀` (Dirichlet)
- Outlet: `∂C/∂x(L, t) = 0` (Neumann)
- Initial: `C(x, 0) = 0` for x > 0

## Report

A comprehensive research report documenting the methodology, results, and analysis is available:
- **Report**: [`docs/reports/report 1 - PINN Baseline/PINN Baseline - Ali Haghighi.pdf`](docs/reports/report%201%20-%20PINN%20Baseline/PINN%20Baseline%20-%20Ali%20Haghighi.pdf)

The report provides detailed theoretical background, implementation details, results analysis, and comparisons with analytical solutions.

## Repository layout

The repo is organized by **research status**, not by method. All source code lives under `code/`:

- **`code/review_ready/`** — Minimal, supervisor-facing scripts. One script per variant (PINN, parametric PINN, DeepONet); self-contained, no cross-module imports. These are the only “review ready” deliverables.
- **`code/mainline_wip/`** — Active work on the main research path: full baseline implementations (PINN, parametric PINN, neural operator) that are still in development.
- **`code/exploratory/`** — Experiments and side ideas (sandboxes, grid searches, alternative loss schemes, etc.). Anything speculative or off the main trajectory lives here.
- **`code/shared/`** — Reusable pieces used by multiple projects (e.g. the analytical solution).

**Outputs** stay with the code that produced them: each project has its own `results/` folder (and for multi-run experiments, subfolders like `results/exp_001/`). Nothing is scattered at the repo root.

**Docs and references:** `docs/reports/` and `docs/slides/` hold LaTeX reports and presentation PDFs; `docs/diagrams/` holds architecture diagrams. `references/` holds literature PDFs.

As work matures, the intended flow is: **exploratory → mainline_wip → review_ready**.

## Baseline Implementation: `code/mainline_wip/pinn_baseline/pinn_baseline.py`

The main implementation is the baseline PINN located in `code/mainline_wip/pinn_baseline/pinn_baseline.py`. This is a minimal, clean, and self-contained implementation (~330 lines) that serves as the foundation for understanding PINNs applied to contaminant transport problems.

### Quick Start

```bash
cd code/mainline_wip/pinn_baseline
python pinn_baseline.py
```

This will train the PINN and generate:
- Collocation points distribution plot saved to `results/collocation_points.png`
- Concentration profiles saved to `results/pinn_baseline_concentration_profiles.png`

### Code Structure and Implementation Details

The baseline implementation (`code/mainline_wip/pinn_baseline/pinn_baseline.py`) is organized into clear sections:

#### 1. **Configuration Section** (Lines 35-68)
All parameters are configurable at the top of the file:
- **Physics parameters**: Advection velocity `U`, dispersion coefficient `D`, inlet concentration `C₀`, domain length `L`, and time horizon `T_phys`
- **Model architecture**: Number of hidden layers, neurons per layer, and activation function
- **Training parameters**: Number of epochs, learning rate, collocation point counts, and loss weights
- **Plotting parameters**: Times to visualize and spatial resolution

The code automatically computes derived dimensionless parameters (Péclet number `Pe`, time scale `T`, etc.) for numerical stability.

#### 2. **Dimensionless Formulation** (Lines 74-80)
The implementation uses dimensionless variables to improve numerical stability:
- `x* = x/L` (dimensionless space)
- `t* = t/T` where `T = L/U` (dimensionless time)
- `C* = C/C₀` (dimensionless concentration)

The PDE in dimensionless form becomes:
```
∂C*/∂t* + ∂C*/∂x* = (1/Pe) * ∂²C*/∂x*²
```
where `Pe = (U*L)/D` is the Péclet number.

#### 3. **Neural Network Architecture** (Lines 86-100)
The `PINN` class implements a fully connected feedforward neural network:
- Input: `(x*, t*)` - dimensionless space and time coordinates
- Output: `C*` - dimensionless concentration
- Architecture: Configurable number of hidden layers with configurable neurons per layer
- Activation: Configurable (default: Tanh)

#### 4. **Loss Functions** (Lines 106-140)
The physics is enforced through multiple loss components:

- **PDE Residual Loss** (`compute_pde_residual`): Computes the residual of the dimensionless PDE using automatic differentiation. The loss is the mean squared residual over collocation points.

- **Initial Condition Loss** (`compute_ic_loss`): Enforces `C*(x*, 0) = 0` at the initial time.

- **Boundary Condition Losses** (`compute_bc_losses`):
  - Inlet: `C*(0, t*) = 1` (Dirichlet)
  - Outlet: `C*(1, t*) = 0` (Dirichlet, far-field approximation)

All losses are computed in dimensionless form and can be weighted independently.

#### 5. **Training Loop** (Lines 146-176)
The `train_pinn` function implements the training process:
- Uses Adam optimizer with configurable learning rate
- Each epoch:
  1. Samples random collocation points `(x*, t*)` in the domain
  2. Computes PDE residual loss
  3. Samples initial condition points and computes IC loss
  4. Samples boundary condition points and computes BC losses
  5. Combines weighted losses and performs backpropagation
- Training continues for the specified number of epochs

#### 6. **Evaluation and Visualization** (Lines 182-320)
- **`predict_concentration`**: Converts dimensional inputs `(x, t)` to dimensionless, evaluates the model, and converts back to dimensional concentration.

- **`plot_concentration_profiles`**: 
  - Generates concentration profiles at multiple time steps
  - Compares PINN predictions with analytical solution (Ogata-Banks)
  - Creates publication-quality plots with proper styling
  - Saves both PNG (to `results/`) and PDF (to `docs/reports/report 1 - PINN Baseline/figs/`) formats

### Key Features of Baseline Implementation

- **Minimal and Clean**: ~384 lines of well-organized, readable code
- **Self-contained**: All functionality in a single file
- **Dimensionless Formulation**: Improves numerical stability and training convergence
- **Reproducible**: Fixed random seeds ensure consistent results
- **Configurable**: All parameters easily adjustable at the top of the file
- **Validated**: Direct comparison with analytical solution (Ogata-Banks)
- **Visualization**: Includes collocation points distribution plot for training analysis

### Output

Running the baseline implementation generates:
- Collocation points distribution visualization
- Concentration profiles at multiple time steps (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 days)
- Comparison plots showing PINN predictions vs. analytical solution
- Results saved to `code/mainline_wip/pinn_baseline/results/pinn_baseline_concentration_profiles.png`
- PDF copy saved to `docs/reports/report 1 - PINN Baseline/figs/pinn_baseline_concentration_profiles.pdf`

See `code/mainline_wip/pinn_baseline/README.md` for additional details.

## Project Structure

Code is organized by maturity:

- **`code/review_ready/`** — Minimal, supervisor-facing scripts (from former baseline_scripts)
  - `pinn_baseline/`, `parametric_pinn/`, `neural_operator_deeponet/`

- **`code/mainline_wip/`** — Main research path, active development
  - `pinn_baseline/` — Baseline PINN (primary focus)
  - `parametric_pinn/` — Parametric PINN
  - `neural_operator_supervised/`, `neural_operator_unsupervised/` — Neural operator baselines

- **`code/exploratory/`** — Experiments and side ideas
  - `parametric_pinn_sandbox/`, `pinn_grid_search/`, `pinn_parameter_study/`
  - `loss_distribution_learning/`, `pinn_distribution_adaptive_adam_lbfgs/`, `volume_of_error_pinn/`

- **`code/shared/`** — Shared utilities
  - `analytical_solution/` — Ogata-Banks analytical solution for validation

- **`docs/reports/`** — LaTeX reports and figures
- **`docs/slides/`** — Presentation PDFs
- **`docs/diagrams/`** — Architecture diagrams
- **`references/`** — Literature PDFs

Each code project has its own `results/` subdirectory.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch ≥2.0.0
- NumPy ≥1.24.0
- Matplotlib ≥3.7.0
- SciPy ≥1.10.0 (for analytical solution)
- imageio ≥2.31.0 (for GIF generation in experimental implementation)

## License

MIT License - see LICENSE file for details.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
- Ogata, A., & Banks, R. B. (1961). A solution of the differential equation of longitudinal dispersion in porous media. US Geological Survey Professional Paper, 411-A.
