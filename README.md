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
- **Report**: [`report/PINN Baseline - Ali Haghighi.pdf`](report/PINN%20Baseline%20-%20Ali%20Haghighi.pdf)

The report provides detailed theoretical background, implementation details, results analysis, and comparisons with analytical solutions.

## Baseline Implementation: `pinn_baseline/pinn_baseline.py`

The main implementation is the baseline PINN located in `pinn_baseline/pinn_baseline.py`. This is a minimal, clean, and self-contained implementation (~330 lines) that serves as the foundation for understanding PINNs applied to contaminant transport problems.

### Quick Start

```bash
cd pinn_baseline
python pinn_baseline.py
```

This will train the PINN and generate concentration profiles saved to `results/plots/pinn_baseline_concentration_profiles.png`.

### Code Structure and Implementation Details

The baseline implementation (`pinn_baseline/pinn_baseline.py`) is organized into clear sections:

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
  - Outlet: `∂C*/∂x*(1, t*) = 0` (Neumann, using automatic differentiation)

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
  - Saves both PNG (to `results/`) and PDF (to `report/figs/`) formats

### Key Features of Baseline Implementation

- **Minimal and Clean**: ~330 lines of well-organized, readable code
- **Self-contained**: All functionality in a single file
- **Dimensionless Formulation**: Improves numerical stability and training convergence
- **Reproducible**: Fixed random seeds ensure consistent results
- **Configurable**: All parameters easily adjustable at the top of the file
- **Validated**: Direct comparison with analytical solution (Ogata-Banks)

### Output

Running the baseline implementation generates:
- Concentration profiles at multiple time steps (200, 400, 600, 800, 1000 days)
- Comparison plots showing PINN predictions vs. analytical solution
- Results saved to `pinn_baseline/results/plots/pinn_baseline_concentration_profiles.png`
- PDF copy saved to `report/figs/pinn_baseline_concentration_profiles.pdf`

See `pinn_baseline/README.md` for additional details.

## Experimental Implementations

### Adaptive Learning Implementation

An advanced implementation with adaptive learning is available in `Experimental/pinn adaptive learning/`:

```bash
cd "Experimental/pinn adaptive learning"
python "pinn adaptive learning.py"
```

This experimental implementation extends the baseline with:

- **Residual-based Adaptive Refinement (RAR)**: Periodically identifies high-residual regions and adds collocation points around them
- **Hybrid Sampling Strategy**: Combines uniform random sampling with adaptive Gaussian-based sampling
- **Enhanced Visualization**: 
  - Residual analysis plots showing where physics violations occur
  - Training animation GIFs showing convergence over epochs
  - Comprehensive comparison with analytical solution
- **Advanced Loss Components**: Optional monotonicity constraints

The implementation is more complex (~1690 lines) and includes research-grade features for investigating adaptive learning strategies.

### Parameter Study Implementation

A parameter study implementation is available in `Experimental/pinn_parameter_study/` for investigating the effects of different hyperparameters and physical parameters on PINN performance.

## Project Structure

- **`pinn_baseline/`** - Main baseline PINN implementation (primary focus)
  - `pinn_baseline.py` - Complete, self-contained implementation
  - `README.md` - Detailed usage instructions
  - `results/` - Generated plots and results

- **`Experimental/pinn adaptive learning/`** - Advanced experimental implementation
  - Adaptive learning with RAR
  - Comprehensive visualization and analysis tools

- **`Experimental/pinn_parameter_study/`** - Parameter study implementation
  - Hyperparameter and physical parameter investigations

- **`analytical_solution/`** - Analytical solution implementation
  - Ogata-Banks analytical solution for 1D contaminant transport
  - Used for validation and comparison

- **`report/`** - Research report and figures
  - LaTeX report with full documentation
  - Generated figures in PDF format

Each folder is self-contained with its code and results in `results/` subdirectories.

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
