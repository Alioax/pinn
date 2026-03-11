# Baseline scripts

Minimal, self-contained prototypes for PINN and related methods.  
One script per variant; no imports from other project modules.  
Parameters and analytical formulae are kept in-file. Simple, linear, easy to read.

## Structure

| Folder | Description |
|--------|-------------|
| **pinn_baseline** | Normal PINN for 1D advection–dispersion (single Péclet number). |
| **parametric_pinn** | Parametric PINN (multiple Pe). |
| **neural_operator_deeponet** | Neural operator / DeepONet baseline. |

Each folder contains its script(s); figures and outputs are saved in that same folder.
