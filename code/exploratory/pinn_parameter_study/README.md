# PINN Parameter Study

This directory contains a parameter study implementation based on the baseline PINN code. It is used to investigate how different values of the dispersion coefficient (D) and advection speed (U) affect the PINN solution.

## Overview

This script is identical to the baseline PINN implementation (`pinn_baseline/pinn_baseline.py`) but is set up in a separate directory to facilitate parameter studies. You can modify the physics parameters (U and D) at the top of the script to explore different scenarios.

## Purpose

- Study the effect of different dispersion coefficients (D) on the solution
- Study the effect of different advection speeds (U) on the solution
- Compare PINN solutions with analytical solutions for various parameter combinations

## How to run

```bash
python pinn_parameter_study.py
```

## Configuration

Edit the physics parameters at the top of `pinn_parameter_study.py`:

- **U**: Advection velocity (m/day) - default: `0.1`
- **D**: Dispersion coefficient (m²/day) - default: `1e-7 * 86400`

All other parameters (model architecture, training parameters, etc.) remain the same as the baseline for consistency.

## Output

Results are saved to `results/pinn_parameter_study_concentration_profiles.png` showing concentration profiles at multiple time steps, comparing PINN predictions with analytical solutions.
