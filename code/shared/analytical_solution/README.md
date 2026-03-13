# Analytical Solution

Implementation of the Ogata-Banks analytical solution for 1D contaminant transport in a semi-infinite domain.

## Overview

This provides the reference analytical solution used for validation and comparison with the PINN results.

## Solution

The Ogata-Banks analytical solution:
```
C(x,t) = (C₀/2) * [erfc((x-ut)/(2√(Dt))) + exp(ux/D) * erfc((x+ut)/(2√(Dt)))]
```

## Features

- Concentration profiles at different times
- Space-time contour plots
- Outlet arrival time analysis
- Handles numerical overflow issues

## How to run

```bash
python analytical_solution.py
```

## Output

Generates several plots saved to `results/plots/`:
- `analytical_solution.png` - Concentration profiles
- `analytical_contour.png` - Space-time contour
- `analytical_outlet.png` - Outlet concentration over time

## Usage

The analytical solution function can be imported and used in other scripts:

```python
from analytical_solution import analytical_solution
C = analytical_solution(x, t)  # x in meters, t in days
```
