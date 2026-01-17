# Simplified and Parametric PINNs for 1D Advection-Dispersion (Dimensionless)

## Objective
Summarize progress on the simplified baseline PINN and the parametric PINN with Peclet number as input. All results are dimensionless. Include PDF plots saved to `report #2/figs`.

## Required plots (PDF)
Save all plots as PDF in `report #2/figs` with the exact filenames below:
- `baseline_loss.pdf` (loss curves for baseline PINN)
- `baseline_profiles.pdf` (C* vs x* at selected t* for baseline PINN, with analytical overlay if enabled)
- `parametric_loss.pdf` (loss curves for parametric PINN)
- `parametric_profiles_pe10.pdf` (C* vs x* at selected t* for Pe = 10)
- `parametric_profiles_pe1e5.pdf` (C* vs x* at selected t* for Pe = 1e5)

## Report structure
1. Executive Summary
   - Progress since last update
   - Key outcomes: simplified baseline PINN, parametric PINN across Pe
   - References to the two scripts
2. Scope
   - Dimensionless formulation only
   - Focus on clarity and stability, not model complexity
3. Methods
   - Dimensionless PDE and domain
   - Baseline PINN setup (inputs x*, t*, output C*)
   - Parametric PINN setup (inputs x*, t*, log Pe)
   - Sampling and loss terms (PDE, IC, inlet, outlet)
   - Analytical overlay note: shape-only reference using dimensionless proxy mapping
4. Results
   - Baseline PINN: loss curve and concentration profiles (PDF figures)
   - Parametric PINN: loss curve and two representative Pe profile plots (PDF figures)
   - Qualitative trends with Pe (diffusion vs advection dominated)
5. Summary and Next Steps
   - Current status and what is working
   - Immediate next steps (e.g., sensitivity to t_final_star, loss weighting, additional Pe values)

## Notes
- Use only dimensionless variables in text and figure captions (x*, t*, C*, Pe).
- Keep the analytical overlay as a qualitative reference, not a strict benchmark.
