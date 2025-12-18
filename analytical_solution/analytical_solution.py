"""
Analytical Solution for 1D Contaminant Transport in an Aquifer

Implements the Ogata-Banks analytical solution for the 1D advection-dispersion equation:
    ∂C/∂t + u*∂C/∂x = D*∂²C/∂x²

Assumptions for semi-infinite domain solution:
1. Semi-infinite spatial domain (x ∈ [0, ∞))
2. Constant inlet concentration C(0,t) = C₀
3. Zero initial condition C(x,0) = 0 for x > 0
4. Far-field condition: C(x,t) → 0 as x → ∞
5. Valid for times before the advective front reaches the domain boundary

Analytical Solution (Ogata-Banks):
    C(x,t) = (C₀/2) * [erfc((x-ut)/(2√(Dt))) + exp(ux/D) * erfc((x+ut)/(2√(Dt)))]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ============================================================================
# Problem Parameters
# ============================================================================

# Physical parameters
U = 0.1  # m/day (advection velocity)
D = 1e-7 * 86400  # m²/day (dispersion coefficient, converted from m²/s)
C_0 = 5.0  # kg/m³ (inlet concentration)

# Spatial domain (semi-infinite, but we plot up to 100 m)
X_MAX = 100.0  # meters
X_PLOT = np.linspace(0, X_MAX, 500)

# ============================================================================
# Analytical Solution
# ============================================================================

def analytical_solution(x, t, U_param=None, D_param=None, C_0_param=None):
    """
    Compute the Ogata-Banks analytical solution for 1D contaminant transport.
    
    Parameters:
        x: spatial coordinate (m)
        t: time (days)
        U_param: advection velocity (m/day). If None, uses module default U.
        D_param: dispersion coefficient (m²/day). If None, uses module default D.
        C_0_param: inlet concentration (kg/m³). If None, uses module default C_0.
    
    Returns:
        C(x,t): concentration (kg/m³)
    """
    # Use provided parameters or fall back to module defaults
    u = U_param if U_param is not None else U
    d = D_param if D_param is not None else D
    c0 = C_0_param if C_0_param is not None else C_0
    
    # Handle t=0 case (initial condition)
    if t == 0:
        return np.zeros_like(x)
    
    # Convert to numpy array if scalar
    x = np.asarray(x)
    t = np.asarray(t)
    
    # Handle scalar inputs
    if x.ndim == 0:
        x = np.array([x])
        scalar_output = True
    else:
        scalar_output = False
    
    if t.ndim == 0:
        t = np.broadcast_to(t, x.shape)
    
    # Compute terms in the analytical solution
    sqrt_Dt = np.sqrt(d * t)
    
    # First term: erfc((x - ut) / (2√(Dt)))
    term1_arg = (x - u * t) / (2 * sqrt_Dt)
    term1 = erfc(term1_arg)
    
    # Second term: exp(ux/D) * erfc((x + ut) / (2√(Dt)))
    # Handle numerical overflow: for large x, exp(ux/D) overflows but erfc becomes very small
    # Use a threshold to avoid overflow
    ux_over_D = u * x / d
    term2_arg = (x + u * t) / (2 * sqrt_Dt)
    
    # For large ux/D, check if erfc term is negligible first
    # If erfc argument is large (>10), erfc is essentially zero, so skip exp calculation
    term2 = np.zeros_like(x)
    mask = term2_arg < 10  # Only compute where erfc might be non-negligible
    
    if np.any(mask):
        # For values that might matter, compute carefully
        # If exp would overflow, the erfc term should make it negligible anyway
        try:
            term2_exp = np.exp(np.clip(ux_over_D[mask], None, 700))  # Clip to avoid overflow
            term2_erfc = erfc(term2_arg[mask])
            term2[mask] = term2_exp * term2_erfc
        except (OverflowError, RuntimeWarning):
            # If still problematic, set to zero (erfc of large argument is ~0)
            term2[mask] = 0.0
    
    # Final solution
    C = (c0 / 2) * (term1 + term2)
    
    # Handle any remaining NaN or inf values (set to zero)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scalar_output:
        return C[0]
    return C

# ============================================================================
# Visualization
# ============================================================================

def plot_concentration_profiles(times=[0, 100, 200, 300, 400, 500, 600, 800, 1000]):
    """
    Plot concentration profiles C(x,t) vs x for selected times.
    """
    plt.figure(figsize=(10, 6))
    
    for t in times:
        C = analytical_solution(X_PLOT, t)
        plt.plot(X_PLOT, C, linewidth=2, label=f't = {t} days')
    
    plt.xlabel('Distance x (m)', fontsize=12)
    plt.ylabel('Concentration C (kg/m³)', fontsize=12)
    plt.title('Analytical Solution: Concentration Profiles at Different Times', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, X_MAX)
    plt.ylim(0, C_0 * 1.1)
    
    # Add vertical line at x=60m to show where concentration becomes negligible for early times
    plt.axvline(x=60, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(62, C_0 * 0.9, 'x ≈ 60m\n(typical front\nfor early times)', 
             fontsize=9, alpha=0.7, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('results/plots/analytical_solution.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'results/plots/analytical_solution.png'")
    plt.show()
    
    # Print diagnostic information
    print("\n" + "="*60)
    print("Concentration Front Analysis")
    print("="*60)
    print("For early times, concentration is effectively zero beyond x≈60m")
    print("because the advective front moves at u=0.1 m/day:")
    print(f"  - At t=600 days: front at x = {0.1*600:.0f}m")
    print(f"  - At t=500 days: front at x = {0.1*500:.0f}m")
    print(f"  - At t=400 days: front at x = {0.1*400:.0f}m")
    print("\nConcentration only reaches x=80-100m at later times (t≥800 days)")
    print("="*60)

def plot_space_time_contour():
    """
    Create a space-time contour plot of the concentration.
    """
    # Create meshgrid
    x_mesh = np.linspace(0, X_MAX, 200)
    t_mesh = np.linspace(0, 1000, 200)
    X_mesh, T_mesh = np.meshgrid(x_mesh, t_mesh)
    
    # Compute concentration for all points
    C_mesh = analytical_solution(X_mesh, T_mesh)
    
    # Create contour plot
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X_mesh, T_mesh, C_mesh, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Concentration C (kg/m³)')
    plt.xlabel('Distance x (m)', fontsize=12)
    plt.ylabel('Time t (days)', fontsize=12)
    plt.title('Analytical Solution: Concentration Contour Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results/plots/analytical_contour.png', dpi=300, bbox_inches='tight')
    print("Contour plot saved to 'results/plots/analytical_contour.png'")
    plt.show()

def analyze_outlet_arrival():
    """
    Analyze when concentration reaches x=100m using analytical solution.
    """
    x_outlet = 100.0  # meters
    times = np.linspace(0, 1500, 2000)
    concentrations = analytical_solution(np.array([x_outlet]), times)
    
    # Find first time when concentration exceeds threshold
    threshold = 0.01  # kg/m³
    idx = np.where(concentrations >= threshold)[0]
    
    if len(idx) > 0:
        t_arrival = times[idx[0]]
        C_arrival = concentrations[idx[0]]
        print(f"\nFirst detection at x=100m (C ≥ {threshold} kg/m³):")
        print(f"  Time: {t_arrival:.2f} days")
        print(f"  Concentration: {C_arrival:.4f} kg/m³")
    
    # Find time when concentration reaches 50% of inlet
    idx_50 = np.where(concentrations >= 0.5 * C_0)[0]
    if len(idx_50) > 0:
        t_50 = times[idx_50[0]]
        print(f"\n50% of inlet concentration (2.5 kg/m³) at x=100m:")
        print(f"  Time: {t_50:.2f} days")
    
    # Plot concentration at outlet over time
    plt.figure(figsize=(10, 6))
    plt.plot(times, concentrations, 'b-', linewidth=2, label='Concentration at x=100m')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold} kg/m³)')
    plt.axhline(y=0.5*C_0, color='g', linestyle='--', label='50% of inlet (2.5 kg/m³)')
    
    if len(idx) > 0:
        plt.axvline(x=t_arrival, color='r', linestyle=':', alpha=0.7)
        plt.plot(t_arrival, C_arrival, 'ro', markersize=10, 
                label=f'First detection: t={t_arrival:.1f} days')
    
    if len(idx_50) > 0:
        plt.axvline(x=t_50, color='g', linestyle=':', alpha=0.7)
        plt.plot(t_50, concentrations[idx_50[0]], 'go', markersize=10,
                label=f'50% inlet: t={t_50:.1f} days')
    
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Concentration C (kg/m³)', fontsize=12)
    plt.title('Analytical Solution: Concentration at Outlet (x=100m) vs Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/analytical_outlet.png', dpi=300, bbox_inches='tight')
    print("\nOutlet analysis plot saved to 'results/plots/analytical_outlet.png'")
    plt.show()
    
    return t_arrival if len(idx) > 0 else None

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Analytical Solution for 1D Contaminant Transport")
    print("="*60)
    print(f"\nParameters:")
    print(f"  Advection velocity: u = {U} m/day")
    print(f"  Dispersion coefficient: D = {D:.6e} m²/day")
    print(f"  Inlet concentration: C₀ = {C_0} kg/m³")
    print(f"\nTheoretical advection time to x=100m: {100.0/U:.1f} days")
    print("="*60)
    
    # Plot concentration profiles
    print("\nGenerating concentration profiles...")
    plot_concentration_profiles()
    
    # Plot space-time contour
    print("\nGenerating space-time contour plot...")
    plot_space_time_contour()
    
    # Analyze outlet arrival
    print("\nAnalyzing concentration at outlet (x=100m)...")
    analyze_outlet_arrival()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)

