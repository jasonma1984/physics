import numpy as np
import scipy.special as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class NestedBlackHoleSystem:
    def __init__(self, mass_outer: float = 1e31, mass_inner: float = 3e30, 
                 inner_position: float = None, c: float = 3e8, G: float = 6.67e-11):
        """
        Initialize nested black hole system with inner BH inside outer BH's horizon
        
        Args:
            mass_outer: Outer (larger) black hole mass (kg)
            mass_inner: Inner (smaller) black hole mass (kg) - exists inside outer BH
            inner_position: Position of inner BH relative to outer center (m)
            c: Speed of light (m/s)
            G: Gravitational constant (m³/kg/s²)
        """
        self.M_outer = mass_outer
        self.M_inner = mass_inner
        self.c = c
        self.G = G
        
        # Schwarzschild radii
        self.rs_outer = 2 * G * mass_outer / (c**2)     # Outer horizon
        self.rs_inner = 2 * G * mass_inner / (c**2)     # Inner horizon
        
        # Position inner BH well inside outer horizon
        if inner_position is None:
            self.inner_pos = 0.3 * self.rs_outer  # 30% of way to outer singularity
        else:
            self.inner_pos = min(inner_position, 0.8 * self.rs_outer)  # Ensure it's inside
            
        # Verify nested configuration
        assert self.inner_pos + self.rs_inner < self.rs_outer, "Inner BH must fit inside outer BH!"
        
        # State transition parameters for nested system
        self.psi0 = 0.0
        self.alpha = 2.0     # Enhanced for nested complexity
        self.n = 1.5         # Modified exponent for dual singularities
        self.E0 = 1e12       # Higher reference energy for extreme conditions
        
        # Enhanced physics constants
        self.hbar = 1.055e-34
        self.beta = 5e-6     # Enhanced state evolution
        self.gamma = 5e-12   # Enhanced stress-energy coupling
        self.delta = 5e-3    # Enhanced nonlinear coupling
        self.kappa = 1e-4    # Inner-outer BH interaction strength
        
        print(f"🕳️ Nested Black Hole System - BH inside BH!")
        print(f"   Outer BH: {self.M_outer:.2e} kg, rs_outer = {self.rs_outer:.2e} m")
        print(f"   Inner BH: {self.M_inner:.2e} kg, rs_inner = {self.rs_inner:.2e} m")
        print(f"   Inner BH position: {self.inner_pos:.2e} m from outer center")
        print(f"   Inner BH is {self.inner_pos/self.rs_outer:.1%} inside outer horizon")

    def nested_gravitational_field(self, r: float) -> Tuple[float, float, float]:
        """
        Calculate gravitational field with nested singularities
        
        Args:
            r: Distance from outer BH center
            
        Returns:
            (field_outer, field_inner, field_combined): Gravitational field components
        """
        # Distance from inner BH
        r_inner = abs(r - self.inner_pos)
        r_inner = max(r_inner, 1e-30)  # Avoid division by zero
        
        # Outer BH field (standard)
        if r > 1e-30:
            field_outer = self.G * self.M_outer / r**2
        else:
            field_outer = np.inf
            
        # Inner BH field (modified by being inside outer horizon)
        field_inner = self.G * self.M_inner / r_inner**2
        
        # Combined field with interaction terms
        # Inner BH creates "bubble" of enhanced curvature inside outer BH
        interaction_factor = 1 + self.kappa * np.exp(-r_inner / (0.1 * self.rs_inner))
        field_combined = field_outer + field_inner * interaction_factor
        
        return field_outer, field_inner, field_combined

    def nested_metric_coefficient(self, r: float) -> float:
        """
        Metric coefficient for nested BH system
        Matter experiences both horizons simultaneously
        """
        if r <= 0:
            return 0.0
            
        # Distance from inner BH
        r_inner = abs(r - self.inner_pos)
        r_inner = max(r_inner, 1e-30)
        
        # Outer BH metric
        g_outer = 1 - self.rs_outer / r if r > self.rs_outer else 0
        
        # Inner BH metric (operates within outer BH)
        g_inner = 1 - self.rs_inner / r_inner if r_inner > self.rs_inner else 0
        
        # Combined metric - multiplicative for nested horizons
        # Being inside outer BH modifies how inner BH affects spacetime
        if r > self.rs_outer:
            # Outside both - normal outer BH dominates
            return g_outer
        elif r_inner > self.rs_inner:
            # Inside outer, outside inner - complex superposition
            return g_inner * (0.1 + 0.9 * np.exp(-(self.rs_outer - r) / self.rs_outer))
        else:
            # Inside both horizons - extreme curvature
            return 0.0

    def energy_density_nested(self, r: float, rho0: float = 1e4) -> float:
        """
        Energy density with contributions from both nested singularities
        """
        if r <= 0:
            return np.inf
            
        r_inner = abs(r - self.inner_pos)
        r_inner = max(r_inner, 1e-30)
        
        # Compression from outer BH
        compression_outer = (self.rs_outer / max(r, 1e-30))**3
        
        # Additional compression from inner BH
        compression_inner = (self.rs_inner / r_inner)**3
        
        # Total compression with interaction effects
        total_compression = compression_outer + compression_inner
        
        # Enhanced energy density due to nested structure
        enhancement = 1 + np.exp(-r / (0.5 * self.rs_outer))  # Enhancement inside outer BH
        
        return rho0 * self.c**2 * total_compression * enhancement

    def state_variable_nested(self, r: float, energy_density: float) -> float:
        """
        State variable for nested BH system - accounts for dual singularities
        """
        if r <= 0:
            return np.inf
            
        try:
            r_inner = abs(r - self.inner_pos)
            r_inner = max(r_inner, 1e-30)
            
            # Energy ratio
            ed_ratio = max(energy_density / self.E0, 1e-10)
            
            # Dual singularity contribution
            outer_contrib = np.exp(self.alpha * r**(-self.n))
            inner_contrib = np.exp(self.alpha * r_inner**(-self.n))
            
            # Nested interaction term
            interaction = 1 + self.kappa * outer_contrib * inner_contrib
            
            psi = (self.psi0 + 1) * (outer_contrib + inner_contrib) * np.log(ed_ratio) * interaction
            return max(psi, 0)
            
        except (OverflowError, ZeroDivisionError):
            return 1e6

    def matter_state_classification_nested(self, psi: float, r: float, energy_density: float) -> str:
        """
        Enhanced matter state classification for nested BH system
        """
        r_inner = abs(r - self.inner_pos)
        
        # Determine location relative to both horizons
        inside_outer = r < self.rs_outer
        inside_inner = r_inner < self.rs_inner
        near_inner = r_inner < 2 * self.rs_inner
        
        # Base classification from psi
        if psi < 0.1:
            base_state = "Normal Matter"
        elif psi < 1.0:
            base_state = "Dense Matter"
        elif psi < 5.0:
            base_state = "Degenerate Matter"
        elif psi < 20.0:
            base_state = "Exotic Matter"
        elif psi < 100.0:
            base_state = "Trans-Planckian Matter"
        else:
            base_state = "Infinite State Matter"
            
        # Add nested BH context
        if inside_inner:
            return f"Inner Singularity {base_state}"
        elif near_inner and inside_outer:
            return f"Dual-Horizon {base_state}"
        elif inside_outer:
            return f"Outer-Trapped {base_state}"
        else:
            return base_state

    def tidal_force_nested(self, r: float, mass_test: float, delta_r: float, psi: float) -> float:
        """
        Tidal forces with contributions from both nested singularities
        """
        if r <= 0:
            return np.inf
            
        r_inner = abs(r - self.inner_pos)
        r_inner = max(r_inner, 1e-30)
        
        # Tidal force from outer BH
        tidal_outer = 2 * self.G * self.M_outer * mass_test * delta_r / r**3
        
        # Tidal force from inner BH
        tidal_inner = 2 * self.G * self.M_inner * mass_test * delta_r / r_inner**3
        
        # State-dependent amplification
        state_factor = 1 + np.sin(psi)**2 + 0.5 * np.cos(psi * r_inner / self.rs_inner)
        
        # Nested interaction enhancement
        nested_factor = 1 + self.kappa * np.exp(-r / self.rs_outer) * np.exp(-r_inner / self.rs_inner)
        
        return (tidal_outer + tidal_inner) * state_factor * nested_factor

    def simulate_matter_approach_nested(self, r_initial: float, r_final: float, 
                                       num_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Simulate matter falling through nested BH system
        """
        # Ensure we sample through both interesting regions
        r_values = np.logspace(np.log10(r_initial), np.log10(max(r_final, 1e-30)), num_steps)
        r_values = r_values[::-1]  # Reverse for infall
        
        psi_values = np.zeros(num_steps)
        matter_states = []
        
        print(f"\n🌀 Simulating matter infall through nested BH system...")
        print(f"   Path: {r_initial:.2e} m → {r_final:.2e} m")
        print(f"   Will cross outer horizon at r = {self.rs_outer:.2e} m")
        print(f"   Will encounter inner BH at r ≈ {self.inner_pos:.2e} m")
        
        for i, r in enumerate(r_values):
            # Calculate energy density for nested system
            energy_dens = self.energy_density_nested(r)
            
            # Calculate nested state variable
            psi = self.state_variable_nested(r, energy_dens)
            psi_values[i] = psi
            
            # Classify matter state in nested context
            state = self.matter_state_classification_nested(psi, r, energy_dens)
            matter_states.append(state)
            
        return r_values, psi_values, matter_states

    def calculate_nested_quantities(self, r_values: np.ndarray, psi_values: np.ndarray) -> dict:
        """
        Calculate physical quantities for nested BH system
        """
        quantities = {
            'tidal_forces': [],
            'energy_densities': [],
            'metric_coefficients': [],
            'gravitational_fields': [],
            'inner_distances': [],
            'horizon_crossings': []
        }
        
        test_mass = 1.0
        delta_r = 0.1
        
        for r, psi in zip(r_values, psi_values):
            if r <= 0:
                break
                
            # Distance from inner BH
            r_inner = abs(r - self.inner_pos)
            quantities['inner_distances'].append(r_inner)
            
            # Nested tidal force
            F_tidal = self.tidal_force_nested(r, test_mass, delta_r, psi)
            quantities['tidal_forces'].append(F_tidal)
            
            # Nested energy density
            energy_dens = self.energy_density_nested(r)
            quantities['energy_densities'].append(energy_dens)
            
            # Metric coefficient
            g_tt = self.nested_metric_coefficient(r)
            quantities['metric_coefficients'].append(g_tt)
            
            # Gravitational field components
            field_outer, field_inner, field_combined = self.nested_gravitational_field(r)
            quantities['gravitational_fields'].append(field_combined)
            
            # Track horizon crossings
            crossing_status = ""
            if r < self.rs_outer and r > self.rs_outer * 0.99:
                crossing_status = "Crossing Outer Horizon"
            elif r_inner < self.rs_inner and r_inner > self.rs_inner * 0.99:
                crossing_status = "Crossing Inner Horizon"
            elif r < self.rs_outer:
                crossing_status = "Inside Outer BH"
            elif r_inner < self.rs_inner:
                crossing_status = "Inside Inner BH"
            quantities['horizon_crossings'].append(crossing_status)
        
        return quantities

    def plot_nested_results(self, r_values, psi_values, matter_states, quantities=None):
        """
        Visualize nested BH simulation results
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # State variable evolution
        axes[0,0].plot(r_values, psi_values, 'b-', linewidth=2, label='ψ (Nested State)')
        axes[0,0].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.7, label='Outer Horizon')
        axes[0,0].axvline(self.inner_pos, color='orange', linestyle='--', alpha=0.7, label='Inner BH Position')
        axes[0,0].set_xscale('log')
        axes[0,0].set_xlabel('Radius (m)')
        axes[0,0].set_ylabel('ψ')
        axes[0,0].set_title('State Variable in Nested BH System')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Matter state transitions
        unique_states = list(set(matter_states))
        state_colors = plt.cm.plasma(np.linspace(0, 1, len(unique_states)))
        for i, state in enumerate(unique_states):
            state_indices = [j for j, s in enumerate(matter_states) if s == state]
            if state_indices:
                axes[0,1].scatter([r_values[j] for j in state_indices], 
                                [i] * len(state_indices), 
                                c=[state_colors[i]], label=state[:20] + "..." if len(state) > 20 else state, 
                                alpha=0.8, s=30)
        axes[0,1].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.5)
        axes[0,1].axvline(self.inner_pos, color='orange', linestyle='--', alpha=0.5)
        axes[0,1].set_xscale('log')
        axes[0,1].set_xlabel('Radius (m)')
        axes[0,1].set_ylabel('Matter State Index')
        axes[0,1].set_title('Nested Matter State Transitions')
        axes[0,1].grid(True)
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        if quantities:
            # Energy density
            axes[1,0].plot(r_values[:len(quantities['energy_densities'])], 
                        quantities['energy_densities'], 'r-', linewidth=2)
            axes[1,0].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.5)
            axes[1,0].axvline(self.inner_pos, color='orange', linestyle='--', alpha=0.5)
            axes[1,0].set_xscale('log')
            axes[1,0].set_yscale('log')
            axes[1,0].set_xlabel('Radius (m)')
            axes[1,0].set_ylabel('Energy Density (J/m³)')
            axes[1,0].set_title('Nested Energy Density')
            axes[1,0].grid(True)
            
            # Tidal forces
            axes[1,1].plot(r_values[:len(quantities['tidal_forces'])], 
                        quantities['tidal_forces'], 'g-', linewidth=2)
            axes[1,1].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.5)
            axes[1,1].axvline(self.inner_pos, color='orange', linestyle='--', alpha=0.5)
            axes[1,1].set_xscale('log')
            axes[1,1].set_yscale('log')
            axes[1,1].set_xlabel('Radius (m)')
            axes[1,1].set_ylabel('Tidal Force (N)')
            axes[1,1].set_title('Nested Tidal Forces')
            axes[1,1].grid(True)
            
            # Distance from inner BH
            axes[2,0].plot(r_values[:len(quantities['inner_distances'])], 
                        quantities['inner_distances'], 'm-', linewidth=2)
            axes[2,0].axhline(self.rs_inner, color='orange', linestyle='--', alpha=0.7, label='Inner Horizon')
            axes[2,0].set_xscale('log')
            axes[2,0].set_yscale('log')
            axes[2,0].set_xlabel('Distance from Outer Center (m)')
            axes[2,0].set_ylabel('Distance from Inner BH (m)')
            axes[2,0].set_title('Proximity to Inner Black Hole')
            axes[2,0].grid(True)
            axes[2,0].legend()
            
            # Metric coefficient
            axes[2,1].plot(r_values[:len(quantities['metric_coefficients'])], 
                        quantities['metric_coefficients'], 'c-', linewidth=2)
            axes[2,1].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.5)
            axes[2,1].axvline(self.inner_pos, color='orange', linestyle='--', alpha=0.5)
            axes[2,1].set_xscale('log')
            axes[2,1].set_xlabel('Radius (m)')
            axes[2,1].set_ylabel('Metric Coefficient g_tt')
            axes[2,1].set_title('Nested Spacetime Metric')
            axes[2,1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_nested_simulation(self, r_initial: float = None, r_final: float = None, 
                             plot_results: bool = True) -> dict:
        """
        Run complete nested BH simulation
        """
        if r_initial is None:
            r_initial = 5 * self.rs_outer  # Start well outside
        if r_final is None:
            r_final = min(1e-30, self.inner_pos * 0.01)  # End near inner singularity
            
        print(f"🌌 Nested Black Hole Evolution Simulation")
        print(f"   Black Hole INSIDE Black Hole!")
        print("="*60)
        print(f"Outer BH: {self.M_outer:.2e} kg, rs = {self.rs_outer:.2e} m")
        print(f"Inner BH: {self.M_inner:.2e} kg, rs = {self.rs_inner:.2e} m")
        print(f"Inner BH position: {self.inner_pos:.2e} m ({self.inner_pos/self.rs_outer:.1%} inside)")
        print(f"Simulation: {r_initial:.2e} m → {r_final:.2e} m")
        print("="*60)
        
        # Run nested simulation
        r_values, psi_values, matter_states = self.simulate_matter_approach_nested(
            r_initial, r_final, num_steps=1500
        )
        
        # Calculate nested quantities
        quantities = self.calculate_nested_quantities(r_values, psi_values)
        
        # Analysis
        print("\n🔍 Nested BH Analysis:")
        print("-" * 50)
        
        key_points = [0, len(r_values)//6, len(r_values)//3, len(r_values)//2, 
                     2*len(r_values)//3, 5*len(r_values)//6, -1]
        
        for i in key_points:
            if i >= len(r_values):
                continue
            r = r_values[i]
            r_inner = abs(r - self.inner_pos)
            
            print(f"r = {r:.2e} m ({r/self.rs_outer:.2f}×rs_outer):")
            print(f"  Distance from inner BH: {r_inner:.2e} m")
            print(f"  Matter state: {matter_states[i]}")
            print(f"  ψ = {psi_values[i]:.2f}")
            
            if i < len(quantities['horizon_crossings']):
                if quantities['horizon_crossings'][i]:
                    print(f"  Status: {quantities['horizon_crossings'][i]}")
                print(f"  Tidal force: {quantities['tidal_forces'][i]:.2e} N")
                print(f"  Energy density: {quantities['energy_densities'][i]:.2e} J/m³")
            print()
        
        if plot_results:
            self.plot_nested_results(r_values, psi_values, matter_states, quantities)
        
        # Statistics
        unique_states = len(set(matter_states))
        inner_states = sum(1 for s in matter_states if 'Inner' in s)
        dual_states = sum(1 for s in matter_states if 'Dual' in s)
        max_psi = max(psi_values)
        
        print("✅ Nested Simulation Complete!")
        print(f"   Unique matter states: {unique_states}")
        print(f"   Inner singularity states: {inner_states}")
        print(f"   Dual-horizon states: {dual_states}")
        print(f"   Maximum ψ: {max_psi:.2f}")
        print(f"   Nested compression ratio: {r_initial/r_values[-1]:.2e}×")
        
        return {
            'r_values': r_values,
            'psi_values': psi_values,
            'matter_states': matter_states,
            'nested_quantities': quantities,
            'system_params': {
                'rs_outer': self.rs_outer,
                'rs_inner': self.rs_inner,
                'inner_position': self.inner_pos,
                'mass_ratio': self.M_inner / self.M_outer
            },
            'simulation_stats': {
                'unique_states': unique_states,
                'inner_states': inner_states,
                'dual_states': dual_states,
                'max_psi': max_psi,
                'compression_ratio': r_initial/r_values[-1]
            }
        }

# Create nested BH system
if __name__ == "__main__":
    # Primary: ~5 solar masses, Secondary: ~1.5 solar masses INSIDE the primary
    nested_system = NestedBlackHoleSystem(
        mass_outer=1e31,   # ~5 solar masses
        mass_inner=3e30,  # ~1.5 solar masses  
        inner_position=None   # Auto-position inside outer BH
    )
    
    # Run nested simulation
    results = nested_system.run_nested_simulation(plot_results=True)
    
    print("\n🚀 Nested BH System Ready for:")
    print("   • Gravitational wave emission from inner BH")
    print("   • Information paradox analysis across dual horizons") 
    print("   • Hawking radiation from nested event horizons")
    print("   • Quantum entanglement between separated singularities")
    print("   • Spacetime topology analysis of nested geometries")