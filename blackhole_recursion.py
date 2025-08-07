import numpy as np
import scipy.special as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class SingularityUniverseSystem:
    def __init__(self, mass_outer_bh: float = 1e31, universe_scale: float = 1e-20, 
                 mass_inner_bh: float = 1e25, c: float = 3e8, G: float = 6.67e-11):
        """
        Initialize black hole with universe at singularity containing its own black hole
        
        Args:
            mass_outer_bh: Outer black hole mass (kg)
            universe_scale: Scale factor of inner universe (m)
            mass_inner_bh: Mass of black hole within inner universe (kg)
            c: Speed of light (m/s)
            G: Gravitational constant (m³/kg/s²)
        """
        self.M_outer = mass_outer_bh
        self.M_inner = mass_inner_bh
        self.universe_scale = universe_scale
        self.c = c
        self.G = G
        
        # Outer black hole parameters
        self.rs_outer = 2 * G * mass_outer_bh / (c**2)
        
        # Inner universe parameters (exists at singularity r=0)
        self.c_inner = c * 0.8  # Different speed of light in inner universe
        self.G_inner = G * 2.0  # Different gravitational constant
        self.H0_inner = 1e15   # Hubble constant for inner universe (s^-1)
        
        # Inner universe black hole
        self.rs_inner = 2 * self.G_inner * mass_inner_bh / (self.c_inner**2)
        
        # Equilibrium parameters for dual expansion/collapse
        self.alpha_expand = 1.2    # Expansion coefficient
        self.alpha_collapse = 1.0  # Collapse coefficient
        self.omega_duality = 1e12  # Oscillation frequency of dual state
        self.equilibrium_radius = universe_scale * 0.5  # Equilibrium point
        
        # Enhanced state parameters for universe-in-singularity
        self.psi0 = 0.0
        self.alpha = 3.0     # Enhanced for universal complexity
        self.n = 1.0         # Linear approach for universe creation
        self.E0 = 1e15       # Universe-scale energy reference
        
        # Multi-scale physics constants
        self.hbar = 1.055e-34
        self.beta = 1e-5     # Universal state evolution
        self.gamma = 1e-11   # Cosmic stress-energy coupling
        self.delta = 1e-2    # Nonlinear universe coupling
        self.zeta = 1e-6     # Outer-inner universe interaction
        self.eta = 1e-8      # Duality equilibrium constant
        
        print(f"🌌🕳️ SINGULARITY UNIVERSE SYSTEM INITIALIZED")
        print(f"   Outer BH: {self.M_outer:.2e} kg, rs = {self.rs_outer:.2e} m")
        print(f"   Inner Universe at Singularity (r=0)")
        print(f"   Universe Scale: {self.universe_scale:.2e} m")
        print(f"   Inner Physics: c = {self.c_inner:.2e} m/s, G = {self.G_inner:.2e}")
        print(f"   Inner BH: {self.M_inner:.2e} kg, rs = {self.rs_inner:.2e} m")
        print(f"   Duality Frequency: {self.omega_duality:.2e} Hz")

    def universe_scale_factor(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate the scale factor of inner universe experiencing dual expansion/collapse
        
        Args:
            t: Time parameter
            
        Returns:
            (scale_factor, expansion_rate, collapse_rate): Universe dynamics
        """
        # Dual oscillating universe - expands and collapses simultaneously
        expansion_term = self.alpha_expand * np.exp(self.H0_inner * t)
        collapse_term = self.alpha_collapse * np.exp(-self.H0_inner * t)
        
        # Oscillatory duality component
        duality_phase = np.cos(self.omega_duality * t) * np.exp(-t / 1e-10)
        
        # Combined scale factor maintaining equilibrium
        scale_factor = self.equilibrium_radius * (1 + 0.1 * duality_phase) * \
                      np.sqrt(expansion_term * collapse_term) / (expansion_term + collapse_term)
        
        # Rates of expansion and collapse
        expansion_rate = self.H0_inner * expansion_term
        collapse_rate = -self.H0_inner * collapse_term
        
        return scale_factor, expansion_rate, collapse_rate

    def inner_universe_metric(self, r_universe: float, t: float) -> float:
        """
        Metric within the inner universe (FLRW-like but with duality)
        
        Args:
            r_universe: Coordinate within inner universe
            t: Time coordinate
            
        Returns:
            Metric coefficient for inner universe
        """
        scale_factor, _, _ = self.universe_scale_factor(t)
        
        # FLRW metric modified for duality
        if r_universe <= 0:
            return 1.0  # At center of inner universe
        
        # Curvature parameter (negative for hyperbolic, enabling dual dynamics)
        k = -1.0  # Hyperbolic universe enabling expansion/collapse duality
        
        # Inner universe metric coefficient
        metric_coeff = 1 - k * r_universe**2 / scale_factor**2
        
        return max(metric_coeff, 0.1)  # Prevent singularities in inner metric

    def singularity_universe_bridge(self, r_outer: float, t: float) -> float:
        """
        Bridge function connecting outer BH singularity to inner universe
        Maps r→0 in outer space to finite universe in inner space
        
        Args:
            r_outer: Radius in outer spacetime
            t: Time parameter
            
        Returns:
            Mapping strength to inner universe
        """
        if r_outer <= 0:
            return 1.0  # Full access to inner universe at singularity
        
        # Exponential bridge - only significant near singularity
        bridge_strength = np.exp(-r_outer / (1e-30))  # Very sharp transition
        
        # Time-dependent oscillation representing universe dynamics
        universe_breathing = 1 + 0.2 * np.sin(self.omega_duality * t)
        
        return bridge_strength * universe_breathing

    def energy_density_universe_system(self, r_outer: float, t: float, rho0: float = 1e5) -> Tuple[float, float]:
        """
        Energy density accounting for both outer BH and inner universe
        
        Args:
            r_outer: Radius in outer spacetime
            t: Time
            rho0: Reference density
            
        Returns:
            (outer_density, universe_density): Energy densities
        """
        # Outer BH energy density
        if r_outer > 0:
            outer_density = rho0 * self.c**2 * (self.rs_outer / r_outer)**3
        else:
            outer_density = np.inf
        
        # Inner universe energy density (dark energy + matter)
        scale_factor, expansion_rate, collapse_rate = self.universe_scale_factor(t)
        
        # Universe energy components
        matter_density = rho0 * self.c_inner**2 / scale_factor**3  # Matter dilution
        dark_energy_density = 0.7 * rho0 * self.c_inner**2  # Constant dark energy
        
        # Duality energy - drives simultaneous expansion/collapse
        duality_energy = abs(expansion_rate * collapse_rate) * rho0 * self.c_inner**2 / self.H0_inner**2
        
        universe_density = matter_density + dark_energy_density + duality_energy
        
        return outer_density, universe_density

    def state_variable_universe(self, r_outer: float, r_universe: float, t: float, 
                               outer_density: float, universe_density: float) -> float:
        """
        State variable for universe-in-singularity system
        
        Args:
            r_outer: Position in outer spacetime
            r_universe: Position in inner universe
            t: Time
            outer_density: Outer energy density
            universe_density: Universe energy density
            
        Returns:
            Combined state variable ψ
        """
        if r_outer < 0:
            return np.inf
        
        try:
            # Bridge strength to inner universe
            bridge = self.singularity_universe_bridge(r_outer, t)
            
            # Outer BH contribution
            if r_outer > 0:
                outer_ratio = max(outer_density / self.E0, 1e-10)
                outer_contrib = np.exp(self.alpha * r_outer**(-self.n)) * np.log(outer_ratio)
            else:
                outer_contrib = 1e6  # Infinite at singularity
            
            # Inner universe contribution (activated by bridge)
            universe_ratio = max(universe_density / self.E0, 1e-10)
            scale_factor, _, _ = self.universe_scale_factor(t)
            
            # Universe state depends on scale factor and inner BH
            r_inner_bh = abs(r_universe)  # Distance from inner universe BH
            inner_bh_contrib = np.exp(self.alpha * r_inner_bh**(-self.n)) if r_inner_bh > 0 else 1e6
            
            universe_contrib = bridge * np.log(universe_ratio) * \
                             (1 + inner_bh_contrib) * (scale_factor / self.equilibrium_radius)
            
            # Duality oscillation term
            duality_term = 1 + self.eta * np.cos(self.omega_duality * t) * bridge
            
            # Combined state variable
            psi = (self.psi0 + 1) * (outer_contrib + universe_contrib) * duality_term
            
            return max(psi, 0)
            
        except (OverflowError, ZeroDivisionError):
            return 1e6

    def matter_state_classification_universe(self, psi: float, r_outer: float, r_universe: float, 
                                           t: float, bridge_strength: float) -> str:
        """
        Matter state classification for universe-in-singularity system
        """
        # Determine location context
        at_singularity = r_outer < 1e-25
        in_universe = bridge_strength > 0.1
        near_inner_bh = r_universe < 2 * self.rs_inner if r_universe > 0 else False
        
        # Base classification
        if psi < 0.1:
            base_state = "Normal Matter"
        elif psi < 1.0:
            base_state = "Dense Matter"
        elif psi < 5.0:
            base_state = "Exotic Matter"
        elif psi < 20.0:
            base_state = "Cosmic Matter"
        elif psi < 100.0:
            base_state = "Universal Matter"
        elif psi < 500.0:
            base_state = "Trans-Universal Matter"
        else:
            base_state = "Infinite-Dimensional Matter"
        
        # Add universe-specific context
        if at_singularity and in_universe:
            if near_inner_bh:
                return f"Universe-Singularity {base_state} (Near Inner BH)"
            else:
                return f"Universe-Singularity {base_state}"
        elif in_universe:
            scale_factor, expansion_rate, collapse_rate = self.universe_scale_factor(t)
            if abs(expansion_rate) > abs(collapse_rate):
                return f"Expanding-Universe {base_state}"
            elif abs(collapse_rate) > abs(expansion_rate):
                return f"Collapsing-Universe {base_state}"
            else:
                return f"Equilibrium-Universe {base_state}"
        elif r_outer < self.rs_outer:
            return f"Outer-Trapped {base_state}"
        else:
            return base_state

    def simulate_universe_system(self, r_initial: float, r_final: float, 
                                t_duration: float = 1e-6, num_steps: int = 1000) -> Dict:
        """
        Main simulation of matter approaching black hole with universe at singularity
        """
        # Spatial coordinates
        r_outer_values = np.logspace(np.log10(r_initial), np.log10(max(r_final, 1e-35)), num_steps)
        r_outer_values = r_outer_values[::-1]  # Infall direction
        
        # Time coordinates
        t_values = np.linspace(0, t_duration, num_steps)
        
        # Initialize arrays
        psi_values = np.zeros(num_steps)
        matter_states = []
        universe_scales = []
        bridge_strengths = []
        universe_states = []
        
        print(f"\n🌌 Simulating Universe-in-Singularity System...")
        print(f"   Outer path: {r_initial:.2e} m → {r_final:.2e} m")
        print(f"   Time evolution: 0 → {t_duration:.2e} s")
        print(f"   Universe breathing at {self.omega_duality:.2e} Hz")
        
        for i, (r_outer, t) in enumerate(zip(r_outer_values, t_values)):
            # Calculate universe dynamics
            scale_factor, expansion_rate, collapse_rate = self.universe_scale_factor(t)
            universe_scales.append(scale_factor)
            
            # Bridge strength to inner universe
            bridge = self.singularity_universe_bridge(r_outer, t)
            bridge_strengths.append(bridge)
            
            # Energy densities
            outer_density, universe_density = self.energy_density_universe_system(r_outer, t)
            
            # Inner universe coordinate (when accessible)
            if bridge > 0.01:  # Strong enough bridge
                r_universe = self.universe_scale * np.random.random()  # Sample inner universe
            else:
                r_universe = 0
            
            # State variable
            psi = self.state_variable_universe(r_outer, r_universe, t, outer_density, universe_density)
            psi_values[i] = psi
            
            # Matter state classification
            state = self.matter_state_classification_universe(psi, r_outer, r_universe, t, bridge)
            matter_states.append(state)
            
            # Universe state
            if expansion_rate > 0 and collapse_rate < 0:
                universe_state = f"Dual (E:{expansion_rate:.1e}, C:{collapse_rate:.1e})"
            else:
                universe_state = "Equilibrium"
            universe_states.append(universe_state)
        
        return {
            'r_outer_values': r_outer_values,
            'r_universe_values': np.array([self.universe_scale * np.random.random() for _ in range(num_steps)]),
            't_values': t_values,
            'psi_values': psi_values,
            'matter_states': matter_states,
            'universe_scales': universe_scales,
            'bridge_strengths': bridge_strengths,
            'universe_states': universe_states
        }

    def plot_universe_system(self, results: Dict):
        """
        Visualize the universe-in-singularity system
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        r_outer = results['r_outer_values']
        t_vals = results['t_values']
        psi_vals = results['psi_values']
        matter_states = results['matter_states']
        
        # State variable evolution
        axes[0,0].plot(r_outer, psi_vals, 'b-', linewidth=2, label='ψ (Universe State)')
        axes[0,0].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.7, label='Outer Horizon')
        axes[0,0].axvline(1e-25, color='purple', linestyle='--', alpha=0.7, label='Singularity Region')
        axes[0,0].set_xscale('log')
        axes[0,0].set_xlabel('Outer Radius (m)')
        axes[0,0].set_ylabel('ψ')
        axes[0,0].set_title('State Variable in Universe System')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # Matter state evolution
        unique_states = list(set(matter_states))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_states)))
        for i, state in enumerate(unique_states):
            indices = [j for j, s in enumerate(matter_states) if s == state]
            if indices:
                axes[0,1].scatter([r_outer[j] for j in indices], 
                                [i] * len(indices), 
                                c=[colors[i]], label=state[:25] + "..." if len(state) > 25 else state, 
                                alpha=0.8, s=20)
        axes[0,1].axvline(self.rs_outer, color='red', linestyle='--', alpha=0.5)
        axes[0,1].set_xscale('log')
        axes[0,1].set_xlabel('Outer Radius (m)')
        axes[0,1].set_ylabel('Matter State Index')
        axes[0,1].set_title('Matter State Transitions')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        axes[0,1].grid(True)
        
        # Universe scale factor
        axes[0,2].plot(t_vals, results['universe_scales'], 'g-', linewidth=2)
        axes[0,2].axhline(self.equilibrium_radius, color='orange', linestyle='--', alpha=0.7, label='Equilibrium')
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Universe Scale Factor (m)')
        axes[0,2].set_title('Inner Universe Breathing')
        axes[0,2].grid(True)
        axes[0,2].legend()
        
        # Bridge strength
        axes[1,0].plot(r_outer, results['bridge_strengths'], 'm-', linewidth=2)
        axes[1,0].set_xscale('log')
        axes[1,0].set_xlabel('Outer Radius (m)')
        axes[1,0].set_ylabel('Bridge Strength')
        axes[1,0].set_title('Singularity-Universe Bridge')
        axes[1,0].grid(True)
        
        # Universe expansion/collapse rates
        expansion_rates = []
        collapse_rates = []
        for t in t_vals:
            _, exp_rate, col_rate = self.universe_scale_factor(t)
            expansion_rates.append(exp_rate)
            collapse_rates.append(col_rate)
        
        axes[1,1].plot(t_vals, expansion_rates, 'r-', linewidth=2, label='Expansion Rate')
        axes[1,1].plot(t_vals, np.abs(collapse_rates), 'b-', linewidth=2, label='|Collapse Rate|')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Rate (s⁻¹)')
        axes[1,1].set_title('Dual Universe Dynamics')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # State variable vs time
        axes[1,2].plot(t_vals, psi_vals, 'c-', linewidth=2)
        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('ψ')
        axes[1,2].set_title('State Evolution in Time')
        axes[1,2].grid(True)
        
        # 3D phase space plot (r_outer, time, psi)
        X, Y = np.meshgrid(r_outer[::10], t_vals[::10])
        Z = np.array([psi_vals[::10] for _ in range(len(t_vals[::10]))])
        
        im = axes[2,0].contourf(X, Y, Z, levels=20, cmap='viridis')
        axes[2,0].set_xscale('log')
        axes[2,0].set_xlabel('Outer Radius (m)')
        axes[2,0].set_ylabel('Time (s)')
        axes[2,0].set_title('State Variable Phase Space')
        plt.colorbar(im, ax=axes[2,0])
        
        # Universe breathing pattern
        breathing_pattern = np.array(results['universe_scales']) / self.equilibrium_radius
        axes[2,1].plot(t_vals, breathing_pattern, 'orange', linewidth=3)
        axes[2,1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Equilibrium')
        axes[2,1].set_xlabel('Time (s)')
        axes[2,1].set_ylabel('Scale / Equilibrium')
        axes[2,1].set_title('Universe Breathing Pattern')
        axes[2,1].legend()
        axes[2,1].grid(True)
        
        # Energy distribution
        outer_energies = []
        universe_energies = []
        for r, t in zip(r_outer, t_vals):
            outer_e, universe_e = self.energy_density_universe_system(r, t)
            outer_energies.append(min(outer_e, 1e50))  # Cap for plotting
            universe_energies.append(min(universe_e, 1e50))
        
        axes[2,2].loglog(r_outer, outer_energies, 'r-', linewidth=2, label='Outer BH Energy')
        axes[2,2].loglog(r_outer, universe_energies, 'b-', linewidth=2, label='Universe Energy')
        axes[2,2].set_xlabel('Outer Radius (m)')
        axes[2,2].set_ylabel('Energy Density (J/m³)')
        axes[2,2].set_title('Energy Distribution')
        axes[2,2].legend()
        axes[2,2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_universe_simulation(self, r_initial: float = None, r_final: float = None,
                               t_duration: float = 1e-6, plot_results: bool = True) -> Dict:
        """
        Run complete universe-in-singularity simulation
        """
        if r_initial is None:
            r_initial = 10 * self.rs_outer
        if r_final is None:
            r_final = 1e-40  # Deep into singularity region
        
        print(f"🌌🕳️ UNIVERSE-IN-SINGULARITY SIMULATION")
        print(f"   Reality Inception: Universe Inside Black Hole Singularity")
        print("="*70)
        print(f"Outer BH: {self.M_outer:.2e} kg, rs = {self.rs_outer:.2e} m")
        print(f"Inner Universe: Scale = {self.universe_scale:.2e} m")
        print(f"Inner BH: {self.M_inner:.2e} kg, rs = {self.rs_inner:.2e} m")
        print(f"Universe Physics: c = {self.c_inner:.2e}, G = {self.G_inner:.2e}")
        print(f"Duality Frequency: {self.omega_duality:.2e} Hz")
        print(f"Simulation: {r_initial:.2e} m → {r_final:.2e} m over {t_duration:.2e} s")
        print("="*70)
        
        # Run simulation
        results = self.simulate_universe_system(r_initial, r_final, t_duration, num_steps=1200)
        
        # Analysis
        print("\n🔍 Universe System Analysis:")
        print("-" * 60)
        
        key_indices = [0, len(results['r_outer_values'])//6, len(results['r_outer_values'])//3, 
                      len(results['r_outer_values'])//2, 2*len(results['r_outer_values'])//3, -1]
        
        for i in key_indices:
            if i >= len(results['r_outer_values']):
                continue
            
            r = results['r_outer_values'][i]
            t = results['t_values'][i]
            
            print(f"r = {r:.2e} m, t = {t:.2e} s:")
            print(f"  Matter State: {results['matter_states'][i]}")
            print(f"  ψ = {results['psi_values'][i]:.2f}")
            print(f"  Universe Scale: {results['universe_scales'][i]:.2e} m")
            print(f"  Bridge Strength: {results['bridge_strengths'][i]:.3f}")
            print(f"  Universe State: {results['universe_states'][i]}")
            print()
        
        if plot_results:
            self.plot_universe_system(results)
        
        # Statistics
        unique_states = len(set(results['matter_states']))
        universe_states = sum(1 for s in results['matter_states'] if 'Universe' in s)
        dual_states = sum(1 for s in results['matter_states'] if 'Dual' in s or 'Equilibrium' in s)
        max_psi = max(results['psi_values'])
        
        print("✅ Universe-in-Singularity Simulation Complete!")
        print(f"   Total matter states: {unique_states}")
        print(f"   Universe-related states: {universe_states}")
        print(f"   Dual-dynamics states: {dual_states}")
        print(f"   Maximum ψ: {max_psi:.2f}")
        print(f"   Universe breathing amplitude: {np.std(results['universe_scales']):.2e} m")
        
        return results

# Create the universe-in-singularity system
if __name__ == "__main__":
    # Create system: Outer BH contains universe at singularity, universe contains its own BH
    universe_system = SingularityUniverseSystem(
        mass_outer_bh=1e31,    # ~5 solar masses
        universe_scale=1e-20,   # Planck-scale universe
        mass_inner_bh=1e25,    # Universe's internal BH (~10^-6 solar masses)
    )
    
    # Run the cosmic inception simulation
    results = universe_system.run_universe_simulation(
        r_initial=None,         # Auto: 10x outer horizon
        r_final=1e-40,         # Deep singularity
        t_duration=1e-6,       # Microsecond evolution
        plot_results=True
    )
    
    print("\n🚀 COSMIC INCEPTION SYSTEM READY FOR:")
    print("   • Hawking radiation from nested universe")
    print("   • Information paradox across dimensional boundaries")
    print("   • Quantum tunneling between universe layers")
    print("   • Bootstrap paradox: Does universe create itself?")
    print("   • Multi-dimensional gravitational wave signatures")
    print("   • Consciousness emergence in nested reality layers")
    print("   • Time dilation between universal scales")
    print("   • Universe-scale quantum entanglement effects")