import numpy as np
import scipy.special as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SingularityMatterStates:
    def __init__(self, mass_bh: float = 1e30, c: float = 3e8, G: float = 6.67e-11):
        """
        Initialize black hole parameters
        
        Args:
            mass_bh: Black hole mass (kg)
            c: Speed of light (m/s)
            G: Gravitational constant (m³/kg/s²)
        """
        self.M = mass_bh
        self.c = c
        self.G = G
        self.rs = 2 * G * mass_bh / (c**2)  # Schwarzschild radius
        
        # State transition parameters
        self.psi0 = 0.0  # Initial state index
        self.alpha = 1.0  # State transition coupling
        self.n = 2.0     # Singularity approach exponent
        self.E0 = 1e10   # Reference energy density
        
        # Physical constants
        self.hbar = 1.055e-34
        self.beta = 1e-6   # State evolution coupling
        self.gamma = 1e-12 # Stress-energy coupling
        self.delta = 1e-3  # Nonlinear state coupling
        
        
    def schwarzschild_metric(self, r: float) -> float:
        """Calculate Schwarzschild metric coefficient"""
        if r <= self.rs:
            return 0.0  # At/inside event horizon
        return 1 - self.rs / r

    def state_variable(self, r: float, energy_density: float) -> float:
        """
        Calculate the state variable ψ with improved numerical stability
        
        Args:
            r: Radial coordinate (m)
            energy_density: Local energy density (J/m³)
        
        Returns:
            State variable ψ
        """
        if r <= 0:
            return np.inf
        
        try:
            # Improved numerical stability
            ed_ratio = max(energy_density / self.E0, 1e-10)
            psi = (self.psi0 + 1) * np.exp(self.alpha * r**(-self.n)) * np.log(ed_ratio)
            return max(psi, 0)
        except (OverflowError, ZeroDivisionError):
            return 1e6  # Large finite value for numerical stability

    def matter_state_classification(self, psi: float, r: float, energy_density: float) -> str:
        """
        Classify matter state based on the state variable ψ, radius, and energy density.
        This is the corrected version that integrates with the physics simulation.
        
        Args:
            psi: State variable from the simulation
            r: Radial coordinate (m)
            energy_density: Energy density (J/m³)
        
        Returns:
            Matter state classification string
        """
        # Calculate effective temperature from energy density (rough approximation)
        # Using Stefan-Boltzmann relation: T ~ (energy_density)^(1/4)
        temp_k = (energy_density / (7.56e-16))**(1/4) if energy_density > 0 else 0
        
        # Calculate pressure from gravitational compression
        if r > 0:
            pressure_pa = self.G * self.M * energy_density / (r * self.c**2)
        else:
            pressure_pa = np.inf
        
        # State classification based on ψ value and physical conditions
        if psi < 0.1:
            return "Normal Matter"
        elif psi < 1.0:
            if temp_k < 273.15:
                return "Compressed Solid"
            elif temp_k < 373.15:
                return "Dense Liquid"
            else:
                return "Dense Gas"
        elif psi < 2.0:
            if temp_k >= 10000:
                return "Plasma State"
            else:
                return "Exotic Dense Matter"
        elif psi < 5.0:
            return "Degenerate Matter"
        elif psi < 10.0:
            if temp_k >= 1e12:
                return "Quark-Gluon Plasma"
            else:
                return "Ultra-Dense Plasma"
        elif psi < 20.0:
            return "Strange Matter"
        elif psi < 50.0:
            return "Planck-Scale Matter"
        elif psi < 100.0:
            return "Trans-Planckian State"
        elif psi < np.pi * 100:
            return "Infinite Density Approach"
        else:
            # At the singularity or beyond known physics
            if temp_k >= 1.416785e32:  # Planck temperature
                return "Planck Singularity State"
            else:
                return "Infinite State Manifold"

    def stress_energy_tensor(self, T0: float, psi: float) -> float:
        """
        Enhanced stress-energy tensor with state dependence
        
        Args:
            T0: Base stress-energy component
            psi: State variable
            
        Returns:
            Enhanced stress-energy component
        """
        try:
            enhancement = (1 + psi**2 / (4 * np.pi**2)) * np.exp(min(psi/np.pi, 50))
            return T0 * enhancement
        except OverflowError:
            return T0 * 1e50

    def infinite_state_amplification(self, psi: float) -> float:
        """
        Infinite state amplification function Ψ(ψ)
        
        Args:
            psi: State variable
            
        Returns:
            Amplification factor
        """
        if psi < np.pi:
            return 1.0
        elif psi < 2*np.pi:
            try:
                return 1.0 / np.tanh(max(psi - np.pi, 1e-10))
            except:
                return 100.0
        else:
            # Factorial approximation for large values
            try:
                return sp.gamma(psi - 2*np.pi + 1)
            except:
                return 1e20

    def tidal_force(self, r: float, mass_test: float, delta_r: float, psi: float) -> float:
        """
        State-dependent tidal force calculation
        
        Args:
            r: Radial coordinate (m)
            mass_test: Test mass (kg)
            delta_r: Spatial separation (m)
            psi: State variable
            
        Returns:
            Tidal force (N)
        """
        if r <= 0:
            return np.inf
        
        base_tidal = 2 * self.G * self.M * mass_test * delta_r / r**3
        state_factor = 1 + np.sin(psi)**2
        amplification = self.infinite_state_amplification(psi)
        
        return base_tidal * state_factor * min(amplification, 1e20)

    def proper_time_dilation(self, r: float) -> float:
        """Calculate proper time dilation factor"""
        metric_coeff = self.schwarzschild_metric(r)
        if metric_coeff <= 0:
            return 0.0
        return np.sqrt(metric_coeff)

    def energy_density(self, r: float, rho0: float = 1e3) -> float:
        """
        Calculate energy density as function of radius with improved stability
        
        Args:
            r: Radial coordinate (m)
            rho0: Reference density (kg/m³)
            
        Returns:
            Energy density (J/m³)
        """
        if r <= 0:
            return np.inf
        
        # Energy density grows as matter is compressed - with numerical safety
        r_safe = max(r, 1e-20)
        compression_factor = min((self.rs / r_safe)**3, 1e20)  # Cap extreme values
        return rho0 * self.c**2 * compression_factor

    def state_evolution_derivatives(self, state: List[float], tau: float, r_func) -> List[float]:
        """
        State evolution equation: dψ/dτ = β∇²ψ + γT_μν·ψ³ + δ·ψ^(ψ/π)
        
        Args:
            state: [ψ, dψ/dτ]
            tau: Proper time
            r_func: Function giving r(τ)
            
        Returns:
            Derivatives [dψ/dτ, d²ψ/dτ²]
        """
        psi, dpsi_dtau = state
        r = r_func(tau)
        
        if r <= 0:
            return [1e6, 1e6]
        
        # Energy density and stress-energy
        energy_dens = self.energy_density(r)
        T_component = energy_dens / self.c**2
        
        # Laplacian approximation (simplified 1D)
        laplacian_psi = -psi / r**2  # Simplified radial Laplacian
        
        # State evolution terms
        diffusion_term = self.beta * laplacian_psi
        stress_term = self.gamma * T_component * psi**3
        
        try:
            nonlinear_term = self.delta * psi**(min(psi/np.pi, 10))
        except:
            nonlinear_term = self.delta * 1e10
        
        d2psi_dtau2 = diffusion_term + stress_term + nonlinear_term
        
        return [dpsi_dtau, d2psi_dtau2]

    def simulate_matter_approach(self, r_initial: float, r_final: float, 
                                num_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Main simulation algorithm for matter approaching singularity
        
        Args:
            r_initial: Starting radius (m)
            r_final: Final radius (m) 
            num_steps: Number of simulation steps
            
        Returns:
            Tuple of (radii, state_variables, matter_states)
        """
        # Radial coordinates with improved numerical safety
        r_values = np.logspace(np.log10(r_initial), np.log10(max(r_final, 1e-20)), num_steps)
        r_values = np.clip(r_values[::-1], 1e-20, None)  # Reverse and clamp
        
        # Initialize arrays
        psi_values = np.zeros(num_steps)
        matter_states = []
        
        # Time coordinate (proper time)
        tau_values = np.linspace(0, 1, num_steps)
        
        for i, r in enumerate(r_values):
            # Calculate current energy density
            energy_dens = self.energy_density(r)
            
            # Calculate state variable
            psi = self.state_variable(r, energy_dens)
            psi_values[i] = psi
            
            # Classify matter state using the corrected method
            state = self.matter_state_classification(psi, r, energy_dens)
            matter_states.append(state)
        
        return r_values, psi_values, matter_states

    def calculate_physical_quantities(self, r_values: np.ndarray, psi_values: np.ndarray) -> dict:
        """
        Calculate all relevant physical quantities
        
        Args:
            r_values: Array of radial coordinates
            psi_values: Array of state variables
            
        Returns:
            Dictionary of physical quantities
        """
        quantities = {
            'tidal_forces': [],
            'energy_densities': [],
            'time_dilations': [],
            'curvature_scalars': [],
            'stress_energy_components': []
        }
        
        test_mass = 1.0  # kg
        delta_r = 0.1    # m
        
        for r, psi in zip(r_values, psi_values):
            if r <= 0:
                break
                
            # Tidal force
            F_tidal = self.tidal_force(r, test_mass, delta_r, psi)
            quantities['tidal_forces'].append(F_tidal)
            
            # Energy density
            energy_dens = self.energy_density(r)
            quantities['energy_densities'].append(energy_dens)
            
            # Time dilation
            time_dilation = self.proper_time_dilation(r)
            quantities['time_dilations'].append(time_dilation)
            
            # Curvature scalar (simplified)
            if r > 0:
                curvature = 48 * self.G**2 * self.M**2 / (r**6 * self.c**4)
                quantities['curvature_scalars'].append(curvature)
            else:
                quantities['curvature_scalars'].append(np.inf)
            
            # Enhanced stress-energy
            base_T = energy_dens / self.c**2
            enhanced_T = self.stress_energy_tensor(base_T, psi)
            quantities['stress_energy_components'].append(enhanced_T)
        
        return quantities

    def plot_results(self, r_values, psi_values, matter_states, quantities=None):
        """
        Visualize simulation results
        
        Args:
            r_values: Radial coordinates
            psi_values: State variables
            matter_states: Matter state classifications
            quantities: Optional physical quantities dict
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # State variable evolution
        axes[0,0].plot(r_values, psi_values, 'b-', linewidth=2, label='ψ (State Variable)')
        axes[0,0].set_xscale('log')
        axes[0,0].set_xlabel('Radius (m)')
        axes[0,0].set_ylabel('ψ')
        axes[0,0].set_title('Matter State Variable Approaching Singularity')
        axes[0,0].grid(True)
        axes[0,0].legend()
        
        # State transitions
        unique_states = list(set(matter_states))
        state_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_states)))
        for i, state in enumerate(unique_states):
            state_indices = [j for j, s in enumerate(matter_states) if s == state]
            if state_indices:
                axes[0,1].scatter([r_values[j] for j in state_indices], 
                                [i] * len(state_indices), 
                                c=[state_colors[i]], label=state, alpha=0.7)
        axes[0,1].set_xscale('log')
        axes[0,1].set_xlabel('Radius (m)')
        axes[0,1].set_ylabel('Matter State')
        axes[0,1].set_title('Matter State Transitions')
        axes[0,1].grid(True)
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if quantities:
            # Energy density
            axes[1,0].plot(r_values[:len(quantities['energy_densities'])], 
                        quantities['energy_densities'], 'r-', linewidth=2)
            axes[1,0].set_xscale('log')
            axes[1,0].set_yscale('log')
            axes[1,0].set_xlabel('Radius (m)')
            axes[1,0].set_ylabel('Energy Density (J/m³)')
            axes[1,0].set_title('Energy Density Growth')
            axes[1,0].grid(True)
            
            # Time dilation
            axes[1,1].plot(r_values[:len(quantities['time_dilations'])], 
                        quantities['time_dilations'], 'g-', linewidth=2)
            axes[1,1].set_xscale('log')
            axes[1,1].set_xlabel('Radius (m)')
            axes[1,1].set_ylabel('Time Dilation Factor')
            axes[1,1].set_title('Proper Time Dilation')
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()

    def run_full_simulation(self, r_initial: float = None, r_final: float = None, 
                        plot_results: bool = True) -> dict:
        """
        Run complete singularity approach simulation
        
        Args:
            r_initial: Starting radius (default: 10 * Schwarzschild radius)
            r_final: Final radius (default: Planck length)
            plot_results: Whether to generate plots
            
        Returns:
            Complete simulation results
        """
        if r_initial is None:
            r_initial = 10 * self.rs
        if r_final is None:
            r_final = 1.616e-35  # Planck length
            
        print(f"🔥 Initializing Black Hole State Evolution Simulation")
        print(f"   Physics-Heavy Neural Net for Collapsing Reality")
        print("="*60)
        print(f"Black hole mass: {self.M:.2e} kg")
        print(f"Schwarzschild radius: {self.rs:.2e} m")
        print(f"Simulation range: {r_initial:.2e} m to {r_final:.2e} m")
        print("="*60)
        
        # Main simulation
        r_values, psi_values, matter_states = self.simulate_matter_approach(
            r_initial, r_final, num_steps=1000
        )
        
        # Calculate physical quantities
        quantities = self.calculate_physical_quantities(r_values, psi_values)
        
        # Print key results
        print("\n🧠 Key Simulation Results:")
        print("-" * 40)
        
        for i in [0, len(r_values)//4, len(r_values)//2, 3*len(r_values)//4, -1]:
            if i >= len(r_values):
                continue
            print(f"r = {r_values[i]:.2e} m:")
            print(f"  Matter State: {matter_states[i]}")
            print(f"  ψ = {psi_values[i]:.3f}")
            if i < len(quantities['tidal_forces']):
                print(f"  Tidal Force: {quantities['tidal_forces'][i]:.2e} N")
                print(f"  Energy Density: {quantities['energy_densities'][i]:.2e} J/m³")
                print(f"  Time Dilation: {quantities['time_dilations'][i]:.6f}")
            print()
        
        # Generate visualizations
        if plot_results:
            self.plot_results(r_values, psi_values, matter_states, quantities)
        
        # Summary statistics
        unique_states = len(set(matter_states))
        infinite_states = sum(1 for s in matter_states if 'Infinite' in s)
        max_psi = max(psi_values)
        
        print("✅ Simulation Analysis Complete!")
        print(f"   Total unique states: {unique_states}")
        print(f"   Maximum ψ value: {max_psi:.2f}")
        print(f"   Infinite states reached: {infinite_states}")
        print(f"   Reality collapse depth: {r_initial/r_values[-1]:.2e}x compression")
        
        return {
            'r_values': r_values,
            'psi_values': psi_values,
            'matter_states': matter_states,
            'physical_quantities': quantities,
            'schwarzschild_radius': self.rs,
            'simulation_stats': {
                'unique_states': unique_states,
                'infinite_states': infinite_states,
                'max_psi': max_psi,
                'compression_ratio': r_initial/r_values[-1]
            }
        }

# Create simulator for a stellar-mass black hole
if __name__ == "__main__":
    simulator = SingularityMatterStates(mass_bh=3e30)  # ~1.5 solar masses

    # Run full simulation with visualization
    results = simulator.run_full_simulation(plot_results=True)

    print("\n🚀 Ready for extensions:")
    print("   • TensorFlow hybrid symbolic-numeric modeling")
    print("   • WebGL-based visual demo")
    print("   • Quantum optimization for ψ collapse prediction")
    print("   • Reverse-horizon escape modeling")
    print("   • Singularity paradox feedback effects")