"""
AMO Physics Parameter Calculator

Complete calculation engine for Atomic, Molecular, and Optical physics parameters.
Computes coupling strengths, Q factors, cooperativities, and other key physics quantities
from system descriptions and physical properties.
"""

import numpy as np
import math
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class SystemType(Enum):
    """Types of quantum optical systems"""
    CAVITY_QED = "cavity_qed"
    OPTOMECHANICS = "optomechanics"
    SQUEEZED_LIGHT = "squeezed_light"
    PHOTON_BLOCKADE = "photon_blockade"
    QUANTUM_MEMORY = "quantum_memory"
    HYBRID_SYSTEM = "hybrid_system"

@dataclass
class PhysicsConstants:
    """Physical constants for calculations"""
    h_bar: float = 1.054571817e-34  # J⋅s
    k_b: float = 1.380649e-23       # J/K
    c: float = 299792458            # m/s
    e: float = 1.602176634e-19      # C
    epsilon_0: float = 8.8541878128e-12  # F/m
    mu_0: float = 1.25663706212e-6   # H/m
    m_e: float = 9.1093837015e-31    # kg
    a_0: float = 5.29177210903e-11   # m (Bohr radius)

@dataclass
class CavityParameters:
    """Parameters for optical cavity systems"""
    length: Optional[float] = None              # m
    finesse: Optional[float] = None            # dimensionless
    mirror_reflectivity: Optional[float] = None # dimensionless
    mode_waist: Optional[float] = None         # m
    wavelength: Optional[float] = None         # m
    refractive_index: Optional[float] = 1.0    # dimensionless
    free_spectral_range: Optional[float] = None # Hz
    
    def __post_init__(self):
        if self.wavelength and self.length and not self.free_spectral_range:
            self.free_spectral_range = PhysicsConstants.c / (2 * self.length * self.refractive_index)

@dataclass
class AtomicParameters:
    """Parameters for atomic systems"""
    transition_frequency: Optional[float] = None    # Hz
    linewidth: Optional[float] = None              # Hz (FWHM)
    dipole_moment: Optional[float] = None          # C⋅m
    mass: Optional[float] = None                   # kg
    quantum_numbers: Optional[Dict[str, float]] = None
    excited_state_lifetime: Optional[float] = None # s
    
    def __post_init__(self):
        if self.excited_state_lifetime and not self.linewidth:
            self.linewidth = 1 / (2 * math.pi * self.excited_state_lifetime)
        elif self.linewidth and not self.excited_state_lifetime:
            self.excited_state_lifetime = 1 / (2 * math.pi * self.linewidth)

@dataclass
class MechanicalParameters:
    """Parameters for mechanical systems"""
    frequency: Optional[float] = None           # Hz
    quality_factor: Optional[float] = None     # dimensionless
    mass: Optional[float] = None               # kg
    effective_mass: Optional[float] = None     # kg
    spring_constant: Optional[float] = None    # N/m
    damping_rate: Optional[float] = None       # Hz
    
    def __post_init__(self):
        if self.frequency and self.quality_factor and not self.damping_rate:
            self.damping_rate = 2 * math.pi * self.frequency / self.quality_factor
        elif self.frequency and self.damping_rate and not self.quality_factor:
            self.quality_factor = 2 * math.pi * self.frequency / self.damping_rate

@dataclass
class CalculationResult:
    """Result of physics parameter calculation"""
    parameter_name: str
    value: float
    units: str
    calculation_method: str
    input_parameters: Dict[str, Any]
    uncertainty: Optional[float] = None
    notes: List[str] = field(default_factory=list)

class AMOPhysicsCalculator:
    """
    Comprehensive calculator for AMO physics parameters.
    Computes coupling strengths, Q factors, cooperativities, and other key quantities.
    """
    
    def __init__(self):
        self.constants = PhysicsConstants()
        self.logger = logging.getLogger('AMOPhysicsCalculator')
        
    def calculate_cavity_qed_parameters(self, 
                                      cavity_params: CavityParameters,
                                      atomic_params: AtomicParameters) -> Dict[str, CalculationResult]:
        """Calculate all relevant cavity QED parameters"""
        results = {}
        
        # Vacuum Rabi coupling strength
        if cavity_params.mode_waist and atomic_params.dipole_moment and cavity_params.wavelength:
            coupling = self.calculate_vacuum_rabi_coupling(cavity_params, atomic_params)
            results['vacuum_rabi_coupling'] = coupling
            
        # Cavity decay rate
        if cavity_params.finesse and cavity_params.free_spectral_range:
            decay_rate = self.calculate_cavity_decay_rate(cavity_params)
            results['cavity_decay_rate'] = decay_rate
            
        # Cooperativity
        if 'vacuum_rabi_coupling' in results and 'cavity_decay_rate' in results and atomic_params.linewidth:
            cooperativity = self.calculate_cooperativity(
                results['vacuum_rabi_coupling'].value,
                results['cavity_decay_rate'].value,
                atomic_params.linewidth
            )
            results['cooperativity'] = cooperativity
            
        # Purcell factor
        if cavity_params.finesse and cavity_params.mode_waist and cavity_params.wavelength:
            purcell = self.calculate_purcell_factor(cavity_params)
            results['purcell_factor'] = purcell
            
        return results
    
    def calculate_vacuum_rabi_coupling(self, 
                                     cavity_params: CavityParameters,
                                     atomic_params: AtomicParameters) -> CalculationResult:
        """
        Calculate vacuum Rabi coupling strength g₀
        g₀ = d√(ω/(2ε₀ħV)) where V is mode volume
        """
        try:
            # Calculate mode volume V = π w₀² L / 2 for Fabry-Perot cavity
            mode_volume = math.pi * (cavity_params.mode_waist**2) * cavity_params.length / 2
            
            # Angular frequency
            omega = 2 * math.pi * self.constants.c / cavity_params.wavelength
            
            # Coupling strength formula
            denominator = 2 * self.constants.epsilon_0 * self.constants.h_bar * mode_volume
            coupling = atomic_params.dipole_moment * math.sqrt(omega / denominator)
            
            return CalculationResult(
                parameter_name="vacuum_rabi_coupling",
                value=coupling,
                units="Hz",
                calculation_method="g₀ = d√(ω/(2ε₀ħV))",
                input_parameters={
                    "dipole_moment": atomic_params.dipole_moment,
                    "mode_waist": cavity_params.mode_waist,
                    "cavity_length": cavity_params.length,
                    "wavelength": cavity_params.wavelength
                },
                notes=["Mode volume calculated for Fabry-Perot cavity"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate vacuum Rabi coupling: {e}")
            return CalculationResult(
                parameter_name="vacuum_rabi_coupling",
                value=0.0,
                units="Hz",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_cavity_decay_rate(self, cavity_params: CavityParameters) -> CalculationResult:
        """
        Calculate cavity decay rate κ = FSR / F
        where FSR is free spectral range and F is finesse
        """
        try:
            decay_rate = 2 * math.pi * cavity_params.free_spectral_range / cavity_params.finesse
            
            return CalculationResult(
                parameter_name="cavity_decay_rate",
                value=decay_rate,
                units="rad/s",
                calculation_method="κ = 2π⋅FSR/F",
                input_parameters={
                    "free_spectral_range": cavity_params.free_spectral_range,
                    "finesse": cavity_params.finesse
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cavity decay rate: {e}")
            return CalculationResult(
                parameter_name="cavity_decay_rate",
                value=0.0,
                units="rad/s",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_cooperativity(self, coupling_strength: float, 
                              cavity_decay_rate: float, 
                              atomic_linewidth: float) -> CalculationResult:
        """
        Calculate cooperativity C = 4g²/(κγ)
        """
        try:
            # Convert linewidth to rad/s if needed
            gamma = 2 * math.pi * atomic_linewidth if atomic_linewidth < 1e6 else atomic_linewidth
            
            cooperativity = 4 * (coupling_strength**2) / (cavity_decay_rate * gamma)
            
            return CalculationResult(
                parameter_name="cooperativity",
                value=cooperativity,
                units="dimensionless",
                calculation_method="C = 4g²/(κγ)",
                input_parameters={
                    "coupling_strength": coupling_strength,
                    "cavity_decay_rate": cavity_decay_rate,
                    "atomic_linewidth": atomic_linewidth
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cooperativity: {e}")
            return CalculationResult(
                parameter_name="cooperativity",
                value=0.0,
                units="dimensionless",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_purcell_factor(self, cavity_params: CavityParameters) -> CalculationResult:
        """
        Calculate Purcell factor F_P = 3λ³Q/(4π²V)
        """
        try:
            # Quality factor from finesse
            Q = cavity_params.finesse * math.pi / 2
            
            # Mode volume
            mode_volume = math.pi * (cavity_params.mode_waist**2) * cavity_params.length / 2
            
            # Purcell factor
            purcell = (3 * (cavity_params.wavelength**3) * Q) / (4 * (math.pi**2) * mode_volume)
            
            return CalculationResult(
                parameter_name="purcell_factor",
                value=purcell,
                units="dimensionless",
                calculation_method="F_P = 3λ³Q/(4π²V)",
                input_parameters={
                    "wavelength": cavity_params.wavelength,
                    "finesse": cavity_params.finesse,
                    "mode_waist": cavity_params.mode_waist,
                    "cavity_length": cavity_params.length
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate Purcell factor: {e}")
            return CalculationResult(
                parameter_name="purcell_factor",
                value=0.0,
                units="dimensionless",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_optomechanical_parameters(self, 
                                          cavity_params: CavityParameters,
                                          mechanical_params: MechanicalParameters,
                                          optical_power: float = None) -> Dict[str, CalculationResult]:
        """Calculate optomechanical system parameters"""
        results = {}
        
        # Single-photon optomechanical coupling
        if cavity_params.length and mechanical_params.frequency:
            g0 = self.calculate_single_photon_optomechanical_coupling(cavity_params, mechanical_params)
            results['single_photon_coupling'] = g0
            
        # Enhanced coupling with optical power
        if 'single_photon_coupling' in results and optical_power:
            enhanced_coupling = self.calculate_enhanced_optomechanical_coupling(
                results['single_photon_coupling'].value, optical_power, cavity_params
            )
            results['enhanced_coupling'] = enhanced_coupling
            
        # Optomechanical cooperativity
        if ('enhanced_coupling' in results and 
            cavity_params.finesse and mechanical_params.damping_rate):
            
            cavity_decay = 2 * math.pi * cavity_params.free_spectral_range / cavity_params.finesse
            cooperativity = self.calculate_optomechanical_cooperativity(
                results['enhanced_coupling'].value,
                cavity_decay,
                mechanical_params.damping_rate
            )
            results['optomechanical_cooperativity'] = cooperativity
            
        return results
    
    def calculate_single_photon_optomechanical_coupling(self, 
                                                       cavity_params: CavityParameters,
                                                       mechanical_params: MechanicalParameters) -> CalculationResult:
        """
        Calculate single-photon optomechanical coupling g₀ = ωc/L × xzpf
        where xzpf is zero-point fluctuation amplitude
        """
        try:
            # Cavity frequency
            omega_c = 2 * math.pi * self.constants.c / cavity_params.wavelength
            
            # Zero-point fluctuation amplitude
            if mechanical_params.effective_mass:
                xzpf = math.sqrt(self.constants.h_bar / (2 * mechanical_params.effective_mass * 
                                                       2 * math.pi * mechanical_params.frequency))
            else:
                # Estimate for typical nano-mechanical resonator
                xzpf = 1e-15  # m, typical value
                
            # Single-photon coupling
            g0 = (omega_c / cavity_params.length) * xzpf
            
            return CalculationResult(
                parameter_name="single_photon_optomechanical_coupling",
                value=g0,
                units="Hz",
                calculation_method="g₀ = (ωc/L) × xzpf",
                input_parameters={
                    "cavity_length": cavity_params.length,
                    "wavelength": cavity_params.wavelength,
                    "mechanical_frequency": mechanical_params.frequency,
                    "effective_mass": mechanical_params.effective_mass,
                    "xzpf": xzpf
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate single-photon optomechanical coupling: {e}")
            return CalculationResult(
                parameter_name="single_photon_optomechanical_coupling",
                value=0.0,
                units="Hz",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_enhanced_optomechanical_coupling(self, 
                                                 g0: float, 
                                                 optical_power: float,
                                                 cavity_params: CavityParameters) -> CalculationResult:
        """
        Calculate enhanced coupling G = g₀√n̄ where n̄ is intracavity photon number
        """
        try:
            # Photon energy
            photon_energy = self.constants.h_bar * 2 * math.pi * self.constants.c / cavity_params.wavelength
            
            # Intracavity photon number (assuming cavity finesse enhancement)
            cavity_enhancement = cavity_params.finesse / math.pi
            intracavity_photons = optical_power * cavity_enhancement / photon_energy
            
            # Enhanced coupling
            enhanced_coupling = g0 * math.sqrt(intracavity_photons)
            
            return CalculationResult(
                parameter_name="enhanced_optomechanical_coupling",
                value=enhanced_coupling,
                units="Hz",
                calculation_method="G = g₀√n̄",
                input_parameters={
                    "g0": g0,
                    "optical_power": optical_power,
                    "intracavity_photons": intracavity_photons,
                    "cavity_enhancement": cavity_enhancement
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced optomechanical coupling: {e}")
            return CalculationResult(
                parameter_name="enhanced_optomechanical_coupling",
                value=0.0,
                units="Hz",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_optomechanical_cooperativity(self, 
                                             enhanced_coupling: float,
                                             cavity_decay_rate: float,
                                             mechanical_damping_rate: float) -> CalculationResult:
        """
        Calculate optomechanical cooperativity C = 4G²/(κγₘ)
        """
        try:
            cooperativity = 4 * (enhanced_coupling**2) / (cavity_decay_rate * mechanical_damping_rate)
            
            return CalculationResult(
                parameter_name="optomechanical_cooperativity",
                value=cooperativity,
                units="dimensionless",
                calculation_method="C = 4G²/(κγₘ)",
                input_parameters={
                    "enhanced_coupling": enhanced_coupling,
                    "cavity_decay_rate": cavity_decay_rate,
                    "mechanical_damping_rate": mechanical_damping_rate
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate optomechanical cooperativity: {e}")
            return CalculationResult(
                parameter_name="optomechanical_cooperativity",
                value=0.0,
                units="dimensionless",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_squeezing_parameters(self, 
                                     chi: float,           # Nonlinear coefficient
                                     kappa: float,         # Cavity decay rate
                                     pump_power: float,    # Pump power
                                     cavity_params: CavityParameters) -> Dict[str, CalculationResult]:
        """Calculate squeezed light generation parameters"""
        results = {}
        
        # Parametric threshold
        threshold = self.calculate_parametric_threshold(chi, kappa)
        results['parametric_threshold'] = threshold
        
        # Squeezing level
        if pump_power < threshold.value:
            squeezing = self.calculate_squeezing_level(pump_power, threshold.value)
            results['squeezing_level'] = squeezing
            
        # Squeezing bandwidth
        bandwidth = self.calculate_squeezing_bandwidth(kappa, pump_power, threshold.value)
        results['squeezing_bandwidth'] = bandwidth
        
        return results
    
    def calculate_parametric_threshold(self, chi: float, kappa: float) -> CalculationResult:
        """Calculate parametric oscillation threshold"""
        try:
            threshold = (kappa**2) / (4 * chi**2)
            
            return CalculationResult(
                parameter_name="parametric_threshold",
                value=threshold,
                units="W",
                calculation_method="P_th = κ²/(4χ²)",
                input_parameters={
                    "chi": chi,
                    "kappa": kappa
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate parametric threshold: {e}")
            return CalculationResult(
                parameter_name="parametric_threshold",
                value=0.0,
                units="W",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_squeezing_level(self, pump_power: float, threshold_power: float) -> CalculationResult:
        """Calculate squeezing level in dB"""
        try:
            pump_ratio = pump_power / threshold_power
            if pump_ratio >= 1:
                squeezing_db = 0.0
                notes = ["Above threshold - no squeezing"]
            else:
                # Squeezing parameter r = asinh(√(P_th/P - 1))
                r = math.asinh(math.sqrt(max(0, 1/pump_ratio - 1)))
                # Squeezing in dB: -10log₁₀(e^(-2r))
                squeezing_db = -10 * math.log10(math.exp(-2 * r))
                notes = ["Below threshold squeezing"]
            
            return CalculationResult(
                parameter_name="squeezing_level",
                value=squeezing_db,
                units="dB",
                calculation_method="S = -10log₁₀(e^(-2r))",
                input_parameters={
                    "pump_power": pump_power,
                    "threshold_power": threshold_power,
                    "pump_ratio": pump_ratio
                },
                notes=notes
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate squeezing level: {e}")
            return CalculationResult(
                parameter_name="squeezing_level",
                value=0.0,
                units="dB",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def calculate_squeezing_bandwidth(self, kappa: float, pump_power: float, threshold_power: float) -> CalculationResult:
        """Calculate squeezing bandwidth"""
        try:
            pump_ratio = pump_power / threshold_power
            if pump_ratio >= 1:
                bandwidth = 0.0
                notes = ["Above threshold - no squeezing bandwidth"]
            else:
                bandwidth = kappa * math.sqrt(1 - pump_ratio) / (2 * math.pi)
                notes = ["Below threshold bandwidth"]
            
            return CalculationResult(
                parameter_name="squeezing_bandwidth",
                value=bandwidth,
                units="Hz",
                calculation_method="Δω = κ√(1-P/P_th)/(2π)",
                input_parameters={
                    "kappa": kappa,
                    "pump_power": pump_power,
                    "threshold_power": threshold_power
                },
                notes=notes
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate squeezing bandwidth: {e}")
            return CalculationResult(
                parameter_name="squeezing_bandwidth",
                value=0.0,
                units="Hz",
                calculation_method="calculation_failed",
                input_parameters={},
                notes=[f"Calculation failed: {e}"]
            )
    
    def estimate_dipole_moment(self, transition_wavelength: float, oscillator_strength: float = 1.0) -> float:
        """Estimate atomic dipole moment from transition wavelength"""
        # Classical electron radius times transition wavelength
        r_e = self.constants.e**2 / (4 * math.pi * self.constants.epsilon_0 * self.constants.m_e * self.constants.c**2)
        dipole_estimate = self.constants.e * math.sqrt(oscillator_strength * r_e * transition_wavelength)
        return dipole_estimate
    
    def get_system_type_from_description(self, description: str) -> SystemType:
        """Identify system type from description text"""
        description_lower = description.lower()
        
        if any(term in description_lower for term in ['cavity qed', 'atom cavity', 'jaynes-cummings']):
            return SystemType.CAVITY_QED
        elif any(term in description_lower for term in ['optomechanical', 'mechanical', 'phonon']):
            return SystemType.OPTOMECHANICS
        elif any(term in description_lower for term in ['squeezed', 'squeezing', 'parametric']):
            return SystemType.SQUEEZED_LIGHT
        elif any(term in description_lower for term in ['blockade', 'antibunching', 'kerr']):
            return SystemType.PHOTON_BLOCKADE
        elif any(term in description_lower for term in ['memory', 'storage', 'retrieval']):
            return SystemType.QUANTUM_MEMORY
        elif any(term in description_lower for term in ['hybrid', 'coupled systems']):
            return SystemType.HYBRID_SYSTEM
        else:
            return SystemType.CAVITY_QED  # Default
    
    def calculate_all_relevant_parameters(self, 
                                        system_description: str,
                                        extracted_params: Dict[str, Any]) -> Dict[str, CalculationResult]:
        """
        Calculate all relevant parameters based on system type and available data
        """
        system_type = self.get_system_type_from_description(system_description)
        results = {}
        
        try:
            if system_type == SystemType.CAVITY_QED:
                # Build cavity and atomic parameters from extracted data
                cavity_params = self._build_cavity_parameters(extracted_params)
                atomic_params = self._build_atomic_parameters(extracted_params)
                
                if cavity_params and atomic_params:
                    cavity_results = self.calculate_cavity_qed_parameters(cavity_params, atomic_params)
                    results.update(cavity_results)
                    
            elif system_type == SystemType.OPTOMECHANICS:
                cavity_params = self._build_cavity_parameters(extracted_params)
                mechanical_params = self._build_mechanical_parameters(extracted_params)
                optical_power = extracted_params.get('optical_power', extracted_params.get('pump_power'))
                
                if cavity_params and mechanical_params:
                    opto_results = self.calculate_optomechanical_parameters(
                        cavity_params, mechanical_params, optical_power
                    )
                    results.update(opto_results)
                    
            elif system_type == SystemType.SQUEEZED_LIGHT:
                chi = extracted_params.get('nonlinear_coefficient', extracted_params.get('chi'))
                kappa = extracted_params.get('cavity_decay_rate', extracted_params.get('kappa'))
                pump_power = extracted_params.get('pump_power')
                cavity_params = self._build_cavity_parameters(extracted_params)
                
                if chi and kappa and pump_power and cavity_params:
                    squeezing_results = self.calculate_squeezing_parameters(
                        chi, kappa, pump_power, cavity_params
                    )
                    results.update(squeezing_results)
                    
        except Exception as e:
            self.logger.error(f"Failed to calculate parameters for {system_type}: {e}")
            
        return results
    
    def _build_cavity_parameters(self, params: Dict[str, Any]) -> Optional[CavityParameters]:
        """Build CavityParameters from extracted parameter dictionary"""
        try:
            return CavityParameters(
                length=params.get('cavity_length'),
                finesse=params.get('finesse', params.get('cavity_finesse')),
                mirror_reflectivity=params.get('mirror_reflectivity'),
                mode_waist=params.get('mode_waist'),
                wavelength=params.get('wavelength', params.get('transition_wavelength')),
                refractive_index=params.get('refractive_index', 1.0),
                free_spectral_range=params.get('free_spectral_range')
            )
        except Exception as e:
            self.logger.warning(f"Failed to build cavity parameters: {e}")
            return None
    
    def _build_atomic_parameters(self, params: Dict[str, Any]) -> Optional[AtomicParameters]:
        """Build AtomicParameters from extracted parameter dictionary"""
        try:
            return AtomicParameters(
                transition_frequency=params.get('transition_frequency', params.get('atomic_frequency')),
                linewidth=params.get('linewidth', params.get('atomic_linewidth')),
                dipole_moment=params.get('dipole_moment'),
                mass=params.get('atomic_mass'),
                excited_state_lifetime=params.get('excited_state_lifetime')
            )
        except Exception as e:
            self.logger.warning(f"Failed to build atomic parameters: {e}")
            return None
    
    def _build_mechanical_parameters(self, params: Dict[str, Any]) -> Optional[MechanicalParameters]:
        """Build MechanicalParameters from extracted parameter dictionary"""
        try:
            return MechanicalParameters(
                frequency=params.get('mechanical_frequency'),
                quality_factor=params.get('mechanical_quality_factor', params.get('mechanical_q')),
                mass=params.get('mechanical_mass'),
                effective_mass=params.get('effective_mass'),
                spring_constant=params.get('spring_constant'),
                damping_rate=params.get('mechanical_damping_rate')
            )
        except Exception as e:
            self.logger.warning(f"Failed to build mechanical parameters: {e}")
            return None