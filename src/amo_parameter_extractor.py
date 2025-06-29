"""
AMO Parameter Extractor

Advanced parameter extraction system for parsing physics values from research solutions.
Extracts coupling strengths, frequencies, decay rates, and other quantum optical parameters
from natural language descriptions and mathematical expressions.
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExtractedParameter:
    """Represents an extracted physics parameter"""
    name: str
    value: float
    units: str
    confidence: float  # 0-1 confidence score
    extraction_method: str
    context: str
    alternative_values: List[float] = field(default_factory=list)

@dataclass
class ExtractionResult:
    """Complete result of parameter extraction"""
    parameters: Dict[str, ExtractedParameter]
    system_type: str
    confidence_score: float
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

class AMOParameterExtractor:
    """
    Advanced parameter extraction engine for AMO physics.
    Parses natural language and mathematical expressions to extract quantitative parameters.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('AMOParameterExtractor')
        self._setup_extraction_patterns()
        self._setup_unit_conversions()
        self._setup_physics_knowledge()
    
    def _setup_extraction_patterns(self):
        """Setup regex patterns for parameter extraction"""
        # Scientific notation pattern
        self.sci_notation = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
        
        # Unit patterns
        self.frequency_units = r'(?:Hz|hz|KHz|khz|MHz|mhz|GHz|ghz|THz|thz|rad/s|s\^-1)'
        self.length_units = r'(?:m|mm|μm|um|nm|pm|cm|km)'
        self.power_units = r'(?:W|mW|μW|uW|nW|pW|kW|MW)'
        self.time_units = r'(?:s|ms|μs|us|ns|ps|fs|as)'
        
        # Parameter patterns with various naming conventions
        self.parameter_patterns = {
            'coupling_strength': [
                rf'(?:coupling|g|g0|g_0)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:rabi|vacuum rabi)\s+(?:frequency|coupling)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:atom[- ]cavity|cavity[- ]atom)\s+coupling\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?'
            ],
            'cavity_decay_rate': [
                rf'(?:κ|kappa|cavity decay|decay rate)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:cavity|optical)\s+(?:linewidth|loss|damping)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?'
            ],
            'atomic_linewidth': [
                rf'(?:γ|gamma|atomic|spontaneous)\s+(?:decay|linewidth|damping)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:natural|excited state)\s+linewidth\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?'
            ],
            'finesse': [
                rf'(?:finesse|F)\s*[:=]?\s*({self.sci_notation})',
                rf'(?:cavity|optical)\s+finesse\s*[:=]?\s*({self.sci_notation})'
            ],
            'quality_factor': [
                rf'(?:Q|quality factor|Q[ -]factor)\s*[:=]?\s*({self.sci_notation})',
                rf'(?:mechanical|cavity)\s+(?:Q|quality)\s*[:=]?\s*({self.sci_notation})'
            ],
            'cooperativity': [
                rf'(?:cooperativity|C|cooperative parameter)\s*[:=]?\s*({self.sci_notation})',
                rf'(?:strong|weak)\s+coupling\s+(?:parameter|cooperativity)\s*[:=]?\s*({self.sci_notation})'
            ],
            'wavelength': [
                rf'(?:wavelength|λ|lambda)\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})',
                rf'(?:optical|laser|probe)\s+wavelength\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})'
            ],
            'frequency': [
                rf'(?:frequency|ω|omega|f)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:transition|optical|laser)\s+frequency\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?'
            ],
            'cavity_length': [
                rf'(?:cavity|resonator)\s+length\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})',
                rf'(?:L|l)\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})'
            ],
            'mode_waist': [
                rf'(?:mode\s+waist|beam\s+waist|w0|w_0)\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})',
                rf'(?:waist|focal)\s+(?:radius|size)\s*[:=]?\s*({self.sci_notation})\s*({self.length_units})'
            ],
            'pump_power': [
                rf'(?:pump|drive|laser)\s+power\s*[:=]?\s*({self.sci_notation})\s*({self.power_units})',
                rf'(?:optical|input)\s+power\s*[:=]?\s*({self.sci_notation})\s*({self.power_units})'
            ],
            'mechanical_frequency': [
                rf'(?:mechanical|vibration|oscillation)\s+frequency\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?',
                rf'(?:ωm|omega_m|f_m)\s*[:=]?\s*({self.sci_notation})\s*({self.frequency_units})?'
            ],
            'temperature': [
                rf'(?:temperature|T)\s*[:=]?\s*({self.sci_notation})\s*(?:K|Kelvin|°K)',
                rf'(?:room|operating)\s+temperature\s*[:=]?\s*({self.sci_notation})\s*(?:K|Kelvin|°K)'
            ],
            'squeezing_level': [
                rf'(?:squeezing|squeezed)\s*[:=]?\s*({self.sci_notation})\s*(?:dB|db)',
                rf'(?:noise\s+reduction|quantum\s+noise)\s*[:=]?\s*({self.sci_notation})\s*(?:dB|db)'
            ],
            'fidelity': [
                rf'(?:fidelity|F)\s*[:=]?\s*({self.sci_notation})',
                rf'(?:storage|retrieval|quantum)\s+fidelity\s*[:=]?\s*({self.sci_notation})'
            ]
        }
        
        # Mathematical expression patterns
        self.math_expressions = {
            'cooperativity_formula': r'(?:C|cooperativity)\s*=\s*(?:4\s*\*?\s*)?g\^?2\s*/\s*\(?\s*(?:κ|kappa)\s*\*?\s*(?:γ|gamma)\s*\)?',
            'rabi_frequency': r'(?:Ω|Omega|rabi)\s*=\s*2\s*\*?\s*g',
            'purcell_factor': r'(?:F_p|purcell)\s*=\s*(?:3\s*\*?\s*)?(?:λ|lambda)\^?3\s*\*?\s*Q\s*/\s*\(?\s*4\s*\*?\s*π\^?2\s*\*?\s*V\s*\)?'
        }
    
    def _setup_unit_conversions(self):
        """Setup unit conversion factors"""
        self.frequency_conversions = {
            'Hz': 1, 'hz': 1,
            'KHz': 1e3, 'khz': 1e3, 'kHz': 1e3,
            'MHz': 1e6, 'mhz': 1e6,
            'GHz': 1e9, 'ghz': 1e9,
            'THz': 1e12, 'thz': 1e12,
            'rad/s': 1/(2*math.pi), 's^-1': 1, 's⁻¹': 1
        }
        
        self.length_conversions = {
            'm': 1, 'mm': 1e-3, 'cm': 1e-2,
            'μm': 1e-6, 'um': 1e-6, 'micron': 1e-6,
            'nm': 1e-9, 'pm': 1e-12, 'km': 1e3
        }
        
        self.power_conversions = {
            'W': 1, 'mW': 1e-3, 'kW': 1e3, 'MW': 1e6,
            'μW': 1e-6, 'uW': 1e-6, 'nW': 1e-9, 'pW': 1e-12
        }
        
        self.time_conversions = {
            's': 1, 'ms': 1e-3, 'μs': 1e-6, 'us': 1e-6,
            'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18
        }
    
    def _setup_physics_knowledge(self):
        """Setup physics knowledge for validation and estimation"""
        # Typical parameter ranges for validation
        self.parameter_ranges = {
            'coupling_strength': (1e3, 1e12),    # Hz
            'cavity_decay_rate': (1e3, 1e8),     # Hz  
            'atomic_linewidth': (1e3, 1e9),      # Hz
            'finesse': (10, 1e7),                # dimensionless
            'quality_factor': (10, 1e12),        # dimensionless
            'cooperativity': (0.01, 1e6),        # dimensionless
            'wavelength': (100e-9, 10e-6),       # m
            'frequency': (1e12, 1e16),           # Hz
            'cavity_length': (1e-6, 1),          # m
            'mode_waist': (1e-9, 1e-3),         # m
            'pump_power': (1e-12, 1),            # W
            'mechanical_frequency': (1e3, 1e9),  # Hz
            'temperature': (0.01, 1000),         # K
            'squeezing_level': (0, 20),          # dB
            'fidelity': (0, 1)                   # dimensionless
        }
        
        # Common atomic transitions
        self.atomic_transitions = {
            'Rb87_D2': {'wavelength': 780e-9, 'linewidth': 6.07e6},
            'Cs_D2': {'wavelength': 852e-9, 'linewidth': 5.22e6},
            'Ca_intercombination': {'wavelength': 657e-9, 'linewidth': 400},
            'Sr_clock': {'wavelength': 698e-9, 'linewidth': 7.5e-3}
        }
    
    def extract_parameters(self, content: str, solution_data: Dict[str, Any] = None) -> ExtractionResult:
        """
        Main parameter extraction function.
        Extracts all relevant physics parameters from content.
        """
        self.logger.info("Starting parameter extraction")
        
        # Clean and preprocess content
        cleaned_content = self._preprocess_content(content)
        
        # Extract parameters using multiple methods
        extracted_params = {}
        warnings = []
        notes = []
        
        # Method 1: Direct pattern matching
        pattern_params = self._extract_with_patterns(cleaned_content)
        extracted_params.update(pattern_params)
        
        # Method 2: Mathematical expression parsing
        math_params = self._extract_from_math_expressions(cleaned_content)
        extracted_params.update(math_params)
        
        # Method 3: Contextual extraction
        context_params = self._extract_with_context(cleaned_content)
        extracted_params.update(context_params)
        
        # Method 4: Use solution data if available
        if solution_data:
            solution_params = self._extract_from_solution_data(solution_data)
            extracted_params.update(solution_params)
        
        # Validate extracted parameters
        validated_params, param_warnings = self._validate_parameters(extracted_params)
        warnings.extend(param_warnings)
        
        # Calculate derived parameters
        derived_params = self._calculate_derived_parameters(validated_params)
        validated_params.update(derived_params)
        
        # Identify system type
        system_type = self._identify_system_type(cleaned_content, validated_params)
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence_score(validated_params)
        
        self.logger.info(f"Extracted {len(validated_params)} parameters with confidence {confidence_score:.2f}")
        
        return ExtractionResult(
            parameters=validated_params,
            system_type=system_type,
            confidence_score=confidence_score,
            warnings=warnings,
            notes=notes
        )
    
    def _preprocess_content(self, content: str) -> str:
        """Clean and preprocess content for extraction"""
        # Convert to lowercase for pattern matching
        content = content.lower()
        
        # Replace unicode characters
        replacements = {
            'ω': 'omega', 'κ': 'kappa', 'γ': 'gamma', 'λ': 'lambda',
            'μ': 'u', '×': '*', '·': '*', '†': '+', '⁻': '^-',
            '₀': '0', '₁': '1', '₂': '2', '₃': '3'
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Clean up spacing around mathematical operators
        content = re.sub(r'\s*([=:×*+\-/\^])\s*', r' \1 ', content)
        
        return content
    
    def _extract_with_patterns(self, content: str) -> Dict[str, ExtractedParameter]:
        """Extract parameters using regex patterns"""
        extracted = {}
        
        for param_name, patterns in self.parameter_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    try:
                        value_str = match.group(1)
                        units_str = match.group(2) if len(match.groups()) > 1 else None
                        
                        # Convert to float
                        value = float(value_str)
                        
                        # Convert units to standard form
                        if units_str and param_name in ['coupling_strength', 'cavity_decay_rate', 'atomic_linewidth', 'frequency', 'mechanical_frequency']:
                            value *= self.frequency_conversions.get(units_str, 1)
                            standard_units = 'Hz'
                        elif units_str and param_name in ['wavelength', 'cavity_length', 'mode_waist']:
                            value *= self.length_conversions.get(units_str, 1)
                            standard_units = 'm'
                        elif units_str and param_name == 'pump_power':
                            value *= self.power_conversions.get(units_str, 1)
                            standard_units = 'W'
                        else:
                            standard_units = units_str or 'dimensionless'
                        
                        # Get context around match
                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 50)
                        context = content[start:end].strip()
                        
                        extracted[param_name] = ExtractedParameter(
                            name=param_name,
                            value=value,
                            units=standard_units,
                            confidence=0.8,  # High confidence for direct pattern match
                            extraction_method='pattern_matching',
                            context=context
                        )
                        
                        break  # Use first successful match
                        
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Failed to extract {param_name} from pattern: {e}")
                        continue
        
        return extracted
    
    def _extract_from_math_expressions(self, content: str) -> Dict[str, ExtractedParameter]:
        """Extract parameters from mathematical expressions"""
        extracted = {}
        
        # Look for cooperativity formula: C = 4g²/(κγ)
        coop_pattern = r'(?:c|cooperativity)\s*=\s*4\s*\*?\s*(?:g|coupling)\^?2\s*/\s*\(?\s*(?:kappa|κ)\s*\*?\s*(?:gamma|γ)\s*\)?'
        if re.search(coop_pattern, content, re.IGNORECASE):
            # Try to extract g, kappa, gamma and calculate cooperativity
            g = self._find_parameter_value(content, ['g', 'coupling'])
            kappa = self._find_parameter_value(content, ['kappa', 'cavity decay'])
            gamma = self._find_parameter_value(content, ['gamma', 'atomic'])
            
            if g and kappa and gamma:
                cooperativity = 4 * (g**2) / (kappa * gamma)
                extracted['cooperativity'] = ExtractedParameter(
                    name='cooperativity',
                    value=cooperativity,
                    units='dimensionless',
                    confidence=0.9,
                    extraction_method='mathematical_formula',
                    context='Calculated from C = 4g²/(κγ)'
                )
        
        # Look for Rabi frequency: Ω = 2g
        rabi_pattern = r'(?:omega|rabi)\s*=\s*2\s*\*?\s*(?:g|coupling)'
        if re.search(rabi_pattern, content, re.IGNORECASE):
            g = self._find_parameter_value(content, ['g', 'coupling'])
            if g:
                rabi_freq = 2 * g
                extracted['rabi_frequency'] = ExtractedParameter(
                    name='rabi_frequency',
                    value=rabi_freq,
                    units='Hz',
                    confidence=0.9,
                    extraction_method='mathematical_formula',
                    context='Calculated from Ω = 2g'
                )
        
        return extracted
    
    def _extract_with_context(self, content: str) -> Dict[str, ExtractedParameter]:
        """Extract parameters using contextual analysis"""
        extracted = {}
        
        # Look for numerical values near physics terms
        physics_terms = {
            'strong coupling': 'coupling_strength',
            'cavity finesse': 'finesse',
            'quality factor': 'quality_factor',
            'decay rate': 'cavity_decay_rate',
            'linewidth': 'atomic_linewidth'
        }
        
        for term, param_name in physics_terms.items():
            pattern = rf'{term}\s+(?:of\s+|is\s+|:\s*)?({self.sci_notation})'
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                try:
                    value = float(match.group(1))
                    
                    # Estimate units based on parameter type and value magnitude
                    if param_name in ['coupling_strength', 'cavity_decay_rate', 'atomic_linewidth']:
                        units = 'Hz'
                        # Apply magnitude-based unit correction
                        if value < 1000:
                            value *= 1e6  # Assume MHz
                        elif value < 1000000:
                            value *= 1e3  # Assume kHz
                    else:
                        units = 'dimensionless'
                    
                    context = match.group(0)
                    
                    extracted[param_name] = ExtractedParameter(
                        name=param_name,
                        value=value,
                        units=units,
                        confidence=0.6,  # Lower confidence for contextual extraction
                        extraction_method='contextual_analysis',
                        context=context
                    )
                    
                except (ValueError, IndexError):
                    continue
        
        return extracted
    
    def _extract_from_solution_data(self, solution_data: Dict[str, Any]) -> Dict[str, ExtractedParameter]:
        """Extract parameters from solution data structure"""
        extracted = {}
        
        # Check for system_parameters section
        if 'system_parameters' in solution_data:
            sys_params = solution_data['system_parameters']
            
            for key, value in sys_params.items():
                if isinstance(value, (int, float)):
                    # Map parameter names
                    param_name = self._map_parameter_name(key)
                    if param_name:
                        # Estimate units based on parameter name and value
                        units = self._estimate_units(param_name, value)
                        
                        extracted[param_name] = ExtractedParameter(
                            name=param_name,
                            value=float(value),
                            units=units,
                            confidence=0.95,  # High confidence for structured data
                            extraction_method='solution_data',
                            context=f'From system_parameters.{key}'
                        )
        
        # Check for key_results section
        if 'key_results' in solution_data:
            results = solution_data['key_results']
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    param_name = self._map_parameter_name(key)
                    if param_name:
                        units = self._estimate_units(param_name, value)
                        
                        extracted[param_name] = ExtractedParameter(
                            name=param_name,
                            value=float(value),
                            units=units,
                            confidence=0.9,
                            extraction_method='solution_data',
                            context=f'From key_results.{key}'
                        )
        
        return extracted
    
    def _find_parameter_value(self, content: str, keywords: List[str]) -> Optional[float]:
        """Find a parameter value by searching for keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}\s*[:=]?\s*({self.sci_notation})'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _map_parameter_name(self, key: str) -> Optional[str]:
        """Map various parameter names to standard names"""
        key_lower = key.lower().replace('_', ' ')
        
        mapping = {
            'coupling strength': 'coupling_strength',
            'g': 'coupling_strength',
            'rabi': 'coupling_strength',
            'kappa': 'cavity_decay_rate',
            'cavity decay': 'cavity_decay_rate',
            'decay rate': 'cavity_decay_rate',
            'gamma': 'atomic_linewidth',
            'atomic decay': 'atomic_linewidth',
            'linewidth': 'atomic_linewidth',
            'finesse': 'finesse',
            'quality factor': 'quality_factor',
            'q factor': 'quality_factor',
            'cooperativity': 'cooperativity',
            'wavelength': 'wavelength',
            'frequency': 'frequency',
            'cavity length': 'cavity_length',
            'mode waist': 'mode_waist',
            'pump power': 'pump_power',
            'mechanical frequency': 'mechanical_frequency',
            'temperature': 'temperature',
            'squeezing': 'squeezing_level',
            'fidelity': 'fidelity'
        }
        
        return mapping.get(key_lower)
    
    def _estimate_units(self, param_name: str, value: float) -> str:
        """Estimate appropriate units based on parameter name and value"""
        if param_name in ['coupling_strength', 'cavity_decay_rate', 'atomic_linewidth', 'frequency', 'mechanical_frequency']:
            return 'Hz'
        elif param_name in ['wavelength', 'cavity_length', 'mode_waist']:
            return 'm'
        elif param_name == 'pump_power':
            return 'W'
        elif param_name == 'temperature':
            return 'K'
        elif param_name == 'squeezing_level':
            return 'dB'
        else:
            return 'dimensionless'
    
    def _validate_parameters(self, parameters: Dict[str, ExtractedParameter]) -> Tuple[Dict[str, ExtractedParameter], List[str]]:
        """Validate extracted parameters against known ranges"""
        validated = {}
        warnings = []
        
        for param_name, param in parameters.items():
            if param_name in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param_name]
                
                if min_val <= param.value <= max_val:
                    validated[param_name] = param
                else:
                    warnings.append(f"{param_name} value {param.value} outside expected range [{min_val}, {max_val}]")
                    # Include with reduced confidence
                    param.confidence *= 0.5
                    validated[param_name] = param
            else:
                validated[param_name] = param
        
        return validated, warnings
    
    def _calculate_derived_parameters(self, parameters: Dict[str, ExtractedParameter]) -> Dict[str, ExtractedParameter]:
        """Calculate derived parameters from extracted ones"""
        derived = {}
        
        # Calculate cooperativity if we have g, kappa, gamma
        if ('coupling_strength' in parameters and 
            'cavity_decay_rate' in parameters and 
            'atomic_linewidth' in parameters):
            
            g = parameters['coupling_strength'].value
            kappa = parameters['cavity_decay_rate'].value
            gamma = parameters['atomic_linewidth'].value
            
            cooperativity = 4 * (g**2) / (kappa * gamma)
            
            derived['cooperativity'] = ExtractedParameter(
                name='cooperativity',
                value=cooperativity,
                units='dimensionless',
                confidence=0.8,
                extraction_method='derived_calculation',
                context='Calculated from C = 4g²/(κγ)'
            )
        
        # Calculate Rabi frequency if we have coupling strength
        if 'coupling_strength' in parameters:
            g = parameters['coupling_strength'].value
            rabi_freq = 2 * g
            
            derived['rabi_frequency'] = ExtractedParameter(
                name='rabi_frequency',
                value=rabi_freq,
                units='Hz',
                confidence=0.9,
                extraction_method='derived_calculation',
                context='Calculated from Ω = 2g'
            )
        
        # Calculate Q factor from finesse if we have finesse
        if 'finesse' in parameters:
            finesse = parameters['finesse'].value
            q_factor = finesse * math.pi / 2  # Approximate relationship
            
            derived['quality_factor'] = ExtractedParameter(
                name='quality_factor',
                value=q_factor,
                units='dimensionless',
                confidence=0.7,
                extraction_method='derived_calculation',
                context='Estimated from Q ≈ F⋅π/2'
            )
        
        return derived
    
    def _identify_system_type(self, content: str, parameters: Dict[str, ExtractedParameter]) -> str:
        """Identify the type of quantum optical system"""
        # Count indicators for different system types
        indicators = {
            'cavity_qed': ['cavity', 'atom', 'jaynes-cummings', 'rabi', 'purcell'],
            'optomechanics': ['mechanical', 'phonon', 'optomechanical', 'cooling', 'membrane'],
            'squeezed_light': ['squeezed', 'squeezing', 'parametric', 'opo', 'quadrature'],
            'photon_blockade': ['blockade', 'antibunching', 'kerr', 'nonlinear', 'correlation'],
            'quantum_memory': ['memory', 'storage', 'retrieval', 'fidelity']
        }
        
        scores = {}
        for system_type, keywords in indicators.items():
            score = sum(1 for keyword in keywords if keyword in content)
            
            # Boost score based on relevant parameters
            if system_type == 'cavity_qed' and 'cooperativity' in parameters:
                score += 2
            elif system_type == 'optomechanics' and 'mechanical_frequency' in parameters:
                score += 2
            elif system_type == 'squeezed_light' and 'squeezing_level' in parameters:
                score += 2
            
            scores[system_type] = score
        
        return max(scores, key=scores.get) if scores else 'general'
    
    def _calculate_confidence_score(self, parameters: Dict[str, ExtractedParameter]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not parameters:
            return 0.0
        
        # Average individual parameter confidences
        total_confidence = sum(param.confidence for param in parameters.values())
        avg_confidence = total_confidence / len(parameters)
        
        # Boost confidence if we have multiple parameters
        parameter_bonus = min(0.2, 0.05 * len(parameters))
        
        # Boost confidence if we have key parameters
        key_params = ['coupling_strength', 'cavity_decay_rate', 'cooperativity']
        key_param_bonus = sum(0.1 for param in key_params if param in parameters)
        
        final_confidence = min(1.0, avg_confidence + parameter_bonus + key_param_bonus)
        
        return final_confidence
    
    def get_parameter_summary(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """Get a summary of extracted parameters suitable for problem scoring"""
        summary = {}
        
        for param_name, param in extraction_result.parameters.items():
            summary[param_name] = param.value
        
        # Add metadata
        summary['_extraction_metadata'] = {
            'system_type': extraction_result.system_type,
            'confidence_score': extraction_result.confidence_score,
            'parameter_count': len(extraction_result.parameters),
            'warnings': extraction_result.warnings
        }
        
        return summary