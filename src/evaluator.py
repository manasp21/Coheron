import numpy as np
import json
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import yaml

@dataclass
class EvaluationResult:
    feasibility: float
    mathematics: float
    novelty: float
    performance: float
    total_score: float
    details: Dict[str, Any]
    benchmarks_matched: List[str]
    warnings: List[str]

class QuantumOpticsEvaluator:
    """Complete physics-aware evaluation system for quantum optics research"""
    
    def __init__(
        self, 
        benchmarks_path: str = "data/benchmarks.json",
        criteria_path: str = "config/evaluation_criteria.yaml"
    ):
        self.benchmarks = self._load_benchmarks(benchmarks_path)
        self.criteria = self._load_criteria(criteria_path)
        self.logger = logging.getLogger('QuantumOpticsEvaluator')
        
        # Load evaluation weights
        self.weights = self.criteria.get('scoring_criteria', {})
        
        # Physical constants
        self.constants = {
            'h_bar': 1.054571817e-34,  # J⋅s
            'k_b': 1.380649e-23,       # J/K
            'c': 299792458,            # m/s
            'e': 1.602176634e-19,      # C
            'epsilon_0': 8.8541878128e-12,  # F/m
            'mu_0': 1.25663706212e-6   # H/m
        }
        
    def _load_benchmarks(self, filepath: str) -> Dict[str, Any]:
        """Load analytical benchmarks and experimental records"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Benchmarks file not found: {filepath}")
            return self._get_default_benchmarks()
            
    def _load_criteria(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation criteria"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Criteria file not found: {filepath}")
            return self._get_default_criteria()
            
    def evaluate_research(self, solution: Dict[str, Any]) -> EvaluationResult:
        """Complete evaluation pipeline with detailed scoring"""
        
        self.logger.info(f"Evaluating research solution: {solution.get('title', 'Untitled')}")
        
        # Individual component scores
        feasibility_result = self._evaluate_feasibility(solution)
        mathematics_result = self._evaluate_mathematics(solution)
        novelty_result = self._evaluate_novelty(solution)
        performance_result = self._evaluate_performance(solution)
        
        # Calculate weighted total score
        scores = {
            'feasibility': feasibility_result['score'],
            'mathematics': mathematics_result['score'], 
            'novelty': novelty_result['score'],
            'performance': performance_result['score']
        }
        
        weights = {
            'feasibility': self.weights.get('feasibility', {}).get('max_score', 0.25),
            'mathematics': self.weights.get('mathematics', {}).get('max_score', 0.30),
            'novelty': self.weights.get('novelty', {}).get('max_score', 0.25),
            'performance': self.weights.get('performance', {}).get('max_score', 0.20)
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        
        # Compile detailed results
        details = {
            'feasibility_details': feasibility_result,
            'mathematics_details': mathematics_result,
            'novelty_details': novelty_result,
            'performance_details': performance_result,
            'component_scores': scores,
            'weights_used': weights
        }
        
        # Collect warnings and matched benchmarks
        warnings = []
        benchmarks_matched = []
        
        for result in [feasibility_result, mathematics_result, novelty_result, performance_result]:
            warnings.extend(result.get('warnings', []))
            benchmarks_matched.extend(result.get('benchmarks_matched', []))
        
        return EvaluationResult(
            feasibility=scores['feasibility'],
            mathematics=scores['mathematics'],
            novelty=scores['novelty'],
            performance=scores['performance'],
            total_score=total_score,
            details=details,
            benchmarks_matched=list(set(benchmarks_matched)),
            warnings=list(set(warnings))
        )
        
    def _evaluate_feasibility(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive feasibility evaluation"""
        content = solution.get('content', '').lower()
        parameters = self._extract_parameters(content)
        
        feasibility_score = 1.0
        warnings = []
        checks_passed = []
        
        # Energy conservation check
        energy_score, energy_warnings = self._check_energy_conservation(content, parameters)
        feasibility_score *= energy_score
        warnings.extend(energy_warnings)
        if energy_score > 0.8:
            checks_passed.append('energy_conservation')
            
        # Fundamental limits check
        limits_score, limits_warnings = self._check_fundamental_limits(content, parameters)
        feasibility_score *= limits_score
        warnings.extend(limits_warnings)
        if limits_score > 0.8:
            checks_passed.append('fundamental_limits')
            
        # Material properties check
        materials_score, materials_warnings = self._check_material_properties(content, parameters)
        feasibility_score *= materials_score
        warnings.extend(materials_warnings)
        if materials_score > 0.8:
            checks_passed.append('material_properties')
            
        # Experimental complexity assessment
        complexity_score, complexity_warnings = self._assess_experimental_complexity(content, parameters)
        feasibility_score *= complexity_score
        warnings.extend(complexity_warnings)
        if complexity_score > 0.8:
            checks_passed.append('experimental_complexity')
            
        return {
            'score': min(feasibility_score, 1.0),
            'warnings': warnings,
            'checks_passed': checks_passed,
            'component_scores': {
                'energy_conservation': energy_score,
                'fundamental_limits': limits_score,
                'material_properties': materials_score,
                'experimental_complexity': complexity_score
            }
        }
        
    def _evaluate_mathematics(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive mathematical evaluation against benchmarks"""
        content = solution.get('content', '').lower()
        parameters = self._extract_parameters(content)
        
        math_score = 0.0
        benchmarks_matched = []
        warnings = []
        
        # Identify physics category
        category = self._identify_physics_category(content)
        
        if category == 'cavity_qed':
            score, matched, warns = self._verify_cavity_qed_math(content, parameters)
            math_score = max(math_score, score)
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
            
        elif category == 'squeezed_light':
            score, matched, warns = self._verify_squeezing_math(content, parameters)
            math_score = max(math_score, score)
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
            
        elif category == 'optomechanics':
            score, matched, warns = self._verify_optomechanics_math(content, parameters)
            math_score = max(math_score, score)
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
            
        elif category == 'photon_blockade':
            score, matched, warns = self._verify_blockade_math(content, parameters)
            math_score = max(math_score, score)
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
            
        # General physics consistency checks
        consistency_score, consistency_warnings = self._check_general_physics_consistency(content, parameters)
        math_score = max(math_score, consistency_score)
        warnings.extend(consistency_warnings)
        
        # Dimensional analysis
        dimensional_score, dimensional_warnings = self._check_dimensional_analysis(content, parameters)
        math_score *= dimensional_score
        warnings.extend(dimensional_warnings)
        
        return {
            'score': min(math_score, 1.0),
            'category': category,
            'benchmarks_matched': benchmarks_matched,
            'warnings': warnings,
            'consistency_score': consistency_score,
            'dimensional_score': dimensional_score
        }
        
    def _verify_cavity_qed_math(self, content: str, parameters: Dict) -> Tuple[float, List[str], List[str]]:
        """Verify cavity QED mathematical expressions"""
        score = 0.0
        matched = []
        warnings = []
        
        # Extract relevant parameters
        g = self._extract_value(content, ['coupling', 'g', 'rabi'])
        kappa = self._extract_value(content, ['kappa', 'cavity decay', 'decay rate'])
        gamma = self._extract_value(content, ['gamma', 'atomic decay', 'spontaneous'])
        
        if g and kappa and gamma:
            # Check vacuum Rabi frequency: Ω_R = 2g
            if 'rabi' in content and 'frequency' in content:
                expected_rabi = 2 * g
                extracted_rabi = self._extract_value(content, ['rabi frequency', 'omega_r'])
                
                if extracted_rabi and abs(extracted_rabi - expected_rabi) / expected_rabi < 0.1:
                    score += 0.3
                    matched.append('vacuum_rabi_frequency')
                else:
                    warnings.append(f"Rabi frequency mismatch: expected {expected_rabi:.2e}, found {extracted_rabi}")
                    
            # Check cooperativity: C = 4g²/(κγ)
            if 'cooperativity' in content or 'cooperation' in content:
                expected_coop = 4 * g**2 / (kappa * gamma)
                extracted_coop = self._extract_value(content, ['cooperativity', 'c =', 'cooperation'])
                
                if extracted_coop and abs(extracted_coop - expected_coop) / expected_coop < 0.2:
                    score += 0.4
                    matched.append('cooperativity')
                else:
                    warnings.append(f"Cooperativity mismatch: expected {expected_coop:.2f}, found {extracted_coop}")
                    
            # Check strong coupling criterion: g > (κ,γ)/2
            strong_coupling_threshold = max(kappa, gamma) / 2
            if g > strong_coupling_threshold:
                score += 0.3
                matched.append('strong_coupling_criterion')
            else:
                warnings.append(f"Weak coupling regime: g={g:.2e} < threshold={strong_coupling_threshold:.2e}")
                
        return score, matched, warnings
        
    def _verify_squeezing_math(self, content: str, parameters: Dict) -> Tuple[float, List[str], List[str]]:
        """Verify squeezed light mathematical expressions"""
        score = 0.0
        matched = []
        warnings = []
        
        # Extract squeezing parameters
        chi = self._extract_value(content, ['chi', 'nonlinear', 'parametric gain'])
        kappa = self._extract_value(content, ['kappa', 'cavity decay'])
        pump_power = self._extract_value(content, ['pump power', 'p_pump'])
        
        if chi and kappa:
            # Check threshold power: P_th = κ²/(4χ²)
            expected_threshold = kappa**2 / (4 * chi**2)
            extracted_threshold = self._extract_value(content, ['threshold', 'p_th', 'p_threshold'])
            
            if extracted_threshold and abs(extracted_threshold - expected_threshold) / expected_threshold < 0.2:
                score += 0.4
                matched.append('parametric_threshold')
            else:
                warnings.append(f"Threshold power mismatch: expected {expected_threshold:.2e}, found {extracted_threshold}")
                
            # Check squeezing parameter for pump below threshold
            if pump_power and pump_power < expected_threshold:
                pump_ratio = pump_power / expected_threshold
                if pump_ratio < 1 and pump_ratio > 0:
                    expected_squeezing = np.asinh(np.sqrt(max(0, 1 - pump_ratio)))
                    squeezing_db = -10 * np.log10(np.exp(-2 * expected_squeezing))
                    
                    extracted_squeezing_db = self._extract_value(content, ['squeezing', 'db', 'decibel'])
                    if extracted_squeezing_db and abs(extracted_squeezing_db - squeezing_db) < 2:
                        score += 0.4
                        matched.append('squeezing_calculation')
                        
        return score, matched, warnings
        
    def _verify_optomechanics_math(self, content: str, parameters: Dict) -> Tuple[float, List[str], List[str]]:
        """Verify optomechanical mathematical expressions"""
        score = 0.0
        matched = []
        warnings = []
        
        # Extract optomechanical parameters
        g0 = self._extract_value(content, ['g0', 'g_0', 'optomechanical coupling'])
        kappa = self._extract_value(content, ['kappa', 'cavity decay'])
        gamma_m = self._extract_value(content, ['gamma_m', 'mechanical decay'])
        omega_m = self._extract_value(content, ['omega_m', 'mechanical frequency'])
        
        if g0 and kappa and gamma_m:
            # Check cooperativity: C = 4g₀²/(κγₘ)
            expected_coop = 4 * g0**2 / (kappa * gamma_m)
            extracted_coop = self._extract_value(content, ['cooperativity', 'c ='])
            
            if extracted_coop and abs(extracted_coop - expected_coop) / expected_coop < 0.2:
                score += 0.4
                matched.append('optomechanical_cooperativity')
            else:
                warnings.append(f"Cooperativity mismatch: expected {expected_coop:.2f}, found {extracted_coop}")
                
        return score, matched, warnings
        
    def _verify_blockade_math(self, content: str, parameters: Dict) -> Tuple[float, List[str], List[str]]:
        """Verify photon blockade mathematical expressions"""
        score = 0.0
        matched = []
        warnings = []
        
        # Extract blockade parameters
        chi = self._extract_value(content, ['chi', 'kerr', 'nonlinear'])
        kappa = self._extract_value(content, ['kappa', 'cavity decay'])
        omega_drive = self._extract_value(content, ['drive', 'omega', 'laser'])
        
        if chi and kappa:
            # Check blockade condition: χ >> κ
            if chi > 10 * kappa:
                score += 0.4
                matched.append('strong_blockade_regime')
            elif chi > kappa:
                score += 0.2
                matched.append('weak_blockade_regime')
            else:
                warnings.append(f"Insufficient nonlinearity for blockade: χ={chi:.2e} not >> κ={kappa:.2e}")
                
        return score, matched, warnings
        
    def _evaluate_novelty(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate research novelty and potential impact"""
        content = solution.get('content', '').lower()
        parameters = self._extract_parameters(content)
        
        novelty_score = 0.0
        novelty_indicators = []
        
        # Parameter regime novelty
        param_novelty = self._assess_parameter_novelty(content, parameters)
        novelty_score += param_novelty * 0.3
        if param_novelty > 0.7:
            novelty_indicators.append('novel_parameter_regime')
            
        # Architectural novelty
        arch_novelty = self._assess_architectural_novelty(content, parameters)
        novelty_score += arch_novelty * 0.4
        if arch_novelty > 0.7:
            novelty_indicators.append('novel_architecture')
            
        # Application novelty
        app_novelty = self._assess_application_novelty(content, parameters)
        novelty_score += app_novelty * 0.3
        if app_novelty > 0.7:
            novelty_indicators.append('novel_application')
            
        return {
            'score': min(novelty_score, 1.0),
            'indicators': novelty_indicators,
            'parameter_novelty': param_novelty,
            'architectural_novelty': arch_novelty,
            'application_novelty': app_novelty
        }
        
    def _assess_parameter_novelty(self, content: str, parameters: Dict) -> float:
        """Assess novelty of parameter regimes"""
        novelty = 0.0
        
        # Check for extreme parameter regimes
        coupling = parameters.get('coupling_strength', 0)
        if coupling > 1e9:  # Very strong coupling
            novelty += 0.3
        elif coupling < 1e3:  # Very weak coupling
            novelty += 0.2
            
        # Check for novel frequency ranges
        frequency = parameters.get('frequency', 0)
        if frequency > 1e15 or frequency < 1e12:  # Extreme frequencies
            novelty += 0.3
            
        # Check for novel material mentions
        novel_materials = ['metamaterial', 'graphene', 'photonic crystal', 'superconducting', 'diamond nv']
        if any(material in content for material in novel_materials):
            novelty += 0.4
            
        return min(novelty, 1.0)
        
    def _assess_architectural_novelty(self, content: str, parameters: Dict) -> float:
        """Assess architectural and system design novelty"""
        novelty = 0.0
        
        # Check for hybrid systems
        hybrid_indicators = ['hybrid', 'coupled', 'multimode', 'network', 'array']
        hybrid_count = sum(1 for indicator in hybrid_indicators if indicator in content)
        novelty += min(hybrid_count * 0.2, 0.6)
        
        # Check for unconventional geometries
        geometry_indicators = ['ring', 'spiral', 'fractal', 'chiral', 'twisted']
        if any(geo in content for geo in geometry_indicators):
            novelty += 0.3
            
        # Check for many-body systems
        if any(term in content for term in ['many-body', 'collective', 'ensemble']):
            novelty += 0.2
            
        return min(novelty, 1.0)
        
    def _assess_application_novelty(self, content: str, parameters: Dict) -> float:
        """Assess application and use case novelty"""
        novelty = 0.0
        
        # Check for emerging applications
        emerging_apps = ['quantum computing', 'quantum internet', 'quantum sensing', 'quantum radar']
        if any(app in content for app in emerging_apps):
            novelty += 0.4
            
        # Check for interdisciplinary applications
        interdisciplinary = ['biology', 'medicine', 'astronomy', 'geology']
        if any(field in content for field in interdisciplinary):
            novelty += 0.5
            
        # Check for fundamental physics tests
        fundamental = ['relativity', 'gravity', 'dark matter', 'cosmology']
        if any(physics in content for physics in fundamental):
            novelty += 0.3
            
        return min(novelty, 1.0)
        
    def _evaluate_performance(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate system performance metrics"""
        content = solution.get('content', '').lower()
        parameters = self._extract_parameters(content)
        category = self._identify_physics_category(content)
        
        performance_score = 0.0
        metrics = {}
        
        if category == 'cavity_qed':
            metrics = self._extract_cavity_qed_metrics(content, parameters)
        elif category == 'squeezed_light':
            metrics = self._extract_squeezing_metrics(content, parameters)
        elif category == 'optomechanics':
            metrics = self._extract_optomechanics_metrics(content, parameters)
            
        # Score based on performance thresholds
        if metrics:
            performance_score = self._score_performance_metrics(category, metrics)
            
        return {
            'score': performance_score,
            'category': category,
            'metrics': metrics
        }
        
    def _extract_cavity_qed_metrics(self, content: str, parameters: Dict) -> Dict[str, float]:
        """Extract cavity QED performance metrics"""
        metrics = {}
        
        cooperativity = self._extract_value(content, ['cooperativity', 'c ='])
        if cooperativity:
            metrics['cooperativity'] = cooperativity
            
        purcell = self._extract_value(content, ['purcell', 'enhancement'])
        if purcell:
            metrics['purcell_factor'] = purcell
            
        return metrics
        
    def _extract_squeezing_metrics(self, content: str, parameters: Dict) -> Dict[str, float]:
        """Extract squeezing performance metrics"""
        metrics = {}
        
        squeezing_db = self._extract_value(content, ['squeezing', 'db'])
        if squeezing_db:
            metrics['squeezing_db'] = abs(squeezing_db)  # Convert to positive dB
            
        bandwidth = self._extract_value(content, ['bandwidth', 'hz', 'mhz'])
        if bandwidth:
            metrics['bandwidth_hz'] = bandwidth
            
        return metrics
        
    def _extract_optomechanics_metrics(self, content: str, parameters: Dict) -> Dict[str, float]:
        """Extract optomechanics performance metrics"""
        metrics = {}
        
        phonons = self._extract_value(content, ['phonon', 'final'])
        if phonons:
            metrics['final_phonon_number'] = phonons
            
        cooperativity = self._extract_value(content, ['cooperativity'])
        if cooperativity:
            metrics['cooperativity'] = cooperativity
            
        return metrics
        
    def _score_performance_metrics(self, category: str, metrics: Dict[str, float]) -> float:
        """Score performance metrics based on thresholds"""
        score = 0.0
        
        criteria = self.criteria.get('scoring_criteria', {}).get('performance', {}).get('metrics', {})
        category_criteria = criteria.get(category, {})
        
        if not category_criteria:
            return 0.5  # Default score if no criteria available
            
        for metric_name, value in metrics.items():
            if metric_name in category_criteria:
                thresholds = category_criteria[metric_name]
                
                # Parse threshold values
                excellent = self._parse_threshold(thresholds.get('excellent', ''))
                good = self._parse_threshold(thresholds.get('good', ''))
                acceptable = self._parse_threshold(thresholds.get('acceptable', ''))
                
                # Score based on thresholds
                if self._meets_threshold(value, excellent):
                    score += 1.0
                elif self._meets_threshold(value, good):
                    score += 0.8
                elif self._meets_threshold(value, acceptable):
                    score += 0.6
                else:
                    score += 0.2
                    
        return min(score / len(metrics) if metrics else 0.5, 1.0)
        
    def _parse_threshold(self, threshold_str: str) -> float:
        """Parse threshold string to numerical value"""
        if not threshold_str:
            return 0.0
            
        # Extract numerical value from strings like "> 1000", "< 0.1", etc.
        matches = re.findall(r'[\d.]+', threshold_str)
        if matches:
            return float(matches[0])
        return 0.0
        
    def _meets_threshold(self, value: float, threshold: float) -> bool:
        """Check if value meets threshold criteria"""
        # This is a simplified check - could be more sophisticated
        return value >= threshold if threshold > 0 else True
        
    # Utility methods for parameter extraction and analysis
    def _extract_parameters(self, content: str) -> Dict[str, float]:
        """Extract numerical parameters from content"""
        parameters = {}
        
        # Common patterns for parameter extraction
        patterns = {
            'coupling_strength': [r'g\s*=\s*([\d.]+e?[+-]?\d*)', r'coupling.*?([\d.]+e?[+-]?\d*)'],
            'cavity_decay': [r'kappa\s*=\s*([\d.]+e?[+-]?\d*)', r'decay.*?([\d.]+e?[+-]?\d*)'],
            'frequency': [r'omega\s*=\s*([\d.]+e?[+-]?\d*)', r'frequency.*?([\d.]+e?[+-]?\d*)'],
            'finesse': [r'finesse\s*=\s*([\d.]+e?[+-]?\d*)', r'f\s*=\s*([\d.]+e?[+-]?\d*)'],
            'power': [r'power\s*=\s*([\d.]+e?[+-]?\d*)', r'p\s*=\s*([\d.]+e?[+-]?\d*)']
        }
        
        for param_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        parameters[param_name] = float(matches[0])
                        break
                    except ValueError:
                        continue
                        
        return parameters
        
    def _extract_value(self, content: str, keywords: List[str]) -> Optional[float]:
        """Extract numerical value associated with keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}.*?([\d.]+e?[+-]?\d*)'
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue
        return None
        
    def _identify_physics_category(self, content: str) -> str:
        """Identify the primary physics category"""
        categories = {
            'cavity_qed': ['cavity', 'atom', 'jaynes-cummings', 'rabi', 'cooperativity'],
            'squeezed_light': ['squeezed', 'squeezing', 'parametric', 'opo', 'quadrature'],
            'optomechanics': ['optomechanical', 'mechanical', 'phonon', 'radiation pressure'],
            'photon_blockade': ['blockade', 'antibunching', 'g2', 'correlation', 'nonlinear'],
            'quantum_metrology': ['metrology', 'sensing', 'n00n', 'fisher', 'precision']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in content)
            category_scores[category] = score
            
        return max(category_scores, key=category_scores.get) if category_scores else 'general'
        
    def _check_energy_conservation(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Check energy conservation principles"""
        score = 1.0
        warnings = []
        
        # Look for energy gain without input
        if any(term in content for term in ['amplification', 'gain']) and 'pump' not in content:
            score *= 0.3
            warnings.append("Energy amplification without pump source")
            
        return score, warnings
        
    def _check_fundamental_limits(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Check fundamental quantum limits"""
        score = 1.0
        warnings = []
        
        # Check uncertainty principle violations
        if 'uncertainty' in content and 'violate' in content:
            score *= 0.1
            warnings.append("Apparent uncertainty principle violation")
            
        return score, warnings
        
    def _check_material_properties(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Check realistic material properties"""
        score = 1.0
        warnings = []
        
        # Check for unrealistic coupling strengths
        coupling = parameters.get('coupling_strength', 0)
        if coupling > 1e12:  # 1 THz seems unrealistic for most systems
            score *= 0.2
            warnings.append(f"Unrealistic coupling strength: {coupling:.2e} Hz")
            
        return score, warnings
        
    def _assess_experimental_complexity(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Assess experimental feasibility"""
        score = 1.0
        warnings = []
        
        complexity_indicators = ['ultrahigh vacuum', 'dilution refrigerator', 'single atom', 'single photon']
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in content)
        
        if complexity_count > 3:
            score *= 0.7
            warnings.append("Very high experimental complexity")
            
        return score, warnings
        
    def _check_general_physics_consistency(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Check general physics consistency"""
        score = 0.5  # Default baseline
        warnings = []
        
        # Check for reasonable parameter orders of magnitude
        if parameters:
            score += 0.3
            
        # Check for physics terminology usage
        physics_terms = ['hamiltonian', 'evolution', 'coherence', 'decoherence', 'quantum']
        term_count = sum(1 for term in physics_terms if term in content)
        score += min(term_count * 0.05, 0.2)
        
        return min(score, 1.0), warnings
        
    def _check_dimensional_analysis(self, content: str, parameters: Dict) -> Tuple[float, List[str]]:
        """Check dimensional consistency"""
        score = 1.0
        warnings = []
        
        # Look for unit mentions
        units = ['hz', 'ghz', 'mhz', 'khz', 'nm', 'μm', 'mm', 'db']
        if not any(unit in content for unit in units):
            score *= 0.8
            warnings.append("No explicit units found")
            
        return score, warnings
        
    def _get_default_benchmarks(self) -> Dict[str, Any]:
        """Provide default analytical benchmarks if file not found"""
        return {
            "analytical_solutions": {
                "jaynes_cummings_rabi": {
                    "parameters": {"coupling_strength": 1e6, "cavity_decay": 1e4, "atomic_decay": 1e3},
                    "expected_results": {"rabi_frequency": 2e6, "cooperativity": 400}
                }
            },
            "experimental_records": {
                "squeezing_records": {"best_squeezing_db": -15},
                "cavity_finesse": {"highest_finesse": 4.2e6}
            }
        }
        
    def _get_default_criteria(self) -> Dict[str, Any]:
        """Provide default evaluation criteria if file not found"""
        return {
            "scoring_criteria": {
                "feasibility": {"max_score": 0.25},
                "mathematics": {"max_score": 0.30},
                "novelty": {"max_score": 0.25},
                "performance": {"max_score": 0.20}
            }
        }