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
        
        # Initialize AMO components for problem-based evaluation
        self.amo_parameter_extractor = None
        self.amo_physics_calculator = None
        self._initialize_amo_components()
        
        # Physical constants
        self.constants = {
            'h_bar': 1.054571817e-34,  # J⋅s
            'k_b': 1.380649e-23,       # J/K
            'c': 299792458,            # m/s
            'e': 1.602176634e-19,      # C
            'epsilon_0': 8.8541878128e-12,  # F/m
            'mu_0': 1.25663706212e-6   # H/m
        }
    
    def _initialize_amo_components(self):
        """Initialize AMO parameter extraction and calculation components"""
        try:
            from .amo_parameter_extractor import AMOParameterExtractor
            from .amo_physics_calculator import AMOPhysicsCalculator
            
            self.amo_parameter_extractor = AMOParameterExtractor()
            self.amo_physics_calculator = AMOPhysicsCalculator()
            self.logger.info("AMO evaluation components initialized")
            
        except ImportError as e:
            self.logger.warning(f"AMO components not available: {e}")
            self.amo_parameter_extractor = None
            self.amo_physics_calculator = None
        
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
    
    def evaluate_for_amo_problem(self, solution: Dict[str, Any], amo_problem) -> EvaluationResult:
        """
        Evaluate solution specifically for an AMO problem with parameter targets.
        This method focuses on parameter achievement rather than generic research quality.
        """
        self.logger.info(f"Evaluating solution for AMO problem: {amo_problem.title}")
        
        # Extract parameters from solution
        extracted_parameters = {}
        extraction_confidence = 0.0
        
        if self.amo_parameter_extractor:
            try:
                extraction_result = self.amo_parameter_extractor.extract_parameters(
                    solution.get('content', ''), solution
                )
                extracted_parameters = self.amo_parameter_extractor.get_parameter_summary(extraction_result)
                extraction_confidence = extraction_result.confidence_score
                
                # Calculate derived parameters
                if self.amo_physics_calculator:
                    physics_results = self.amo_physics_calculator.calculate_all_relevant_parameters(
                        solution.get('content', ''), extracted_parameters
                    )
                    
                    # Add calculated parameters
                    for param_name, calc_result in physics_results.items():
                        if param_name not in extracted_parameters:
                            extracted_parameters[param_name] = calc_result.value
                
            except Exception as e:
                self.logger.warning(f"Parameter extraction failed: {e}")
        
        # Score against problem targets
        problem_results = amo_problem.calculate_total_score(extracted_parameters)
        target_score = problem_results['total_score']
        
        # Evaluate individual components with AMO focus
        feasibility_result = self._evaluate_amo_feasibility(solution, amo_problem, extracted_parameters)
        mathematics_result = self._evaluate_amo_mathematics(solution, amo_problem, extracted_parameters)
        novelty_result = self._evaluate_amo_novelty(solution, amo_problem, extracted_parameters)
        performance_result = self._evaluate_amo_performance(solution, amo_problem, extracted_parameters, problem_results)
        
        # Weight scores with emphasis on parameter achievement
        amo_weights = {
            'target_achievement': 0.4,  # Primary focus on meeting targets
            'feasibility': 0.2,
            'mathematics': 0.2,
            'novelty': 0.1,
            'performance': 0.1
        }
        
        # Calculate weighted total score
        total_score = (
            target_score * amo_weights['target_achievement'] +
            feasibility_result['score'] * amo_weights['feasibility'] +
            mathematics_result['score'] * amo_weights['mathematics'] +
            novelty_result['score'] * amo_weights['novelty'] +
            performance_result['score'] * amo_weights['performance']
        )
        
        # Compile detailed results
        details = {
            'amo_problem_results': problem_results,
            'extracted_parameters': extracted_parameters,
            'extraction_confidence': extraction_confidence,
            'target_achievement_score': target_score,
            'feasibility_details': feasibility_result,
            'mathematics_details': mathematics_result,
            'novelty_details': novelty_result,
            'performance_details': performance_result,
            'amo_weights_used': amo_weights
        }
        
        # Collect warnings and benchmarks
        warnings = []
        benchmarks_matched = []
        
        for result in [feasibility_result, mathematics_result, novelty_result, performance_result]:
            warnings.extend(result.get('warnings', []))
            benchmarks_matched.extend(result.get('benchmarks_matched', []))
        
        # Add parameter-specific warnings
        missing_targets = amo_problem.get_missing_targets(extracted_parameters)
        if missing_targets:
            warnings.append(f"Missing target parameters: {', '.join(missing_targets)}")
        
        return EvaluationResult(
            feasibility=feasibility_result['score'],
            mathematics=mathematics_result['score'],
            novelty=novelty_result['score'],
            performance=performance_result['score'],
            total_score=total_score,
            details=details,
            benchmarks_matched=list(set(benchmarks_matched)),
            warnings=list(set(warnings))
        )
    
    def _evaluate_amo_feasibility(self, solution: Dict[str, Any], amo_problem, 
                                 extracted_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate feasibility specifically for AMO problem context"""
        content = solution.get('content', '').lower()
        
        feasibility_score = 1.0
        warnings = []
        checks_passed = []
        
        # Check physics constraints from problem
        for constraint in amo_problem.constraints:
            is_valid, message = constraint.validate(extracted_parameters)
            if not is_valid:
                feasibility_score *= 0.7  # Penalty for constraint violation
                warnings.append(f"Constraint violation: {message}")
            else:
                checks_passed.append(constraint.name)
        
        # Standard physics checks
        energy_score, energy_warnings = self._check_energy_conservation(content, extracted_parameters)
        feasibility_score *= energy_score
        warnings.extend(energy_warnings)
        
        limits_score, limits_warnings = self._check_fundamental_limits(content, extracted_parameters)
        feasibility_score *= limits_score
        warnings.extend(limits_warnings)
        
        # AMO-specific parameter range checks
        range_score, range_warnings = self._check_amo_parameter_ranges(extracted_parameters)
        feasibility_score *= range_score
        warnings.extend(range_warnings)
        
        return {
            'score': min(feasibility_score, 1.0),
            'warnings': warnings,
            'checks_passed': checks_passed,
            'constraint_compliance': len(checks_passed) / max(len(amo_problem.constraints), 1)
        }
    
    def _evaluate_amo_mathematics(self, solution: Dict[str, Any], amo_problem,
                                 extracted_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate mathematical correctness with focus on parameter calculations"""
        content = solution.get('content', '').lower()
        
        math_score = 0.0
        benchmarks_matched = []
        warnings = []
        
        # Check parameter consistency using physics calculator
        if self.amo_physics_calculator and extracted_parameters:
            try:
                calculated_params = self.amo_physics_calculator.calculate_all_relevant_parameters(
                    content, extracted_parameters
                )
                
                # Check consistency between extracted and calculated parameters
                consistency_score = self._check_parameter_consistency(
                    extracted_parameters, calculated_params
                )
                math_score += consistency_score * 0.4
                
                if consistency_score > 0.8:
                    benchmarks_matched.append('parameter_consistency')
                
            except Exception as e:
                warnings.append(f"Parameter calculation failed: {e}")
        
        # Standard mathematical checks
        category = amo_problem.category
        if category == 'cavity_qed':
            score, matched, warns = self._verify_cavity_qed_math(content, extracted_parameters)
            math_score += score * 0.3
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
        elif category == 'squeezed_light':
            score, matched, warns = self._verify_squeezing_math(content, extracted_parameters)
            math_score += score * 0.3
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
        elif category == 'optomechanics':
            score, matched, warns = self._verify_optomechanics_math(content, extracted_parameters)
            math_score += score * 0.3
            benchmarks_matched.extend(matched)
            warnings.extend(warns)
        
        # General physics consistency
        consistency_score, consistency_warnings = self._check_general_physics_consistency(content, extracted_parameters)
        math_score += consistency_score * 0.3
        warnings.extend(consistency_warnings)
        
        return {
            'score': min(math_score, 1.0),
            'category': category,
            'benchmarks_matched': benchmarks_matched,
            'warnings': warnings,
            'parameter_consistency': consistency_score if 'consistency_score' in locals() else 0.0
        }
    
    def _evaluate_amo_novelty(self, solution: Dict[str, Any], amo_problem,
                             extracted_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate novelty in context of AMO problem solving"""
        content = solution.get('content', '').lower()
        
        novelty_score = 0.0
        novelty_indicators = []
        
        # Parameter regime novelty
        param_novelty = self._assess_amo_parameter_novelty(extracted_parameters, amo_problem)
        novelty_score += param_novelty * 0.4
        if param_novelty > 0.7:
            novelty_indicators.append('novel_parameter_regime')
        
        # Approach novelty for the specific problem
        approach_novelty = self._assess_amo_approach_novelty(content, amo_problem)
        novelty_score += approach_novelty * 0.4
        if approach_novelty > 0.7:
            novelty_indicators.append('novel_approach')
        
        # Standard novelty assessment
        standard_novelty = self._assess_application_novelty(content, extracted_parameters)
        novelty_score += standard_novelty * 0.2
        
        return {
            'score': min(novelty_score, 1.0),
            'indicators': novelty_indicators,
            'parameter_novelty': param_novelty,
            'approach_novelty': approach_novelty,
            'application_novelty': standard_novelty
        }
    
    def _evaluate_amo_performance(self, solution: Dict[str, Any], amo_problem,
                                 extracted_parameters: Dict[str, float],
                                 problem_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance specifically for AMO problem targets"""
        
        # Primary score comes from target parameter achievement
        target_score = problem_results['total_score']
        
        # Bonus for exceeding targets
        parameter_scores = problem_results.get('parameter_scores', {})
        excellence_bonus = 0.0
        
        for param_name, param_result in parameter_scores.items():
            if param_result.get('score', 0) > 0.95:  # Exceptional achievement
                excellence_bonus += 0.1
        
        excellence_bonus = min(excellence_bonus, 0.3)  # Cap bonus
        
        # Check for breakthrough achievement
        breakthrough_achieved = amo_problem.is_solved(extracted_parameters)
        breakthrough_bonus = 0.2 if breakthrough_achieved else 0.0
        
        performance_score = min(target_score + excellence_bonus + breakthrough_bonus, 1.0)
        
        return {
            'score': performance_score,
            'target_achievement': target_score,
            'excellence_bonus': excellence_bonus,
            'breakthrough_bonus': breakthrough_bonus,
            'breakthrough_achieved': breakthrough_achieved,
            'parameter_details': parameter_scores
        }
    
    def _check_parameter_consistency(self, extracted_params: Dict[str, float],
                                   calculated_results: Dict[str, Any]) -> float:
        """Check consistency between extracted and calculated parameters"""
        if not extracted_params or not calculated_results:
            return 0.5  # Neutral score if no comparison possible
        
        consistency_score = 1.0
        comparisons = 0
        
        for param_name, calc_result in calculated_results.items():
            if param_name in extracted_params:
                extracted_val = extracted_params[param_name]
                calculated_val = calc_result.value
                
                if extracted_val > 0 and calculated_val > 0:
                    # Check relative agreement
                    relative_error = abs(extracted_val - calculated_val) / max(extracted_val, calculated_val)
                    if relative_error < 0.1:  # Within 10%
                        consistency_score *= 1.0
                    elif relative_error < 0.5:  # Within 50%
                        consistency_score *= 0.8
                    else:
                        consistency_score *= 0.5
                    
                    comparisons += 1
        
        return consistency_score if comparisons > 0 else 0.7  # Default score if no comparisons
    
    def _check_amo_parameter_ranges(self, parameters: Dict[str, float]) -> Tuple[float, List[str]]:
        """Check if AMO parameters are in reasonable physical ranges"""
        score = 1.0
        warnings = []
        
        # AMO-specific parameter ranges
        amo_ranges = {
            'coupling_strength': (1e3, 1e12),      # Hz
            'cavity_decay_rate': (1e2, 1e8),       # Hz
            'atomic_linewidth': (1e3, 1e9),        # Hz
            'cooperativity': (0.01, 1e6),          # dimensionless
            'quality_factor': (10, 1e12),          # dimensionless
            'finesse': (10, 1e7),                  # dimensionless
            'squeezing_level': (0, 25),            # dB
            'fidelity': (0, 1),                    # fraction
            'temperature': (0.001, 1000),          # K
            'wavelength': (100e-9, 10e-6),         # m
            'frequency': (1e10, 1e16),             # Hz
        }
        
        for param_name, value in parameters.items():
            if param_name.startswith('_'):
                continue  # Skip metadata
                
            if param_name in amo_ranges:
                min_val, max_val = amo_ranges[param_name]
                
                if not (min_val <= value <= max_val):
                    score *= 0.8
                    warnings.append(f"{param_name} = {value:.2e} outside reasonable range [{min_val:.2e}, {max_val:.2e}]")
        
        return score, warnings
    
    def _assess_amo_parameter_novelty(self, parameters: Dict[str, float], amo_problem) -> float:
        """Assess novelty of parameter values in context of specific AMO problem"""
        novelty = 0.0
        
        # Check if parameters exceed current records
        for param_name, value in parameters.items():
            if param_name in amo_problem.target_parameters:
                target_param = amo_problem.target_parameters[param_name]
                
                # Check against current record if available
                if target_param.current_record:
                    try:
                        # Simple parsing of current record
                        record_value = float(re.findall(r'[\d.]+e?[+-]?\d*', target_param.current_record)[0])
                        
                        # Bonus for exceeding records
                        if target_param.target_operator in ['>', '>='] and value > record_value:
                            novelty += 0.3
                        elif target_param.target_operator in ['<', '<='] and value < record_value:
                            novelty += 0.3
                            
                    except (ValueError, IndexError):
                        pass  # Couldn't parse record
                
                # Bonus for extreme parameter regimes
                if param_name == 'coupling_strength' and value > 1e9:
                    novelty += 0.2
                elif param_name == 'cooperativity' and value > 1000:
                    novelty += 0.2
                elif param_name == 'quality_factor' and value > 1e8:
                    novelty += 0.2
        
        return min(novelty, 1.0)
    
    def _assess_amo_approach_novelty(self, content: str, amo_problem) -> float:
        """Assess novelty of approach for the specific AMO problem"""
        novelty = 0.0
        
        # Novel system architectures
        novel_architectures = [
            'hybrid', 'multimode', 'network', 'coupled array', 'distributed',
            'metamaterial', 'photonic crystal', 'plasmonic', 'superconducting'
        ]
        
        architecture_count = sum(1 for arch in novel_architectures if arch in content)
        novelty += min(architecture_count * 0.15, 0.5)
        
        # Novel physics mechanisms
        novel_mechanisms = [
            'many-body', 'collective', 'nonlinear', 'squeezed', 'entangled',
            'topological', 'synthetic', 'engineered', 'tailored'
        ]
        
        mechanism_count = sum(1 for mech in novel_mechanisms if mech in content)
        novelty += min(mechanism_count * 0.1, 0.3)
        
        # Problem-specific novelty
        if amo_problem.category == 'cavity_qed' and 'room temperature' in content:
            novelty += 0.3  # Room temp cavity QED is highly novel
        elif amo_problem.category == 'quantum_memory' and 'millisecond' in content:
            novelty += 0.2  # Long storage times are challenging
        
        return min(novelty, 1.0)
            
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