"""
AMO Physics Problem Definition System

Complete data structures and validation for Atomic, Molecular, and Optical physics problems
that can be solved through evolutionary optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
import yaml
import json
from pathlib import Path
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PhysicsParameter:
    """Represents a specific physics parameter with target values and constraints"""
    symbol: str
    target: Union[float, str]  # e.g., "> 1000e6" or "0.999"
    units: str
    description: str
    weight: float = 1.0
    type: str = "optimization"  # "optimization", "constraint", "benchmark"
    current_record: Optional[str] = None
    target_operator: Optional[str] = None  # ">", "<", ">=", "<=", "="
    target_number: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.target, str):
            self.target_operator, self.target_number = self._parse_target_string(self.target)
    
    def _parse_target_string(self, target_str: str) -> tuple:
        """Parse target strings like '> 1000e6' into operator and number"""
        target_str = target_str.strip()
        
        if target_str.startswith('>='):
            return '>=', float(target_str[2:].strip())
        elif target_str.startswith('<='):
            return '<=', float(target_str[2:].strip())
        elif target_str.startswith('>'):
            return '>', float(target_str[1:].strip())
        elif target_str.startswith('<'):
            return '<', float(target_str[1:].strip())
        elif target_str.startswith('='):
            return '=', float(target_str[1:].strip())
        else:
            return '=', float(target_str)
    
    def evaluate_achievement(self, achieved_value: float, use_relative_scoring: bool = True) -> float:
        """Evaluate how well the achieved value meets the target (0-1 score)"""
        if achieved_value is None or achieved_value == 0:
            return 0.0
            
        if self.target_operator == '>':
            if achieved_value > self.target_number:
                if use_relative_scoring:
                    # Reward exceeding target with bonus up to 1.0
                    return min(1.0, 0.9 + 0.1 * (achieved_value / self.target_number - 1))
                return 1.0
            else:
                # Linear scoring based on how close we are to target
                return max(0.0, achieved_value / self.target_number * 0.9)
                
        elif self.target_operator == '>=':
            if achieved_value >= self.target_number:
                if use_relative_scoring:
                    return min(1.0, 0.9 + 0.1 * (achieved_value / self.target_number - 1))
                return 1.0
            else:
                return max(0.0, achieved_value / self.target_number * 0.9)
                
        elif self.target_operator == '<':
            if achieved_value < self.target_number:
                return 1.0
            else:
                # Penalty for exceeding limit
                return max(0.1, self.target_number / achieved_value)
                
        elif self.target_operator == '<=':
            if achieved_value <= self.target_number:
                return 1.0
            else:
                return max(0.1, self.target_number / achieved_value)
                
        elif self.target_operator == '=':
            relative_error = abs(achieved_value - self.target_number) / max(self.target_number, 1e-10)
            return max(0.0, 1.0 - relative_error)
        
        return 0.0
    
    def is_target_achieved(self, achieved_value: float, threshold: float = 0.9) -> bool:
        """Check if target is considered achieved"""
        return self.evaluate_achievement(achieved_value) >= threshold

@dataclass
class PhysicsConstraint:
    """Represents a physics constraint that must be satisfied"""
    name: str
    description: str
    constraint_type: str  # "fundamental", "experimental", "material", "practical"
    validation_function: Optional[str] = None
    constraint_expression: Optional[str] = None
    penalty_weight: float = 1.0
    
    def validate(self, parameters: Dict[str, float]) -> tuple[bool, str]:
        """Validate constraint against extracted parameters"""
        if self.validation_function:
            # Would implement custom validation functions
            return True, "Custom validation not implemented"
        
        if self.constraint_expression:
            # Simple expression validation
            try:
                # This is a simplified validator - could be much more sophisticated
                if "energy conservation" in self.description.lower():
                    return self._validate_energy_conservation(parameters)
                elif "uncertainty" in self.description.lower():
                    return self._validate_uncertainty_principle(parameters)
                elif "coupling" in self.description.lower():
                    return self._validate_coupling_limits(parameters)
                else:
                    return True, "No specific validation implemented"
            except Exception as e:
                return False, f"Validation error: {e}"
        
        return True, "No validation criteria specified"
    
    def _validate_energy_conservation(self, parameters: Dict[str, float]) -> tuple[bool, str]:
        """Validate energy conservation"""
        # Look for pump and output energies
        pump_power = parameters.get('pump_power', 0)
        output_power = parameters.get('output_power', 0)
        
        if pump_power > 0 and output_power > pump_power:
            return False, "Output power exceeds pump power (energy conservation violation)"
        
        return True, "Energy conservation satisfied"
    
    def _validate_uncertainty_principle(self, parameters: Dict[str, float]) -> tuple[bool, str]:
        """Validate uncertainty principle"""
        delta_x = parameters.get('position_uncertainty', 0)
        delta_p = parameters.get('momentum_uncertainty', 0)
        
        if delta_x > 0 and delta_p > 0:
            hbar = 1.054571817e-34
            if delta_x * delta_p < hbar / 2:
                return False, "Uncertainty principle violation: Δx·Δp < ℏ/2"
        
        return True, "Uncertainty principle satisfied"
    
    def _validate_coupling_limits(self, parameters: Dict[str, float]) -> tuple[bool, str]:
        """Validate realistic coupling strength limits"""
        coupling = parameters.get('coupling_strength', 0)
        
        if coupling > 1e13:  # 10 THz seems unrealistic for most systems
            return False, f"Unrealistic coupling strength: {coupling:.2e} Hz"
        
        return True, "Coupling strength within realistic limits"

@dataclass
class AMOProblem:
    """Complete definition of an AMO physics problem to be solved"""
    id: str
    title: str
    category: str  # cavity_qed, squeezed_light, optomechanics, etc.
    description: str
    physics_challenge: str
    
    # Target physics parameters to optimize
    target_parameters: Dict[str, PhysicsParameter] = field(default_factory=dict)
    
    # Physics constraints that must be satisfied
    constraints: List[PhysicsConstraint] = field(default_factory=list)
    
    # Success criteria thresholds
    breakthrough_threshold: float = 0.9  # All targets met exceptionally well
    significant_progress_threshold: float = 0.7  # Substantial improvement
    
    # Problem metadata
    domain: str = "quantum_optics"
    subdomain: str = "atomic_molecular_optical"
    difficulty_level: str = "hard"  # easy, medium, hard, expert, unsolved
    experimental_status: str = "open"  # solved, partially_solved, open, impossible
    
    # Context and guidance
    background_context: str = ""
    experimental_constraints: List[str] = field(default_factory=list)
    theoretical_framework: str = ""
    
    # References and related work
    references: List[str] = field(default_factory=list)
    related_problems: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Success metrics and tracking
    current_best_solution: Optional[Dict[str, Any]] = None
    solution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_total_score(self, achieved_parameters: Dict[str, float], 
                            include_constraints: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive score based on achieved parameter values"""
        results = {
            'parameter_scores': {},
            'constraint_violations': [],
            'total_score': 0.0,
            'weighted_score': 0.0,
            'achievement_count': 0,
            'total_parameters': len(self.target_parameters)
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        achievements = 0
        
        # Evaluate each target parameter
        for param_name, param_def in self.target_parameters.items():
            if param_name in achieved_parameters:
                achieved_value = achieved_parameters[param_name]
                score = param_def.evaluate_achievement(achieved_value)
                weighted_score = score * param_def.weight
                
                results['parameter_scores'][param_name] = {
                    'achieved': achieved_value,
                    'target': param_def.target_number,
                    'target_operator': param_def.target_operator,
                    'score': score,
                    'weighted_score': weighted_score,
                    'target_achieved': param_def.is_target_achieved(achieved_value)
                }
                
                total_weighted_score += weighted_score
                total_weight += param_def.weight
                
                if param_def.is_target_achieved(achieved_value):
                    achievements += 1
                    
                logger.debug(f"Parameter {param_name}: achieved={achieved_value}, "
                           f"target={param_def.target}, score={score:.3f}")
        
        # Check constraints
        if include_constraints:
            for constraint in self.constraints:
                is_valid, message = constraint.validate(achieved_parameters)
                if not is_valid:
                    results['constraint_violations'].append({
                        'constraint': constraint.name,
                        'message': message,
                        'penalty': constraint.penalty_weight
                    })
                    # Apply penalty to total score
                    total_weighted_score *= (1.0 - 0.1 * constraint.penalty_weight)
        
        # Calculate final scores
        if total_weight > 0:
            weighted_score = total_weighted_score / total_weight
        else:
            weighted_score = 0.0
            
        results['total_score'] = min(weighted_score, 1.0)
        results['weighted_score'] = weighted_score
        results['achievement_count'] = achievements
        
        return results
    
    def is_solved(self, achieved_parameters: Dict[str, float]) -> bool:
        """Check if the problem is considered solved"""
        results = self.calculate_total_score(achieved_parameters)
        return results['total_score'] >= self.breakthrough_threshold
    
    def is_significant_progress(self, achieved_parameters: Dict[str, float]) -> bool:
        """Check if significant progress has been made"""
        results = self.calculate_total_score(achieved_parameters)
        return results['total_score'] >= self.significant_progress_threshold
    
    def get_missing_targets(self, achieved_parameters: Dict[str, float]) -> List[str]:
        """Get list of targets not yet achieved"""
        missing = []
        for param_name, param_def in self.target_parameters.items():
            if param_name in achieved_parameters:
                achieved_value = achieved_parameters[param_name]
                if not param_def.is_target_achieved(achieved_value):
                    missing.append(param_name)
            else:
                missing.append(param_name)
        return missing
    
    def get_optimization_priority(self, achieved_parameters: Dict[str, float]) -> List[tuple]:
        """Get parameters ranked by optimization priority"""
        priorities = []
        
        for param_name, param_def in self.target_parameters.items():
            if param_name in achieved_parameters:
                achieved_value = achieved_parameters[param_name]
                current_score = param_def.evaluate_achievement(achieved_value)
                gap = 1.0 - current_score
                weighted_gap = gap * param_def.weight
                priorities.append((param_name, weighted_gap, current_score))
            else:
                # Missing parameter gets highest priority
                priorities.append((param_name, param_def.weight, 0.0))
        
        # Sort by weighted gap (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities
    
    def update_best_solution(self, solution_data: Dict[str, Any], achieved_parameters: Dict[str, float]):
        """Update current best solution if this one is better"""
        current_score = self.calculate_total_score(achieved_parameters)['total_score']
        
        if (self.current_best_solution is None or 
            current_score > self.current_best_solution.get('score', 0)):
            
            self.current_best_solution = {
                'solution': solution_data,
                'parameters': achieved_parameters,
                'score': current_score,
                'timestamp': self._get_timestamp()
            }
            
            # Add to solution history
            self.solution_history.append(self.current_best_solution.copy())
            
            # Keep only last 100 solutions to manage memory
            if len(self.solution_history) > 100:
                self.solution_history = self.solution_history[-100:]
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

class AMOProblemValidator:
    """Validates AMO problem definitions for correctness and completeness"""
    
    @staticmethod
    def validate_problem(problem: AMOProblem) -> Dict[str, Any]:
        """Comprehensive validation of an AMO problem"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Required field validation
        required_fields = ['id', 'title', 'category', 'description', 'physics_challenge']
        for field in required_fields:
            if not getattr(problem, field):
                validation_results['errors'].append(f"Required field '{field}' is empty")
                validation_results['is_valid'] = False
        
        # Target parameters validation
        if not problem.target_parameters:
            validation_results['errors'].append("No target parameters defined")
            validation_results['is_valid'] = False
        else:
            for param_name, param in problem.target_parameters.items():
                if not param.symbol:
                    validation_results['warnings'].append(f"Parameter '{param_name}' has no symbol")
                if not param.description:
                    validation_results['warnings'].append(f"Parameter '{param_name}' has no description")
                if param.weight <= 0:
                    validation_results['errors'].append(f"Parameter '{param_name}' has invalid weight: {param.weight}")
                    validation_results['is_valid'] = False
        
        # Category validation
        valid_categories = ['cavity_qed', 'squeezed_light', 'photon_blockade', 
                          'quantum_metrology', 'optomechanics', 'quantum_memory', 'hybrid_systems']
        if problem.category not in valid_categories:
            validation_results['warnings'].append(f"Category '{problem.category}' not in standard list: {valid_categories}")
        
        # Threshold validation
        if not 0 <= problem.breakthrough_threshold <= 1:
            validation_results['errors'].append(f"Invalid breakthrough threshold: {problem.breakthrough_threshold}")
            validation_results['is_valid'] = False
        
        if not 0 <= problem.significant_progress_threshold <= 1:
            validation_results['errors'].append(f"Invalid progress threshold: {problem.significant_progress_threshold}")
            validation_results['is_valid'] = False
            
        if problem.significant_progress_threshold >= problem.breakthrough_threshold:
            validation_results['warnings'].append("Progress threshold should be lower than breakthrough threshold")
        
        # Physics validity checks
        physics_validation = AMOProblemValidator._validate_physics_consistency(problem)
        validation_results['warnings'].extend(physics_validation)
        
        # Completeness suggestions
        if not problem.background_context:
            validation_results['suggestions'].append("Consider adding background context")
        if not problem.references:
            validation_results['suggestions'].append("Consider adding relevant references")
        if not problem.keywords:
            validation_results['suggestions'].append("Consider adding keywords for searchability")
        
        return validation_results
    
    @staticmethod
    def _validate_physics_consistency(problem: AMOProblem) -> List[str]:
        """Check physics consistency of the problem definition"""
        warnings = []
        
        # Check for reasonable parameter ranges based on category
        if problem.category == 'cavity_qed':
            for param_name, param in problem.target_parameters.items():
                if 'coupling' in param_name.lower():
                    if param.target_number > 1e12:
                        warnings.append(f"Very high coupling strength target: {param.target_number} Hz")
                elif 'cooperativity' in param_name.lower():
                    if param.target_number > 10000:
                        warnings.append(f"Very high cooperativity target: {param.target_number}")
        
        elif problem.category == 'squeezed_light':
            for param_name, param in problem.target_parameters.items():
                if 'squeezing' in param_name.lower() and 'db' in param.units.lower():
                    if param.target_number > 20:
                        warnings.append(f"Very high squeezing target: {param.target_number} dB")
        
        return warnings

class AMOProblemLoader:
    """Loads and manages AMO physics problems from YAML definitions"""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> AMOProblem:
        """Load an AMO problem from a YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return AMOProblemLoader._create_problem_from_dict(data)
    
    @staticmethod
    def load_from_dict(data: dict) -> AMOProblem:
        """Create an AMO problem from a dictionary"""
        return AMOProblemLoader._create_problem_from_dict(data)
    
    @staticmethod
    def _create_problem_from_dict(data: dict) -> AMOProblem:
        """Internal method to create AMOProblem from dictionary data"""
        
        # Parse target parameters
        target_parameters = {}
        if 'target_parameters' in data:
            for param_name, param_data in data['target_parameters'].items():
                target_parameters[param_name] = PhysicsParameter(
                    symbol=param_data.get('symbol', param_name),
                    target=param_data['target'],
                    units=param_data.get('units', ''),
                    description=param_data.get('description', ''),
                    current_record=param_data.get('current_record'),
                    weight=param_data.get('weight', 1.0),
                    type=param_data.get('type', 'optimization')
                )
        
        # Parse constraints
        constraints = []
        if 'constraints' in data:
            for constraint_data in data['constraints']:
                if isinstance(constraint_data, str):
                    constraints.append(PhysicsConstraint(
                        name="unnamed_constraint",
                        description=constraint_data,
                        constraint_type="general"
                    ))
                else:
                    constraints.append(PhysicsConstraint(
                        name=constraint_data.get('name', 'unnamed'),
                        description=constraint_data['description'],
                        constraint_type=constraint_data.get('type', 'general'),
                        validation_function=constraint_data.get('validation_function'),
                        constraint_expression=constraint_data.get('expression'),
                        penalty_weight=constraint_data.get('penalty_weight', 1.0)
                    ))
        
        return AMOProblem(
            id=data['id'],
            title=data['title'],
            category=data['category'],
            description=data['description'],
            physics_challenge=data['physics_challenge'],
            target_parameters=target_parameters,
            constraints=constraints,
            breakthrough_threshold=data.get('breakthrough_threshold', 0.9),
            significant_progress_threshold=data.get('significant_progress_threshold', 0.7),
            domain=data.get('domain', 'quantum_optics'),
            subdomain=data.get('subdomain', 'atomic_molecular_optical'),
            difficulty_level=data.get('difficulty_level', 'hard'),
            experimental_status=data.get('experimental_status', 'open'),
            background_context=data.get('background_context', ''),
            experimental_constraints=data.get('experimental_constraints', []),
            theoretical_framework=data.get('theoretical_framework', ''),
            references=data.get('references', []),
            related_problems=data.get('related_problems', []),
            keywords=data.get('keywords', [])
        )
    
    @staticmethod
    def save_to_yaml(problem: AMOProblem, file_path: str):
        """Save an AMO problem to a YAML file"""
        data = AMOProblemLoader._problem_to_dict(problem)
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    
    @staticmethod
    def _problem_to_dict(problem: AMOProblem) -> dict:
        """Convert AMOProblem to dictionary for serialization"""
        data = {
            'id': problem.id,
            'title': problem.title,
            'category': problem.category,
            'description': problem.description,
            'physics_challenge': problem.physics_challenge,
            'target_parameters': {},
            'constraints': [],
            'breakthrough_threshold': problem.breakthrough_threshold,
            'significant_progress_threshold': problem.significant_progress_threshold,
            'domain': problem.domain,
            'subdomain': problem.subdomain,
            'difficulty_level': problem.difficulty_level,
            'experimental_status': problem.experimental_status,
            'background_context': problem.background_context,
            'experimental_constraints': problem.experimental_constraints,
            'theoretical_framework': problem.theoretical_framework,
            'references': problem.references,
            'related_problems': problem.related_problems,
            'keywords': problem.keywords
        }
        
        # Convert target parameters
        for param_name, param in problem.target_parameters.items():
            data['target_parameters'][param_name] = {
                'symbol': param.symbol,
                'target': param.target,
                'units': param.units,
                'description': param.description,
                'weight': param.weight,
                'type': param.type
            }
            if param.current_record:
                data['target_parameters'][param_name]['current_record'] = param.current_record
        
        # Convert constraints
        for constraint in problem.constraints:
            constraint_dict = {
                'name': constraint.name,
                'description': constraint.description,
                'type': constraint.constraint_type,
                'penalty_weight': constraint.penalty_weight
            }
            if constraint.validation_function:
                constraint_dict['validation_function'] = constraint.validation_function
            if constraint.constraint_expression:
                constraint_dict['expression'] = constraint.constraint_expression
            data['constraints'].append(constraint_dict)
        
        return data

class AMOProblemLibrary:
    """Manages a collection of AMO physics problems"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.library_path = library_path or "data/amo_problems"
        self.problems: Dict[str, AMOProblem] = {}
        self.categories: Dict[str, List[str]] = {}
        self._load_library()
    
    def _load_library(self):
        """Load all problems from the library directory"""
        library_dir = Path(self.library_path)
        if not library_dir.exists():
            logger.warning(f"AMO problem library directory not found: {library_dir}")
            library_dir.mkdir(parents=True, exist_ok=True)
            self._create_example_problems()
            return
        
        for yaml_file in library_dir.glob("*.yaml"):
            try:
                problem = AMOProblemLoader.load_from_yaml(str(yaml_file))
                self.add_problem(problem)
                logger.info(f"Loaded AMO problem: {problem.title}")
            except Exception as e:
                logger.error(f"Failed to load problem from {yaml_file}: {e}")
    
    def _create_example_problems(self):
        """Create example problems if library is empty"""
        example_problems = self._get_example_problems()
        for problem in example_problems:
            self.add_problem(problem)
            # Save to file
            filename = f"{problem.id}.yaml"
            filepath = Path(self.library_path) / filename
            AMOProblemLoader.save_to_yaml(problem, str(filepath))
    
    def add_problem(self, problem: AMOProblem):
        """Add a new problem to the library"""
        self.problems[problem.id] = problem
        
        # Update categories index
        if problem.category not in self.categories:
            self.categories[problem.category] = []
        if problem.id not in self.categories[problem.category]:
            self.categories[problem.category].append(problem.id)
    
    def get_problem(self, problem_id: str) -> Optional[AMOProblem]:
        """Get a specific problem by ID"""
        return self.problems.get(problem_id)
    
    def get_problems_by_category(self, category: str) -> List[AMOProblem]:
        """Get all problems in a specific category"""
        problem_ids = self.categories.get(category, [])
        return [self.problems[pid] for pid in problem_ids if pid in self.problems]
    
    def get_unsolved_problems(self) -> List[AMOProblem]:
        """Get all problems marked as unsolved"""
        return [p for p in self.problems.values() if p.experimental_status == "open"]
    
    def get_problems_by_difficulty(self, difficulty: str) -> List[AMOProblem]:
        """Get problems by difficulty level"""
        return [p for p in self.problems.values() if p.difficulty_level == difficulty]
    
    def search_problems(self, keywords: List[str]) -> List[AMOProblem]:
        """Search problems by keywords"""
        results = []
        for problem in self.problems.values():
            # Check in title, description, and keywords
            text_to_search = f"{problem.title} {problem.description} {' '.join(problem.keywords)}".lower()
            if any(keyword.lower() in text_to_search for keyword in keywords):
                results.append(problem)
        return results
    
    def list_problems(self) -> List[str]:
        """List all available problem IDs"""
        return list(self.problems.keys())
    
    def get_all_problems_info(self) -> Dict[str, Dict[str, Any]]:
        """Get basic info for all problems"""
        return {
            problem_id: {
                'title': problem.title,
                'category': problem.category,
                'difficulty_level': problem.difficulty_level,
                'description': problem.description
            }
            for problem_id, problem in self.problems.items()
        }
    
    def get_problem_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the problem library"""
        summary = {
            'total_problems': len(self.problems),
            'by_category': {},
            'by_status': {},
            'by_difficulty': {},
            'categories': list(self.categories.keys())
        }
        
        for problem in self.problems.values():
            # Count by category
            if problem.category not in summary['by_category']:
                summary['by_category'][problem.category] = 0
            summary['by_category'][problem.category] += 1
            
            # Count by status
            if problem.experimental_status not in summary['by_status']:
                summary['by_status'][problem.experimental_status] = 0
            summary['by_status'][problem.experimental_status] += 1
            
            # Count by difficulty
            if problem.difficulty_level not in summary['by_difficulty']:
                summary['by_difficulty'][problem.difficulty_level] = 0
            summary['by_difficulty'][problem.difficulty_level] += 1
        
        return summary
    
    def validate_all_problems(self) -> Dict[str, Any]:
        """Validate all problems in the library"""
        validation_summary = {
            'total_problems': len(self.problems),
            'valid_problems': 0,
            'invalid_problems': 0,
            'problems_with_warnings': 0,
            'validation_details': {}
        }
        
        for problem_id, problem in self.problems.items():
            validation_result = AMOProblemValidator.validate_problem(problem)
            validation_summary['validation_details'][problem_id] = validation_result
            
            if validation_result['is_valid']:
                validation_summary['valid_problems'] += 1
            else:
                validation_summary['invalid_problems'] += 1
            
            if validation_result['warnings']:
                validation_summary['problems_with_warnings'] += 1
        
        return validation_summary
    
    def _get_example_problems(self) -> List[AMOProblem]:
        """Create example AMO problems for demonstration"""
        problems = []
        
        # Room temperature cavity QED
        problems.append(AMOProblem(
            id="room_temp_cavity_qed",
            title="Room Temperature Strong Coupling in Cavity QED",
            category="cavity_qed",
            description="Achieve strong coupling regime between single atoms and optical cavity modes at room temperature",
            physics_challenge="Design a cavity QED system that maintains g > √(κγ) at 300K without cryogenic cooling",
            target_parameters={
                "coupling_strength": PhysicsParameter(
                    symbol="g", target="> 1000e6", units="Hz",
                    description="Atom-cavity coupling strength", weight=0.4
                ),
                "quality_factor": PhysicsParameter(
                    symbol="Q", target="> 1e7", units="",
                    description="Cavity quality factor", weight=0.3
                ),
                "cooperativity": PhysicsParameter(
                    symbol="C", target="> 1000", units="",
                    description="Cooperativity C = 4g²/(κγ)", weight=0.3
                )
            },
            constraints=[
                PhysicsConstraint(
                    name="room_temperature", 
                    description="System must operate at T = 300K",
                    constraint_type="experimental"
                ),
                PhysicsConstraint(
                    name="energy_conservation",
                    description="Must conserve energy in all processes",
                    constraint_type="fundamental"
                )
            ],
            background_context="Current cavity QED experiments require cryogenic cooling to achieve strong coupling. Room temperature operation would revolutionize quantum technologies.",
            keywords=["cavity QED", "strong coupling", "room temperature", "single atom"],
            difficulty_level="expert"
        ))
        
        # Quantum memory fidelity
        problems.append(AMOProblem(
            id="quantum_memory_fidelity",
            title="Ultra-High Fidelity Quantum Memory",
            category="quantum_memory",
            description="Design quantum memory system with >99.9% fidelity for 1ms storage time",
            physics_challenge="Achieve near-perfect quantum state storage and retrieval over millisecond timescales",
            target_parameters={
                "storage_fidelity": PhysicsParameter(
                    symbol="F_s", target="> 0.999", units="",
                    description="Storage fidelity", weight=0.4
                ),
                "retrieval_fidelity": PhysicsParameter(
                    symbol="F_r", target="> 0.999", units="",
                    description="Retrieval fidelity", weight=0.4
                ),
                "storage_time": PhysicsParameter(
                    symbol="τ", target="> 1e-3", units="s",
                    description="Storage time", weight=0.2
                )
            },
            difficulty_level="hard",
            keywords=["quantum memory", "fidelity", "storage time"]
        ))
        
        return problems