import re
import numpy as np
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging for the quantum research system"""
    
    # Create logger
    logger = logging.getLogger('QuantumOpticsAI')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration completeness and correctness"""
    errors = []
    
    # Required top-level keys
    required_keys = ['api', 'current_model', 'models', 'evolution', 'evaluation']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    # API configuration validation
    if 'api' in config:
        api_config = config['api']
        required_api_keys = ['provider', 'api_key', 'base_url']
        for key in required_api_keys:
            if key not in api_config:
                errors.append(f"Missing API config key: {key}")
    
    # Model validation
    if 'current_model' in config and 'models' in config:
        current_model = config['current_model']
        if current_model not in config['models']:
            errors.append(f"Current model '{current_model}' not found in models configuration")
    
    # Evolution parameters validation
    if 'evolution' in config:
        evolution = config['evolution']
        required_evolution = ['population_size', 'max_generations', 'mutation_rate', 'crossover_rate']
        for key in required_evolution:
            if key not in evolution:
                errors.append(f"Missing evolution parameter: {key}")
            elif not isinstance(evolution[key], (int, float)):
                errors.append(f"Evolution parameter '{key}' must be numeric")
    
    return len(errors) == 0, errors

def extract_physics_parameters(text: str) -> Dict[str, float]:
    """Extract physics parameters with units from text"""
    parameters = {}
    
    # Common physics parameter patterns
    patterns = {
        'frequency': [
            r'(\d+\.?\d*)\s*(?:Hz|hz|hertz)',
            r'(\d+\.?\d*)\s*(?:GHz|ghz|gigahertz)',
            r'(\d+\.?\d*)\s*(?:MHz|mhz|megahertz)',
            r'(\d+\.?\d*)\s*(?:kHz|khz|kilohertz)',
            r'(\d+\.?\d*)\s*(?:THz|thz|terahertz)'
        ],
        'coupling_strength': [
            r'g\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'coupling.*?(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'rabi.*?(\d+\.?\d*(?:e[+-]?\d+)?)'
        ],
        'decay_rate': [
            r'κ\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'kappa\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'γ\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'gamma\s*=\s*(\d+\.?\d*(?:e[+-]?\d+)?)',
            r'decay.*?(\d+\.?\d*(?:e[+-]?\d+)?)'
        ],
        'power': [
            r'(\d+\.?\d*)\s*(?:W|watt|watts)',
            r'(\d+\.?\d*)\s*(?:mW|milliwatt|milliwatts)',
            r'(\d+\.?\d*)\s*(?:μW|microwatt|microwatts)',
            r'(\d+\.?\d*)\s*(?:nW|nanowatt|nanowatts)'
        ],
        'wavelength': [
            r'(\d+\.?\d*)\s*(?:nm|nanometer|nanometers)',
            r'(\d+\.?\d*)\s*(?:μm|micrometer|micrometers)',
            r'(\d+\.?\d*)\s*(?:mm|millimeter|millimeters)'
        ],
        'temperature': [
            r'(\d+\.?\d*)\s*(?:K|kelvin)',
            r'(\d+\.?\d*)\s*(?:mK|millikelvin)',
            r'(\d+\.?\d*)\s*(?:μK|microkelvin)'
        ]
    }
    
    for param_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the first match and convert to float
                    value = float(matches[0])
                    parameters[param_type] = value
                    break  # Use first successful match for each parameter type
                except ValueError:
                    continue
    
    return parameters

def calculate_physics_consistency(parameters: Dict[str, float]) -> Dict[str, Any]:
    """Check physics consistency of extracted parameters"""
    consistency = {
        'valid': True,
        'warnings': [],
        'dimensionless_ratios': {},
        'physical_regimes': []
    }
    
    # Extract key parameters
    g = parameters.get('coupling_strength', 0)
    kappa = parameters.get('decay_rate', 0)
    frequency = parameters.get('frequency', 0)
    
    if g and kappa:
        # Calculate cooperativity (assuming atomic decay ~ kappa for simplicity)
        cooperativity = 4 * g**2 / (kappa**2)
        consistency['dimensionless_ratios']['cooperativity'] = cooperativity
        
        # Determine coupling regime
        if g > kappa:
            consistency['physical_regimes'].append('strong_coupling')
        elif g > kappa / 2:
            consistency['physical_regimes'].append('intermediate_coupling')
        else:
            consistency['physical_regimes'].append('weak_coupling')
            consistency['warnings'].append('System operates in weak coupling regime')
    
    # Check for unrealistic values
    if g > 1e12:  # 1 THz coupling seems unrealistic
        consistency['valid'] = False
        consistency['warnings'].append(f'Unrealistic coupling strength: {g:.2e} Hz')
    
    if frequency > 1e16:  # Beyond optical frequencies
        consistency['warnings'].append(f'Very high frequency: {frequency:.2e} Hz')
    
    return consistency

def generate_solution_hash(content: str) -> str:
    """Generate unique hash for solution content"""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def calculate_content_similarity(content1: str, content2: str) -> float:
    """Calculate semantic similarity between two research contents"""
    # Simple approach: normalize and calculate word overlap
    words1 = set(re.findall(r'\b\w+\b', content1.lower()))
    words2 = set(re.findall(r'\b\w+\b', content2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def extract_key_concepts(text: str) -> List[str]:
    """Extract key quantum optics concepts from text"""
    # Physics keywords to look for
    quantum_keywords = [
        'quantum', 'qubit', 'superposition', 'entanglement', 'coherence',
        'decoherence', 'fidelity', 'gate', 'measurement', 'state'
    ]
    
    optics_keywords = [
        'photon', 'laser', 'cavity', 'mirror', 'beam', 'interference',
        'reflection', 'transmission', 'polarization', 'wavelength'
    ]
    
    physics_keywords = [
        'coupling', 'frequency', 'amplitude', 'phase', 'energy',
        'momentum', 'angular', 'orbital', 'spin', 'magnetic'
    ]
    
    all_keywords = quantum_keywords + optics_keywords + physics_keywords
    
    # Find keywords in text
    found_concepts = []
    text_lower = text.lower()
    
    for keyword in all_keywords:
        if keyword in text_lower:
            found_concepts.append(keyword)
    
    # Also extract compound terms
    compound_patterns = [
        r'cavity qed', r'quantum dot', r'single photon', r'photon blockade',
        r'rabi frequency', r'vacuum rabi', r'jaynes.?cummings', r'spin squeezing',
        r'quantum metrology', r'heisenberg limit', r'shot noise', r'fisher information'
    ]
    
    for pattern in compound_patterns:
        matches = re.findall(pattern, text_lower)
        found_concepts.extend(matches)
    
    return list(set(found_concepts))  # Remove duplicates

def format_scientific_notation(value: float, precision: int = 2) -> str:
    """Format numbers in scientific notation for display"""
    if abs(value) >= 1000 or abs(value) < 0.01:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"

def create_evolution_plot(evolution_data: List[Dict], save_path: Optional[str] = None) -> None:
    """Create evolution progress visualization"""
    if not evolution_data:
        return
    
    generations = [d['generation'] for d in evolution_data]
    best_scores = [d['best_score'] for d in evolution_data]
    avg_scores = [d['average_score'] for d in evolution_data]
    diversity = [d['diversity_index'] for d in evolution_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Score evolution
    ax1.plot(generations, best_scores, 'b-', label='Best Score', linewidth=2)
    ax1.plot(generations, avg_scores, 'r--', label='Average Score', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('Evolution Progress - Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Diversity evolution
    ax2.plot(generations, diversity, 'g-', label='Diversity Index', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity Index')
    ax2.set_title('Population Diversity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def create_category_analysis(solutions_data: List[Dict], save_path: Optional[str] = None) -> None:
    """Create category-wise analysis visualization"""
    if not solutions_data:
        return
    
    categories = [s.get('category', 'unknown') for s in solutions_data]
    scores = [s.get('total_score', 0) for s in solutions_data]
    
    # Create category performance analysis
    category_stats = {}
    for cat, score in zip(categories, scores):
        if cat not in category_stats:
            category_stats[cat] = []
        category_stats[cat].append(score)
    
    # Calculate statistics
    cat_names = list(category_stats.keys())
    cat_means = [np.mean(category_stats[cat]) for cat in cat_names]
    cat_stds = [np.std(category_stats[cat]) for cat in cat_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of average scores
    bars = ax1.bar(cat_names, cat_means, yerr=cat_stds, capsize=5)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Average Performance by Category')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean in zip(bars, cat_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom')
    
    # Box plot for distribution
    score_lists = [category_stats[cat] for cat in cat_names]
    ax2.boxplot(score_lists, labels=cat_names)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Score Distribution')
    ax2.set_title('Score Distribution by Category')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def export_results_report(
    evolution_data: List[Dict],
    best_solutions: List[Dict],
    breakthroughs: List[Dict],
    output_path: str
) -> None:
    """Generate comprehensive results report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_generations': len(evolution_data),
            'best_score': max(d['best_score'] for d in evolution_data) if evolution_data else 0,
            'total_solutions': len(best_solutions),
            'breakthrough_count': len(breakthroughs),
            'categories_explored': len(set(s.get('category', '') for s in best_solutions))
        },
        'evolution_progress': evolution_data,
        'top_solutions': best_solutions[:10],  # Top 10 solutions
        'breakthrough_discoveries': breakthroughs,
        'analysis': {
            'convergence_rate': _calculate_convergence_rate(evolution_data),
            'diversity_trend': _calculate_diversity_trend(evolution_data),
            'category_performance': _analyze_category_performance(best_solutions)
        }
    }
    
    # Save report
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

def _calculate_convergence_rate(evolution_data: List[Dict]) -> float:
    """Calculate overall convergence rate"""
    if len(evolution_data) < 2:
        return 0.0
    
    first_score = evolution_data[0]['best_score']
    last_score = evolution_data[-1]['best_score']
    generations = len(evolution_data)
    
    return (last_score - first_score) / generations if generations > 0 else 0.0

def _calculate_diversity_trend(evolution_data: List[Dict]) -> str:
    """Analyze diversity trend over generations"""
    if len(evolution_data) < 3:
        return "insufficient_data"
    
    diversities = [d['diversity_index'] for d in evolution_data]
    
    # Simple trend analysis
    first_half = np.mean(diversities[:len(diversities)//2])
    second_half = np.mean(diversities[len(diversities)//2:])
    
    if second_half > first_half * 1.1:
        return "increasing"
    elif second_half < first_half * 0.9:
        return "decreasing"
    else:
        return "stable"

def _analyze_category_performance(solutions_data: List[Dict]) -> Dict[str, Any]:
    """Analyze performance by category"""
    category_scores = {}
    
    for solution in solutions_data:
        category = solution.get('category', 'unknown')
        score = solution.get('total_score', 0)
        
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
    
    analysis = {}
    for category, scores in category_scores.items():
        analysis[category] = {
            'count': len(scores),
            'average_score': np.mean(scores),
            'max_score': max(scores),
            'std_dev': np.std(scores)
        }
    
    return analysis

def validate_quantum_physics(content: str) -> Dict[str, Any]:
    """Validate quantum physics principles in content"""
    validation = {
        'valid': True,
        'violations': [],
        'principles_mentioned': [],
        'confidence': 0.5
    }
    
    # Check for fundamental principles
    principles = {
        'uncertainty_principle': ['uncertainty', 'heisenberg', 'Δx', 'Δp'],
        'superposition': ['superposition', 'coherent state', 'linear combination'],
        'entanglement': ['entanglement', 'bell state', 'epr', 'non-local'],
        'measurement': ['measurement', 'collapse', 'observable', 'eigenstate'],
        'conservation': ['conservation', 'energy conservation', 'momentum conservation']
    }
    
    content_lower = content.lower()
    
    for principle, keywords in principles.items():
        if any(keyword.lower() in content_lower for keyword in keywords):
            validation['principles_mentioned'].append(principle)
    
    # Check for potential violations
    violations = [
        ('faster than light', 'faster.*light|superluminal'),
        ('perpetual motion', 'perpetual.*motion|infinite.*energy'),
        ('uncertainty violation', 'violate.*uncertainty|bypass.*uncertainty'),
        ('information paradox', 'clone.*quantum|copy.*quantum.*state')
    ]
    
    for violation_name, pattern in violations:
        if re.search(pattern, content_lower):
            validation['violations'].append(violation_name)
            validation['valid'] = False
    
    # Calculate confidence based on physics content
    physics_terms = len(validation['principles_mentioned'])
    total_terms = len(principles)
    validation['confidence'] = min(0.9, 0.3 + (physics_terms / total_terms) * 0.6)
    
    return validation

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default return value"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def load_json_safe(filepath: str, default: Any = None) -> Any:
    """Safely load JSON file with default fallback"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load {filepath}: {e}")
        return default if default is not None else {}

def save_json_safe(data: Any, filepath: str) -> bool:
    """Safely save data to JSON file"""
    try:
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Could not save to {filepath}: {e}")
        return False

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix