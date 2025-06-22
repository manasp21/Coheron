#!/usr/bin/env python3
"""
Basic tests for the Coheron system
"""

import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all core modules can be imported"""
    try:
        from llm_interface import OpenRouterInterface, GenerationResult
        from evaluator import QuantumOpticsEvaluator, EvaluationResult
        from research_generator import QuantumResearchGenerator, ResearchPrompt
        from evolution_controller import QuantumResearchEvolver
        from database import ResearchDatabase
        from utils import setup_logging, validate_config
        print("✅ All imports successful")
        assert True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False

def test_config_validation():
    """Test configuration validation"""
    from utils import validate_config
    
    # Valid config
    valid_config = {
        'api': {'provider': 'openrouter', 'api_key': 'test', 'base_url': 'test'},
        'current_model': 'test_model',
        'models': {'test_model': {}},
        'evolution': {
            'population_size': 10,
            'max_generations': 20,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7
        },
        'evaluation': {}
    }
    
    is_valid, errors = validate_config(valid_config)
    assert is_valid, f"Valid config failed validation: {errors}"
    
    # Invalid config
    invalid_config = {'api': {}}
    is_valid, errors = validate_config(invalid_config)
    assert not is_valid, "Invalid config passed validation"
    assert len(errors) > 0, "No errors reported for invalid config"
    
    print("✅ Config validation tests passed")

def test_evaluator_initialization():
    """Test evaluator can be initialized"""
    try:
        from evaluator import QuantumOpticsEvaluator
        evaluator = QuantumOpticsEvaluator()
        assert evaluator is not None
        print("✅ Evaluator initialization successful")
    except Exception as e:
        print(f"❌ Evaluator initialization failed: {e}")
        assert False

def test_research_generator():
    """Test research generator functionality"""
    try:
        from research_generator import QuantumResearchGenerator
        generator = QuantumResearchGenerator()
        
        # Test getting available categories
        stats = generator.get_category_statistics()
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        print("✅ Research generator tests passed")
    except Exception as e:
        print(f"❌ Research generator test failed: {e}")
        assert False

def test_database_creation():
    """Test database can be created"""
    try:
        from database import ResearchDatabase
        
        # Use a test database
        test_db_path = "test_research.db"
        db = ResearchDatabase(test_db_path)
        
        # Test database stats
        stats = db.get_database_stats()
        assert isinstance(stats, dict)
        
        # Cleanup
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
            
        print("✅ Database creation test passed")
    except Exception as e:
        print(f"❌ Database creation test failed: {e}")
        assert False

def test_physics_parameter_extraction():
    """Test physics parameter extraction"""
    try:
        from utils import extract_physics_parameters, calculate_physics_consistency
        
        test_text = """
        The system has a coupling strength g = 1e6 Hz and cavity decay κ = 1e4 Hz.
        The frequency is 2.84e14 Hz and power is 100 mW.
        """
        
        params = extract_physics_parameters(test_text)
        assert 'coupling_strength' in params
        assert 'frequency' in params
        assert params['coupling_strength'] == 1e6
        
        # Test physics consistency
        consistency = calculate_physics_consistency(params)
        assert 'valid' in consistency
        assert 'warnings' in consistency
        
        print("✅ Physics parameter extraction tests passed")
    except Exception as e:
        print(f"❌ Physics parameter extraction test failed: {e}")
        assert False

def test_evaluation_system():
    """Test the evaluation system with sample content"""
    try:
        from evaluator import QuantumOpticsEvaluator
        
        evaluator = QuantumOpticsEvaluator()
        
        test_solution = {
            'content': '''
            Design a cavity QED system for strong coupling between single atom and optical mode.
            The system uses a Fabry-Perot cavity with length L = 100 μm and finesse F = 10000.
            The coupling strength is g = 1e6 Hz, with cavity decay κ = 1e4 Hz.
            The cooperativity is C = 4g²/(κγ) = 400, satisfying strong coupling g > (κ,γ)/2.
            ''',
            'category': 'cavity_qed',
            'title': 'Test Cavity QED System',
            'description': 'A test system for evaluation'
        }
        
        result = evaluator.evaluate_research(test_solution)
        
        # Check result structure
        assert hasattr(result, 'total_score')
        assert hasattr(result, 'feasibility')
        assert hasattr(result, 'mathematics')
        assert hasattr(result, 'novelty')
        assert hasattr(result, 'performance')
        
        # Check score ranges
        assert 0 <= result.total_score <= 1
        assert 0 <= result.feasibility <= 1
        assert 0 <= result.mathematics <= 1
        assert 0 <= result.novelty <= 1
        assert 0 <= result.performance <= 1
        
        print(f"✅ Evaluation test passed (score: {result.total_score:.3f})")
        
    except Exception as e:
        print(f"❌ Evaluation system test failed: {e}")
        assert False

if __name__ == "__main__":
    """Run tests manually"""
    print("🧪 Running Coheron Tests\n")
    
    tests = [
        test_imports,
        test_config_validation,
        test_evaluator_initialization,
        test_research_generator,
        test_database_creation,
        test_physics_parameter_extraction,
        test_evaluation_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)