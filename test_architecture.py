#!/usr/bin/env python3
"""
Architecture test to verify clean code works without external dependencies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_dataclass_structure():
    """Test that our EvolutionStats dataclass works correctly"""
    print("Testing EvolutionStats dataclass...")
    
    try:
        # Read the evolution_controller.py file to check dataclass structure
        evolver_code = Path('src/evolution_controller.py').read_text()
        
        # Check that EvolutionStats is defined as a dataclass
        if '@dataclass\nclass EvolutionStats:' in evolver_code:
            print("✅ EvolutionStats is properly defined as dataclass")
        else:
            print("❌ EvolutionStats dataclass definition not found")
            return False
        
        # Check that it has the expected fields
        expected_fields = [
            'generation: int',
            'population_size: int', 
            'best_score: float',
            'average_score: float',
            'diversity_index: float',
            'new_solutions: int',
            'breakthrough_solutions: int',
            'convergence_rate: float'
        ]
        
        missing_fields = []
        for field in expected_fields:
            if field not in evolver_code:
                missing_fields.append(field)
        
        if not missing_fields:
            print("✅ EvolutionStats has all expected fields")
            return True
        else:
            print(f"❌ EvolutionStats missing fields: {missing_fields}")
            return False
            
    except Exception as e:
        print(f"❌ EvolutionStats test failed: {e}")
        return False

def test_class_names():
    """Test that class names have been updated correctly"""
    print("Testing class name updates...")
    
    try:
        # Read llm_interface.py file to check class names
        llm_code = Path('src/llm_interface.py').read_text()
        
        # Check that A4FInterface class exists
        if 'class A4FInterface:' in llm_code:
            print("✅ A4FInterface class found")
        else:
            print("❌ A4FInterface class not found")
            return False
        
        # Check that OpenRouterInterface class doesn't exist
        if 'class OpenRouterInterface:' in llm_code:
            print("❌ OpenRouterInterface still exists (should be removed)")
            return False
        else:
            print("✅ OpenRouterInterface correctly removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Class name test failed: {e}")
        return False

def test_evolution_controller_structure():
    """Test evolution controller clean structure"""
    print("Testing evolution controller structure...")
    
    try:
        # Check that problematic functions are removed
        evolver_code = Path('src/evolution_controller.py').read_text()
        
        # Check that create_evolution_stats function is removed
        if 'def create_evolution_stats(' in evolver_code:
            print("❌ create_evolution_stats function still exists (should be removed)")
            return False
        else:
            print("✅ create_evolution_stats function correctly removed")
        
        # Check that _safe_asdict method is removed
        if 'def _safe_asdict(' in evolver_code:
            print("❌ _safe_asdict method still exists (should be removed)")
            return False
        else:
            print("✅ _safe_asdict method correctly removed")
        
        # Check that A4FInterface is imported instead of OpenRouterInterface
        if 'from llm_interface import A4FInterface' in evolver_code:
            print("✅ A4FInterface correctly imported")
        else:
            print("❌ A4FInterface not imported")
            return False
        
        # Check that asdict is used properly
        if 'self.evolution_history.append(asdict(stats))' in evolver_code:
            print("✅ EvolutionStats properly serialized with asdict()")
        else:
            print("❌ EvolutionStats not properly serialized")
            return False
        
        # Check that debugging code is reduced
        debug_indicators = ['DEBUG:', 'traceback.print_exc()', 'import traceback', 'sys.exc_info()']
        debug_count = sum(1 for indicator in debug_indicators if indicator in evolver_code)
        
        if debug_count > 2:  # Allow minimal debugging
            print(f"❌ Too much debugging code still present ({debug_count} indicators)")
            return False
        else:
            print(f"✅ Debugging code cleaned up ({debug_count} indicators remaining)")
        
        return True
        
    except Exception as e:
        print(f"❌ Evolution controller test failed: {e}")
        return False

def test_main_py_structure():
    """Test main.py clean structure"""
    print("Testing main.py structure...")
    
    try:
        main_code = Path('src/main.py').read_text()
        
        # Check that A4FInterface is imported
        if 'from llm_interface import A4FInterface' in main_code:
            print("✅ A4FInterface correctly imported in main.py")
        else:
            print("❌ A4FInterface not imported in main.py")
            return False
        
        # Check that OpenRouterInterface is not imported
        if 'OpenRouterInterface' in main_code:
            print("❌ OpenRouterInterface still referenced in main.py")
            return False
        else:
            print("✅ OpenRouterInterface correctly removed from main.py")
        
        # Check for excessive debugging code
        if 'traceback.print_exception' in main_code:
            print("❌ Excessive debugging code still in main.py")
            return False
        else:
            print("✅ Debugging code cleaned up in main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Main.py test failed: {e}")
        return False

def main():
    """Run all architecture tests"""
    print("🧪 Running Coheron Architecture Tests")
    print("=" * 50)
    
    tests = [
        test_dataclass_structure,
        test_class_names,
        test_evolution_controller_structure,
        test_main_py_structure
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All architecture tests passed! Clean code restoration successful.")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Further cleanup needed.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)