#!/usr/bin/env python3
"""
Quick test to verify the research generator fix
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_research_generator():
    """Test that research generator doesn't crash on concept extraction"""
    print("Testing research generator...")
    
    try:
        from research_generator import QuantumResearchGenerator
        
        # Create generator
        generator = QuantumResearchGenerator()
        
        # Test extract_key_concepts with minimal solution
        test_solution = {
            'id': 'test_001',
            'content': 'Test cavity QED system',
            'category': 'cavity_qed',
            'title': 'Test Solution',
            'description': 'A test quantum optical system',
            'generation': 0
        }
        
        # This should not crash
        concepts = generator._extract_key_concepts(test_solution)
        print(f"✅ Key concepts extracted: {concepts}")
        
        # Test crossover that was causing the 'approach_1' error
        solution1 = test_solution.copy()
        solution2 = {
            'id': 'test_002',
            'content': 'Test squeezed light system',
            'category': 'squeezed_light',
            'title': 'Squeezed Light Test',
            'description': 'A test squeezed light system',
            'generation': 0
        }
        
        # This should also not crash
        crossover_prompt = generator.crossover_solutions(solution1, solution2)
        print(f"✅ Crossover prompt generated: {crossover_prompt.content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Research generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test"""
    print("🧪 Testing Research Generator Fix")
    print("=" * 40)
    
    success = test_research_generator()
    
    print("=" * 40)
    if success:
        print("✅ Fix successful! Research generator should work now.")
    else:
        print("❌ Fix failed. Issue still exists.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)