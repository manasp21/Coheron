#!/usr/bin/env python3
"""
Test demo mode without external dependencies
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_demo_test():
    """Test evolution in demo mode without external dependencies"""
    print("🧪 Testing Coheron Demo Mode")
    print("=" * 50)
    
    try:
        # Mock the missing modules for demo mode
        import types
        import random
        
        # Create mock modules
        openai_mock = types.ModuleType('openai')
        openai_mock.OpenAI = lambda **kwargs: None
        sys.modules['openai'] = openai_mock
        
        dotenv_mock = types.ModuleType('dotenv')
        dotenv_mock.load_dotenv = lambda: None
        sys.modules['dotenv'] = dotenv_mock
        
        # Now import our modules
        from evolution_controller import QuantumResearchEvolver
        
        print("✅ Successfully imported QuantumResearchEvolver")
        
        # Initialize in demo mode
        evolver = QuantumResearchEvolver(demo_mode=True)
        print("✅ Initialized evolver in demo mode")
        
        # Set small population for testing
        evolver.population_size = 3
        
        # Run a short evolution
        print("🧬 Starting evolution...")
        start_time = time.time()
        
        best_solutions = evolver.evolve_research(2)  # Just 2 generations
        
        evolution_time = time.time() - start_time
        
        print(f"✅ Evolution completed in {evolution_time:.2f}s")
        print(f"📊 Generated {len(best_solutions)} solutions")
        
        if best_solutions:
            best = best_solutions[0]
            print(f"🏆 Best solution:")
            print(f"   Title: {best.title}")
            print(f"   Category: {best.category}")
            print(f"   Score: {best.evaluation_result.total_score:.3f}")
            print(f"   Generation: {best.generation}")
        
        # Print evolution summary
        summary = evolver.get_evolution_summary()
        if summary:
            print(f"\n📈 Evolution Summary:")
            print(f"   Total generations: {summary.get('total_generations', 0)}")
            print(f"   Best score: {summary.get('best_score', 0):.3f}")
            print(f"   Breakthroughs: {summary.get('total_breakthroughs', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run demo test"""
    success = run_demo_test()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Demo mode works perfectly!")
        print("The clean architecture restoration was successful.")
    else:
        print("❌ Demo mode failed.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)