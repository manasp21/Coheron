#!/usr/bin/env python3
"""
Coheron Results Viewer
View your quantum optics research results
"""

import json
import sys
from pathlib import Path

def view_results(results_dir="results"):
    """Display results from the specified directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory '{results_dir}' not found!")
        print("Run Coheron first to generate results.")
        return False
    
    print("="*60)
    print("🔬 COHERON RESEARCH RESULTS VIEWER")
    print("="*60)
    
    # Check for evolution state
    evolution_file = results_path / "evolution_state.json"
    if evolution_file.exists():
        try:
            with open(evolution_file, 'r', encoding='utf-8') as f:
                evolution_data = json.load(f)
            
            print(f"\n📊 EVOLUTION SUMMARY:")
            print(f"   Generations completed: {evolution_data.get('current_generation', 0)}")
            
            summary = evolution_data.get('summary', {})
            print(f"   Best score achieved: {summary.get('best_score', 0):.3f}")
            print(f"   Total breakthroughs: {summary.get('total_breakthroughs', 0)}")
            print(f"   Model used: {summary.get('model_used', 'unknown')}")
            
            # Show top solutions
            best_solutions = evolution_data.get('best_solutions', [])[:3]
            if best_solutions:
                print(f"\n🏆 TOP RESEARCH SOLUTIONS:")
                print("-" * 40)
                for i, sol in enumerate(best_solutions):
                    eval_result = sol.get('evaluation_result', {})
                    print(f"\n{i+1}. Score: {eval_result.get('total_score', 0):.3f}")
                    print(f"   Title: {sol.get('title', 'Unknown')}")
                    print(f"   Category: {sol.get('category', 'unknown').replace('_', ' ').title()}")
                    print(f"   Generation: {sol.get('generation', 0)}")
                    
                    # Show content preview
                    content = sol.get('content', '')
                    if content:
                        preview = content[:150] + "..." if len(content) > 150 else content
                        print(f"   Preview: {preview}")
            
            # Show evolution history
            history = evolution_data.get('evolution_history', [])
            if len(history) > 1:
                print(f"\n📈 EVOLUTION PROGRESS:")
                print("-" * 30)
                for stats in history[-5:]:  # Show last 5 generations
                    gen = stats.get('generation', 0)
                    best = stats.get('best_score', 0)
                    avg = stats.get('average_score', 0)
                    div = stats.get('diversity_index', 0)
                    print(f"   Gen {gen}: Best={best:.3f}, Avg={avg:.3f}, Diversity={div:.3f}")
            
        except Exception as e:
            print(f"Error reading evolution data: {e}")
    
    # Check for research report
    report_file = results_path / "research_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            print(f"\n📄 RESEARCH REPORT:")
            timestamp = report_data.get('timestamp', 'unknown')
            print(f"   Generated: {timestamp}")
            
            analysis = report_data.get('analysis', {})
            if analysis:
                print(f"   Convergence rate: {analysis.get('convergence_rate', 0):.3f}")
                print(f"   Diversity trend: {analysis.get('diversity_trend', 'unknown')}")
        
        except Exception as e:
            print(f"Error reading report: {e}")
    
    # List all files in results directory
    print(f"\n📁 FILES IN RESULTS DIRECTORY:")
    try:
        for file_path in results_path.iterdir():
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"   📄 {file_path.name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    print(f"\n📂 Full path: {results_path.absolute()}")
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results"
    
    success = view_results(results_dir)
    
    if not success:
        print("\nUsage: python view_results.py [results_directory]")
        print("Example: python view_results.py results")

if __name__ == "__main__":
    main()