#!/usr/bin/env python3
"""
Coheron - Main Application

An evolutionary AI system for breakthrough quantum optics research discovery.
Inspired by AlphaEvolve with specialized quantum physics evaluation.
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evolution_controller import QuantumResearchEvolver
from llm_interface import OpenRouterInterface
from evaluator import QuantumOpticsEvaluator
from database import ResearchDatabase
from utils import (
    setup_logging, validate_config, create_evolution_plot, 
    create_category_analysis, export_results_report, format_duration
)

def main():
    """Main application entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_file = "logs/quantum_research.log" if args.log_file else None
    logger = setup_logging(args.log_level, log_file)
    
    logger.info("🚀 Starting Coheron")
    logger.info(f"Command: {' '.join(sys.argv)}")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            sys.exit(1)
            
        # Execute command
        if args.command == 'evolve':
            run_evolution(args, logger)
        elif args.command == 'evaluate':
            run_evaluation(args, logger)
        elif args.command == 'analyze':
            run_analysis(args, logger)
        elif args.command == 'test':
            run_test(args, logger)
        elif args.command == 'benchmark':
            run_benchmark(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description='Coheron - Evolutionary Discovery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evolve --generations 30 --model deepseek/deepseek-r1
  %(prog)s evaluate --content "research.txt" --category cavity_qed
  %(prog)s analyze --database --plots
  %(prog)s test --model anthropic/claude-3.5-sonnet
  %(prog)s benchmark --compare-models
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evolution command
    evolve_parser = subparsers.add_parser('evolve', help='Run research evolution')
    evolve_parser.add_argument('--generations', type=int, default=20, 
                              help='Number of evolution generations (default: 20)')
    evolve_parser.add_argument('--population', type=int, default=10,
                              help='Population size (default: 10)')
    evolve_parser.add_argument('--model', type=str, 
                              help='Override model (e.g., anthropic/claude-3.5-sonnet)')
    evolve_parser.add_argument('--category', type=str,
                              help='Focus on specific category (cavity_qed, squeezed_light, etc.)')
    evolve_parser.add_argument('--save-interval', type=int, default=5,
                              help='Save progress every N generations (default: 5)')
    evolve_parser.add_argument('--output-dir', type=str, default='results',
                              help='Output directory for results (default: results)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate research content')
    eval_parser.add_argument('--content', type=str, required=True,
                            help='Research content (text or file path)')
    eval_parser.add_argument('--category', type=str, default='general',
                            help='Research category for evaluation')
    eval_parser.add_argument('--detailed', action='store_true',
                            help='Show detailed evaluation breakdown')
    eval_parser.add_argument('--save-result', type=str,
                            help='Save evaluation result to file')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Analyze research database')
    analysis_parser.add_argument('--database', action='store_true',
                                help='Analyze database contents')
    analysis_parser.add_argument('--evolution', type=str,
                                help='Analyze evolution from file')
    analysis_parser.add_argument('--plots', action='store_true',
                                help='Generate visualization plots')
    analysis_parser.add_argument('--export', type=str,
                                help='Export analysis to file')
    analysis_parser.add_argument('--category', type=str,
                                help='Focus analysis on specific category')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test system components')
    test_parser.add_argument('--model', type=str,
                            help='Test specific model')
    test_parser.add_argument('--evaluator', action='store_true',
                            help='Test evaluation system')
    test_parser.add_argument('--database', action='store_true',
                            help='Test database operations')
    test_parser.add_argument('--full', action='store_true',
                            help='Run full system test')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--compare-models', action='store_true',
                                 help='Compare multiple models')
    benchmark_parser.add_argument('--physics-accuracy', action='store_true',
                                 help='Test physics accuracy')
    benchmark_parser.add_argument('--speed', action='store_true',
                                 help='Benchmark generation speed')
    benchmark_parser.add_argument('--models', nargs='+',
                                 help='Specific models to benchmark')
    
    # Global arguments
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', action='store_true',
                       help='Enable file logging')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress non-error output')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    return parser

def validate_environment() -> bool:
    """Validate environment and dependencies"""
    
    # Check for OpenRouter API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("   Please set your OpenRouter API key:")
        print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return False
    
    # Check required directories
    required_dirs = ['config', 'data', 'logs']
    for directory in required_dirs:
        Path(directory).mkdir(exist_ok=True)
    
    # Check configuration files
    config_files = ['config/config.yaml', 'config/prompts.yaml', 'config/evaluation_criteria.yaml']
    missing_configs = [f for f in config_files if not Path(f).exists()]
    
    if missing_configs:
        print(f"❌ Error: Missing configuration files: {missing_configs}")
        return False
    
    return True

def run_evolution(args: argparse.Namespace, logger) -> None:
    """Run the research evolution process"""
    
    logger.info(f"🧬 Starting evolution with {args.generations} generations")
    
    # Initialize evolution controller
    evolver = QuantumResearchEvolver(args.config)
    
    # Switch model if specified
    if args.model:
        try:
            evolver.llm.switch_model(args.model)
            logger.info(f"Switched to model: {args.model}")
        except ValueError as e:
            logger.error(f"Model switch failed: {e}")
            return
    
    # Override population size if specified
    if args.population:
        evolver.population_size = args.population
        
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize database for tracking
    db = ResearchDatabase()
    
    logger.info(f"Model: {evolver.llm.current_model}")
    logger.info(f"Population: {evolver.population_size}")
    logger.info(f"Category focus: {args.category or 'All categories'}")
    
    start_time = time.time()
    
    try:
        # Run evolution
        best_solutions = evolver.evolve_research(args.generations)
        
        # Save results
        evolution_summary = evolver.get_evolution_summary()
        
        # Store results in database
        for solution in best_solutions[:10]:  # Store top 10
            solution_dict = {
                'id': solution.id,
                'content': solution.content,
                'category': solution.category,
                'title': solution.title,
                'description': solution.description,
                'generation': solution.generation,
                'parent_ids': solution.parent_ids,
                'mutation_type': solution.mutation_type,
                'evaluation_result': {
                    'total_score': solution.evaluation_result.total_score,
                    'feasibility': solution.evaluation_result.feasibility,
                    'mathematics': solution.evaluation_result.mathematics,
                    'novelty': solution.evaluation_result.novelty,
                    'performance': solution.evaluation_result.performance,
                    'details': solution.evaluation_result.details
                },
                'generation_metadata': solution.generation_metadata,
                'timestamp': solution.timestamp
            }
            db.store_solution(solution_dict)
        
        # Save evolution state
        evolver.save_evolution_state(output_dir / 'evolution_state.json')
        
        # Generate plots if requested
        if not args.quiet:
            create_evolution_plot(
                evolver.evolution_history, 
                output_dir / 'evolution_progress.png'
            )
            
            create_category_analysis(
                [vars(sol) for sol in best_solutions],
                output_dir / 'category_analysis.png'
            )
        
        # Generate report
        export_results_report(
            [vars(stats) for stats in evolver.evolution_history],
            [vars(sol) for sol in best_solutions[:10]],
            [vars(sol) for sol in evolver.breakthrough_solutions],
            output_dir / 'research_report.json'
        )
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"✅ Evolution completed in {format_duration(total_time)}")
        logger.info(f"📊 Best score: {best_solutions[0].evaluation_result.total_score:.3f}")
        logger.info(f"🏆 Breakthroughs: {len(evolver.breakthrough_solutions)}")
        logger.info(f"💾 Results saved to: {output_dir}")
        
        # Display top results
        if not args.quiet:
            print(f"\n🏆 Top 3 Research Solutions:")
            for i, solution in enumerate(best_solutions[:3]):
                print(f"{i+1}. Score: {solution.evaluation_result.total_score:.3f}")
                print(f"   Title: {solution.title}")
                print(f"   Category: {solution.category}")
                print(f"   Generation: {solution.generation}")
                print()
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise

def run_evaluation(args: argparse.Namespace, logger) -> None:
    """Run evaluation on provided content"""
    
    logger.info("🔍 Running research evaluation")
    
    # Load content
    if Path(args.content).exists():
        with open(args.content, 'r') as f:
            content = f.read()
        logger.info(f"Loaded content from: {args.content}")
    else:
        content = args.content
        logger.info("Using provided content string")
    
    # Initialize evaluator
    evaluator = QuantumOpticsEvaluator()
    
    # Prepare solution data
    solution_data = {
        'content': content,
        'category': args.category,
        'title': 'Evaluation Target',
        'description': content[:200] + "..." if len(content) > 200 else content
    }
    
    # Evaluate
    start_time = time.time()
    result = evaluator.evaluate_research(solution_data)
    eval_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 Evaluation Results ({eval_time:.2f}s)")
    print(f"Overall Score: {result.total_score:.3f}")
    print(f"  Feasibility:  {result.feasibility:.3f}")
    print(f"  Mathematics:  {result.mathematics:.3f}")
    print(f"  Novelty:      {result.novelty:.3f}")
    print(f"  Performance:  {result.performance:.3f}")
    
    if result.benchmarks_matched:
        print(f"\n✅ Benchmarks matched: {', '.join(result.benchmarks_matched)}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    if args.detailed:
        print(f"\n📋 Detailed Analysis:")
        for component, details in result.details.items():
            print(f"  {component}:")
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.3f}")
                elif isinstance(value, list) and len(value) < 10:
                    print(f"    {key}: {', '.join(map(str, value))}")
    
    # Save result if requested
    if args.save_result:
        result_data = {
            'evaluation_result': vars(result),
            'input_content': content,
            'category': args.category,
            'timestamp': time.time()
        }
        
        with open(args.save_result, 'w') as f:
            import json
            json.dump(result_data, f, indent=2)
        
        logger.info(f"💾 Result saved to: {args.save_result}")

def run_analysis(args: argparse.Namespace, logger) -> None:
    """Run analysis on research data"""
    
    logger.info("📈 Running research analysis")
    
    if args.database:
        # Analyze database
        db = ResearchDatabase()
        
        print("📊 Database Statistics:")
        stats = db.get_database_stats()
        for table, count in stats.items():
            print(f"  {table}: {count} records")
        
        print("\n🏆 Best Solutions:")
        best_solutions = db.get_best_solutions(5, args.category)
        for i, solution in enumerate(best_solutions):
            print(f"{i+1}. Score: {solution['total_score']:.3f} - {solution['title']}")
        
        print("\n📈 Category Performance:")
        category_stats = db.get_category_statistics()
        for stat in category_stats:
            print(f"  {stat['category']}: {stat['average_score']:.3f} avg ({stat['total_solutions']} solutions)")
        
        if args.plots:
            # Generate database analysis plots
            evolution_history = db.get_evolution_history()
            if evolution_history:
                create_evolution_plot(evolution_history, 'analysis_evolution.png')
                logger.info("📊 Evolution plot saved to analysis_evolution.png")
        
        if args.export:
            success = db.export_research_data(args.export)
            if success:
                logger.info(f"💾 Database exported to: {args.export}")
            else:
                logger.error("Export failed")

def run_test(args: argparse.Namespace, logger) -> None:
    """Run system tests"""
    
    logger.info("🧪 Running system tests")
    
    tests_passed = 0
    tests_total = 0
    
    if args.model or args.full:
        # Test model interface
        print("Testing LLM interface...")
        tests_total += 1
        try:
            llm = OpenRouterInterface(args.config)
            if args.model:
                llm.switch_model(args.model)
            
            # Test generation
            test_prompt = "Design a simple cavity QED system."
            result = llm.generate_research(test_prompt)
            
            if result.content and len(result.content) > 50:
                print("✅ LLM interface test passed")
                tests_passed += 1
            else:
                print("❌ LLM interface test failed: insufficient content")
                
        except Exception as e:
            print(f"❌ LLM interface test failed: {e}")
    
    if args.evaluator or args.full:
        # Test evaluator
        print("Testing evaluator...")
        tests_total += 1
        try:
            evaluator = QuantumOpticsEvaluator()
            
            test_solution = {
                'content': 'Test cavity QED system with coupling strength g = 1e6 Hz',
                'category': 'cavity_qed',
                'title': 'Test Solution'
            }
            
            result = evaluator.evaluate_research(test_solution)
            
            if 0 <= result.total_score <= 1:
                print("✅ Evaluator test passed")
                tests_passed += 1
            else:
                print("❌ Evaluator test failed: invalid score range")
                
        except Exception as e:
            print(f"❌ Evaluator test failed: {e}")
    
    if args.database or args.full:
        # Test database
        print("Testing database...")
        tests_total += 1
        try:
            db = ResearchDatabase("test_db.db")
            
            # Test solution storage
            test_solution_data = {
                'id': 'test_001',
                'content': 'Test content',
                'category': 'test',
                'title': 'Test Solution',
                'generation': 0,
                'parent_ids': [],
                'mutation_type': 'test',
                'evaluation_result': {'total_score': 0.5},
                'generation_metadata': {},
                'timestamp': time.time()
            }
            
            success = db.store_solution(test_solution_data)
            
            if success:
                print("✅ Database test passed")
                tests_passed += 1
                # Cleanup
                Path("test_db.db").unlink(missing_ok=True)
            else:
                print("❌ Database test failed")
                
        except Exception as e:
            print(f"❌ Database test failed: {e}")
    
    print(f"\n🧪 Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        logger.info("✅ All tests passed")
    else:
        logger.warning(f"⚠️  {tests_total - tests_passed} tests failed")

def run_benchmark(args: argparse.Namespace, logger) -> None:
    """Run system benchmarks"""
    
    logger.info("⚡ Running benchmarks")
    
    if args.compare_models:
        # Compare multiple models
        models_to_test = args.models or [
            'deepseek/deepseek-r1-0528:free',
            'anthropic/claude-3.5-sonnet',
            'openai/gpt-4-turbo'
        ]
        
        print("🏁 Model Comparison Benchmark")
        print(f"Testing models: {', '.join(models_to_test)}")
        
        test_prompt = "Design a cavity QED system for strong coupling between single atom and optical mode."
        
        for model in models_to_test:
            print(f"\nTesting {model}...")
            try:
                llm = OpenRouterInterface(args.config)
                llm.switch_model(model)
                
                start_time = time.time()
                result = llm.generate_research(test_prompt)
                generation_time = time.time() - start_time
                
                print(f"  ⏱️  Generation time: {generation_time:.2f}s")
                print(f"  📝 Content length: {len(result.content)} chars")
                print(f"  🎯 Tokens used: {result.tokens_used}")
                print(f"  💰 Cost estimate: ${result.cost_estimate:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    if args.physics_accuracy:
        # Test physics accuracy
        print("\n🔬 Physics Accuracy Benchmark")
        
        test_cases = [
            ("Calculate vacuum Rabi frequency for g = 1e6 Hz", "cavity_qed"),
            ("Derive squeezing parameter for degenerate OPO", "squeezed_light"),
            ("Analyze photon blockade conditions", "photon_blockade")
        ]
        
        evaluator = QuantumOpticsEvaluator()
        
        for prompt, category in test_cases:
            print(f"\nTesting: {prompt}")
            
            try:
                llm = OpenRouterInterface(args.config)
                result = llm.generate_research(prompt)
                
                solution_data = {
                    'content': result.content,
                    'category': category,
                    'title': prompt
                }
                
                eval_result = evaluator.evaluate_research(solution_data)
                
                print(f"  🎯 Total Score: {eval_result.total_score:.3f}")
                print(f"  📐 Mathematics: {eval_result.mathematics:.3f}")
                print(f"  ⚖️  Feasibility: {eval_result.feasibility:.3f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")

if __name__ == '__main__':
    main()