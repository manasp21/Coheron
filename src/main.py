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
from dataclasses import asdict
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evolution_controller import QuantumResearchEvolver
from llm_interface import A4FInterface
from evaluator import QuantumOpticsEvaluator
from database import ResearchDatabase
from amo_problem_solver import AMOProblemSolver
from amo_problem_definition import AMOProblemLibrary
from utils import (
    setup_logging, validate_config, create_evolution_plot, 
    create_category_analysis, export_results_report, format_duration,
    clean_unicode_for_console
)

def main():
    """Main application entry point"""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging with Unicode-safe handler
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
        elif args.command == 'solve':
            run_amo_solve(args, logger)
        elif args.command == 'problems':
            run_amo_problems(args, logger)
        elif args.command == 'create-problem':
            run_create_problem(args, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
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
    evolve_parser.add_argument('--demo', action='store_true',
                              help='Demo mode: use mock responses instead of API calls')
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
    
    # AMO Problem Solving commands
    solve_parser = subparsers.add_parser('solve', help='Solve AMO physics problems')
    solve_parser.add_argument('problem_id', help='Problem ID to solve')
    solve_parser.add_argument('--generations', type=int, default=50,
                             help='Maximum generations for evolution (default: 50)')
    solve_parser.add_argument('--population', type=int, default=20,
                             help='Population size (default: 20)')
    solve_parser.add_argument('--output-dir', type=str, default='results/amo_solving',
                             help='Output directory for results')
    solve_parser.add_argument('--model', type=str,
                             help='Override model for problem solving')
    
    # Problems management
    problems_parser = subparsers.add_parser('problems', help='Manage AMO problems')
    problems_parser.add_argument('--list', action='store_true',
                                help='List available problems')
    problems_parser.add_argument('--show', type=str,
                                help='Show details of specific problem')
    problems_parser.add_argument('--category', type=str,
                                help='Filter by category')
    problems_parser.add_argument('--difficulty', type=str,
                                help='Filter by difficulty level')
    problems_parser.add_argument('--status', type=str,
                                help='Filter by status (open, solved, etc.)')
    
    # Create problem
    create_problem_parser = subparsers.add_parser('create-problem', help='Create new AMO problem')
    create_problem_parser.add_argument('--template', type=str,
                                      help='Use problem template file')
    create_problem_parser.add_argument('--interactive', action='store_true',
                                      help='Interactive problem creation')
    create_problem_parser.add_argument('--validate-only', action='store_true',
                                      help='Only validate problem definition')
    
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
    
    # Check for A4F API key
    if not os.getenv('A4F_API_KEY'):
        print("❌ Error: A4F_API_KEY environment variable not set")
        print("   Please set your A4F API key:")
        print("   export A4F_API_KEY='your-api-key-here'")
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
    evolver = QuantumResearchEvolver(args.config, demo_mode=getattr(args, 'demo', False))
    
    # Switch model if specified
    if args.model:
        if getattr(args, 'demo', False):
            logger.warning("Model switching not supported in demo mode")
        else:
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
    
    model_name = "demo-mode" if getattr(args, 'demo', False) else evolver.llm.current_model
    logger.info(f"Model: {model_name}")
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
                [asdict(sol) for sol in best_solutions],
                output_dir / 'category_analysis.png'
            )
        
        # Generate report
        export_results_report(
            evolver.evolution_history,  # Already dictionaries
            [asdict(sol) for sol in best_solutions[:10]],
            [asdict(sol) for sol in evolver.breakthrough_solutions],
            output_dir / 'research_report.json'
        )
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"✅ Evolution completed in {format_duration(total_time)}")
        logger.info(f"📊 Best score: {best_solutions[0].evaluation_result.total_score:.3f}")
        logger.info(f"🏆 Breakthroughs: {len(evolver.breakthrough_solutions)}")
        
        # Show detailed results info (with ASCII-safe fallback)
        def safe_print(text):
            """Print text with ASCII fallback for Windows"""
            try:
                print(text)
            except UnicodeEncodeError:
                print(clean_unicode_for_console(text))
        
        safe_print(f"\n" + "="*60)
        safe_print(f"🎉 COHERON EVOLUTION COMPLETE!")
        safe_print(f"="*60)
        safe_print(f"⏱️  Total time: {format_duration(total_time)}")
        safe_print(f"🧬 Generations: {len(evolver.evolution_history)}")
        safe_print(f"📊 Best score achieved: {best_solutions[0].evaluation_result.total_score:.3f}")
        safe_print(f"🏆 Breakthroughs discovered: {len(evolver.breakthrough_solutions)}")
        
        # Show saved files
        safe_print(f"\n📁 RESULTS SAVED TO:")
        safe_print(f"   📂 Directory: {output_dir.absolute()}")
        safe_print(f"   📄 Evolution state: evolution_state.json")
        safe_print(f"   📄 Research report: research_report.json")
        if not args.quiet:
            safe_print(f"   📊 Evolution plot: evolution_progress.png")
            safe_print(f"   📈 Category analysis: category_analysis.png")
        
        # Display top results with more detail
        safe_print(f"\n🏆 TOP {min(3, len(best_solutions))} RESEARCH SOLUTIONS:")
        safe_print("-" * 60)
        for i, solution in enumerate(best_solutions[:3]):
            safe_print(f"\n{i+1}. SCORE: {solution.evaluation_result.total_score:.3f}")
            safe_print(f"   📋 TITLE: {solution.title}")
            safe_print(f"   🔬 CATEGORY: {solution.category.replace('_', ' ').title()}")
            safe_print(f"   🧬 GENERATION: {solution.generation}")
            safe_print(f"   📊 BREAKDOWN:")
            safe_print(f"      - Feasibility: {solution.evaluation_result.feasibility:.3f}")
            safe_print(f"      - Mathematics: {solution.evaluation_result.mathematics:.3f}")
            safe_print(f"      - Novelty: {solution.evaluation_result.novelty:.3f}")
            safe_print(f"      - Performance: {solution.evaluation_result.performance:.3f}")
            
            # Show content preview
            content_preview = solution.content[:200] + "..." if len(solution.content) > 200 else solution.content
            safe_print(f"   📝 CONTENT PREVIEW:")
            safe_print(f"      {content_preview}")
        
        # Show evolution progress
        if len(evolver.evolution_history) > 1:
            safe_print(f"\n📈 EVOLUTION PROGRESS:")
            safe_print("-" * 40)
            for i, stats in enumerate(evolver.evolution_history):
                safe_print(f"   Gen {stats['generation']}: Best={stats['best_score']:.3f}, Avg={stats['average_score']:.3f}, Diversity={stats['diversity_index']:.3f}")
        
        logger.info(f"💾 All results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise

def run_evaluation(args: argparse.Namespace, logger) -> None:
    """Run evaluation on provided content"""
    
    logger.info("🔍 Running research evaluation")
    
    # Load content
    if Path(args.content).exists():
        with open(args.content, 'r', encoding='utf-8') as f:
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
            'evaluation_result': asdict(result),
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
            llm = A4FInterface(args.config)
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
                llm = A4FInterface(args.config)
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
                llm = A4FInterface(args.config)
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

def run_amo_solve(args: argparse.Namespace, logger) -> None:
    """Run AMO problem solving"""
    
    logger.info(f"🔬 Starting AMO problem solving for: {args.problem_id}")
    
    try:
        # Initialize problem solver
        solver = AMOProblemSolver(
            config_path=args.config,
            results_dir=args.output_dir
        )
        
        # Check if problem exists
        available_problems = solver.list_available_problems()
        problem_exists = any(p['id'] == args.problem_id for p in available_problems)
        
        if not problem_exists:
            logger.error(f"❌ Problem '{args.problem_id}' not found")
            print(f"\nAvailable problems:")
            for problem in available_problems:
                print(f"  • {problem['id']}: {problem['title']}")
            return
        
        # Get problem details
        problem = solver.problem_library.get_problem(args.problem_id)
        print(f"\n🎯 Problem: {problem.title}")
        print(f"📋 Category: {problem.category}")
        print(f"🔬 Challenge: {problem.physics_challenge}")
        print(f"🎲 Difficulty: {problem.difficulty_level}")
        print(f"📊 Targets: {len(problem.target_parameters)} parameters")
        
        # Show target parameters
        print(f"\n🎯 Target Parameters:")
        for param_name, param in problem.target_parameters.items():
            print(f"  • {param_name}: {param.target_value} {param.units}")
            print(f"    {param.description}")
        
        print(f"\n🚀 Starting evolution with {args.generations} generations...")
        print(f"👥 Population size: {args.population}")
        
        start_time = time.time()
        
        # Solve the problem
        session = solver.solve_problem(
            args.problem_id,
            max_generations=args.generations,
            population_size=args.population,
            save_results=True
        )
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"\n✅ Problem solving completed in {format_duration(elapsed_time)}")
        print(f"🧬 Total generations: {session.total_generations}")
        print(f"📊 Total evaluations: {session.total_evaluations}")
        print(f"🏆 Breakthrough achieved: {'Yes' if session.breakthrough_achieved else 'No'}")
        
        if session.best_solution:
            print(f"🎯 Best score: {session.best_solution.problem_score:.3f}")
            print(f"⚖️  Physics consistency: {session.best_solution.physics_consistency:.3f}")
            print(f"🔍 Extraction confidence: {session.best_solution.extraction_confidence:.3f}")
            
            # Show parameter achievements
            if session.best_solution.parameters:
                print(f"\n📊 Parameter Achievements:")
                problem_results = problem.calculate_total_score(session.best_solution.parameters)
                
                for param_name, param_def in problem.target_parameters.items():
                    if param_name in session.best_solution.parameters:
                        achieved_value = session.best_solution.parameters[param_name]
                        score = param_def.evaluate_achievement(achieved_value)
                        achieved = param_def.is_target_achieved(achieved_value)
                        status = "✅" if achieved else "❌"
                        
                        print(f"  {status} {param_name}: {achieved_value:.2e} {param_def.units}")
                        print(f"      Target: {param_def.target_value} (Score: {score:.3f})")
                    else:
                        print(f"  ❓ {param_name}: Not detected")
            
            # Show solution preview
            print(f"\n📝 Best Solution Preview:")
            content_preview = session.best_solution.content[:300] + "..." if len(session.best_solution.content) > 300 else session.best_solution.content
            print(f"   {content_preview}")
        
        # Show evolution progress
        if session.evolution_progress:
            print(f"\n📈 Evolution Progress:")
            for i, progress in enumerate(session.evolution_progress[-5:]):  # Last 5 generations
                achieved_count = sum(progress.target_achievements.values())
                total_targets = len(progress.target_achievements)
                print(f"  Gen {progress.generation}: Score={progress.best_score:.3f}, Targets={achieved_count}/{total_targets}")
        
        print(f"\n💾 Results saved to: {Path(args.output_dir).absolute()}")
        
    except Exception as e:
        logger.error(f"❌ Problem solving failed: {e}")
        raise

def run_amo_problems(args: argparse.Namespace, logger) -> None:
    """Manage AMO problems"""
    
    logger.info("📚 Managing AMO problems")
    
    try:
        # Initialize problem library
        problem_library = AMOProblemLibrary()
        
        if args.list:
            # List problems with optional filtering
            problems = problem_library.list_problems()
            
            if args.category:
                problems = [pid for pid in problems 
                          if problem_library.get_problem(pid).category == args.category]
            
            if args.difficulty:
                problems = [pid for pid in problems 
                          if problem_library.get_problem(pid).difficulty_level == args.difficulty]
            
            if args.status:
                problems = [pid for pid in problems 
                          if problem_library.get_problem(pid).experimental_status == args.status]
            
            print(f"\n📚 Available AMO Problems ({len(problems)} total):")
            print("=" * 70)
            
            for problem_id in problems:
                problem = problem_library.get_problem(problem_id)
                if problem:
                    status_icon = {
                        'open': '🔓',
                        'partially_solved': '🔄', 
                        'solved': '✅',
                        'impossible': '❌'
                    }.get(problem.experimental_status, '❓')
                    
                    difficulty_icon = {
                        'easy': '🟢',
                        'medium': '🟡',
                        'hard': '🔴',
                        'expert': '🟣',
                        'unsolved': '⚫'
                    }.get(problem.difficulty_level, '❓')
                    
                    print(f"\n{status_icon} {difficulty_icon} {problem.id}")
                    print(f"   📋 {problem.title}")
                    print(f"   🔬 Category: {problem.category}")
                    print(f"   🎯 Targets: {len(problem.target_parameters)} parameters")
                    print(f"   📝 {problem.description[:100]}{'...' if len(problem.description) > 100 else ''}")
        
        elif args.show:
            # Show detailed problem information
            problem = problem_library.get_problem(args.show)
            
            if not problem:
                print(f"❌ Problem '{args.show}' not found")
                return
            
            print(f"\n🔬 Problem Details: {problem.id}")
            print("=" * 50)
            print(f"📋 Title: {problem.title}")
            print(f"🔬 Category: {problem.category}")
            print(f"🎲 Difficulty: {problem.difficulty_level}")
            print(f"📊 Status: {problem.experimental_status}")
            print(f"\n📝 Description:")
            print(f"   {problem.description}")
            print(f"\n🎯 Physics Challenge:")
            print(f"   {problem.physics_challenge}")
            
            print(f"\n🎯 Target Parameters ({len(problem.target_parameters)}):")
            for param_name, param in problem.target_parameters.items():
                print(f"  • {param_name} ({param.symbol})")
                print(f"    Target: {param.target_value} {param.units}")
                print(f"    Weight: {param.weight}")
                print(f"    Description: {param.description}")
                if param.current_record:
                    print(f"    Current record: {param.current_record}")
                print()
            
            if problem.constraints:
                print(f"⚖️  Constraints ({len(problem.constraints)}):")
                for constraint in problem.constraints:
                    print(f"  • {constraint.name} ({constraint.constraint_type})")
                    print(f"    {constraint.description}")
                print()
            
            if problem.background_context:
                print(f"📚 Background:")
                print(f"   {problem.background_context}")
                print()
            
            if problem.references:
                print(f"📖 References:")
                for ref in problem.references:
                    print(f"  • {ref}")
                print()
            
            if problem.keywords:
                print(f"🏷️  Keywords: {', '.join(problem.keywords)}")
                print()
            
            # Show current best solution if available
            if problem.current_best_solution:
                best = problem.current_best_solution
                print(f"🏆 Current Best Solution:")
                print(f"   Score: {best['score']:.3f}")
                print(f"   Timestamp: {time.ctime(best['timestamp'])}")
                print()
        
        else:
            # Show summary statistics
            summary = problem_library.get_problem_summary()
            
            print(f"\n📚 AMO Problem Library Summary")
            print("=" * 40)
            print(f"📊 Total problems: {summary['total_problems']}")
            print(f"🔬 Categories: {len(summary['categories'])}")
            
            print(f"\n📊 By Category:")
            for category, count in summary['by_category'].items():
                print(f"  • {category}: {count} problems")
            
            print(f"\n📊 By Status:")
            for status, count in summary['by_status'].items():
                print(f"  • {status}: {count} problems")
            
            print(f"\n📊 By Difficulty:")
            for difficulty, count in summary['by_difficulty'].items():
                print(f"  • {difficulty}: {count} problems")
            
            print(f"\n💡 Use --list to see all problems")
            print(f"💡 Use --show <problem_id> to see problem details")
    
    except Exception as e:
        logger.error(f"❌ Problem management failed: {e}")
        raise

def run_create_problem(args: argparse.Namespace, logger) -> None:
    """Create new AMO problem"""
    
    logger.info("📝 Creating new AMO problem")
    
    try:
        if args.template:
            # Load from template file
            if not Path(args.template).exists():
                print(f"❌ Template file not found: {args.template}")
                return
            
            from amo_problem_definition import AMOProblemLoader, AMOProblemValidator
            
            if args.validate_only:
                # Just validate the template
                try:
                    problem = AMOProblemLoader.load_from_yaml(args.template)
                    validation_result = AMOProblemValidator.validate_problem(problem)
                    
                    if validation_result['is_valid']:
                        print("✅ Problem definition is valid!")
                    else:
                        print("❌ Problem definition has errors:")
                        for error in validation_result['errors']:
                            print(f"   • {error}")
                    
                    if validation_result['warnings']:
                        print("⚠️  Warnings:")
                        for warning in validation_result['warnings']:
                            print(f"   • {warning}")
                    
                    if validation_result['suggestions']:
                        print("💡 Suggestions:")
                        for suggestion in validation_result['suggestions']:
                            print(f"   • {suggestion}")
                
                except Exception as e:
                    print(f"❌ Failed to load problem: {e}")
                
                return
            
            # Load and add to library
            try:
                problem = AMOProblemLoader.load_from_yaml(args.template)
                problem_library = AMOProblemLibrary()
                problem_library.add_problem(problem)
                
                print(f"✅ Problem '{problem.id}' added to library!")
                print(f"📋 Title: {problem.title}")
                print(f"🔬 Category: {problem.category}")
                print(f"🎯 Targets: {len(problem.target_parameters)} parameters")
                
            except Exception as e:
                print(f"❌ Failed to create problem: {e}")
        
        elif args.interactive:
            # Interactive problem creation
            print("🛠️  Interactive AMO Problem Creation")
            print("=" * 40)
            
            problem_data = {}
            
            # Basic information
            problem_data['id'] = input("Problem ID: ").strip()
            problem_data['title'] = input("Title: ").strip()
            
            print("\nAvailable categories:")
            categories = ['cavity_qed', 'squeezed_light', 'photon_blockade', 
                         'quantum_metrology', 'optomechanics', 'quantum_memory', 'hybrid_systems']
            for i, cat in enumerate(categories, 1):
                print(f"  {i}. {cat}")
            
            cat_choice = input(f"Select category (1-{len(categories)}): ").strip()
            try:
                problem_data['category'] = categories[int(cat_choice) - 1]
            except (ValueError, IndexError):
                problem_data['category'] = 'cavity_qed'
            
            problem_data['description'] = input("Description: ").strip()
            problem_data['physics_challenge'] = input("Physics challenge: ").strip()
            
            # Target parameters
            problem_data['target_parameters'] = {}
            print("\nDefine target parameters (press Enter with empty name to finish):")
            
            while True:
                param_name = input("Parameter name: ").strip()
                if not param_name:
                    break
                
                symbol = input(f"Symbol for {param_name}: ").strip()
                target = input(f"Target value (e.g., '> 1000e6'): ").strip()
                units = input(f"Units: ").strip()
                description = input(f"Description: ").strip()
                weight = input(f"Weight (default 1.0): ").strip() or "1.0"
                
                problem_data['target_parameters'][param_name] = {
                    'symbol': symbol,
                    'target': target,
                    'units': units,
                    'description': description,
                    'weight': float(weight),
                    'type': 'optimization'
                }
            
            # Save to file
            output_file = f"data/amo_problems/{problem_data['id']}.yaml"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            from amo_problem_definition import AMOProblemLoader
            
            try:
                # Create problem object and save
                problem = AMOProblemLoader.load_from_dict(problem_data)
                AMOProblemLoader.save_to_yaml(problem, output_file)
                
                print(f"\n✅ Problem saved to: {output_file}")
                print(f"📋 Use 'python main.py problems --show {problem_data['id']}' to view details")
                
            except Exception as e:
                print(f"❌ Failed to save problem: {e}")
        
        else:
            print("💡 Use --template <file> to load from YAML template")
            print("💡 Use --interactive for guided problem creation")
            print("💡 Use --validate-only with --template to just validate")
    
    except Exception as e:
        logger.error(f"❌ Problem creation failed: {e}")
        raise

if __name__ == '__main__':
    main()