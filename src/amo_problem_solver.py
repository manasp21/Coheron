"""
AMO Problem Solver Controller

Core orchestration system for AMO physics problem solving.
Manages the evolution of solutions toward specific physics targets,
tracks progress, and detects breakthroughs.
"""

import json
import time
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from amo_problem_definition import AMOProblem, AMOProblemLibrary
from amo_parameter_extractor import AMOParameterExtractor, ExtractionResult
from amo_physics_calculator import AMOPhysicsCalculator
from research_generator import QuantumResearchGenerator, ResearchPrompt
from evaluator import QuantumOpticsEvaluator
from evolution_controller import QuantumResearchEvolver

logger = logging.getLogger(__name__)

@dataclass
class SolutionCandidate:
    """Represents a solution candidate being evolved"""
    id: str
    content: str
    parameters: Dict[str, float]
    problem_score: float
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    extraction_confidence: float = 0.0
    physics_consistency: float = 0.0
    breakthrough_achieved: bool = False
    notes: List[str] = field(default_factory=list)

@dataclass
class EvolutionProgress:
    """Tracks evolution progress for a problem"""
    problem_id: str
    generation: int
    population_size: int
    best_score: float
    best_solution: Optional[SolutionCandidate]
    target_achievements: Dict[str, bool]
    convergence_history: List[float] = field(default_factory=list)
    breakthrough_generation: Optional[int] = None
    stagnation_count: int = 0

@dataclass
class ProblemSolvingSession:
    """Complete problem solving session data"""
    problem: AMOProblem
    start_time: float
    end_time: Optional[float] = None
    total_generations: int = 0
    total_evaluations: int = 0
    best_solution: Optional[SolutionCandidate] = None
    breakthrough_achieved: bool = False
    solution_history: List[SolutionCandidate] = field(default_factory=list)
    evolution_progress: List[EvolutionProgress] = field(default_factory=list)

class AMOProblemSolver:
    """
    Core controller for AMO physics problem solving.
    Orchestrates evolutionary optimization toward specific physics targets.
    """
    
    def __init__(self, 
                 problem_library: Optional[AMOProblemLibrary] = None,
                 config_path: str = "config/config.yaml",
                 results_dir: str = "results/amo_solving"):
        
        self.logger = logging.getLogger('AMOProblemSolver')
        
        # Initialize components
        self.problem_library = problem_library or AMOProblemLibrary()
        self.parameter_extractor = AMOParameterExtractor()
        self.physics_calculator = AMOPhysicsCalculator()
        self.research_generator = QuantumResearchGenerator()
        self.evaluator = QuantumOpticsEvaluator()
        self.evolver = QuantumResearchEvolver(config_path=config_path)
        
        # Setup results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Evolution parameters
        self.max_generations = 100
        self.population_size = 20
        self.convergence_threshold = 0.95
        self.stagnation_limit = 15
        self.target_achievement_threshold = 0.9
        
        # Current session
        self.current_session: Optional[ProblemSolvingSession] = None
    
    def solve_problem(self, 
                     problem_id: str, 
                     max_generations: Optional[int] = None,
                     population_size: Optional[int] = None,
                     save_results: bool = True) -> ProblemSolvingSession:
        """
        Main problem solving function.
        Evolves solutions toward the specified physics targets.
        """
        # Get problem from library
        problem = self.problem_library.get_problem(problem_id)
        if not problem:
            raise ValueError(f"Problem '{problem_id}' not found in library")
        
        self.logger.info(f"Starting problem solving for: {problem.title}")
        
        # Override parameters if specified
        if max_generations is not None:
            self.max_generations = max_generations
        if population_size is not None:
            self.population_size = population_size
        
        # Initialize session
        session = ProblemSolvingSession(
            problem=problem,
            start_time=time.time()
        )
        self.current_session = session
        
        try:
            # Generate initial population
            initial_population = self._generate_initial_population(problem)
            session.total_evaluations += len(initial_population)
            
            # Evaluate initial population
            evaluated_population = self._evaluate_population_for_problem(initial_population, problem)
            
            # Start evolution loop
            current_population = evaluated_population
            generation = 0
            
            while generation < self.max_generations:
                self.logger.info(f"Generation {generation + 1}/{self.max_generations}")
                
                # Track progress
                progress = self._track_generation_progress(current_population, problem, generation)
                session.evolution_progress.append(progress)
                
                # Check for breakthrough
                if progress.best_solution and progress.best_solution.breakthrough_achieved:
                    self.logger.info(f"Breakthrough achieved at generation {generation + 1}!")
                    session.breakthrough_achieved = True
                    session.best_solution = progress.best_solution
                    break
                
                # Check for convergence
                if self._check_convergence(session.evolution_progress):
                    self.logger.info(f"Convergence achieved at generation {generation + 1}")
                    break
                
                # Check for stagnation
                if progress.stagnation_count >= self.stagnation_limit:
                    self.logger.info(f"Evolution stagnated after {self.stagnation_limit} generations")
                    break
                
                # Evolve to next generation
                next_population = self._evolve_generation(current_population, problem)
                session.total_evaluations += len(next_population)
                
                # Evaluate new population
                current_population = self._evaluate_population_for_problem(next_population, problem)
                
                generation += 1
            
            # Finalize session
            session.end_time = time.time()
            session.total_generations = generation + 1
            
            if not session.best_solution and session.evolution_progress:
                session.best_solution = session.evolution_progress[-1].best_solution
            
            # Update problem with best solution
            if session.best_solution:
                problem.update_best_solution(
                    {'content': session.best_solution.content},
                    session.best_solution.parameters
                )
            
            self.logger.info(f"Problem solving completed after {session.total_generations} generations")
            self.logger.info(f"Best score achieved: {session.best_solution.problem_score if session.best_solution else 0:.3f}")
            
            # Save results if requested
            if save_results:
                self._save_session_results(session)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Problem solving failed: {e}")
            session.end_time = time.time()
            if save_results:
                self._save_session_results(session)
            raise
    
    def _generate_initial_population(self, problem: AMOProblem) -> List[SolutionCandidate]:
        """Generate initial population of solution candidates"""
        population = []
        
        # Generate problem-targeted prompts
        targeted_prompts = self._generate_problem_targeted_prompts(problem, self.population_size)
        
        for i, prompt in enumerate(targeted_prompts):
            try:
                # Generate solution using research generator
                solution_content = self._generate_solution_from_prompt(prompt)
                
                candidate = SolutionCandidate(
                    id=f"gen0_sol{i}",
                    content=solution_content,
                    parameters={},
                    problem_score=0.0,
                    generation=0,
                    notes=[f"Generated from: {prompt.mutation_type or 'initial'}"]
                )
                
                population.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate initial solution {i}: {e}")
                continue
        
        self.logger.info(f"Generated initial population of {len(population)} solutions")
        return population
    
    def _generate_problem_targeted_prompts(self, problem: AMOProblem, count: int) -> List[ResearchPrompt]:
        """Generate prompts specifically targeted at solving the given problem"""
        prompts = []
        
        # Base prompt from problem description
        base_prompt = ResearchPrompt(
            content=f"Design a {problem.category.replace('_', ' ')} system to solve: {problem.physics_challenge}. "
                   f"Specific requirements: {problem.description}. "
                   f"Target parameters to achieve: {', '.join(problem.target_parameters.keys())}.",
            category=problem.category,
            system_prompt=f"You are an expert in {problem.category.replace('_', ' ')} physics. "
                         f"Focus on achieving the specific numerical targets.",
            generation=0
        )
        
        # Add variations of the base prompt
        for i in range(count // 2):
            prompts.append(base_prompt)
        
        # Generate additional exploratory prompts
        for i in range(count - len(prompts)):
            # Create variations by emphasizing different aspects
            aspects = [
                "experimental feasibility", "theoretical novelty", 
                "parameter optimization", "technical implementation"
            ]
            aspect = random.choice(aspects)
            
            varied_prompt = ResearchPrompt(
                content=base_prompt.content + f" Pay special attention to {aspect}.",
                category=problem.category,
                system_prompt=base_prompt.system_prompt,
                generation=0,
                mutation_type=f"aspect_emphasis_{aspect}"
            )
            prompts.append(varied_prompt)
        
        return prompts
    
    def _generate_solution_from_prompt(self, prompt: ResearchPrompt) -> str:
        """Generate solution content from a research prompt"""
        # This would interface with the AI model to generate content
        # For now, return a placeholder that would be replaced with actual AI generation
        return f"[AI-generated solution for: {prompt.content[:100]}...]"
    
    def _evaluate_population_for_problem(self, 
                                       population: List[SolutionCandidate], 
                                       problem: AMOProblem) -> List[SolutionCandidate]:
        """Evaluate population candidates against the specific problem targets"""
        evaluated_population = []
        
        for candidate in population:
            try:
                # Extract parameters from solution content
                extraction_result = self.parameter_extractor.extract_parameters(candidate.content)
                candidate.parameters = self.parameter_extractor.get_parameter_summary(extraction_result)
                candidate.extraction_confidence = extraction_result.confidence_score
                
                # Calculate physics consistency using physics calculator
                physics_results = self.physics_calculator.calculate_all_relevant_parameters(
                    candidate.content, candidate.parameters
                )
                
                # Add calculated parameters
                for param_name, calc_result in physics_results.items():
                    if param_name not in candidate.parameters:
                        candidate.parameters[param_name] = calc_result.value
                
                # Score against problem targets
                problem_results = problem.calculate_total_score(candidate.parameters)
                candidate.problem_score = problem_results['total_score']
                
                # Check if breakthrough achieved
                candidate.breakthrough_achieved = problem.is_solved(candidate.parameters)
                
                # Calculate physics consistency score
                candidate.physics_consistency = self._calculate_physics_consistency(candidate, problem)
                
                # Add notes about achievements
                if candidate.breakthrough_achieved:
                    candidate.notes.append("Breakthrough: All targets achieved!")
                
                missing_targets = problem.get_missing_targets(candidate.parameters)
                if missing_targets:
                    candidate.notes.append(f"Missing targets: {', '.join(missing_targets)}")
                
                evaluated_population.append(candidate)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate candidate {candidate.id}: {e}")
                # Keep candidate with zero score
                candidate.problem_score = 0.0
                evaluated_population.append(candidate)
        
        # Sort by problem score
        evaluated_population.sort(key=lambda x: x.problem_score, reverse=True)
        
        return evaluated_population
    
    def _calculate_physics_consistency(self, candidate: SolutionCandidate, problem: AMOProblem) -> float:
        """Calculate physics consistency score for a candidate"""
        consistency_score = 0.0
        
        # Check parameter extraction confidence
        consistency_score += candidate.extraction_confidence * 0.3
        
        # Check against physics constraints
        violations = 0
        for constraint in problem.constraints:
            is_valid, message = constraint.validate(candidate.parameters)
            if not is_valid:
                violations += 1
                candidate.notes.append(f"Constraint violation: {message}")
        
        # Penalty for constraint violations
        if problem.constraints:
            constraint_score = max(0, 1 - violations / len(problem.constraints))
            consistency_score += constraint_score * 0.4
        else:
            consistency_score += 0.4  # No constraints to violate
        
        # Check parameter ranges (basic physics sanity)
        valid_params = 0
        total_params = 0
        
        for param_name, value in candidate.parameters.items():
            if param_name.startswith('_'):
                continue  # Skip metadata
            
            total_params += 1
            
            # Basic range checks
            if param_name in ['coupling_strength', 'cavity_decay_rate', 'atomic_linewidth']:
                if 1e3 <= value <= 1e12:  # Reasonable frequency range
                    valid_params += 1
            elif param_name in ['cooperativity']:
                if 0.01 <= value <= 1e6:  # Reasonable cooperativity range
                    valid_params += 1
            elif param_name in ['fidelity']:
                if 0 <= value <= 1:  # Fidelity must be 0-1
                    valid_params += 1
            else:
                valid_params += 1  # Assume valid if no specific check
        
        if total_params > 0:
            range_score = valid_params / total_params
            consistency_score += range_score * 0.3
        else:
            consistency_score += 0.3  # No parameters to check
        
        return min(consistency_score, 1.0)
    
    def _track_generation_progress(self, 
                                 population: List[SolutionCandidate], 
                                 problem: AMOProblem, 
                                 generation: int) -> EvolutionProgress:
        """Track progress for the current generation"""
        
        best_candidate = max(population, key=lambda x: x.problem_score) if population else None
        
        # Check target achievements
        target_achievements = {}
        if best_candidate:
            for param_name, param_def in problem.target_parameters.items():
                if param_name in best_candidate.parameters:
                    achieved = param_def.is_target_achieved(
                        best_candidate.parameters[param_name], 
                        self.target_achievement_threshold
                    )
                    target_achievements[param_name] = achieved
                else:
                    target_achievements[param_name] = False
        
        # Calculate stagnation
        stagnation_count = 0
        if hasattr(self, 'current_session') and self.current_session.evolution_progress:
            recent_scores = [p.best_score for p in self.current_session.evolution_progress[-5:]]
            if len(recent_scores) >= 3 and all(abs(s - recent_scores[0]) < 0.01 for s in recent_scores):
                stagnation_count = len([p for p in self.current_session.evolution_progress[-self.stagnation_limit:] 
                                      if abs(p.best_score - recent_scores[0]) < 0.01])
        
        progress = EvolutionProgress(
            problem_id=problem.id,
            generation=generation,
            population_size=len(population),
            best_score=best_candidate.problem_score if best_candidate else 0.0,
            best_solution=best_candidate,
            target_achievements=target_achievements,
            stagnation_count=stagnation_count
        )
        
        # Check for breakthrough
        if best_candidate and best_candidate.breakthrough_achieved:
            progress.breakthrough_generation = generation
        
        progress.convergence_history.append(progress.best_score)
        
        return progress
    
    def _check_convergence(self, progress_history: List[EvolutionProgress]) -> bool:
        """Check if evolution has converged"""
        if len(progress_history) < 10:
            return False
        
        # Check if best score is above convergence threshold
        latest_score = progress_history[-1].best_score
        if latest_score >= self.convergence_threshold:
            return True
        
        # Check for score plateau
        recent_scores = [p.best_score for p in progress_history[-10:]]
        score_variance = np.var(recent_scores)
        if score_variance < 0.001:  # Very small variance indicates convergence
            return True
        
        return False
    
    def _evolve_generation(self, 
                         current_population: List[SolutionCandidate], 
                         problem: AMOProblem) -> List[SolutionCandidate]:
        """Evolve current population to next generation"""
        next_population = []
        
        # Elite retention - keep best solutions
        elite_count = max(1, int(0.2 * self.population_size))
        elites = current_population[:elite_count]
        
        for elite in elites:
            # Create copy with new ID
            elite_copy = SolutionCandidate(
                id=f"gen{elite.generation + 1}_elite_{len(next_population)}",
                content=elite.content,
                parameters=elite.parameters.copy(),
                problem_score=elite.problem_score,
                generation=elite.generation + 1,
                parent_ids=[elite.id],
                extraction_confidence=elite.extraction_confidence,
                physics_consistency=elite.physics_consistency,
                breakthrough_achieved=elite.breakthrough_achieved,
                notes=elite.notes.copy() + ["Elite retention"]
            )
            next_population.append(elite_copy)
        
        # Generate mutations and crossovers
        while len(next_population) < self.population_size:
            if random.random() < 0.7:  # 70% mutations
                parent = self._select_parent(current_population)
                mutated = self._mutate_solution(parent, problem)
                if mutated:
                    next_population.append(mutated)
            else:  # 30% crossovers
                parent1 = self._select_parent(current_population)
                parent2 = self._select_parent(current_population)
                if parent1.id != parent2.id:
                    crossed = self._crossover_solutions(parent1, parent2, problem)
                    if crossed:
                        next_population.append(crossed)
        
        return next_population[:self.population_size]
    
    def _select_parent(self, population: List[SolutionCandidate]) -> SolutionCandidate:
        """Select parent using tournament selection"""
        tournament_size = min(5, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.problem_score)
    
    def _mutate_solution(self, parent: SolutionCandidate, problem: AMOProblem) -> Optional[SolutionCandidate]:
        """Create a mutation of the parent solution"""
        try:
            # Create solution data structure for mutation
            solution_data = {
                'id': parent.id,
                'content': parent.content,
                'category': problem.category,
                'generation': parent.generation
            }
            
            # Generate mutation prompt
            mutation_prompt = self.research_generator.mutate_solution(solution_data)
            
            # Generate new content (would use AI model)
            mutated_content = self._generate_solution_from_prompt(mutation_prompt)
            
            # Create new candidate
            mutated = SolutionCandidate(
                id=f"gen{parent.generation + 1}_mut_{int(time.time() * 1000) % 10000}",
                content=mutated_content,
                parameters={},
                problem_score=0.0,
                generation=parent.generation + 1,
                parent_ids=[parent.id],
                notes=[f"Mutation of {parent.id}", f"Strategy: {mutation_prompt.mutation_type}"]
            )
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Failed to mutate solution: {e}")
            return None
    
    def _crossover_solutions(self, 
                           parent1: SolutionCandidate, 
                           parent2: SolutionCandidate, 
                           problem: AMOProblem) -> Optional[SolutionCandidate]:
        """Create crossover of two parent solutions"""
        try:
            # Create solution data structures
            solution1 = {
                'id': parent1.id,
                'content': parent1.content,
                'category': problem.category,
                'generation': parent1.generation
            }
            
            solution2 = {
                'id': parent2.id,
                'content': parent2.content,
                'category': problem.category,
                'generation': parent2.generation
            }
            
            # Generate crossover prompt
            crossover_prompt = self.research_generator.crossover_solutions(solution1, solution2)
            
            # Generate new content (would use AI model)
            crossed_content = self._generate_solution_from_prompt(crossover_prompt)
            
            # Create new candidate
            crossed = SolutionCandidate(
                id=f"gen{max(parent1.generation, parent2.generation) + 1}_cross_{int(time.time() * 1000) % 10000}",
                content=crossed_content,
                parameters={},
                problem_score=0.0,
                generation=max(parent1.generation, parent2.generation) + 1,
                parent_ids=[parent1.id, parent2.id],
                notes=[f"Crossover of {parent1.id} and {parent2.id}"]
            )
            
            return crossed
            
        except Exception as e:
            self.logger.warning(f"Failed to crossover solutions: {e}")
            return None
    
    def _save_session_results(self, session: ProblemSolvingSession):
        """Save session results to file"""
        try:
            timestamp = int(session.start_time)
            filename = f"amo_session_{session.problem.id}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert session to serializable format
            session_data = {
                'problem_id': session.problem.id,
                'problem_title': session.problem.title,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'total_generations': session.total_generations,
                'total_evaluations': session.total_evaluations,
                'breakthrough_achieved': session.breakthrough_achieved,
                'best_solution': {
                    'id': session.best_solution.id,
                    'content': session.best_solution.content,
                    'parameters': session.best_solution.parameters,
                    'problem_score': session.best_solution.problem_score,
                    'generation': session.best_solution.generation,
                    'breakthrough_achieved': session.best_solution.breakthrough_achieved
                } if session.best_solution else None,
                'evolution_summary': [
                    {
                        'generation': p.generation,
                        'best_score': p.best_score,
                        'target_achievements': p.target_achievements,
                        'stagnation_count': p.stagnation_count
                    } for p in session.evolution_progress
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session results saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session results: {e}")
    
    def get_problem_solving_status(self) -> Optional[Dict[str, Any]]:
        """Get current problem solving status"""
        if not self.current_session:
            return None
        
        session = self.current_session
        latest_progress = session.evolution_progress[-1] if session.evolution_progress else None
        
        return {
            'problem_title': session.problem.title,
            'current_generation': latest_progress.generation if latest_progress else 0,
            'max_generations': self.max_generations,
            'best_score': latest_progress.best_score if latest_progress else 0.0,
            'breakthrough_achieved': session.breakthrough_achieved,
            'target_achievements': latest_progress.target_achievements if latest_progress else {},
            'total_evaluations': session.total_evaluations,
            'elapsed_time': time.time() - session.start_time
        }
    
    def list_available_problems(self) -> List[Dict[str, Any]]:
        """List all available problems in the library"""
        problems = []
        
        for problem_id in self.problem_library.list_problems():
            problem = self.problem_library.get_problem(problem_id)
            if problem:
                problems.append({
                    'id': problem.id,
                    'title': problem.title,
                    'category': problem.category,
                    'difficulty': problem.difficulty_level,
                    'status': problem.experimental_status,
                    'target_count': len(problem.target_parameters),
                    'description': problem.description[:100] + "..." if len(problem.description) > 100 else problem.description
                })
        
        return problems