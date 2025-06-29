import random
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from llm_interface import A4FInterface, GenerationResult
from evaluator import QuantumOpticsEvaluator, EvaluationResult
from research_generator import QuantumResearchGenerator, ResearchPrompt

@dataclass
class EvolutionStats:
    generation: int
    population_size: int
    best_score: float
    average_score: float
    diversity_index: float
    new_solutions: int
    breakthrough_solutions: int
    convergence_rate: float

@dataclass
class ResearchSolution:
    id: str
    content: str
    category: str
    title: str
    description: str
    generation: int
    parent_ids: List[str]
    mutation_type: str
    evaluation_result: EvaluationResult
    generation_metadata: Dict[str, Any]
    timestamp: float

class QuantumResearchEvolver:
    """Main evolution controller inspired by AlphaEvolve for quantum optics research"""
    
    def __init__(self, config_path: str = "config/config.yaml", demo_mode: bool = False):
        # Initialize core components
        self.demo_mode = demo_mode
        self.logger = logging.getLogger('QuantumResearchEvolver')
        
        if not demo_mode:
            self.llm = A4FInterface(config_path)
        else:
            self.llm = None
        
        # AMO problem-solving mode
        self.amo_mode = False
        self.current_amo_problem = None
        self.amo_convergence_history = []
        self.amo_breakthrough_threshold = 0.9
            
        self.evaluator = QuantumOpticsEvaluator()
        self.generator = QuantumResearchGenerator()
        
        # Load evolution parameters
        if not demo_mode:
            self.config = self.llm.config
        else:
            import yaml
            with open(Path(config_path), 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        self.evolution_params = self.config.get('evolution', {})
        self.evaluation_params = self.config.get('evaluation', {})
        
        # Evolution state
        self.current_generation = 0
        self.population: List[ResearchSolution] = []
        self.evolution_history: List[Dict[str, Any]] = []
        self.best_solutions: List[ResearchSolution] = []
        self.breakthrough_solutions: List[ResearchSolution] = []
        
        # Evolution parameters
        self.population_size = self.evolution_params.get('population_size', 10)
        self.max_generations = self.evolution_params.get('max_generations', 50)
        self.mutation_rate = self.evolution_params.get('mutation_rate', 0.3)
        self.crossover_rate = self.evolution_params.get('crossover_rate', 0.7)
        self.elite_retention = self.evolution_params.get('elite_retention', 0.2)
        self.diversity_threshold = self.evolution_params.get('diversity_threshold', 0.1)
        self.stagnation_limit = self.evolution_params.get('stagnation_limit', 10)
        
        # Evaluation thresholds
        self.elite_score = self.evaluation_params.get('thresholds', {}).get('elite_score', 0.8)
        self.breakthrough_score = self.evaluation_params.get('thresholds', {}).get('breakthrough_score', 0.9)
        
    def evolve_research(self, num_generations: Optional[int] = None) -> List[ResearchSolution]:
        """Main evolution loop for quantum optics research discovery"""
        if num_generations is None:
            num_generations = self.max_generations
            
        self.logger.info(f"Starting quantum optics research evolution for {num_generations} generations")
        model_name = "demo-mode" if self.demo_mode else (self.llm.current_model if self.llm else "unknown")
        self.logger.info(f"Population size: {self.population_size}, Model: {model_name}")
        
        # Initialize population
        if not self.population:
            self._initialize_population()
        start_time = time.time()
        stagnation_counter = 0
        previous_best_score = 0.0
        
        for generation in range(self.current_generation, self.current_generation + num_generations):
            generation_start = time.time()
            self.logger.info(f"\n🧬 Generation {generation + 1}/{self.current_generation + num_generations}")
            
            # Generate new candidate solutions
            candidates = self._generate_generation_candidates()
            
            # Evaluate all candidates
            evaluated_candidates = self._evaluate_candidates(candidates)
            
            # Update population through selection
            self._update_population(evaluated_candidates)
            
            # Calculate generation statistics
            stats = self._calculate_generation_stats(generation + 1)
            self.evolution_history.append(asdict(stats))
            
            # Check for breakthroughs
            breakthroughs = self._identify_breakthroughs()
            if breakthroughs:
                self.logger.info(f"🏆 {len(breakthroughs)} breakthrough solution(s) discovered!")
                self.breakthrough_solutions.extend(breakthroughs)
                
            # Check for convergence or stagnation
            improvement = stats.best_score - previous_best_score
            if improvement < 0.001:  # Minimal improvement threshold
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            previous_best_score = stats.best_score
            
            # Log generation summary
            generation_time = time.time() - generation_start
            self._log_generation_summary(stats, generation_time, improvement)
            
            # Check stopping criteria
            if stagnation_counter >= self.stagnation_limit:
                self.logger.info(f"🛑 Stopping due to stagnation ({self.stagnation_limit} generations)")
                break
                
            if stats.best_score >= 0.95:  # Near-perfect solution
                self.logger.info(f"🎯 Stopping due to near-perfect solution (score: {stats.best_score:.3f})")
                break
                
            self.current_generation = generation + 1
            
        total_time = time.time() - start_time
        self._log_evolution_summary(total_time)
        
        # Return best solutions sorted by score
        return sorted(self.population, key=lambda x: x.evaluation_result.total_score, reverse=True)
        
    def _initialize_population(self) -> None:
        """Initialize population with diverse research prompts"""
        self.logger.info("🌱 Initializing population with seed research")
        
        # Load seed research data
        seed_data = self._load_seed_research()
        
        # Generate initial prompts
        initial_prompts = self.generator.generate_initial_prompts(self.population_size)
        
        # Convert to solutions
        for i, prompt in enumerate(initial_prompts):
            # Generate research content using LLM
            if self.demo_mode:
                generation_result = self._generate_demo_research(prompt.category)
            else:
                generation_result = self.llm.generate_research(
                    prompt.content, 
                    prompt.system_prompt
                )
            
            # Parse and structure the generated content
            solution_data = self._parse_generated_content(generation_result.content, prompt.category)
            
            # Evaluate the solution
            evaluation_result = self.evaluator.evaluate_research(solution_data)
            
            # Create solution object
            solution = ResearchSolution(
                id=f"gen0_sol{i:03d}",
                content=generation_result.content,
                category=prompt.category,
                title=solution_data.get('title', f'Research Solution {i+1}'),
                description=solution_data.get('description', ''),
                generation=0,
                parent_ids=[],
                mutation_type='initial',
                evaluation_result=evaluation_result,
                generation_metadata={
                    'model': generation_result.model,
                    'tokens_used': generation_result.tokens_used,
                    'generation_time': generation_result.generation_time,
                    'prompt_category': prompt.category
                },
                timestamp=time.time()
            )
            
            self.population.append(solution)
            
        self.logger.info(f"✅ Initialized population with {len(self.population)} solutions")
        
    def _generate_generation_candidates(self) -> List[ResearchPrompt]:
        """Generate candidate solutions for current generation"""
        candidates = []
        
        # Calculate number of each type of candidate
        num_mutations = int(self.population_size * self.mutation_rate)
        num_crossovers = int(self.population_size * self.crossover_rate)
        num_explorations = max(1, self.population_size - num_mutations - num_crossovers)
        
        # Generate mutations from best solutions
        elite_solutions = self._get_elite_solutions()
        for _ in range(num_mutations):
            parent = random.choice(elite_solutions)
            parent_dict = self._solution_to_dict(parent)
            mutation_prompt = self.generator.mutate_solution(parent_dict)
            candidates.append(mutation_prompt)
            
        # Generate crossovers
        for _ in range(num_crossovers):
            parent1, parent2 = random.sample(elite_solutions, 2)
            parent1_dict = self._solution_to_dict(parent1)
            parent2_dict = self._solution_to_dict(parent2)
            crossover_prompt = self.generator.crossover_solutions(parent1_dict, parent2_dict)
            candidates.append(crossover_prompt)
            
        # Generate exploratory solutions
        exploration_prompts = self.generator.generate_exploration_prompts(num_explorations)
        candidates.extend(exploration_prompts)
        
        self.logger.info(f"Generated {len(candidates)} candidates: {num_mutations} mutations, {num_crossovers} crossovers, {num_explorations} explorations")
        return candidates
        
    def _evaluate_candidates(self, candidates: List[ResearchPrompt]) -> List[ResearchSolution]:
        """Evaluate all candidate solutions"""
        evaluated_solutions = []
        
        for i, prompt in enumerate(candidates):
            try:
                # Adapt prompt for current model
                adapted_prompt = self.generator.adapt_prompt_for_model(
                    prompt, 
                    'analytical' if self.demo_mode else self.llm.model_config.get('reasoning_style', 'analytical')
                )
                
                # Generate research content
                if self.demo_mode:
                    generation_result = self._generate_demo_research(prompt.category)
                else:
                    generation_result = self.llm.generate_research(
                        adapted_prompt.content,
                        adapted_prompt.system_prompt
                    )
                
                # Parse generated content
                solution_data = self._parse_generated_content(generation_result.content, prompt.category)
                
                # Evaluate solution
                evaluation_result = self.evaluator.evaluate_research(solution_data)
                
                # Create solution object
                solution = ResearchSolution(
                    id=f"gen{self.current_generation + 1}_sol{i:03d}",
                    content=generation_result.content,
                    category=prompt.category,
                    title=solution_data.get('title', f'Solution {i+1}'),
                    description=solution_data.get('description', ''),
                    generation=self.current_generation + 1,
                    parent_ids=[prompt.parent_id] if prompt.parent_id else [],
                    mutation_type=prompt.mutation_type or 'unknown',
                    evaluation_result=evaluation_result,
                    generation_metadata={
                        'model': generation_result.model,
                        'tokens_used': generation_result.tokens_used,
                        'generation_time': generation_result.generation_time,
                        'prompt_category': prompt.category,
                        'mutation_type': prompt.mutation_type
                    },
                    timestamp=time.time()
                )
                
                evaluated_solutions.append(solution)
                
                # Log progress
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Evaluated {i + 1}/{len(candidates)} candidates")
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate candidate {i}: {e}")
                continue
                
        self.logger.info(f"✅ Successfully evaluated {len(evaluated_solutions)}/{len(candidates)} candidates")
        return evaluated_solutions
        
    def _update_population(self, new_solutions: List[ResearchSolution]) -> None:
        """Update population through selection and replacement"""
        
        # Combine current population with new solutions
        all_solutions = self.population + new_solutions
        
        # Sort by total score
        all_solutions.sort(key=lambda x: x.evaluation_result.total_score, reverse=True)
        
        # Select elite solutions to retain
        num_elite = int(self.population_size * self.elite_retention)
        elite_solutions = all_solutions[:num_elite]
        
        # Fill remaining slots with diverse solutions
        remaining_slots = self.population_size - num_elite
        diverse_solutions = self._select_diverse_solutions(
            all_solutions[num_elite:], 
            remaining_slots
        )
        
        # Update population
        self.population = elite_solutions + diverse_solutions
        
        # Update best solutions tracking
        self._update_best_solutions()
        
    def _select_diverse_solutions(self, candidates: List[ResearchSolution], num_select: int) -> List[ResearchSolution]:
        """Select diverse solutions to maintain population diversity"""
        if len(candidates) <= num_select:
            return candidates
            
        selected = []
        remaining = candidates.copy()
        
        # Select first solution (highest scoring among candidates)
        if remaining:
            selected.append(remaining.pop(0))
            
        # Select remaining solutions based on diversity
        while len(selected) < num_select and remaining:
            best_candidate = None
            max_diversity = -1
            
            for candidate in remaining:
                # Calculate diversity score based on category and content similarity
                diversity = self._calculate_diversity_score(candidate, selected)
                
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = candidate
                    
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
                
        return selected
        
    def _calculate_diversity_score(self, candidate: ResearchSolution, selected: List[ResearchSolution]) -> float:
        """Calculate diversity score for solution selection"""
        if not selected:
            return 1.0
            
        # Category diversity
        categories_selected = [sol.category for sol in selected]
        category_penalty = categories_selected.count(candidate.category) * 0.2
        
        # Content similarity (simplified)
        content_similarities = []
        for selected_sol in selected:
            similarity = self._calculate_content_similarity(candidate.content, selected_sol.content)
            content_similarities.append(similarity)
            
        avg_similarity = np.mean(content_similarities) if content_similarities else 0
        
        # Combine factors
        diversity_score = 1.0 - category_penalty - avg_similarity
        return max(0.0, diversity_score)
    
    def _generate_demo_research(self, category: str) -> 'GenerationResult':
        """Generate mock research content for demo mode"""
        import random
        
        demo_contents = {
            'cavity_qed': [
                "# Strong Coupling Cavity QED System\n\nA novel cavity QED configuration achieving g/κ = 15.2 through optimized mirror curvature R = 5.2 mm and cavity length L = 12.8 μm. The system operates with 87Rb atoms at λ = 780 nm, demonstrating coherent atom-photon interactions with cooperativity C = 8.7.",
                "# Ultra-High Finesse Optical Cavity\n\nDemonstration of F = 850,000 finesse cavity using crystalline coating technology. Cavity supports strong coupling regime with single atoms, enabling quantum state transfer efficiency η = 0.94. Operating parameters: wavelength 637 nm, cavity length 380 μm."
            ],
            'squeezed_light': [
                "# Quadrature Squeezed Light Generation\n\nGeneration of -12.3 dB quadrature squeezed vacuum using PPKTP crystal in optical parametric oscillator. Pump power P = 2.1 W at 775 nm, cavity finesse F = 185. Measured squeezing bandwidth Δf = 8.5 MHz with detection efficiency η = 0.89.",
                "# Spin Squeezed Atomic Ensemble\n\nSpin squeezing parameter ξ² = 0.31 achieved in ensemble of 10⁶ cold 87Rb atoms using one-axis twisting. Interaction strength χ = 2π × 0.8 Hz, squeezing time t = 165 ms. Demonstrates 5.1 dB improvement in phase sensitivity."
            ],
            'photon_blockade': [
                "# Quantum Dot Photon Blockade\n\nStrong photon blockade in InAs quantum dot with g = 12.4 μeV, κ = 8.1 μeV, γ = 1.2 μeV. Second-order correlation g²(0) = 0.03 at resonance. System operates at 4.2 K with single photon purity > 99.7%.",
                "# Atom-Cavity Photon Blockade\n\nUnconventional photon blockade in weakly coupled atom-cavity system. Cooperativity C = 0.12, detuning Δ = 2.5κ. Demonstrates antibunching g²(0) = 0.18 through destructive interference of two-photon pathways."
            ],
            'quantum_metrology': [
                "# Atomic Interferometry Gravimeter\n\nCold atom gravimeter achieving sensitivity Δg/g = 3.2 × 10⁻⁹ using 87Rb atoms. Interrogation time T = 800 ms, free-fall height h = 1.2 m. Shot-noise limited performance with 10⁶ atoms per measurement cycle.",
                "# Optical Clock with Entangled Atoms\n\nStrontium optical lattice clock with Ramsey spectroscopy using spin-squeezed ensemble. Achieves fractional frequency stability σ_y(τ) = 1.4 × 10⁻¹⁸/√τ. Clock laser at 698 nm with linewidth < 1 Hz."
            ],
            'optomechanics': [
                "# Ground State Cooling Optomechanics\n\nSideband cooling of 42 MHz mechanical mode to n̄ = 0.07 phonons using cavity optomechanics. Cooperativity C = 2.1, effective detuning Δ_eff = ω_m. Cavity finesse F = 14,000, input power P = 120 μW.",
                "# Squeezed Mechanical Motion\n\nGeneration of mechanical squeezed states in levitated nanoparticle. Squeezing parameter r = 0.8, mechanical frequency ω_m = 2π × 150 kHz. Feedback cooling to motional ground state followed by parametric driving."
            ]
        }
        
        content = random.choice(demo_contents.get(category, demo_contents['cavity_qed']))
        
        # Return mock GenerationResult
        from llm_interface import GenerationResult
        return GenerationResult(
            content=content,
            model="demo-mode",
            tokens_used=random.randint(200, 800),
            cost_estimate=0.0,
            generation_time=random.uniform(0.5, 2.0),
            metadata={'demo_mode': True}
        )
        
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity between two solutions"""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
        
    def _calculate_generation_stats(self, generation: int) -> EvolutionStats:
        """Calculate comprehensive statistics for current generation"""
        if not self.population:
            return EvolutionStats(generation, 0, 0.0, 0.0, 0.0, 0, 0, 0.0)
        
        # Extract scores
        scores = [sol.evaluation_result.total_score for sol in self.population]
        
        # Basic statistics
        best_score = max(scores)
        average_score = float(np.mean(scores))
        
        # Diversity index (based on category distribution)
        categories = [sol.category for sol in self.population]
        unique_categories = set(categories)
        diversity_index = len(unique_categories) / len(categories) if categories else 0
        
        # Count new solutions and breakthroughs
        current_gen_solutions = [sol for sol in self.population if sol.generation == generation]
        new_solutions = len(current_gen_solutions)
        breakthrough_solutions = len([sol for sol in current_gen_solutions 
                                    if sol.evaluation_result.total_score >= self.breakthrough_score])
        
        # Convergence rate (improvement over last generation)
        convergence_rate = 0.0
        if len(self.evolution_history) > 0:
            prev_best = self.evolution_history[-1]['best_score']
            convergence_rate = (best_score - prev_best) / prev_best if prev_best > 0 else 0
            
        return EvolutionStats(
            generation=generation,
            population_size=len(self.population),
            best_score=best_score,
            average_score=average_score,
            diversity_index=diversity_index,
            new_solutions=new_solutions,
            breakthrough_solutions=breakthrough_solutions,
            convergence_rate=convergence_rate
        )
        
    def _identify_breakthroughs(self) -> List[ResearchSolution]:
        """Identify breakthrough solutions in current generation"""
        breakthroughs = []
        
        for solution in self.population:
            if (solution.evaluation_result.total_score >= self.breakthrough_score and
                solution.generation == self.current_generation + 1):
                breakthroughs.append(solution)
                
        return breakthroughs
        
    def _get_elite_solutions(self) -> List[ResearchSolution]:
        """Get elite solutions for breeding"""
        sorted_pop = sorted(self.population, 
                          key=lambda x: x.evaluation_result.total_score, 
                          reverse=True)
        num_elite = max(2, int(len(self.population) * self.elite_retention))
        return sorted_pop[:num_elite]
        
    def _update_best_solutions(self) -> None:
        """Update tracking of best solutions"""
        # Keep top 10 solutions overall
        all_time_best = sorted(self.population, 
                             key=lambda x: x.evaluation_result.total_score, 
                             reverse=True)[:10]
        self.best_solutions = all_time_best
        
    def _solution_to_dict(self, solution: ResearchSolution) -> Dict[str, Any]:
        """Convert solution object to dictionary for processing"""
        return {
            'id': solution.id,
            'content': solution.content,
            'category': solution.category,
            'title': solution.title,
            'description': solution.description,
            'generation': solution.generation,
            'system_parameters': {},  # Would be extracted from content
            'evaluation_result': asdict(solution.evaluation_result)
        }
        
    def _parse_generated_content(self, content: str, category: str) -> Dict[str, Any]:
        """Parse LLM-generated content into structured format"""
        # This is a simplified parser - could be more sophisticated
        lines = content.split('\n')
        
        # Extract title (look for first line that looks like a title)
        title = ''
        for line in lines[:5]:
            if line.strip() and not line.startswith('#'):
                title = line.strip()
                break
                
        # Extract description (first paragraph)
        description = ''
        paragraph_lines = []
        for line in lines:
            if line.strip():
                paragraph_lines.append(line.strip())
            elif paragraph_lines:
                break
        description = ' '.join(paragraph_lines[:3])  # First 3 sentences
        
        return {
            'content': content,
            'title': title or f"Quantum {category.replace('_', ' ').title()} Research",
            'description': description[:200] + "..." if len(description) > 200 else description,
            'category': category,
            'theoretical_framework': '',  # Could be extracted with more parsing
            'system_parameters': {},      # Could be extracted with regex
            'key_results': {},           # Could be extracted with NLP
            'experimental_considerations': {}
        }
        
    def _load_seed_research(self) -> Dict[str, Any]:
        """Load seed research data"""
        try:
            with open('data/seed_research.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load seed research: {e}")
            return {}
            
    def _log_generation_summary(self, stats: EvolutionStats, generation_time: float, improvement: float) -> None:
        """Log summary of generation results"""
        self.logger.info(f"📊 Generation {stats.generation} Summary:")
        self.logger.info(f"   Best Score: {stats.best_score:.3f} (↑{improvement:+.3f})")
        self.logger.info(f"   Avg Score:  {stats.average_score:.3f}")
        self.logger.info(f"   Diversity:  {stats.diversity_index:.3f}")
        self.logger.info(f"   New Solutions: {stats.new_solutions}")
        if stats.breakthrough_solutions > 0:
            self.logger.info(f"   🏆 Breakthroughs: {stats.breakthrough_solutions}")
        self.logger.info(f"   Time: {generation_time:.1f}s")
        
    def _log_evolution_summary(self, total_time: float) -> None:
        """Log final evolution summary"""
        self.logger.info(f"\n🎯 Evolution Complete!")
        self.logger.info(f"   Total Generations: {len(self.evolution_history)}")
        self.logger.info(f"   Total Time: {total_time:.1f}s")
        if self.population:
            best_solution = max(self.population, key=lambda x: x.evaluation_result.total_score)
            self.logger.info(f"   Best Score: {best_solution.evaluation_result.total_score:.3f}")
        else:
            self.logger.info(f"   Best Score: N/A")
        self.logger.info(f"   Breakthroughs: {len(self.breakthrough_solutions)}")
        model_used = "demo-mode" if self.demo_mode else (self.llm.current_model if self.llm else "unknown")
        self.logger.info(f"   Model Used: {model_used}")
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        if not self.evolution_history:
            return {}
            
        return {
            'total_generations': len(self.evolution_history),
            'best_score': max((stats['best_score'] for stats in self.evolution_history), default=0.0),
            'final_score': self.evolution_history[-1]['best_score'] if self.evolution_history else 0.0,
            'total_breakthroughs': len(self.breakthrough_solutions),
            'convergence_history': [stats['best_score'] for stats in self.evolution_history],
            'diversity_history': [stats['diversity_index'] for stats in self.evolution_history],
            'model_used': "demo-mode" if self.demo_mode else (self.llm.current_model if self.llm else "unknown"),
            'best_solutions': [asdict(sol) for sol in (self.best_solutions[:5] if self.best_solutions else [])]
        }
        
    def save_evolution_state(self, filepath: str = "data/evolution_state.json") -> None:
        """Save current evolution state"""
        state = {
            'current_generation': self.current_generation,
            'population': [asdict(sol) for sol in self.population],
            'evolution_history': self.evolution_history,  # Already dictionaries
            'best_solutions': [asdict(sol) for sol in self.best_solutions],
            'breakthrough_solutions': [asdict(sol) for sol in self.breakthrough_solutions],
            'summary': self.get_evolution_summary()
        }
        
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"💾 Evolution state saved to {filepath}")
    
    def set_amo_problem_mode(self, amo_problem) -> None:
        """Set evolution controller to AMO problem-solving mode"""
        self.amo_mode = True
        self.current_amo_problem = amo_problem
        self.amo_convergence_history = []
        self.logger.info(f"🔬 AMO problem mode enabled: {amo_problem.title}")
    
    def evolve_for_amo_problem(self, amo_problem, max_generations: int = 50) -> List[ResearchSolution]:
        """
        Evolve solutions specifically for an AMO problem.
        Uses parameter-targeted evolution with convergence detection.
        """
        self.set_amo_problem_mode(amo_problem)
        
        self.logger.info(f"🧬 Starting AMO problem evolution for: {amo_problem.title}")
        self.logger.info(f"🎯 Target parameters: {list(amo_problem.target_parameters.keys())}")
        
        # Generate initial population targeted at the problem
        initial_prompts = self.generator.generate_problem_targeted_prompts(
            self._amo_problem_to_dict(amo_problem), 
            self.population_size
        )
        
        # Initialize population with problem-targeted solutions
        self.population = []
        for i, prompt in enumerate(initial_prompts):
            try:
                if not self.demo_mode:
                    result = self.llm.generate_research(prompt.content, prompt.system_prompt)
                else:
                    result = self._generate_demo_content(prompt.category)
                
                # Evaluate using AMO problem-specific evaluation
                evaluation = self.evaluator.evaluate_for_amo_problem(
                    {
                        'content': result.content,
                        'category': prompt.category,
                        'title': f"AMO Solution {i+1}",
                        'description': result.content[:200] + "..." if len(result.content) > 200 else result.content
                    },
                    amo_problem
                )
                
                solution = ResearchSolution(
                    id=f"amo_gen0_sol{i}",
                    content=result.content,
                    category=prompt.category,
                    title=f"AMO Solution {i+1}",
                    description=evaluation.details.get('amo_problem_results', {}).get('achievement_count', 0),
                    generation=0,
                    parent_ids=[],
                    mutation_type='amo_initial',
                    evaluation_result=evaluation,
                    generation_metadata={'amo_mode': True, 'problem_id': amo_problem.id},
                    timestamp=time.time()
                )
                
                self.population.append(solution)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate initial AMO solution {i}: {e}")
        
        # Evolution loop with AMO-specific logic
        for generation in range(max_generations):
            self.current_generation = generation
            
            # Calculate AMO progress
            amo_progress = self._calculate_amo_progress(generation)
            self.amo_convergence_history.append(amo_progress)
            
            self.logger.info(f"🧬 AMO Generation {generation + 1}/{max_generations}")
            self.logger.info(f"🎯 Best score: {amo_progress['best_score']:.3f}")
            self.logger.info(f"🏆 Targets achieved: {amo_progress['targets_achieved']}/{amo_progress['total_targets']}")
            
            # Check for breakthrough
            if amo_progress['breakthrough_achieved']:
                self.logger.info(f"🎉 BREAKTHROUGH! Problem solved at generation {generation + 1}")
                break
            
            # Check for convergence
            if self._check_amo_convergence():
                self.logger.info(f"📈 AMO evolution converged at generation {generation + 1}")
                break
            
            # Generate next generation with AMO-specific strategies
            self._evolve_amo_generation()
            
            # Update best solutions
            self._update_best_solutions()
        
        self.logger.info(f"✅ AMO evolution completed after {self.current_generation + 1} generations")
        
        # Final progress report
        final_progress = self.amo_convergence_history[-1] if self.amo_convergence_history else {}
        self.logger.info(f"🎯 Final score: {final_progress.get('best_score', 0):.3f}")
        self.logger.info(f"🏆 Final targets: {final_progress.get('targets_achieved', 0)}/{final_progress.get('total_targets', 0)}")
        
        return self.best_solutions
    
    def _amo_problem_to_dict(self, amo_problem) -> Dict[str, Any]:
        """Convert AMO problem to dictionary for prompt generation"""
        return {
            'id': amo_problem.id,
            'title': amo_problem.title,
            'category': amo_problem.category,
            'description': amo_problem.description,
            'physics_challenge': amo_problem.physics_challenge,
            'target_parameters': {
                name: {
                    'target_value': param.target_value,
                    'units': param.units,
                    'description': param.description,
                    'weight': param.weight
                }
                for name, param in amo_problem.target_parameters.items()
            }
        }
    
    def _calculate_amo_progress(self, generation: int) -> Dict[str, Any]:
        """Calculate progress specifically for AMO problem solving"""
        if not self.population or not self.current_amo_problem:
            return {
                'generation': generation,
                'best_score': 0.0,
                'targets_achieved': 0,
                'total_targets': 0,
                'breakthrough_achieved': False,
                'convergence_indicator': 0.0
            }
        
        # Find best solution
        best_solution = max(self.population, key=lambda x: x.evaluation_result.total_score)
        best_score = best_solution.evaluation_result.total_score
        
        # Check target achievements
        amo_details = best_solution.evaluation_result.details.get('amo_problem_results', {})
        targets_achieved = amo_details.get('achievement_count', 0)
        total_targets = amo_details.get('total_parameters', len(self.current_amo_problem.target_parameters))
        
        # Check for breakthrough
        breakthrough_achieved = best_score >= self.amo_breakthrough_threshold
        
        # Calculate convergence indicator
        convergence_indicator = self._calculate_convergence_indicator()
        
        progress = {
            'generation': generation,
            'best_score': best_score,
            'targets_achieved': targets_achieved,
            'total_targets': total_targets,
            'breakthrough_achieved': breakthrough_achieved,
            'convergence_indicator': convergence_indicator,
            'best_solution_id': best_solution.id,
            'parameter_scores': amo_details.get('parameter_scores', {})
        }
        
        return progress
    
    def _check_amo_convergence(self) -> bool:
        """Check if AMO evolution has converged"""
        if len(self.amo_convergence_history) < 10:
            return False
        
        # Check for score plateau
        recent_scores = [p['best_score'] for p in self.amo_convergence_history[-10:]]
        score_variance = np.var(recent_scores)
        
        if score_variance < 0.001:  # Very small variance
            return True
        
        # Check for diminishing returns
        if len(self.amo_convergence_history) >= 20:
            early_avg = np.mean([p['best_score'] for p in self.amo_convergence_history[-20:-10]])
            recent_avg = np.mean([p['best_score'] for p in self.amo_convergence_history[-10:]])
            
            improvement = recent_avg - early_avg
            if improvement < 0.01:  # Less than 1% improvement
                return True
        
        return False
    
    def _calculate_convergence_indicator(self) -> float:
        """Calculate convergence indicator for AMO evolution"""
        if len(self.amo_convergence_history) < 5:
            return 0.0
        
        # Score stability
        recent_scores = [p['best_score'] for p in self.amo_convergence_history[-5:]]
        score_stability = 1.0 - np.std(recent_scores)
        
        # Target consistency
        recent_targets = [p['targets_achieved'] for p in self.amo_convergence_history[-5:]]
        target_stability = 1.0 - (np.std(recent_targets) / max(recent_targets[-1], 1))
        
        return min((score_stability + target_stability) / 2, 1.0)
    
    def _evolve_amo_generation(self) -> None:
        """Evolve generation with AMO-specific strategies"""
        if not self.population or not self.current_amo_problem:
            return
        
        new_population = []
        
        # Elite retention
        elite_count = max(1, int(0.2 * self.population_size))
        elites = sorted(self.population, key=lambda x: x.evaluation_result.total_score, reverse=True)[:elite_count]
        new_population.extend(elites)
        
        # Determine strategies based on current progress
        latest_progress = self.amo_convergence_history[-1] if self.amo_convergence_history else {}
        targets_achieved = latest_progress.get('targets_achieved', 0)
        total_targets = latest_progress.get('total_targets', 1)
        completion_ratio = targets_achieved / total_targets if total_targets > 0 else 0
        
        # Strategy selection based on progress
        if completion_ratio < 0.3:
            # Early stage - focus on exploration
            strategy_weights = {'parameter_optimization': 0.4, 'mutation': 0.3, 'breakthrough': 0.3}
        elif completion_ratio < 0.7:
            # Mid stage - focus on parameter optimization
            strategy_weights = {'parameter_optimization': 0.6, 'mutation': 0.2, 'breakthrough': 0.2}
        else:
            # Late stage - focus on breakthrough seeking
            strategy_weights = {'parameter_optimization': 0.3, 'mutation': 0.2, 'breakthrough': 0.5}
        
        # Generate new solutions based on strategies
        while len(new_population) < self.population_size:
            strategy = np.random.choice(
                list(strategy_weights.keys()),
                p=list(strategy_weights.values())
            )
            
            try:
                if strategy == 'parameter_optimization':
                    new_solution = self._generate_parameter_optimization_solution()
                elif strategy == 'breakthrough':
                    new_solution = self._generate_breakthrough_solution()
                else:  # mutation
                    new_solution = self._generate_mutation_solution()
                
                if new_solution:
                    new_population.append(new_solution)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate {strategy} solution: {e}")
        
        # Update population
        self.population = new_population[:self.population_size]
    
    def _generate_parameter_optimization_solution(self) -> Optional[ResearchSolution]:
        """Generate solution focused on optimizing specific parameters"""
        if not self.population or not self.current_amo_problem:
            return None
        
        # Select parent solution
        parent = random.choice(self.population[:min(5, len(self.population))])  # Top 5
        
        # Identify missing targets
        amo_details = parent.evaluation_result.details.get('amo_problem_results', {})
        parameter_scores = amo_details.get('parameter_scores', {})
        
        missing_targets = []
        for param_name in self.current_amo_problem.target_parameters.keys():
            if param_name not in parameter_scores or parameter_scores[param_name].get('score', 0) < 0.8:
                missing_targets.append(param_name)
        
        if not missing_targets:
            missing_targets = list(self.current_amo_problem.target_parameters.keys())
        
        # Generate optimization prompts
        optimization_prompts = self.generator.generate_parameter_optimization_prompts(
            {
                'content': parent.content,
                'category': parent.category,
                'id': parent.id,
                'generation': parent.generation,
                'parameters': amo_details.get('extracted_parameters', {})
            },
            {name: param.__dict__ for name, param in self.current_amo_problem.target_parameters.items()},
            missing_targets[:3]  # Focus on top 3 missing
        )
        
        if not optimization_prompts:
            return None
        
        prompt = random.choice(optimization_prompts)
        
        # Generate content
        if not self.demo_mode:
            result = self.llm.generate_research(prompt.content, prompt.system_prompt)
        else:
            result = self._generate_demo_content(prompt.category)
        
        # Evaluate
        evaluation = self.evaluator.evaluate_for_amo_problem(
            {
                'content': result.content,
                'category': prompt.category,
                'title': f"Optimized {prompt.mutation_type}",
                'description': result.content[:200] + "..." if len(result.content) > 200 else result.content
            },
            self.current_amo_problem
        )
        
        return ResearchSolution(
            id=f"amo_gen{self.current_generation + 1}_opt_{int(time.time() * 1000) % 10000}",
            content=result.content,
            category=prompt.category,
            title=f"Optimized {prompt.mutation_type}",
            description=evaluation.details.get('amo_problem_results', {}).get('achievement_count', 0),
            generation=self.current_generation + 1,
            parent_ids=[parent.id],
            mutation_type=prompt.mutation_type,
            evaluation_result=evaluation,
            generation_metadata={'amo_mode': True, 'strategy': 'parameter_optimization'},
            timestamp=time.time()
        )
    
    def _generate_breakthrough_solution(self) -> Optional[ResearchSolution]:
        """Generate solution using breakthrough-seeking strategies"""
        if not self.current_amo_problem:
            return None
        
        # Get current best score
        best_score = max((s.evaluation_result.total_score for s in self.population), default=0.0)
        
        # Generate breakthrough prompts
        breakthrough_prompts = self.generator.generate_breakthrough_seeking_prompts(
            self._amo_problem_to_dict(self.current_amo_problem),
            best_score
        )
        
        if not breakthrough_prompts:
            return None
        
        prompt = random.choice(breakthrough_prompts)
        
        # Generate content
        if not self.demo_mode:
            result = self.llm.generate_research(prompt.content, prompt.system_prompt)
        else:
            result = self._generate_demo_content(prompt.category)
        
        # Evaluate
        evaluation = self.evaluator.evaluate_for_amo_problem(
            {
                'content': result.content,
                'category': prompt.category,
                'title': f"Breakthrough {prompt.mutation_type}",
                'description': result.content[:200] + "..." if len(result.content) > 200 else result.content
            },
            self.current_amo_problem
        )
        
        return ResearchSolution(
            id=f"amo_gen{self.current_generation + 1}_break_{int(time.time() * 1000) % 10000}",
            content=result.content,
            category=prompt.category,
            title=f"Breakthrough {prompt.mutation_type}",
            description=evaluation.details.get('amo_problem_results', {}).get('achievement_count', 0),
            generation=self.current_generation + 1,
            parent_ids=[],
            mutation_type=prompt.mutation_type,
            evaluation_result=evaluation,
            generation_metadata={'amo_mode': True, 'strategy': 'breakthrough'},
            timestamp=time.time()
        )
    
    def _generate_mutation_solution(self) -> Optional[ResearchSolution]:
        """Generate solution using standard mutation"""
        if not self.population:
            return None
        
        # Select parent
        parent = self._select_parent_for_mutation()
        
        # Create mutation prompt
        solution_data = {
            'id': parent.id,
            'content': parent.content,
            'category': parent.category,
            'generation': parent.generation
        }
        
        mutation_prompt = self.generator.mutate_solution(solution_data)
        
        # Generate content
        if not self.demo_mode:
            result = self.llm.generate_research(mutation_prompt.content, mutation_prompt.system_prompt)
        else:
            result = self._generate_demo_content(mutation_prompt.category)
        
        # Evaluate
        evaluation = self.evaluator.evaluate_for_amo_problem(
            {
                'content': result.content,
                'category': mutation_prompt.category,
                'title': f"Mutated {parent.title}",
                'description': result.content[:200] + "..." if len(result.content) > 200 else result.content
            },
            self.current_amo_problem
        )
        
        return ResearchSolution(
            id=f"amo_gen{self.current_generation + 1}_mut_{int(time.time() * 1000) % 10000}",
            content=result.content,
            category=mutation_prompt.category,
            title=f"Mutated {parent.title}",
            description=evaluation.details.get('amo_problem_results', {}).get('achievement_count', 0),
            generation=self.current_generation + 1,
            parent_ids=[parent.id],
            mutation_type=mutation_prompt.mutation_type,
            evaluation_result=evaluation,
            generation_metadata={'amo_mode': True, 'strategy': 'mutation'},
            timestamp=time.time()
        )
    
    def _select_parent_for_mutation(self) -> ResearchSolution:
        """Select parent solution for mutation using tournament selection"""
        tournament_size = min(5, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.evaluation_result.total_score)
    
    def get_amo_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of AMO evolution"""
        if not self.amo_mode or not self.amo_convergence_history:
            return {}
        
        final_progress = self.amo_convergence_history[-1]
        
        return {
            'problem_id': self.current_amo_problem.id if self.current_amo_problem else None,
            'problem_title': self.current_amo_problem.title if self.current_amo_problem else None,
            'total_generations': len(self.amo_convergence_history),
            'breakthrough_achieved': final_progress.get('breakthrough_achieved', False),
            'final_score': final_progress.get('best_score', 0.0),
            'targets_achieved': final_progress.get('targets_achieved', 0),
            'total_targets': final_progress.get('total_targets', 0),
            'convergence_history': [p['best_score'] for p in self.amo_convergence_history],
            'target_progress': [p['targets_achieved'] for p in self.amo_convergence_history],
            'parameter_details': final_progress.get('parameter_scores', {}),
            'best_solution_id': final_progress.get('best_solution_id'),
            'model_used': "demo-mode" if self.demo_mode else (self.llm.current_model if self.llm else "unknown")
        }