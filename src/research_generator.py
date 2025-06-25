import random
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ResearchPrompt:
    content: str
    category: str
    system_prompt: str
    mutation_type: Optional[str] = None
    parent_id: Optional[str] = None
    generation: int = 0

class QuantumResearchGenerator:
    """Generate quantum optics research prompts for evolution system"""
    
    def __init__(self, prompts_path: str = "config/prompts.yaml"):
        self.prompts_path = Path(prompts_path)
        self.logger = logging.getLogger('QuantumResearchGenerator')
        self.prompts_config = self._load_prompts()
        
        # Extract prompt categories and templates
        self.categories = self.prompts_config.get('categories', {})
        self.evolution_prompts = self.prompts_config.get('evolution_prompts', {})
        self.model_prompts = self.prompts_config.get('model_specific_prompts', {})
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompt configuration with validation"""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate the configuration structure
            self._validate_prompts_config(config)
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load prompts from {self.prompts_path}: {e}")
            return self._get_default_prompts()
    
    def _validate_prompts_config(self, config: Dict[str, Any]) -> None:
        """Validate prompts configuration for template consistency"""
        try:
            # Check required sections
            required_sections = ['categories', 'evolution_prompts']
            for section in required_sections:
                if section not in config:
                    self.logger.warning(f"Missing required section: {section}")
            
            # Validate evolution prompts templates
            if 'evolution_prompts' in config:
                evolution_prompts = config['evolution_prompts']
                
                # Check mutation strategies templates
                if 'mutation_strategies' in evolution_prompts:
                    for strategy, templates in evolution_prompts['mutation_strategies'].items():
                        for i, template in enumerate(templates):
                            try:
                                # Test template with dummy variables
                                test_vars = {
                                    'system_description': 'test',
                                    'parameter_name': 'test',
                                    'objective_function': 'test',
                                    'constraint': 'test',
                                    'new_environment': 'test',
                                    'scale_factor': '1',
                                    'particle_number': '100',
                                    'approach_1': 'test',
                                    'approach_2': 'test',
                                    'field_1': 'test',
                                    'field_2': 'test'
                                }
                                template.format(**test_vars)
                            except KeyError as e:
                                self.logger.warning(f"Template validation failed for {strategy}[{i}]: missing variable {e}")
                            except Exception as e:
                                self.logger.warning(f"Template validation error for {strategy}[{i}]: {e}")
                
                # Check exploration prompts templates
                if 'exploration_prompts' in evolution_prompts:
                    for strategy, templates in evolution_prompts['exploration_prompts'].items():
                        for i, template in enumerate(templates):
                            try:
                                # Test template with dummy variables
                                test_vars = {
                                    'application_area': 'test',
                                    'quantum_system_type': 'test',
                                    'exotic_material': 'test',
                                    'unusual_geometry': 'test',
                                    'biological_system': 'test',
                                    'classical_analog': 'test',
                                    'other_field': 'test',
                                    'other_physics_field': 'test'
                                }
                                template.format(**test_vars)
                            except KeyError as e:
                                self.logger.warning(f"Exploration template validation failed for {strategy}[{i}]: missing variable {e}")
                            except Exception as e:
                                self.logger.warning(f"Exploration template validation error for {strategy}[{i}]: {e}")
            
            self.logger.info("Prompts configuration validation completed")
            
        except Exception as e:
            self.logger.warning(f"Prompts validation failed: {e}")
            
    def generate_initial_prompts(self, num_prompts: int = 10) -> List[ResearchPrompt]:
        """Generate initial research prompts from base categories"""
        prompts = []
        
        # Get all available categories and their prompts
        all_prompts = []
        for category, config in self.categories.items():
            system_prompt = config.get('system_prompt', '')
            
            # Collect all prompt levels (basic, advanced, experimental, etc.)
            for level, prompt_list in config.get('prompts', {}).items():
                for prompt_text in prompt_list:
                    all_prompts.append((prompt_text, category, system_prompt))
        
        # Sample random prompts
        selected = random.sample(all_prompts, min(num_prompts, len(all_prompts)))
        
        for i, (prompt_text, category, system_prompt) in enumerate(selected):
            prompts.append(ResearchPrompt(
                content=prompt_text,
                category=category,
                system_prompt=system_prompt,
                generation=0
            ))
            
        self.logger.info(f"Generated {len(prompts)} initial research prompts")
        return prompts
        
    def mutate_solution(self, solution: Dict[str, Any], mutation_strategies: List[str] = None) -> ResearchPrompt:
        """Generate mutated research prompt from existing solution"""
        
        try:
            if mutation_strategies is None:
                available_strategies = list(self.evolution_prompts.get('mutation_strategies', {}).keys())
                if not available_strategies:
                    # Fallback if no strategies available
                    mutation_strategies = ['parameter_mutation']
                else:
                    mutation_strategies = available_strategies
                
            # Select random mutation strategy
            strategy = random.choice(mutation_strategies)
            
            # Get mutation templates for the strategy
            if strategy not in self.evolution_prompts.get('mutation_strategies', {}):
                # Fallback if strategy not found
                self.logger.warning(f"Mutation strategy {strategy} not found. Using fallback.")
                mutation_prompt = self._create_fallback_mutation(solution)
            else:
                mutation_templates = self.evolution_prompts['mutation_strategies'][strategy]
                if not mutation_templates:
                    # Fallback if no templates
                    mutation_prompt = self._create_fallback_mutation(solution)
                else:
                    # Select random template
                    template = random.choice(mutation_templates)
                    
                    # Extract information from solution for mutation
                    system_description = self._extract_system_description(solution)
                    
                    # Generate mutation prompt based on strategy
                    mutation_prompt = self._apply_mutation_strategy(template, strategy, solution, system_description)
            
            category = solution.get('category', 'general')
            system_prompt = self._get_system_prompt_for_category(category)
            
            return ResearchPrompt(
                content=mutation_prompt,
                category=category,
                system_prompt=system_prompt,
                mutation_type=strategy,
                parent_id=solution.get('id'),
                generation=solution.get('generation', 0) + 1
            )
            
        except Exception as e:
            self.logger.error(f"Failed to mutate solution: {e}. Using fallback.")
            return self._create_fallback_prompt(solution, 'mutation')
    
    def _create_fallback_mutation(self, solution: Dict[str, Any]) -> str:
        """Create fallback mutation prompt when templates fail"""
        system_description = self._extract_system_description(solution)
        return f"Improve and modify the following quantum optical system: {system_description}. Focus on enhancing its quantum properties and performance characteristics."
    
    def _create_fallback_prompt(self, solution: Dict[str, Any], prompt_type: str) -> ResearchPrompt:
        """Create fallback research prompt when all else fails"""
        category = solution.get('category', 'cavity_qed')
        if prompt_type == 'mutation':
            content = self._create_fallback_mutation(solution)
        else:
            content = f"Design an innovative quantum optical system in the {category} domain."
        
        return ResearchPrompt(
            content=content,
            category=category,
            system_prompt=self._get_system_prompt_for_category(category),
            mutation_type=f'fallback_{prompt_type}',
            parent_id=solution.get('id', 'unknown'),
            generation=solution.get('generation', 0) + 1
        )
        
    def crossover_solutions(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> ResearchPrompt:
        """Generate research prompt by combining two solutions"""
        
        # Extract key concepts from both solutions
        concept1 = self._extract_key_concepts(solution1)
        concept2 = self._extract_key_concepts(solution2)
        
        # Use concept fusion mutation template
        fusion_templates = self.evolution_prompts['mutation_strategies']['concept_fusion']
        template = random.choice(fusion_templates)
        
        # Create crossover prompt with proper template variables
        try:
            # Prepare all possible template variables
            template_vars = {
                'approach_1': concept1,
                'approach_2': concept2,
                'field_1': solution1.get('category', 'quantum optics').replace('_', ' '),
                'field_2': solution2.get('category', 'quantum optics').replace('_', ' ')
            }
            
            crossover_prompt = template.format(**template_vars)
            
        except (KeyError, AttributeError) as e:
            # Fallback if template doesn't have expected placeholders
            self.logger.warning(f"Template formatting failed: {e}. Using fallback.")
            crossover_prompt = f"Combine these two quantum optics approaches: {concept1} and {concept2}. Create a novel hybrid system that leverages the strengths of both approaches."
        
        # Choose dominant category
        category = random.choice([solution1.get('category', 'general'), solution2.get('category', 'general')])
        system_prompt = self._get_system_prompt_for_category(category)
        
        return ResearchPrompt(
            content=crossover_prompt,
            category=category,
            system_prompt=system_prompt,
            mutation_type='crossover',
            parent_id=f"{solution1.get('id', 'unknown')}+{solution2.get('id', 'unknown')}",
            generation=max(solution1.get('generation', 0), solution2.get('generation', 0)) + 1
        )
        
    def generate_exploration_prompts(self, num_prompts: int = 3) -> List[ResearchPrompt]:
        """Generate exploratory research prompts for novelty"""
        
        prompts = []
        exploration_strategies = list(self.evolution_prompts.get('exploration_prompts', {}).keys())
        
        for _ in range(num_prompts):
            strategy = random.choice(exploration_strategies)
            templates = self.evolution_prompts['exploration_prompts'][strategy]
            template = random.choice(templates)
            
            # Fill template with random parameters
            filled_prompt = self._fill_exploration_template(template, strategy)
            
            # Random category for exploration
            category = random.choice(list(self.categories.keys()))
            system_prompt = self._get_system_prompt_for_category(category)
            
            prompts.append(ResearchPrompt(
                content=filled_prompt,
                category=category,
                system_prompt=system_prompt,
                mutation_type=f'exploration_{strategy}',
                generation=0
            ))
            
        return prompts
        
    def _apply_mutation_strategy(self, template: str, strategy: str, solution: Dict[str, Any], system_description: str) -> str:
        """Apply specific mutation strategy to generate prompt"""
        
        try:
            if strategy == 'parameter_mutation':
                # Extract a parameter to mutate
                parameters = solution.get('system_parameters', {})
                if parameters:
                    param_name = random.choice(list(parameters.keys()))
                    objective = self._get_random_objective()
                else:
                    param_name = "coupling strength"
                    objective = "quantum efficiency"
                
                template_vars = {
                    'system_description': system_description,
                    'parameter_name': param_name,
                    'objective_function': objective
                }
                return template.format(**template_vars)
                    
            elif strategy == 'constraint_exploration':
                constraint = self._get_random_constraint()
                environment = self._get_random_environment()
                template_vars = {
                    'constraint': constraint,
                    'system_description': system_description,
                    'new_environment': environment
                }
                return template.format(**template_vars)
                
            elif strategy == 'scale_variation':
                scale_factor = random.choice([0.1, 0.5, 2, 10, 100])
                particle_number = random.choice([10, 100, 1000, 10000])
                template_vars = {
                    'system_description': system_description,
                    'scale_factor': scale_factor,
                    'particle_number': particle_number
                }
                return template.format(**template_vars)
                
            else:
                # Default case - return template with system description
                return template.format(system_description=system_description)
                
        except (KeyError, AttributeError) as e:
            # Fallback for any template formatting issues
            self.logger.warning(f"Template formatting failed for {strategy}: {e}. Using fallback.")
            return f"Modify and improve the following quantum optical system: {system_description}. Focus on enhancing its performance and exploring new capabilities."
            
    def _fill_exploration_template(self, template: str, strategy: str) -> str:
        """Fill exploration template with random parameters"""
        
        try:
            if strategy == 'novel_phenomena':
                application_area = random.choice([
                    'quantum computing', 'quantum sensing', 'quantum communication',
                    'quantum metrology', 'quantum simulation', 'quantum cryptography'
                ])
                quantum_system_type = random.choice([
                    'cavity QED systems', 'optomechanical systems', 'atomic ensembles',
                    'photonic systems', 'hybrid quantum systems'
                ])
                template_vars = {
                    'application_area': application_area,
                    'quantum_system_type': quantum_system_type
                }
                return template.format(**template_vars)
                
            elif strategy == 'unconventional_systems':
                exotic_material = random.choice([
                    'graphene', 'metamaterials', 'photonic crystals', 'superconducting circuits',
                    'diamond NV centers', 'quantum dots', 'topological insulators'
                ])
                unusual_geometry = random.choice([
                    'fractal structures', 'chiral geometries', 'twisted configurations',
                    'ring resonators', 'spiral cavities', 'hierarchical arrays'
                ])
                biological_system = random.choice([
                    'photosynthetic complexes', 'microtubules', 'protein folding',
                    'neural networks', 'bird navigation', 'enzyme catalysis'
                ])
                classical_analog = random.choice([
                    'pendulum oscillators', 'coupled springs', 'wave interference',
                    'acoustic resonators', 'electronic circuits', 'fluid dynamics'
                ])
                template_vars = {
                    'exotic_material': exotic_material,
                    'unusual_geometry': unusual_geometry,
                    'biological_system': biological_system,
                    'classical_analog': classical_analog
                }
                return template.format(**template_vars)
                
            elif strategy == 'interdisciplinary':
                other_field = random.choice([
                    'condensed matter physics', 'astrophysics', 'biophysics',
                    'materials science', 'chemistry', 'neuroscience'
                ])
                other_physics_field = random.choice([
                    'high energy physics', 'general relativity', 'statistical mechanics',
                    'plasma physics', 'solid state physics', 'nuclear physics'
                ])
                template_vars = {
                    'other_field': other_field,
                    'other_physics_field': other_physics_field
                }
                return template.format(**template_vars)
                
        except (KeyError, AttributeError) as e:
            # Fallback for any template formatting issues
            self.logger.warning(f"Exploration template formatting failed for {strategy}: {e}. Using fallback.")
            return f"Explore novel quantum optics approaches in the {strategy} domain. Focus on breakthrough physics and innovative applications."
            
        # Default - return template as is if no specific strategy matches
        return template
        
    def _extract_system_description(self, solution: Dict[str, Any]) -> str:
        """Extract concise system description from solution"""
        title = solution.get('title', '')
        description = solution.get('description', '')
        category = solution.get('category', '')
        
        if title:
            return f"{title}: {description[:100]}..." if len(description) > 100 else f"{title}: {description}"
        elif description:
            return description[:150] + "..." if len(description) > 150 else description
        else:
            return f"quantum {category} system"
            
    def _extract_key_concepts(self, solution: Dict[str, Any]) -> str:
        """Extract key physical concepts from solution"""
        title = solution.get('title', '')
        description = solution.get('description', '')
        content = solution.get('content', '')
        category = solution.get('category', '')
        
        # Try to extract meaningful concepts from title first
        if title and len(title.strip()) > 0:
            return title.strip()
        
        # If no title, try description
        if description and len(description.strip()) > 0:
            # Take first sentence or up to 100 chars
            first_part = description.split('.')[0]
            if len(first_part) <= 100:
                return first_part.strip()
            else:
                return description[:100].strip() + "..."
        
        # If no description, try content
        if content and len(content.strip()) > 0:
            # Take first line or up to 150 chars
            first_line = content.split('\n')[0]
            if len(first_line) <= 150:
                return first_line.strip()
            else:
                return content[:150].strip() + "..."
        
        # Fallback based on category
        category_defaults = {
            'cavity_qed': 'cavity QED system with strong atom-photon coupling',
            'squeezed_light': 'squeezed light generation using parametric processes',
            'photon_blockade': 'photon blockade in nonlinear optical systems',
            'quantum_metrology': 'quantum-enhanced sensing and metrology',
            'optomechanics': 'optomechanical coupling and cooling'
        }
        
        return category_defaults.get(category, f"{category} quantum optical system")
            
    def _get_system_prompt_for_category(self, category: str) -> str:
        """Get appropriate system prompt for research category"""
        if category in self.categories:
            return self.categories[category].get('system_prompt', '')
        else:
            return "You are a quantum optics expert. Provide rigorous theoretical analysis with mathematical derivations."
            
    def _get_random_objective(self) -> str:
        """Get random optimization objective"""
        objectives = [
            'quantum efficiency', 'coupling strength', 'coherence time',
            'fidelity', 'squeezing level', 'entanglement generation rate',
            'sensitivity', 'bandwidth', 'stability', 'scalability'
        ]
        return random.choice(objectives)
        
    def _get_random_constraint(self) -> str:
        """Get random experimental constraint"""
        constraints = [
            'room temperature operation', 'single photon level',
            'fiber-compatible wavelengths', 'CMOS-compatible fabrication',
            'magnetic field insensitive', 'vibration tolerant',
            'low power consumption', 'rapid switching capability'
        ]
        return random.choice(constraints)
        
    def _get_random_environment(self) -> str:
        """Get random experimental environment"""
        environments = [
            'space conditions', 'underwater environment', 'high magnetic fields',
            'cryogenic temperatures', 'mobile platforms', 'noisy environments',
            'high radiation conditions', 'extreme miniaturization'
        ]
        return random.choice(environments)
        
    def adapt_prompt_for_model(self, prompt: ResearchPrompt, model_reasoning_style: str) -> ResearchPrompt:
        """Adapt prompt for specific model reasoning style"""
        
        if model_reasoning_style not in self.model_prompts:
            return prompt  # Return unchanged if style not recognized
            
        style_config = self.model_prompts[model_reasoning_style]
        adapted_system_prompt = style_config.get('system_prompt', prompt.system_prompt)
        instruction_suffix = style_config.get('instruction_suffix', '')
        
        adapted_content = prompt.content + instruction_suffix
        
        return ResearchPrompt(
            content=adapted_content,
            category=prompt.category,
            system_prompt=adapted_system_prompt,
            mutation_type=prompt.mutation_type,
            parent_id=prompt.parent_id,
            generation=prompt.generation
        )
        
    def get_category_statistics(self) -> Dict[str, int]:
        """Get statistics about available prompt categories"""
        stats = {}
        for category, config in self.categories.items():
            total_prompts = 0
            for level, prompt_list in config.get('prompts', {}).items():
                total_prompts += len(prompt_list)
            stats[category] = total_prompts
        return stats
        
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Provide default prompts if config file not found"""
        return {
            'categories': {
                'cavity_qed': {
                    'system_prompt': 'You are a quantum optics expert specializing in cavity QED.',
                    'prompts': {
                        'basic': [
                            'Design a cavity QED system for strong coupling between single atom and optical mode.'
                        ]
                    }
                }
            },
            'evolution_prompts': {
                'mutation_strategies': {
                    'parameter_mutation': [
                        'Modify the following quantum optical system by changing one key parameter: {system_description}'
                    ]
                }
            }
        }