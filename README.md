# Coheron

**An evolutionary AI system for breakthrough quantum optics research discovery**

Inspired by AlphaEvolve, Coheron uses advanced language models to generate, evaluate, and evolve quantum optics research solutions through physics-aware evaluation and evolutionary algorithms. The system automatically discovers novel research approaches by evolving populations of solutions across generations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![A4F API](https://img.shields.io/badge/A4F-Compatible-green.svg)](https://www.a4f.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🧬 Evolutionary Research Discovery**: AlphaEvolve-inspired architecture for systematic research evolution
- **🔬 Physics-Aware Evaluation**: Rigorous evaluation against analytical benchmarks and experimental feasibility
- **🔄 Flexible Model Switching**: Easy switching between Gemini, Claude, GPT-4, and other models via A4F
- **📊 Comprehensive Analytics**: Track research evolution, breakthrough discoveries, and model performance
- **🎯 Specialized Categories**: Focus on cavity QED, squeezed light, photon blockade, quantum metrology, and optomechanics
- **💾 Research Database**: SQLite-based storage for evolution tracking and lineage analysis
- **🖥️ Windows Compatible**: Unicode-safe logging and console output for all platforms

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- A4F API key ([Get one here](https://www.a4f.co/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Coheron
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   # Set your A4F API key
   export A4F_API_KEY="ddc-a4f-your-api-key-here"
   
   # Or on Windows:
   set A4F_API_KEY=ddc-a4f-your-api-key-here
   ```

4. **Run your first evolution**
   ```bash
   python src/main.py evolve --generations 10
   ```

## 📖 Usage Guide

### Basic Commands

```bash
# Run research evolution (default: Gemini 2.5 Flash)
python src/main.py evolve --generations 20

# Switch to a different model  
python src/main.py evolve --model "anthropic/claude-3.5-sonnet" --generations 15

# Focus on specific research category
python src/main.py evolve --category cavity_qed --generations 25

# Evaluate research content
python src/main.py evaluate --content "Design a cavity QED system..." --category cavity_qed

# Analyze results and generate plots
python src/main.py analyze --database --plots

# Test system components
python src/main.py test --full

# Benchmark different models
python src/main.py benchmark --compare-models
```

### Evolution Parameters Explained

**Generations**: Number of evolutionary cycles. Each generation:
- Evaluates current population solutions
- Selects elite solutions for survival
- Creates new solutions through mutation and crossover
- Introduces novel exploratory solutions

**Population**: Number of research solutions maintained per generation. Larger populations explore more diverse approaches but require more computational resources.

### Advanced Usage Examples

#### Custom Evolution Parameters
```bash
python src/main.py evolve \
  --generations 50 \
  --population 15 \
  --model "anthropic/claude-3.5-sonnet" \
  --output-dir custom_results \
  --save-interval 3
```

#### Demo Mode (No API Key Required)
```bash
# Test the system without API calls
python src/main.py evolve --demo --generations 5 --population 3
```

#### Detailed Research Evaluation
```bash
python src/main.py evaluate \
  --content research_paper.txt \
  --category quantum_metrology \
  --detailed \
  --save-result evaluation_results.json
```

#### Database Analysis
```bash
python src/main.py analyze \
  --database \
  --category squeezed_light \
  --plots \
  --export analysis_export.json
```

## 🎨 Custom Prompts and Configuration

### Custom Research Prompts

Edit `config/prompts.yaml` to customize research generation:

```yaml
# Add your custom quantum optics research prompts
custom_research_prompts:
  cavity_qed:
    base_prompt: |
      Design a novel cavity QED system that achieves strong coupling between {atom_type} and {cavity_mode}.
      Consider {coupling_mechanism} and optimize for {target_metric}.
      
      Requirements:
      - Coupling strength: g > {min_coupling} Hz
      - Quality factor: Q > {min_quality}
      - Operating temperature: {temperature}
      
      Provide detailed analysis including:
      1. System architecture and parameters
      2. Theoretical predictions
      3. Experimental feasibility
      4. Performance benchmarks

    mutation_prompts:
      - "Modify the cavity geometry to enhance {parameter}"
      - "Explore alternative {component} designs"
      - "Optimize for {new_metric} instead of {old_metric}"
      
    crossover_prompt: |
      Combine the following two cavity QED approaches:
      
      Approach 1: {approach_1}
      Approach 2: {approach_2}
      
      Create a hybrid system that leverages the strengths of both approaches.

# Custom evaluation criteria
custom_evaluation:
  cavity_qed:
    physics_checks:
      - "Verify coupling strength calculation: g = ..."
      - "Check cooperativity C = 4g²/(κγ)"
      - "Validate quality factor estimates"
    
    benchmarks:
      - name: "Strong coupling threshold"
        condition: "g > sqrt(κγ/4)"
        points: 10
      
      - name: "Room temperature operation"
        condition: "kT < ħω"
        points: 15
```

### Custom Research Categories

Add new categories by extending the configuration:

```yaml
# In config/config.yaml
research_categories:
  # Existing categories...
  
  quantum_sensing:
    description: "Quantum-enhanced sensing and metrology"
    focus_areas:
      - "Atomic interferometry"
      - "Spin squeezing"
      - "Quantum Fisher information"
    
    prompts:
      base: "Design a quantum sensing protocol that..."
      mutations: ["Enhance sensitivity by...", "Reduce decoherence through..."]
    
    evaluation_weights:
      sensitivity: 0.4
      robustness: 0.3
      practicality: 0.3
```

### Model Configuration

Configure different models in `config/config.yaml`:

```yaml
api:
  provider: "a4f"
  api_key: "${A4F_API_KEY}"
  base_url: "https://api.a4f.co/v1"

# Current model (change this to switch models)
current_model: "provider-5/gemini-2.5-flash-preview-04-17"

# Available models
models:
  "provider-5/gemini-2.5-flash-preview-04-17":
    name: "Gemini 2.5 Flash"
    max_tokens: 8192
    temperature: 0.7
    cost_per_1k_tokens: 0.0005
    strengths: ["Fast", "Cost-effective", "Good reasoning"]
    
  "anthropic/claude-3.5-sonnet":
    name: "Claude 3.5 Sonnet"
    max_tokens: 4096
    temperature: 0.7
    cost_per_1k_tokens: 0.003
    strengths: ["Physics expertise", "Analytical", "Detailed explanations"]
    
  "openai/gpt-4-turbo":
    name: "GPT-4 Turbo"
    max_tokens: 4096
    temperature: 0.7
    cost_per_1k_tokens: 0.01
    strengths: ["Structured output", "Mathematical precision", "Code generation"]
```

### Custom Evaluation Criteria

Extend `config/evaluation_criteria.yaml`:

```yaml
# Custom scoring rubrics
custom_scoring:
  feasibility:
    energy_conservation:
      weight: 0.3
      check: "Verify energy balance in all processes"
      scoring:
        perfect: "All energy terms accounted for"
        good: "Minor approximations justified"
        poor: "Energy conservation violated"
    
    material_properties:
      weight: 0.4
      check: "Realistic material parameters"
      benchmarks:
        - "Refractive index within known range"
        - "Loss rates experimentally achievable"
        - "Temperature requirements realistic"
  
  novelty:
    parameter_space:
      weight: 0.5
      check: "Explores new parameter regimes"
      criteria:
        - "Previously unexplored coupling strengths"
        - "Novel material combinations"
        - "Innovative system architectures"

# Custom physics benchmarks
physics_benchmarks:
  cavity_qed:
    analytical_solutions:
      jaynes_cummings:
        description: "Two-level atom in cavity"
        parameters: ["g", "ω_a", "ω_c", "κ", "γ"]
        solutions:
          vacuum_rabi: "Ω_R = 2g"
          cooperativity: "C = 4g²/(κγ)"
          critical_coupling: "g_c = √(κγ)/2"
        
      dressed_states:
        description: "Strong coupling eigenstates"
        formulas:
          splitting: "2g√(n+1)"
          frequencies: "ω_± = (ω_a + ω_c)/2 ± g√(n+1)"
```

### Seed Research Customization

Provide high-quality starting examples in `data/seed_research.json`:

```json
{
  "cavity_qed": [
    {
      "title": "Ultra-strong coupling in circuit QED",
      "content": "Design of superconducting circuit achieving g/ω = 0.2...",
      "category": "cavity_qed",
      "parameters": {
        "coupling_strength": 1e9,
        "quality_factor": 1e6,
        "cooperativity": 1000
      },
      "evaluation_scores": {
        "feasibility": 0.9,
        "mathematics": 0.95,
        "novelty": 0.8,
        "performance": 0.85
      }
    }
  ],
  
  "custom_category": [
    {
      "title": "Your custom research example",
      "content": "Detailed description of your approach...",
      "category": "custom_category",
      "parameters": {...},
      "evaluation_scores": {...}
    }
  ]
}
```

## 🏗️ System Architecture

### Core Components

```
Coheron/
├── src/
│   ├── main.py                 # CLI interface and application entry
│   ├── evolution_controller.py # Main evolution loop (AlphaEvolve-inspired)
│   ├── llm_interface.py        # A4F API integration with model switching
│   ├── research_generator.py   # Quantum research prompt generation
│   ├── evaluator.py            # Physics-aware evaluation system
│   ├── database.py             # Research storage and tracking
│   └── utils.py                # Utility functions and helpers
├── config/
│   ├── config.yaml             # Model and system configuration
│   ├── prompts.yaml            # Research prompt templates
│   └── evaluation_criteria.yaml # Scoring rubrics and benchmarks
├── data/
│   ├── seed_research.json      # Initial high-quality examples
│   ├── benchmarks.json         # Analytical solution benchmarks
│   └── research_history.db     # Evolution tracking database
└── results/                    # Generated results and analysis
```

### Evolution Process Flow

1. **Initialization Phase**
   - Load seed research examples from `data/seed_research.json`
   - Generate initial population using research prompts
   - Establish baseline evaluation scores

2. **Generation Loop** (Repeated for specified generations)
   - **Selection**: Choose elite solutions based on evaluation scores
   - **Mutation**: Create variations of successful solutions
   - **Crossover**: Combine features from multiple parent solutions
   - **Exploration**: Generate novel solutions for diversity
   - **Evaluation**: Score all candidates using physics-aware criteria
   - **Survival**: Select best solutions for next generation

3. **Analysis and Export**
   - Track evolution progress and identify breakthroughs
   - Generate visualization plots and comprehensive reports
   - Store results in database for future analysis

## 🎯 Research Categories Deep Dive

### Cavity QED
**Focus**: Single atom-photon interactions in optical cavities

Key Research Areas:
- **Strong Coupling**: g > √(κγ) regimes where atom-photon interaction dominates
- **Collective Effects**: Superradiance and subradiance in multi-atom systems  
- **Quantum Memory**: Storing and retrieving photonic quantum states
- **Deterministic Gates**: Single-photon controlled phase gates

Example Prompts:
```yaml
cavity_qed_prompts:
  - "Design a cavity QED system achieving g/κ > 100 with single atoms"
  - "Propose a quantum memory protocol with >95% fidelity"
  - "Optimize cavity geometry for enhanced atom-photon coupling"
```

### Squeezed Light
**Focus**: Quantum noise reduction below the standard quantum limit

Key Research Areas:
- **Parametric Oscillators**: OPO and OPA based squeezing generation
- **Spin Squeezing**: Collective atomic state squeezing for enhanced sensitivity
- **Multimode Entanglement**: Einstein-Podolsky-Rosen correlations
- **Quantum Sensing**: Gravitational wave detection and atomic clocks

Example Benchmarks:
```yaml
squeezed_light_benchmarks:
  squeezing_parameter: 
    excellent: "> 10 dB"
    good: "5-10 dB"
    poor: "< 5 dB"
  
  detection_efficiency:
    threshold: "> 90%"
    justification: "Losses rapidly degrade squeezing"
```

### Photon Blockade
**Focus**: Single-photon nonlinear optics and deterministic photon sources

Key Research Areas:
- **Conventional Blockade**: Strong Kerr nonlinearity (U >> κ)
- **Unconventional Blockade**: Interference-based mechanisms
- **Single Photon Sources**: On-demand photon generation
- **Photonic Quantum Gates**: Two-photon controlled operations

### Quantum Metrology  
**Focus**: Quantum-enhanced sensing beyond classical limits

Key Research Areas:
- **N00N States**: Quantum superposition for phase sensing
- **Atomic Clocks**: Optical lattice and ion trap frequency standards
- **Quantum Fisher Information**: Fundamental sensitivity bounds
- **Heisenberg Scaling**: 1/N sensitivity enhancement

### Optomechanics
**Focus**: Quantum coupling between light and mechanical motion

Key Research Areas:
- **Ground State Cooling**: Laser cooling of mechanical oscillators
- **Quantum State Transfer**: Optical-to-mechanical quantum state mapping
- **Hybrid Systems**: Coupling to spins, atoms, and superconducting circuits
- **Quantum Transduction**: Microwave-to-optical frequency conversion

## 📊 Physics-Aware Evaluation System

The evaluation system provides rigorous scoring across four key dimensions:

### 1. Feasibility (25%) - Physical Realizability
- **Energy Conservation**: Verify all energy transfers and conservation laws
- **Fundamental Limits**: Check against quantum limits (uncertainty principle, no-cloning)
- **Material Properties**: Realistic parameters for existing or proposed materials
- **Experimental Complexity**: Assessment of required experimental capabilities

**Example Checks**:
```python
# Energy conservation in parametric processes
def check_energy_conservation(pump_freq, signal_freq, idler_freq):
    return abs(pump_freq - signal_freq - idler_freq) < tolerance

# Uncertainty principle compliance  
def check_uncertainty_limit(delta_x, delta_p, hbar):
    return delta_x * delta_p >= hbar / 2
```

### 2. Mathematics (30%) - Analytical Correctness
- **Benchmark Verification**: Comparison with known analytical solutions
- **Dimensional Analysis**: Units and scaling consistency
- **Approximation Validity**: Justification of approximations used
- **Derivation Logic**: Step-by-step mathematical reasoning

**Benchmark Examples**:
```yaml
jaynes_cummings_benchmarks:
  vacuum_rabi_splitting: "2g"
  dressed_state_energies: "ω₀ ± g√(n+1)"
  cooperativity: "4g²/(κγ)"

harmonic_oscillator_benchmarks:
  ground_state_energy: "ħω/2"
  zero_point_fluctuations: "√(ħ/(2mω))"
  quantum_correlation_function: "⟨n⟩ + 1"
```

### 3. Novelty (25%) - Research Innovation
- **Parameter Exploration**: Novel regimes or unexplored parameter spaces
- **Architectural Innovation**: New system designs or configurations
- **Application Breakthroughs**: Potential for significant impact
- **Interdisciplinary Connections**: Links between different physics areas

**Novelty Scoring**:
```python
def evaluate_novelty(solution, database):
    # Check parameter space coverage
    param_novelty = assess_parameter_uniqueness(solution.parameters, database)
    
    # Evaluate architectural innovation
    design_novelty = analyze_system_architecture(solution.design)
    
    # Assess breakthrough potential
    impact_score = evaluate_impact_potential(solution.applications)
    
    return weighted_average([param_novelty, design_novelty, impact_score])
```

### 4. Performance (20%) - Quantitative Metrics
- **Category-Specific Metrics**: Coupling strength, squeezing level, efficiency, etc.
- **Experimental Records**: Comparison with state-of-the-art results
- **Optimization Potential**: Theoretical limits and improvement pathways
- **Scalability**: Extension to larger systems or higher performance

**Performance Thresholds**:
```yaml
performance_benchmarks:
  cavity_qed:
    strong_coupling: "g > √(κγ)"
    cooperativity: "C > 1"
    single_atom_coupling: "g > 10 MHz"
  
  squeezed_light:
    squeezing_level: "> 10 dB"
    detection_efficiency: "> 90%"
    bandwidth: "> 1 MHz"
  
  optomechanics:
    cooling_efficiency: "n_f < 1"
    coupling_rate: "g > κ_m"
    quality_factor: "Q > 10⁶"
```

## 🎨 Visualization and Analysis Tools

### Evolution Progress Tracking
```bash
# Generate comprehensive evolution plots
python src/main.py analyze --database --plots --export detailed_analysis.json

# View results interactively  
python view_results.py results/
```

Generated visualizations include:
- **Score Evolution**: Best and average scores across generations
- **Diversity Tracking**: Population diversity over time
- **Category Performance**: Comparative analysis across research areas
- **Breakthrough Timeline**: Discovery of high-impact solutions

### Model Performance Comparison
```bash
# Benchmark multiple models on physics problems
python src/main.py benchmark \
  --compare-models \
  --models "provider-5/gemini-2.5-flash-preview-04-17" "anthropic/claude-3.5-sonnet" \
  --physics-accuracy
```

### Custom Analysis Scripts
```python
# Example custom analysis
from database import ResearchDatabase
from utils import create_evolution_plot, export_results_report

db = ResearchDatabase()
solutions = db.get_solutions_by_category("cavity_qed")
analysis = analyze_breakthrough_patterns(solutions)
create_custom_visualization(analysis)
```

## 🔧 Advanced Configuration Options

### Custom Model Integration
```yaml
# Add new models to config/config.yaml
models:
  "custom/your-model":
    name: "Your Custom Model"
    max_tokens: 4096
    temperature: 0.7
    cost_per_1k_tokens: 0.002
    api_endpoint: "custom_endpoint"
    headers:
      Authorization: "Bearer ${CUSTOM_API_KEY}"
    strengths: ["Domain expertise", "Fast inference"]
```

### Evolution Algorithm Tuning
```yaml
evolution:
  population_size: 15              # Solutions per generation
  max_generations: 50              # Total evolution cycles
  mutation_rate: 0.3               # Fraction undergoing mutation
  crossover_rate: 0.7              # Fraction created via crossover
  elite_retention: 0.2             # Top solutions preserved
  exploration_rate: 0.1            # Novel solutions per generation
  diversity_pressure: 0.15         # Penalty for similar solutions
  convergence_threshold: 0.001     # Stop if improvement < threshold
```

### Custom Physics Validation
```python
# Add to src/evaluator.py
def custom_physics_check(solution_content):
    """Implement domain-specific physics validation"""
    
    # Extract parameters
    params = extract_physics_parameters(solution_content)
    
    # Custom validation rules
    if 'coupling_strength' in params:
        g = params['coupling_strength']
        # Check realistic coupling strength
        if g > 1e12:  # > 1 THz unrealistic
            return False, "Coupling strength too high"
    
    # Quantum mechanics consistency
    if violates_uncertainty_principle(params):
        return False, "Violates uncertainty principle"
    
    return True, "Physics checks passed"
```

## 📈 Model Comparison Guide

### Current Model Performance on Quantum Physics

| Model | Physics Accuracy | Mathematical Rigor | Novel Insights | Cost Efficiency | Best Use Cases |
|-------|------------------|-------------------|----------------|-----------------|----------------|
| **Gemini 2.5 Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Rapid prototyping, exploration |
| **Claude 3.5 Sonnet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Complex analysis, detailed derivations |
| **GPT-4 Turbo** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Structured outputs, systematic approaches |

### Model Selection Guidelines

**For Exploration & Testing**:
- Use Gemini 2.5 Flash (free/low-cost)
- Enable demo mode for offline testing
- Focus on broad parameter sweeps

**For Production Research**:
- Use Claude 3.5 Sonnet for physics depth
- Enable detailed evaluation and logging
- Generate comprehensive reports

**For Specific Tasks**:
- **Mathematical Derivations**: Claude 3.5 Sonnet
- **System Design**: GPT-4 Turbo  
- **Parameter Optimization**: Gemini 2.5 Flash
- **Novel Concepts**: Claude 3.5 Sonnet

## 🧪 Testing and Validation

### System Testing
```bash
# Test all components
python src/main.py test --full

# Test specific components
python src/main.py test --evaluator    # Physics evaluation system
python src/main.py test --database     # Data storage and retrieval
python src/main.py test --model "provider-5/gemini-2.5-flash-preview-04-17"

# Test Unicode handling (Windows)
python test_unicode_fix.py

# Test demo mode (no API required)
python test_demo.py
```

### Physics Accuracy Validation
```bash
# Test physics accuracy across categories
python src/main.py benchmark --physics-accuracy

# Compare model physics understanding
python src/main.py benchmark --compare-models --physics-accuracy
```

### Custom Validation Scripts
```python
# Validate custom research content
def validate_research(content_file, category):
    with open(content_file, 'r') as f:
        content = f.read()
    
    evaluator = QuantumOpticsEvaluator()
    result = evaluator.evaluate_research({
        'content': content,
        'category': category,
        'title': 'Custom Research'
    })
    
    return result
```

## 🐛 Troubleshooting Guide

### Common Issues and Solutions

**API Connection Errors**
```bash
# Verify API key
echo $A4F_API_KEY

# Test API connectivity
python src/main.py test --model "provider-5/gemini-2.5-flash-preview-04-17"

# Check API quota and usage
curl -H "Authorization: Bearer $A4F_API_KEY" https://api.a4f.co/v1/models
```

**Unicode/Encoding Errors (Windows)**
```bash
# Test Unicode fixes
python test_unicode_fix.py

# If issues persist, use demo mode
python src/main.py evolve --demo --generations 5
```

**Memory Issues**
```bash
# Reduce computational load
python src/main.py evolve --population 5 --generations 10

# Use streaming for large datasets
python src/main.py analyze --database --streaming
```

**Database Corruption**
```bash
# Reset database (warning: loses history)
rm data/research_history.db

# Backup before operations
cp data/research_history.db data/backup_$(date +%Y%m%d).db
```

**Model Performance Issues**
```bash
# Check model availability
python src/main.py benchmark --models "provider-5/gemini-2.5-flash-preview-04-17"

# Switch to alternative model
python src/main.py evolve --model "anthropic/claude-3.5-sonnet"

# Enable detailed logging
python src/main.py evolve --log-level DEBUG --log-file
```

### Performance Optimization

**Speed Optimization**:
- Use faster models (Gemini Flash) for exploration
- Reduce population size and generations
- Enable parallel processing in config
- Use category focus to narrow search space

**Quality Optimization**:
- Use higher-capability models (Claude Sonnet)
- Increase population size for diversity
- Enable detailed evaluation scoring
- Provide high-quality seed research

**Cost Optimization**:
- Use free models for testing
- Implement request caching
- Batch API calls efficiently
- Monitor usage with detailed logging

## 🚀 Advanced Use Cases

### Multi-Domain Research
```bash
# Explore quantum sensing applications
python src/main.py evolve --category quantum_metrology --generations 30

# Cross-domain optimization (optomechanics + sensing)
python custom_scripts/cross_domain_evolution.py
```

### Automated Literature Review
```bash
# Evaluate existing research papers
for paper in papers/*.txt; do
    python src/main.py evaluate --content "$paper" --detailed --save-result "evaluations/$(basename $paper .txt).json"
done

# Analyze evaluation results
python analyze_literature.py evaluations/
```

### Custom Research Pipelines
```python
# research_pipeline.py
from evolution_controller import QuantumResearchEvolver
from evaluator import QuantumOpticsEvaluator

def run_custom_pipeline():
    # Stage 1: Broad exploration
    evolver = QuantumResearchEvolver()
    evolver.population_size = 20
    initial_solutions = evolver.evolve_research(15)
    
    # Stage 2: Focused optimization  
    top_solutions = initial_solutions[:5]
    evolver.seed_solutions = top_solutions
    evolver.population_size = 10
    final_solutions = evolver.evolve_research(25)
    
    # Stage 3: Detailed analysis
    evaluator = QuantumOpticsEvaluator()
    detailed_results = []
    for solution in final_solutions[:3]:
        result = evaluator.detailed_evaluation(solution)
        detailed_results.append(result)
    
    return detailed_results
```

## 📚 Complete Examples

### Example 1: Cavity QED Breakthrough Discovery
```bash
# Set up for cavity QED research
export A4F_API_KEY="your-api-key"

# Run focused evolution
python src/main.py evolve \
  --category cavity_qed \
  --model "anthropic/claude-3.5-sonnet" \
  --generations 25 \
  --population 12 \
  --output-dir cavity_qed_results

# Analyze breakthroughs
python src/main.py analyze \
  --database \
  --category cavity_qed \
  --plots \
  --export cavity_qed_analysis.json

# View results
python view_results.py cavity_qed_results/
```

### Example 2: Multi-Model Comparison Study
```bash
# Test different models on same problem
models=("provider-5/gemini-2.5-flash-preview-04-17" "anthropic/claude-3.5-sonnet" "openai/gpt-4-turbo")

for model in "${models[@]}"; do
    echo "Testing $model..."
    python src/main.py evolve \
      --model "$model" \
      --generations 15 \
      --population 8 \
      --output-dir "comparison_$model"
done

# Compare results
python compare_models.py comparison_*/
```

### Example 3: Custom Research Evaluation
```bash
# Prepare research content
cat > my_research.txt << EOF
We propose a novel cavity QED system using diamond nitrogen-vacancy centers 
coupled to a photonic crystal cavity. The system achieves strong coupling 
with g = 50 MHz, cavity decay κ = 10 MHz, and NV transition linewidth γ = 15 MHz.
The cooperativity C = 4g²/(κγ) ≈ 67 enables efficient quantum state transfer
between NV spin states and cavity photons.
EOF

# Evaluate the research
python src/main.py evaluate \
  --content my_research.txt \
  --category cavity_qed \
  --detailed \
  --save-result my_evaluation.json

# View detailed results
cat my_evaluation.json | python -m json.tool
```

### Example 4: Automated Research Pipeline
```python
#!/usr/bin/env python3
"""
Automated research discovery pipeline
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from evolution_controller import QuantumResearchEvolver
from database import ResearchDatabase
from utils import export_results_report

def main():
    """Run automated research pipeline"""
    
    # Initialize systems
    evolver = QuantumResearchEvolver()
    db = ResearchDatabase()
    
    # Configuration
    categories = ['cavity_qed', 'squeezed_light', 'optomechanics']
    models = ['provider-5/gemini-2.5-flash-preview-04-17', 'anthropic/claude-3.5-sonnet']
    
    all_results = {}
    
    for category in categories:
        print(f"Exploring {category}...")
        category_results = {}
        
        for model in models:
            print(f"  Using model: {model}")
            
            # Configure evolution
            evolver.llm.switch_model(model)
            evolver.population_size = 10
            
            # Run evolution
            solutions = evolver.evolve_research(20)
            
            # Store results
            category_results[model] = {
                'best_score': solutions[0].evaluation_result.total_score,
                'breakthrough_count': len([s for s in solutions if s.evaluation_result.total_score > 0.8]),
                'solutions': [solution.__dict__ for solution in solutions[:5]]
            }
            
            # Save to database
            for solution in solutions[:10]:
                db.store_solution(solution.__dict__)
        
        all_results[category] = category_results
    
    # Export comprehensive report
    with open('automated_research_report.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Automated research pipeline completed!")
    print("Results saved to automated_research_report.json")

if __name__ == '__main__':
    main()
```

## 📄 License and Contributing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributing Guidelines
1. **Fork the repository** and create your feature branch
2. **Add tests** for new functionality
3. **Follow physics validation** standards
4. **Document new features** comprehensively
5. **Submit pull requests** with detailed descriptions

### Research Validation Standards
- All physics claims must be backed by analytical benchmarks
- Mathematical derivations should be step-by-step verifiable
- Experimental proposals must cite realistic parameters
- Novel concepts require thorough feasibility analysis

## 🙏 Acknowledgments

- **AlphaEvolve Architecture**: Inspiration for evolutionary discovery methods
- **A4F Platform**: Flexible access to multiple language models
- **Quantum Optics Community**: Benchmarks and validation from leading research groups
- **Open Source Contributors**: Community contributions to physics validation and testing

## 📞 Support and Community

### Getting Help
1. **GitHub Issues**: Report bugs and request features
2. **GitHub Discussions**: Ask questions and share research results
3. **Documentation**: See `docs/` for detailed technical guides
4. **Community**: Join the quantum AI research community

### Research Sharing
- Share your breakthrough discoveries with the community
- Contribute high-quality seed research examples
- Submit physics validation improvements
- Propose new research categories and evaluation criteria

---

## 🎯 Quick Start Commands

**Immediate Usage** (copy and paste):
```bash
# Set your API key
export A4F_API_KEY="ddc-a4f-your-api-key-here"

# Run your first quantum research evolution
python src/main.py evolve --generations 15 --population 8

# View your results
python view_results.py results/

# Test without API (demo mode)
python src/main.py evolve --demo --generations 5 --population 3
```

**Ready to discover breakthrough quantum optics research?** 🚀

Start your evolution today and let Coheron guide you to novel physics discoveries through the power of evolutionary AI!