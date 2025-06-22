# Coheron

**An evolutionary AI system for breakthrough quantum optics research discovery**

Inspired by AlphaEvolve, this system uses advanced language models to generate, evaluate, and evolve quantum optics research solutions through physics-aware evaluation and evolutionary algorithms.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Compatible-green.svg)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🧬 Evolutionary Research Discovery**: Inspired by AlphaEvolve architecture for systematic research evolution
- **🔬 Physics-Aware Evaluation**: Rigorous evaluation against analytical benchmarks and experimental feasibility
- **🔄 Flexible Model Switching**: Easy switching between DeepSeek, Claude, GPT-4, and other models via OpenRouter
- **📊 Comprehensive Analytics**: Track research evolution, breakthrough discoveries, and model performance
- **🎯 Specialized Categories**: Focus on cavity QED, squeezed light, photon blockade, quantum metrology, and optomechanics
- **💾 Research Database**: SQLite-based storage for evolution tracking and lineage analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenRouter API key ([Get one here](https://openrouter.ai/keys))

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
   cp .env.template .env
   # Edit .env and add your OpenRouter API key
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

4. **Run your first evolution**
   ```bash
   python src/main.py evolve --generations 10
   ```

## 📖 Usage

### Basic Commands

```bash
# Run research evolution (default: DeepSeek R1 free)
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

### Advanced Usage

#### Custom Evolution Parameters
```bash
python src/main.py evolve \
  --generations 50 \
  --population 15 \
  --model "deepseek/deepseek-r1" \
  --output-dir custom_results \
  --save-interval 3
```

#### Detailed Evaluation
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

## 🏗️ Architecture

### Core Components

```
Coheron/
├── src/
│   ├── main.py                 # CLI interface and application entry
│   ├── evolution_controller.py # Main evolution loop (AlphaEvolve-inspired)
│   ├── llm_interface.py        # OpenRouter integration with model switching
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

### Evolution Process

1. **Initialization**: Load seed research examples and generate initial population
2. **Generation Loop**: 
   - Generate mutations and crossovers from elite solutions
   - Create exploratory solutions for novelty
   - Evaluate all candidates using physics-aware scoring
   - Select best solutions for next generation
3. **Analysis**: Track progress, identify breakthroughs, and export results

## 🎯 Research Categories

### Cavity QED
- Single atom strong coupling
- Collective effects and superradiance
- Quantum memory protocols
- Deterministic photon generation

### Squeezed Light
- Parametric oscillator squeezing
- Spin squeezing in atomic ensembles
- Multimode entanglement
- Gravitational wave detection enhancement

### Photon Blockade
- Kerr nonlinearity systems
- Unconventional blockade mechanisms
- Single photon sources
- Photonic quantum gates

### Quantum Metrology
- N00N state interferometry
- Atomic clock protocols
- Quantum Fisher information
- Heisenberg-limited sensing

### Optomechanics
- Ground state cooling
- Quantum state transfer
- Hybrid quantum systems
- Quantum transduction

## 🔧 Configuration

### Model Switching
Change models instantly by editing `config/config.yaml`:

```yaml
current_model: "deepseek/deepseek-r1-0528:free"  # Free model
# current_model: "anthropic/claude-3.5-sonnet"   # For complex analysis
# current_model: "openai/gpt-4-turbo"            # For structured solutions
```

### Evolution Parameters
```yaml
evolution:
  population_size: 10
  max_generations: 50
  mutation_rate: 0.3
  crossover_rate: 0.7
  elite_retention: 0.2
```

### Evaluation Weights
```yaml
evaluation:
  weights:
    feasibility: 0.25    # Physical realizability
    mathematics: 0.30    # Mathematical correctness
    novelty: 0.25        # Research novelty
    performance: 0.20    # Performance metrics
```

## 📊 Evaluation System

The physics-aware evaluator scores research on four key dimensions:

### 1. Feasibility (25%)
- Energy conservation compliance
- Fundamental physics limits
- Realistic material properties
- Experimental complexity assessment

### 2. Mathematics (30%)
- Analytical benchmark verification
- Dimensional analysis consistency
- Known solution comparison
- Mathematical derivation correctness

### 3. Novelty (25%)
- Parameter regime exploration
- Architectural innovation
- Application breakthrough potential
- Interdisciplinary connections

### 4. Performance (20%)
- Category-specific metrics
- Experimental record comparison
- Optimization potential
- Scalability assessment

## 🎨 Visualization and Analysis

Generate comprehensive analysis plots:

```bash
# Evolution progress tracking
python src/main.py analyze --database --plots

# Category performance comparison
python src/main.py analyze --category-analysis

# Model performance benchmarking
python src/main.py benchmark --compare-models --models deepseek/deepseek-r1 anthropic/claude-3.5-sonnet
```

## 📈 Model Comparison

Easily compare different models for quantum research:

| Model | Physics Strength | Reasoning Style | Cost | Best For |
|-------|-----------------|-----------------|------|----------|
| DeepSeek R1 (Free) | High | Step-by-step | Free | Testing & exploration |
| DeepSeek R1 (Paid) | High | Detailed | Low | Production use |
| Claude 3.5 Sonnet | Very High | Analytical | Medium | Complex physics |
| GPT-4 Turbo | High | Structured | High | Systematic analysis |
| Gemini Pro | Medium | Creative | Low | Novel approaches |

## 🤝 Integration Points

The system is designed with three main customization points:

### 1. Research Prompts (`config/prompts.yaml`)
Add your domain-specific quantum optics research questions and exploration strategies.

### 2. Evaluation System (`src/evaluator.py`)
Implement your analytical solution benchmarks and scoring criteria.

### 3. Seed Research (`data/seed_research.json`)
Provide high-quality starting examples from your research area.

## 🐛 Troubleshooting

### Common Issues

**API Key Error**
```bash
# Make sure your OpenRouter API key is set
export OPENROUTER_API_KEY="your-key-here"
# Or add it to your .env file
```

**Model Not Found**
```bash
# Check available models
python src/main.py test --model
# Update config with valid model name
```

**Database Issues**
```bash
# Reset database if corrupted
rm data/research_history.db
# System will create a new one automatically
```

**Memory Issues**
```bash
# Reduce population size for large models
python src/main.py evolve --population 5 --generations 20
```

### Performance Optimization

- Use free models for testing and paid models for production
- Reduce population size and generations for faster iterations
- Enable logging to monitor progress and debug issues
- Use category focus to narrow search space

## 📚 Examples

### Example 1: Cavity QED Research
```bash
python src/main.py evolve \
  --category cavity_qed \
  --model "deepseek/deepseek-r1-0528:free" \
  --generations 15 \
  --population 8
```

### Example 2: Multi-Model Comparison
```bash
# Test with DeepSeek
python src/main.py evolve --model "deepseek/deepseek-r1-0528:free" --generations 10

# Test with Claude
python src/main.py evolve --model "anthropic/claude-3.5-sonnet" --generations 10

# Compare results
python src/main.py analyze --database --export comparison.json
```

### Example 3: Custom Research Evaluation
```bash
# Evaluate your own research content
echo "Design a cavity QED system with strong coupling..." > my_research.txt
python src/main.py evaluate \
  --content my_research.txt \
  --category cavity_qed \
  --detailed \
  --save-result my_evaluation.json
```

## 🛠️ Development

### Running Tests
```bash
# Full system test
python src/main.py test --full

# Component-specific tests
python src/main.py test --evaluator
python src/main.py test --database
python src/main.py test --model "deepseek/deepseek-r1-0528:free"
```

### Adding New Models
1. Add model configuration to `config/config.yaml`
2. Test with `python src/main.py test --model "new-model-name"`
3. Benchmark with `python src/main.py benchmark --models "new-model-name"`

### Extending Evaluation
1. Add new physics categories to `config/prompts.yaml`
2. Implement category-specific evaluation in `src/evaluator.py`
3. Add benchmark problems to `data/benchmarks.json`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the AlphaEvolve architecture for evolutionary AI discovery
- Built on OpenRouter for flexible model access
- Quantum optics benchmarks from leading research groups
- Community contributions to quantum physics validation

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Open an issue on GitHub
2. **Discussions**: Use GitHub Discussions for questions
3. **Email**: Contact the development team
4. **Documentation**: See `docs/` for detailed guides

---

**Ready to discover breakthrough quantum optics research? Start your evolution today!** 🚀

```bash
export OPENROUTER_API_KEY="your-key-here"
python src/main.py evolve --generations 20
```