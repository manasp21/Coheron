# Coheron

**An evolutionary AI system for solving challenging problems in Atomic, Molecular, and Optical Physics**

Coheron uses advanced language models and evolutionary algorithms to tackle specific physics challenges in quantum optics, cavity QED, optomechanics, and quantum sensing. Instead of generating generic research ideas, Coheron evolves solutions toward measurable physics targets like "achieve g/κ > 1000 at room temperature" or "generate 20 dB squeezed light."

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![A4F API](https://img.shields.io/badge/A4F-Compatible-green.svg)](https://www.a4f.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What Makes Coheron Unique

🎯 **Problem-Focused Evolution**: Evolves solutions toward specific physics targets rather than generating generic ideas

🔬 **Real Physics Challenges**: Tackle unsolved problems like room temperature strong coupling and 20 dB squeezing  

📊 **Measurable Progress**: Track progress with physics parameters (coupling strengths, Q factors, cooperativities)

🧬 **Breakthrough Detection**: Automatically identifies when solutions achieve target parameters

📚 **Physics Knowledge Base**: Comprehensive database of atomic data, cavity designs, and experimental records

## Quick Demo

```bash
# Install and test the system
git clone <repository-url>
cd Coheron
pip install -r requirements.txt

# See available physics problems
python src/main.py list-problems

# Solve a specific problem (demo mode - no API key needed)
python src/main.py solve-problem room_temp_cavity_qed --demo --generations 10

# Get problem details
python src/main.py problem-info room_temp_cavity_qed
```

## Installation Instructions

### Prerequisites

- **Python 3.9+** (required)
- **API Key** (optional - can use demo mode without API)

### Windows Installation

#### Option 1: Native Windows
```powershell
# Install Python from python.org if not already installed
# Verify Python installation
python --version

# Clone repository
git clone <repository-url>
cd Coheron

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### Option 2: Windows Subsystem for Linux (WSL)
```bash
# In WSL terminal
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone and setup
git clone <repository-url>
cd Coheron
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### Option 3: Anaconda/Miniconda (Windows)
```powershell
# Create conda environment
conda create -n coheron python=3.11
conda activate coheron

# Clone repository
git clone <repository-url>
cd Coheron

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

### macOS Installation

#### Option 1: Homebrew
```bash
# Install Python via Homebrew
brew install python git

# Clone repository
git clone <repository-url>
cd Coheron

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### Option 2: System Python
```bash
# Use built-in Python (macOS 10.15+)
git clone <repository-url>
cd Coheron

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### Option 3: Conda (macOS)
```bash
# Install Miniconda if not already installed
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Create environment
conda create -n coheron python=3.11
conda activate coheron

# Clone and install
git clone <repository-url>
cd Coheron
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

### Linux Installation

#### Ubuntu/Debian
```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv git

# Clone repository
git clone <repository-url>
cd Coheron

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### CentOS/RHEL/Fedora
```bash
# For CentOS/RHEL
sudo yum install python3 python3-pip git
# OR for Fedora
sudo dnf install python3 python3-pip git

# Clone repository
git clone <repository-url>
cd Coheron

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

#### Arch Linux
```bash
# Install dependencies
sudo pacman -S python python-pip git

# Clone repository
git clone <repository-url>
cd Coheron

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python src/main.py list-problems
```

## API Configuration (Optional)

Coheron can work in demo mode without an API key, but for full functionality:

### Get API Key
1. Sign up at [A4F.co](https://www.a4f.co/)
2. Get your API key (format: `ddc-a4f-...`)

### Set API Key

**Windows (Command Prompt):**
```cmd
set A4F_API_KEY=ddc-a4f-your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:A4F_API_KEY = "ddc-a4f-your-api-key-here"
```

**macOS/Linux:**
```bash
export A4F_API_KEY="ddc-a4f-your-api-key-here"
```

**Permanent Setup (add to your shell profile):**
```bash
# For bash users (add to ~/.bashrc)
echo 'export A4F_API_KEY="ddc-a4f-your-api-key-here"' >> ~/.bashrc

# For zsh users (add to ~/.zshrc)  
echo 'export A4F_API_KEY="ddc-a4f-your-api-key-here"' >> ~/.zshrc

# For fish users (add to ~/.config/fish/config.fish)
echo 'set -x A4F_API_KEY "ddc-a4f-your-api-key-here"' >> ~/.config/fish/config.fish
```

## Physics Problems Available

Coheron comes with 6 challenging AMO physics problems ready to solve:

### 1. Room Temperature Cavity QED (`room_temp_cavity_qed`)
**Challenge**: Achieve strong coupling regime (g > √κγ) at 300K
- **Target**: Coupling strength g > 1 GHz, Q > 10M, Cooperativity > 1000
- **Difficulty**: Expert
- **Applications**: Quantum computing, single photon sources

### 2. 20 dB Squeezed Light (`squeezed_light_20db`)
**Challenge**: Generate optical squeezed states with >20 dB noise reduction
- **Target**: Squeezing > 20 dB, Detection efficiency > 99%, Bandwidth > 100 MHz
- **Difficulty**: Expert  
- **Applications**: Gravitational wave detection, quantum sensing

### 3. Ground State Cooling (`optomech_ground_state`)
**Challenge**: Cool macroscopic oscillator to quantum ground state
- **Target**: Phonon number < 0.1, Mass > 1 picogram, Cooperativity > 100
- **Difficulty**: Expert
- **Applications**: Macroscopic quantum mechanics, quantum transduction

### 4. Single Photon Purity (`single_photon_purity`)
**Challenge**: Deterministic single photons with 99.9% purity
- **Target**: Purity > 99.9%, Collection efficiency > 90%, Rate > 1 GHz
- **Difficulty**: Hard
- **Applications**: Quantum computing, quantum communication

### 5. Quantum Memory Fidelity (`quantum_memory_fidelity`)
**Challenge**: Store quantum states with >99.9% fidelity for milliseconds
- **Target**: Storage fidelity > 99.9%, Retrieval fidelity > 99.9%, Time > 1 ms
- **Difficulty**: Hard
- **Applications**: Quantum networks, quantum repeaters

### 6. Heisenberg-Limited Sensing (`quantum_sensing_limit`)
**Challenge**: Achieve 1/N sensitivity scaling with entangled particles
- **Target**: Enhancement > 10x SQL, Particles > 1000, Coherence > 1 ms
- **Difficulty**: Expert
- **Applications**: Atomic clocks, magnetometry, navigation

## Usage Guide

### Basic Commands

```bash
# List all available physics problems
python src/main.py list-problems

# Get detailed information about a specific problem
python src/main.py problem-info room_temp_cavity_qed

# Solve a problem (demo mode - no API needed)
python src/main.py solve-problem room_temp_cavity_qed --demo --generations 20

# Solve with full AI assistance (requires API key)
python src/main.py solve-problem squeezed_light_20db --generations 50 --model "anthropic/claude-3.5-sonnet"

# Track progress of a solving session
python src/main.py solution-progress room_temp_cavity_qed

# Test the system
python src/main.py test --all
```

### Advanced Usage

```bash
# Solve multiple problems in parallel
python src/main.py solve-problem room_temp_cavity_qed --generations 30 --population 15 &
python src/main.py solve-problem squeezed_light_20db --generations 30 --population 15 &

# Compare different AI models on same problem
python src/main.py solve-problem single_photon_purity --model "provider-5/gemini-2.5-flash-preview-04-17" --generations 25
python src/main.py solve-problem single_photon_purity --model "anthropic/claude-3.5-sonnet" --generations 25

# Export results for analysis
python src/main.py export-results room_temp_cavity_qed --format json --output cavity_qed_results.json

# Create custom problem library
python src/main.py create-problem-library my_problems/
```

### Example: Solving Room Temperature Cavity QED

```bash
# Step 1: Understand the problem
python src/main.py problem-info room_temp_cavity_qed

# Step 2: Start solving (demo mode first)
python src/main.py solve-problem room_temp_cavity_qed --demo --generations 15

# Step 3: Try with AI assistance
export A4F_API_KEY="your-api-key"
python src/main.py solve-problem room_temp_cavity_qed --generations 30 --model "anthropic/claude-3.5-sonnet"

# Step 4: Monitor progress
python src/main.py solution-progress room_temp_cavity_qed

# Step 5: Export breakthrough solutions
python src/main.py export-solutions room_temp_cavity_qed --breakthrough-only
```

## How It Works

### 1. Problem Definition
Each physics problem is defined with:
- **Target Parameters**: Specific measurable goals (e.g., g > 1 GHz)
- **Constraints**: Physics limitations (energy conservation, uncertainty principle)
- **Evaluation Criteria**: How solutions are scored
- **Background Context**: Physics knowledge and current records

### 2. Solution Evolution
The system evolves solutions through generations:
1. **Generate** initial solution population
2. **Evaluate** each solution against physics targets
3. **Select** best performing solutions
4. **Mutate** and crossover to create new solutions
5. **Repeat** until breakthrough achieved or generations complete

### 3. Physics Evaluation
Solutions are scored on:
- **Parameter Achievement**: How well they meet target physics parameters
- **Physics Validity**: Compliance with fundamental physics laws
- **Experimental Feasibility**: Realistic with current technology
- **Innovation**: Novel approaches and improvements

### 4. Breakthrough Detection
System automatically detects when solutions:
- Achieve all target parameters (breakthrough threshold)
- Make significant progress (progress threshold)
- Discover novel physics regimes
- Exceed current experimental records

## Physics Knowledge Database

Coheron includes comprehensive physics knowledge:

### Atomic Data
- **Alkali Atoms**: Rb87, Cs133, Li6 with transition wavelengths, linewidths, magic wavelengths
- **Alkaline Earth**: Ca40, Sr88 with clock transitions and intercombination lines
- **Rydberg States**: Scaling laws and applications

### Cavity Designs
- **Fabry-Perot Cavities**: Linear and ring resonators with typical parameters
- **Microresonators**: Microspheres, microtoroids, photonic crystals
- **Superconducting Cavities**: 3D resonators for circuit QED

### Experimental Records
- **Cavity QED**: Strongest coupling (1.5×10^7 Hz), highest cooperativity (2.7×10^5)
- **Squeezed Light**: Best squeezing (-15 dB), continuous wave records
- **Optomechanics**: Lowest phonon numbers, highest cooperativities
- **Quantum Memory**: Longest storage times, highest fidelities

### Physical Constants
- **Fundamental Constants**: All CODATA values with full precision
- **Derived Constants**: Rydberg energy, Bohr magneton, etc.
- **Typical Ranges**: Expected parameter ranges for different physics regimes

## Troubleshooting

### Common Issues

**"Python command not found"**
- Windows: Make sure Python is in your PATH or use `py` instead of `python`
- macOS/Linux: Use `python3` instead of `python`

**"Module not found" errors**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**API connection errors**
```bash
# Verify API key is set
echo $A4F_API_KEY  # Linux/macOS
echo %A4F_API_KEY% # Windows

# Test with demo mode first
python src/main.py solve-problem room_temp_cavity_qed --demo --generations 5
```

**Unicode/encoding errors (Windows)**
```bash
# Set proper encoding
set PYTHONIOENCODING=utf-8

# Or use demo mode
python src/main.py solve-problem room_temp_cavity_qed --demo
```

**Permission errors**
```bash
# Don't use sudo with pip in virtual environment
# If using system Python, create virtual environment instead
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Platform-Specific Tips

**Windows:**
- Use PowerShell or Command Prompt (not Git Bash for environment variables)
- Install Python from python.org for best compatibility
- Consider using WSL for Linux-like environment

**macOS:**
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for Python: `brew install python`
- M1/M2 Macs: All dependencies are compatible

**Linux:**
- Install python3-dev for some dependencies: `sudo apt install python3-dev`
- Use system package manager for Python: `sudo apt install python3 python3-pip`
- SELinux users: May need to set appropriate contexts

## Adding Custom Problems

### 1. Create Problem YAML File
```yaml
# my_custom_problem.yaml
id: my_problem
title: "My Custom Physics Problem"
category: quantum_optics
description: "Solve my specific physics challenge"

target_parameters:
  coupling_strength:
    symbol: g
    target: "> 5e9"
    units: Hz
    description: "Coupling strength parameter"
    weight: 0.5
    type: optimization

constraints:
  - name: energy_conservation
    description: "Energy must be conserved"
    type: fundamental
    penalty_weight: 1.0

breakthrough_threshold: 0.9
significant_progress_threshold: 0.75
difficulty_level: hard
```

### 2. Add to Problem Library
```bash
# Copy to problems directory
cp my_custom_problem.yaml data/amo_problems/

# Verify it loads correctly
python src/main.py list-problems
python src/main.py problem-info my_problem
```

### 3. Solve Your Custom Problem
```bash
python src/main.py solve-problem my_problem --generations 25
```

## Available AI Models

| Model | Physics Expertise | Speed | Cost | Best For |
|-------|------------------|-------|------|----------|
| **Gemini 2.5 Flash** | High | Very Fast | Low | Exploration, testing |
| **Claude 3.5 Sonnet** | Excellent | Fast | Medium | Complex physics, analysis |
| **GPT-4 Turbo** | High | Medium | High | Structured solutions |

### Model Selection Guide

**For Learning/Testing:**
```bash
python src/main.py solve-problem room_temp_cavity_qed --demo
```

**For Exploration:**
```bash
python src/main.py solve-problem squeezed_light_20db --model "provider-5/gemini-2.5-flash-preview-04-17"
```

**For Deep Physics:**
```bash
python src/main.py solve-problem quantum_sensing_limit --model "anthropic/claude-3.5-sonnet"
```

## Contributing

We welcome contributions to expand Coheron's physics problem-solving capabilities!

### Ways to Contribute

1. **Add New Physics Problems**: Submit YAML files for unsolved physics challenges
2. **Improve Physics Knowledge**: Expand the atomic data and experimental records database
3. **Enhance Evaluation**: Improve physics validation and scoring algorithms
4. **Cross-Platform Testing**: Test installation and usage on different OS configurations

### Development Setup
```bash
# Fork and clone your fork
git clone https://github.com/yourusername/Coheron.git
cd Coheron

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Run tests
python -m pytest tests/
python src/main.py test --all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/manasp21/Coheron/issues)
- **Discussions**: Ask questions and share results in [GitHub Discussions](https://github.com/manasp21/Coheron/discussions)
- **Documentation**: See source code and comments for detailed technical information

## Acknowledgments

- **Quantum Optics Community**: For providing experimental benchmarks and validation data
- **A4F Platform**: For flexible access to multiple AI models
- **Open Source Community**: For Python packages that make this possible

---

## Ready to Solve Physics Problems?

```bash
# Quick start - try it now!
git clone <repository-url>
cd Coheron
pip install -r requirements.txt

# See what problems you can solve
python src/main.py list-problems

# Start solving (no API key needed for demo)
python src/main.py solve-problem room_temp_cavity_qed --demo --generations 15
```

**Transform physics challenges into breakthrough solutions with evolutionary AI!**
