"""
Quantum Optics Research AI - Core Package

An evolutionary AI system inspired by AlphaEvolve that generates breakthrough 
quantum optics research using OpenRouter API integration, physics-aware evaluation,
and evolutionary research loops.

Components:
- llm_interface: OpenRouter integration with model switching
- evaluator: Physics-aware evaluation system
- research_generator: Quantum research prompt generation
- evolution_controller: Main research evolution loop
- database: Research storage and tracking
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Quantum Optics Research Team"

from .llm_interface import OpenRouterInterface, GenerationResult
from .evaluator import QuantumOpticsEvaluator, EvaluationResult

__all__ = [
    "OpenRouterInterface",
    "GenerationResult", 
    "QuantumOpticsEvaluator",
    "EvaluationResult"
]