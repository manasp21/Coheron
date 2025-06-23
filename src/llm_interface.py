import openai
import yaml
import os
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GenerationResult:
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    generation_time: float
    metadata: Dict[str, Any] = None

class OpenRouterInterface:
    """Complete OpenRouter interface with model switching and optimization"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.client = self._initialize_client()
        self.current_model = self.config['current_model']
        self.model_config = self.config['models'][self.current_model]
        self.logger = self._setup_logging()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'model_usage': {},
            'error_count': 0
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with error handling"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['api', 'current_model', 'models']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
                    
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
            
    def _initialize_client(self) -> openai.OpenAI:
        """Initialize OpenAI client with OpenRouter"""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        return openai.OpenAI(
            base_url=self.config['api']['base_url'],
            api_key=api_key
        )
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('QuantumOpticsAI')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'quantum_ai.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def switch_model(self, model_name: str) -> None:
        """Switch to different model with validation"""
        if model_name not in self.config['models']:
            available = list(self.config['models'].keys())
            raise ValueError(
                f"Model '{model_name}' not configured. "
                f"Available models: {available}"
            )
        
        old_model = self.current_model
        self.current_model = model_name
        self.model_config = self.config['models'][model_name]
        
        self.logger.info(f"Switched from {old_model} to {model_name}")
        print(f"✅ Model switched: {old_model} → {model_name}")
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with descriptions"""
        models = []
        for name, config in self.config['models'].items():
            models.append({
                'name': name,
                'description': config.get('description', 'No description'),
                'cost_per_token': config.get('cost_per_token', 0),
                'max_tokens': config.get('max_tokens', 4000),
                'physics_strength': config.get('physics_strength', 'unknown')
            })
        return models
        
    def generate_research(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GenerationResult:
        """Generate research with full error handling and metrics"""
        
        start_time = time.time()
        
        try:
            # Adapt prompt for current model
            adapted_prompt = self._adapt_prompt(prompt)
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": adapted_prompt})
            
            # Use model-specific or override parameters
            params = {
                'model': self.current_model,
                'messages': messages,
                'temperature': temperature or self.model_config['temperature'],
                'max_tokens': max_tokens or self.model_config['max_tokens'],
                'top_p': self.model_config['top_p']
            }
            
            # Make API call with retry logic
            response = self._make_api_call_with_retry(params)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            tokens_used = response.usage.total_tokens
            cost_estimate = tokens_used * self.model_config.get('cost_per_token', 0)
            
            # Update statistics
            self._update_stats(tokens_used, cost_estimate)
            
            # Create result
            result = GenerationResult(
                content=response.choices[0].message.content,
                model=self.current_model,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                generation_time=generation_time,
                metadata={
                    'prompt_length': len(prompt),
                    'adapted_prompt_length': len(adapted_prompt),
                    'system_prompt_used': system_prompt is not None,
                    'parameters_used': params
                }
            )
            
            self.logger.info(
                f"Generated research: {tokens_used} tokens, "
                f"${cost_estimate:.4f}, {generation_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.stats['error_count'] += 1
            self.logger.error(f"Generation failed: {e}")
            raise
            
    def _make_api_call_with_retry(self, params: Dict[str, Any]) -> Any:
        """Make API call with exponential backoff retry"""
        max_retries = self.config['api'].get('max_retries', 3)
        timeout = self.config['api'].get('timeout', 300)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                    
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
                
    def _adapt_prompt(self, prompt: str) -> str:
        """Adapt prompt based on model reasoning style"""
        style = self.model_config.get('reasoning_style', 'standard')
        
        adaptations = {
            'step_by_step': (
                f"Think step by step and solve this quantum optics problem:\n\n"
                f"{prompt}\n\n"
                f"Break down your solution into clear steps."
            ),
            'detailed': (
                f"Provide a detailed analysis of this quantum optics problem:\n\n"
                f"{prompt}\n\n"
                f"Include mathematical derivations and physical intuition."
            ),
            'analytical': (
                f"Analyze this quantum optics problem systematically:\n\n"
                f"{prompt}\n\n"
                f"Provide rigorous theoretical treatment."
            ),
            'structured': (
                f"Structure your solution to this quantum optics problem:\n\n"
                f"{prompt}\n\n"
                f"1. Problem Analysis\n"
                f"2. Theoretical Framework\n"
                f"3. Mathematical Solution\n"
                f"4. Physical Interpretation"
            ),
            'creative': (
                f"Explore creative solutions to this quantum optics challenge:\n\n"
                f"{prompt}\n\n"
                f"Consider novel approaches and unconventional methods."
            ),
            'concise': (
                f"Solve this quantum optics problem concisely:\n\n"
                f"{prompt}\n\n"
                f"Focus on key results and essential physics."
            )
        }
        
        return adaptations.get(style, prompt)
        
    def _update_stats(self, tokens: int, cost: float) -> None:
        """Update usage statistics"""
        self.stats['total_requests'] += 1
        self.stats['total_tokens'] += tokens
        self.stats['total_cost'] += cost
        
        if self.current_model not in self.stats['model_usage']:
            self.stats['model_usage'][self.current_model] = {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
            
        self.stats['model_usage'][self.current_model]['requests'] += 1
        self.stats['model_usage'][self.current_model]['tokens'] += tokens
        self.stats['model_usage'][self.current_model]['cost'] += cost
        
    def batch_generate(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[GenerationResult]:
        """Generate multiple research solutions with concurrency control"""
        import concurrent.futures
        import threading
        
        results = []
        semaphore = threading.Semaphore(max_concurrent)
        
        def generate_single(prompt: str) -> GenerationResult:
            with semaphore:
                return self.generate_research(prompt, system_prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_prompt = {
                executor.submit(generate_single, prompt): prompt 
                for prompt in prompts
            }
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch generation failed for prompt: {e}")
                    
        return results
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'current_model': self.current_model,
            'model_config': self.model_config,
            'available_models': self.get_available_models(),
            'statistics': self.stats.copy(),
            'api_config': self.config['api']
        }
        
    def save_stats(self, filepath: str = "data/usage_stats.json") -> None:
        """Save usage statistics to file"""
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def load_stats(self, filepath: str = "data/usage_stats.json") -> None:
        """Load usage statistics from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        except FileNotFoundError:
            self.logger.info("No existing stats file found, starting fresh")