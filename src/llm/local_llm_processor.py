#!/usr/bin/env python3
"""
Local LLM Processor Module

Local Phi-3 Mini model integration with Pi 5 optimization, context management,
and structured output generation for meeting analysis.

Author: Claude AI Assistant
Date: 2024-07-15
Version: 1.0
"""

import os
import sys
import logging
import threading
import time
import json
import uuid
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import warnings

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
    import accelerate
    import numpy as np
except ImportError as e:
    print(f"Warning: Required dependencies not installed: {e}")
    print("Please install with: pip install transformers torch accelerate")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelType(Enum):
    """Supported LLM model types"""
    PHI3_MINI = "microsoft/Phi-3-mini-4k-instruct"
    PHI3_MINI_128K = "microsoft/Phi-3-mini-128k-instruct"
    PHI3_SMALL = "microsoft/Phi-3-small-8k-instruct"
    PHI3_MEDIUM = "microsoft/Phi-3-medium-4k-instruct"


class ProcessingMode(Enum):
    """Processing modes for LLM inference"""
    STANDARD = "standard"
    OPTIMIZED = "optimized"  # Pi 5 optimized
    STREAMING = "streaming"  # Real-time streaming
    BATCH = "batch"          # Batch processing


class OutputFormat(Enum):
    """Output format types"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"
    MARKDOWN = "markdown"


@dataclass
class LLMConfig:
    """Configuration for LLM processing"""
    
    # Model settings
    model_type: ModelType = ModelType.PHI3_MINI
    model_path: Optional[str] = None  # Local path if cached
    device: str = "cpu"  # Pi 5 uses CPU
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Context settings
    max_input_length: int = 3000  # Leave room for generation
    context_window: int = 4096    # Phi-3 Mini context window
    
    # Performance settings
    batch_size: int = 1
    num_threads: int = 4  # Pi 5 has 4 cores
    use_cache: bool = True
    
    # Optimization settings
    torch_dtype: str = "float32"  # Better Pi 5 compatibility
    trust_remote_code: bool = True
    use_flash_attention: bool = False  # Not available on Pi 5
    
    # Output settings
    output_format: OutputFormat = OutputFormat.TEXT
    stream_output: bool = False
    
    # Safety settings
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_type": self.model_type.value,
            "model_path": self.model_path,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_input_length": self.max_input_length,
            "context_window": self.context_window,
            "batch_size": self.batch_size,
            "num_threads": self.num_threads,
            "use_cache": self.use_cache,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "use_flash_attention": self.use_flash_attention,
            "output_format": self.output_format.value,
            "stream_output": self.stream_output,
            "do_sample": self.do_sample
        }


@dataclass
class LLMResult:
    """Result from LLM processing"""
    
    # Generation result
    text: str
    
    # Timing information
    processing_time: float
    tokens_generated: int
    tokens_per_second: float
    
    # Input context
    input_text: str
    input_tokens: int
    
    # Quality metrics
    confidence: float = 0.0
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    
    # Generation metadata
    generation_config: Dict[str, Any] = field(default_factory=dict)
    model_used: str = ""
    
    # Status
    success: bool = True
    error_message: str = ""
    
    # Additional metadata
    memory_usage: float = 0.0
    max_memory_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "text": self.text,
            "processing_time": self.processing_time,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "input_text": self.input_text,
            "input_tokens": self.input_tokens,
            "confidence": self.confidence,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "generation_config": self.generation_config,
            "model_used": self.model_used,
            "success": self.success,
            "error_message": self.error_message,
            "memory_usage": self.memory_usage,
            "max_memory_usage": self.max_memory_usage
        }


class LocalLLMProcessor:
    """Main local LLM processing engine"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_initialized = False
        
        # Model info
        self.model_info = {}
        
        # Performance tracking
        self.stats = {
            "total_generations": 0,
            "total_tokens_generated": 0,
            "total_processing_time": 0.0,
            "average_tokens_per_second": 0.0,
            "memory_usage_peak": 0.0,
            "error_count": 0,
            "model_load_time": 0.0
        }
        
        # Threading
        self.generation_lock = threading.Lock()
        
        # Callbacks
        self.result_callbacks: List[callable] = []
        self.error_callbacks: List[callable] = []
        
        logger.info(f"LocalLLMProcessor initialized with {self.config.model_type.value}")
        
    def initialize(self) -> bool:
        """Initialize the local LLM model"""
        try:
            logger.info("Loading local LLM model...")
            start_time = time.time()
            
            # Set torch settings for Pi 5
            torch.set_num_threads(self.config.num_threads)
            if self.config.torch_dtype == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
                
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_type.value,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_type.value,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map={"": self.config.device},
                use_cache=self.config.use_cache
            )
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" else -1,
                torch_dtype=torch_dtype
            )
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            # Store model info
            self.model_info = {
                "model_type": self.config.model_type.value,
                "model_size": self._get_model_size(),
                "device": self.config.device,
                "load_time": load_time,
                "context_window": self.config.context_window,
                "parameters": self._get_model_parameters()
            }
            
            self.is_initialized = True
            logger.info(f"LLM model loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            self._call_error_callbacks(str(e))
            return False
            
    def _get_model_size(self) -> str:
        """Get model size information"""
        try:
            if self.model:
                param_count = sum(p.numel() for p in self.model.parameters())
                if param_count > 1e9:
                    return f"{param_count / 1e9:.1f}B parameters"
                elif param_count > 1e6:
                    return f"{param_count / 1e6:.1f}M parameters"
                else:
                    return f"{param_count / 1e3:.1f}K parameters"
            return "Unknown"
        except Exception as e:
            logger.warning(f"Could not get model size: {e}")
            return "Unknown"
            
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameter information"""
        try:
            if self.model and hasattr(self.model, 'config'):
                config = self.model.config
                return {
                    "hidden_size": getattr(config, "hidden_size", 0),
                    "num_attention_heads": getattr(config, "num_attention_heads", 0),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", 0),
                    "vocab_size": getattr(config, "vocab_size", 0),
                    "max_position_embeddings": getattr(config, "max_position_embeddings", 0)
                }
            return {}
        except Exception as e:
            logger.warning(f"Could not get model parameters: {e}")
            return {}
            
    def generate(self, prompt: str, **kwargs) -> LLMResult:
        """Generate text using the LLM"""
        if not self.is_initialized:
            return LLMResult(
                text="",
                processing_time=0.0,
                tokens_generated=0,
                tokens_per_second=0.0,
                input_text=prompt,
                input_tokens=0,
                success=False,
                error_message="LLM not initialized",
                model_used=self.config.model_type.value
            )
            
        process_start = time.time()
        
        try:
            with self.generation_lock:
                # Prepare input
                input_text = self._prepare_input(prompt)
                input_tokens = len(self.tokenizer.encode(input_text))
                
                # Check input length
                if input_tokens > self.config.max_input_length:
                    input_text = self._truncate_input(input_text, self.config.max_input_length)
                    input_tokens = len(self.tokenizer.encode(input_text))
                
                # Prepare generation config
                generation_config = self._prepare_generation_config(kwargs)
                
                # Generate
                logger.debug(f"Generating with {input_tokens} input tokens")
                
                outputs = self.pipeline(
                    input_text,
                    **generation_config,
                    return_full_text=False
                )
                
                # Extract result
                if isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0]["generated_text"]
                else:
                    generated_text = str(outputs)
                    
                # Clean up output
                generated_text = self._clean_output(generated_text)
                
                # Calculate metrics
                processing_time = time.time() - process_start
                tokens_generated = len(self.tokenizer.encode(generated_text))
                tokens_per_second = tokens_generated / processing_time if processing_time > 0 else 0
                
                # Create result
                result = LLMResult(
                    text=generated_text,
                    processing_time=processing_time,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    input_text=input_text,
                    input_tokens=input_tokens,
                    generation_config=generation_config,
                    model_used=self.config.model_type.value,
                    success=True
                )
                
                # Calculate quality metrics
                result.confidence = self._calculate_confidence(generated_text)
                result.coherence_score = self._calculate_coherence(generated_text)
                result.relevance_score = self._calculate_relevance(input_text, generated_text)
                
                # Update statistics
                self._update_stats(result)
                
                # Call callbacks
                self._call_result_callbacks(result)
                
                return result
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.stats["error_count"] += 1
            self._call_error_callbacks(str(e))
            
            return LLMResult(
                text="",
                processing_time=time.time() - process_start,
                tokens_generated=0,
                tokens_per_second=0.0,
                input_text=prompt,
                input_tokens=0,
                success=False,
                error_message=str(e),
                model_used=self.config.model_type.value
            )
            
    def _prepare_input(self, prompt: str) -> str:
        """Prepare input text for generation"""
        try:
            # Apply Phi-3 chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Simple format for Phi-3
                return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
                
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            return prompt
            
    def _truncate_input(self, text: str, max_tokens: int) -> str:
        """Truncate input to fit within token limit"""
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
                
            # Truncate from the beginning, keeping the end
            truncated_tokens = tokens[-max_tokens:]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
        except Exception as e:
            logger.warning(f"Failed to truncate input: {e}")
            return text[:max_tokens * 4]  # Rough approximation
            
    def _prepare_generation_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare generation configuration"""
        config = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": kwargs.get("do_sample", self.config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        return config
        
    def _clean_output(self, text: str) -> str:
        """Clean generated output"""
        try:
            # Remove special tokens
            text = text.replace("<|end|>", "")
            text = text.replace("<|user|>", "")
            text = text.replace("<|assistant|>", "")
            
            # Strip whitespace
            text = text.strip()
            
            # Remove incomplete sentences at the end
            if text and not text.endswith(('.', '!', '?', ':', ';')):
                sentences = text.split('.')
                if len(sentences) > 1:
                    text = '.'.join(sentences[:-1]) + '.'
                    
            return text
            
        except Exception as e:
            logger.warning(f"Failed to clean output: {e}")
            return text
            
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for generated text"""
        try:
            # Simple heuristic based on text characteristics
            if not text:
                return 0.0
                
            # Length factor
            length_factor = min(len(text) / 100, 1.0)
            
            # Completeness factor (ends with punctuation)
            completeness_factor = 1.0 if text.endswith(('.', '!', '?')) else 0.7
            
            # Coherence factor (no repetition)
            words = text.split()
            unique_words = set(words)
            repetition_factor = len(unique_words) / len(words) if words else 0
            
            confidence = (length_factor + completeness_factor + repetition_factor) / 3
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5
            
    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score for generated text"""
        try:
            # Simple coherence metric
            if not text:
                return 0.0
                
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0.8
                
            # Check for logical flow (simple heuristic)
            coherence_score = 0.8  # Base score
            
            # Penalize very short sentences
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:
                coherence_score -= 0.2
                
            return min(1.0, max(0.0, coherence_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate coherence: {e}")
            return 0.5
            
    def _calculate_relevance(self, input_text: str, output_text: str) -> float:
        """Calculate relevance score between input and output"""
        try:
            # Simple relevance metric based on word overlap
            if not input_text or not output_text:
                return 0.0
                
            input_words = set(input_text.lower().split())
            output_words = set(output_text.lower().split())
            
            if not input_words:
                return 0.0
                
            overlap = len(input_words.intersection(output_words))
            relevance = overlap / len(input_words)
            
            return min(1.0, max(0.0, relevance))
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance: {e}")
            return 0.5
            
    def _update_stats(self, result: LLMResult):
        """Update processing statistics"""
        self.stats["total_generations"] += 1
        
        if result.success:
            self.stats["total_tokens_generated"] += result.tokens_generated
            self.stats["total_processing_time"] += result.processing_time
            
            # Update average tokens per second
            if self.stats["total_processing_time"] > 0:
                self.stats["average_tokens_per_second"] = (
                    self.stats["total_tokens_generated"] / self.stats["total_processing_time"]
                )
                
    def _call_result_callbacks(self, result: LLMResult):
        """Call result callbacks"""
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
                
    def _call_error_callbacks(self, error_message: str):
        """Call error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
                
    def add_result_callback(self, callback: callable):
        """Add result callback"""
        self.result_callbacks.append(callback)
        
    def add_error_callback(self, callback: callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return self.model_info.copy()
        
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            "is_initialized": self.is_initialized,
            "model_info": self.get_model_info(),
            "stats": self.get_stats(),
            "config": self.config.to_dict()
        }
        
    def shutdown(self):
        """Shutdown LLM processor"""
        logger.info("Shutting down LLM processor...")
        self.is_initialized = False
        
        # Clear model from memory
        if self.model:
            del self.model
            self.model = None
            
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("LLM processor shutdown complete")
        
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Factory functions
def create_phi3_processor(device: str = "cpu") -> LocalLLMProcessor:
    """Create Phi-3 Mini processor"""
    config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        device=device,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    return LocalLLMProcessor(config)


def create_optimized_processor(device: str = "cpu") -> LocalLLMProcessor:
    """Create optimized processor for Pi 5"""
    config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        device=device,
        max_new_tokens=256,  # Reduced for performance
        temperature=0.6,     # Slightly more focused
        top_p=0.8,          # Slightly more focused
        top_k=30,           # Reduced for performance
        num_threads=4,      # Pi 5 has 4 cores
        batch_size=1,       # Single batch for Pi 5
        torch_dtype="float32",  # Better Pi 5 compatibility
        use_cache=True
    )
    return LocalLLMProcessor(config)


def create_meeting_processor(device: str = "cpu") -> LocalLLMProcessor:
    """Create processor optimized for meeting analysis"""
    config = LLMConfig(
        model_type=ModelType.PHI3_MINI,
        device=device,
        max_new_tokens=1024,  # Longer for summaries
        temperature=0.5,      # More focused for analysis
        top_p=0.8,           # More focused
        repetition_penalty=1.2,  # Avoid repetition
        max_input_length=3000,   # Longer meetings
        output_format=OutputFormat.STRUCTURED
    )
    return LocalLLMProcessor(config)


# Testing and demonstration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Local LLM Processor Test")
    parser.add_argument("--prompt", type=str, required=True, help="Test prompt")
    parser.add_argument("--model", type=str, default="phi3-mini", 
                       choices=["phi3-mini", "phi3-small"], help="Model to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()
    
    # Create processor
    if args.model == "phi3-mini":
        processor = create_optimized_processor(args.device)
    else:
        processor = create_meeting_processor(args.device)
        
    # Update config
    processor.config.max_new_tokens = args.max_tokens
    
    try:
        # Initialize
        if not processor.initialize():
            print("Failed to initialize processor")
            sys.exit(1)
            
        print(f"Processor status: {processor.get_status()}")
        
        # Generate
        print(f"Generating response to: {args.prompt}")
        result = processor.generate(args.prompt)
        
        if result.success:
            print(f"Response: {result.text}")
            print(f"Processing time: {result.processing_time:.3f}s")
            print(f"Tokens generated: {result.tokens_generated}")
            print(f"Tokens per second: {result.tokens_per_second:.1f}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Coherence: {result.coherence_score:.3f}")
            print(f"Relevance: {result.relevance_score:.3f}")
        else:
            print(f"Generation failed: {result.error_message}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        processor.shutdown()