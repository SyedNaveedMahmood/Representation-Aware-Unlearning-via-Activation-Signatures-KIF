 # Module 0: Enhanced Model Setup & Quantization

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kif_setup.log")
    ]
)
logger = logging.getLogger('KIF-Module0')

@dataclass
class ModelConfig:
    """Enhanced configuration for model setup and quantization"""
    model_id: str = "Qwen/Qwen3-14B"
    output_dir: Path = Path("outputs/model")
    load_in_4bit: bool = True
    quant_type: str = "nf4"  # Explicitly use NF4 for better quality
    compute_dtype: str = "bfloat16"
    double_quant: bool = True
    device_map: str = "auto"
    warmup_prompt: str = "Hello, world!"
    warmup_new_tokens: int = 10
    use_cache: bool = True
    seed: int = 42
    max_memory_usage: float = 0.85  # Maximum GPU memory threshold
    
    # Memory settings
    mem_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_split_size_mb": 128,  # Prevent memory fragmentation
        "pytorch_cuda_alloc_conf": "max_split_size_mb:128"
    })
    
    def __post_init__(self):
        """Validate configuration and set environment variables"""
        self.output_dir = Path(self.output_dir)
        
        # Set environment variables for optimized memory use
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.mem_config["pytorch_cuda_alloc_conf"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "model_id": self.model_id,
            "quant": {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.double_quant,
            },
            "device_map": self.device_map,
            "output_dir": str(self.output_dir),
            "warmup_prompt": self.warmup_prompt,
            "warmup_new_tokens": self.warmup_new_tokens,
            "max_memory_usage": self.max_memory_usage,
            "seed": self.seed
        }


class ModelManager:
    """Enhanced model manager with robust error handling and verification"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.verify_environment()
    
    def verify_environment(self) -> None:
        """Verify system environment and requirements"""
        logger.info("Verifying environment...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Using CPU (this will be slow)")
        else:
            # Check GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} with {gpu_mem:.2f} GB memory")
            
            if gpu_mem < 16:
                logger.warning(f"GPU memory ({gpu_mem:.2f} GB) is less than recommended 16 GB")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.config.output_dir / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info("Environment verification complete")
    
    def setup_quantization(self) -> BitsAndBytesConfig:
        """Configure quantization settings with validation"""
        try:
            # Ensure proper bitsandbytes version
            import bitsandbytes as bnb
            bnb_version = bnb.__version__
            
            if not bnb_version.startswith("0.43."):
                logger.warning(f"Using bitsandbytes {bnb_version} - version 0.43.x is recommended for stability")
            
            logger.info(f"Configuring 4-bit quantization ({self.config.quant_type})")
            
            # Create quantization configuration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.compute_dtype),
                bnb_4bit_use_double_quant=self.config.double_quant
            )
            
            return bnb_config
            
        except ImportError as e:
            logger.error(f"Failed to import bitsandbytes: {e}")
            logger.error("Please install bitsandbytes==0.43.* for optimal quantization")
            raise
    
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and quantize model with enhanced error handling"""
        try:
            logger.info(f"Loading tokenizer from {self.config.model_id}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
            
            # Ensure tokenizer has necessary special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Set up quantization
            bnb_config = self.setup_quantization()
            
            # Monitor memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
                logger.info(f"GPU memory usage before model loading: {mem_before:.2%}")
            
            logger.info(f"Loading and quantizing model from {self.config.model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                torch_dtype=getattr(torch, self.config.compute_dtype),
                use_cache=self.config.use_cache
            )
            
            # Monitor memory after loading
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
                logger.info(f"GPU memory usage after model loading: {mem_after:.2%}")
            
            self.model = model
            self.tokenizer = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise
    
    def save_model(self) -> None:
        """Save model and tokenizer with verification"""
        if self.model is None or self.tokenizer is None:
            logger.error("Cannot save model or tokenizer: not loaded")
            return
        
        try:
            logger.info(f"Saving model to {self.config.output_dir}")
            
            # Save model with safetensors format
            self.model.save_pretrained(
                self.config.output_dir,
                safe_serialization=True
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Verify saved files
            model_file = self.config.output_dir / "model.safetensors"
            if not model_file.exists():
                logger.warning(f"Expected model file {model_file} not found")
            
            logger.info(f"✅ Model and tokenizer saved to {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def verify_model(self) -> bool:
        """Verify model is functioning correctly with test inference"""
        if self.model is None or self.tokenizer is None:
            logger.error("Cannot verify model: not loaded")
            return False
        
        try:
            logger.info("Running test inference for verification...")
            
            # Prepare input
            inputs = self.tokenizer(self.config.warmup_prompt, return_tensors="pt").to(self.model.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.warmup_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode output
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify output is reasonable
            if len(text) <= len(self.config.warmup_prompt):
                logger.warning("Model output suspiciously short - possible issue with generation")
                return False
            
            logger.info(f"✅ Warm-up successful, output: {text}")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def report_memory_usage(self) -> Dict[str, Any]:
        """Report detailed memory usage statistics"""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats["allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory_stats["reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            memory_stats["max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Get per-device memory
            memory_stats["per_device"] = {}
            for i in range(torch.cuda.device_count()):
                memory_stats["per_device"][i] = {
                    "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
                }
            
            logger.info(f"GPU memory allocated: {memory_stats['allocated_gb']:.2f} GB")
            logger.info(f"GPU memory reserved: {memory_stats['reserved_gb']:.2f} GB")
        
        # CPU memory via psutil if available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_stats["cpu_gb"] = process.memory_info().rss / (1024**3)
            logger.info(f"CPU memory (RSS): {memory_stats['cpu_gb']:.2f} GB")
        except ImportError:
            logger.info("psutil not available, skipping CPU memory stats")
        
        return memory_stats


def run_module0() -> tuple[Optional[PreTrainedModel], Optional[PreTrainedTokenizer]]:
    """Main function to run Module 0 with comprehensive error handling"""
    try:
        # Initialize with user-friendly starting message
        logger.info("=" * 50)
        logger.info("Starting KIF Module 0: Model Setup & Quantization")
        logger.info("=" * 50)
        
        # Create configuration
        cfg = ModelConfig()
        
        # Initialize model manager
        manager = ModelManager(cfg)
        
        # Load model and tokenizer
        model, tokenizer = manager.load_model_and_tokenizer()
        
        # Save model and tokenizer
        manager.save_model()
        
        # Verify model functioning
        if not manager.verify_model():
            logger.error("Model verification failed - investigate before proceeding")
            return None, None
        
        # Report memory usage
        memory_stats = manager.report_memory_usage()
        
        logger.info("=" * 50)
        logger.info("Module 0 completed successfully")
        logger.info("=" * 50)
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Module 0 failed with error: {e}", exc_info=True)
        return None, None


if __name__ == "__main__":
    model, tokenizer = run_module0()