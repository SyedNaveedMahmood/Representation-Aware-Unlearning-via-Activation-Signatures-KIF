#MODULE D V2 Plus

# Module D: Knowledge Suppression Capsule Forger - FIXED VERSION (Hardened Loader)
# Implements dimension mismatch fixes, robust signature handling, and safe CPU-first model load.

import os
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm
import pickle
import gzip
import higher
from higher.patch import monkeypatch as make_functional

# Imports for compression
try:
    import compress_pickle
except ImportError:
    class CompressPickleFallback:
        def load(self, filename):
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        def dump(self, data, filename, compression="gzip", compresslevel=3, **kwargs):
            with gzip.open(filename, 'wb', compresslevel=compresslevel) as f:
                pickle.dump(data, f)
    compress_pickle = CompressPickleFallback()

# Model loading imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kif_module_d.log")
    ]
)
logger = logging.getLogger('KIF-ModuleD')

# ------------------------------ CUDA safety helpers ------------------------------
def _cuda_looks_usable() -> bool:
    """Return True if CUDA is available and mem_get_info works without raising."""
    if not torch.cuda.is_available():
        return False
    try:
        idx = torch.cuda.current_device()
        _ = torch.cuda.mem_get_info(idx)  # where your original crash happened
        x = torch.empty(1, device=f"cuda:{idx}")
        torch.cuda.synchronize()
        return True
    except Exception as e:
        logger.warning(f"[CUDA] Unusable CUDA detected, falling back to CPU. Reason: {e}")
        return False

# ----------------------------------- Config -------------------------------------
@dataclass
class CapsuleConfig:
    """Configuration for Knowledge Suppression Capsule"""
    # I/O paths
    model_dir: str = "outputs/model"
    signatures_file: str = "outputs/signatures/top_signatures.pkl.gz"
    prompts_file: str = "outputs/datasets/prompts.jsonl"
    output_dir: Path = Path("outputs/capsules")
    
    # Capsule architecture
    adapter_type: str = "ia3"  # "ia3" or "lora"
    scaling_factor_init: float = -1.0  # Initial suppression strength
    max_scaling_factor: float = -5.0   # Maximum suppression strength
    
    # LoRA specific (if adapter_type == "lora")
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 2
    warmup_steps: int = 50
    
    # Loss weights
    suppression_weight: float = 1.0
    integrity_weight: float = 0.3
    
    # Evaluation
    eval_batch_size: int = 4
    max_new_tokens: int = 30
    
    # Device settings (AUTO with CUDA probing)
    device: str = "auto"
    use_half_precision: bool = True
    memory_threshold_mb: float = 6000
    
    # MEND integration - simplified for now
    use_mend: bool = False  # Disable until properly implemented
    mend_learning_rate: float = 5e-4
    mend_steps: int = 5
    
    # Stage 1: Validation settings
    allow_dimension_projection: bool = True
    strict_validation: bool = True
    min_prompts_for_training: int = 3
    
    # NEW: Advanced dimension handling
    force_dimension_match: bool = True
    signature_vector_max_dim: int = 10000  # Maximum allowed signature dimension
    use_learnable_projection: bool = True  # Use learnable projection matrices

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if _cuda_looks_usable() else "cpu"
        if self.device != "cuda":
            self.use_half_precision = False
        logger.info(f"[Config] Using device={self.device}, fp16={self.use_half_precision}")

# ------------------------------ Signature Utils ---------------------------------
def validate_signature_vector(signature_vector: np.ndarray, config: CapsuleConfig) -> np.ndarray:
    """
    Validate and preprocess signature vector to prevent dimension issues.
    """
    if signature_vector.ndim != 1:
        if signature_vector.ndim == 2 and signature_vector.shape[0] == 1:
            signature_vector = signature_vector.flatten()
            logger.warning(f"Flattened 2D signature vector from shape {signature_vector.shape}")
        else:
            raise ValueError(f"Signature must be 1D vector, got shape {signature_vector.shape}")
    
    sig_dim = signature_vector.shape[0]
    
    # Check for extremely large dimensions that might indicate data corruption
    if sig_dim > config.signature_vector_max_dim:
        logger.error(f"Signature dimension {sig_dim} exceeds maximum allowed {config.signature_vector_max_dim}")
        logger.error("This likely indicates corrupted signature data or incorrect processing")
        raise ValueError(
            f"Signature dimension {sig_dim} is too large (max: {config.signature_vector_max_dim}). "
            "This suggests the signature vector contains flattened weights or corrupted data."
        )
    
    # Check for invalid values
    if np.any(~np.isfinite(signature_vector)):
        logger.warning("Signature vector contains non-finite values, cleaning...")
        signature_vector = np.nan_to_num(signature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Normalize to prevent gradient issues
    norm = np.linalg.norm(signature_vector)
    if norm > 0:
        signature_vector = signature_vector / norm
    else:
        logger.warning("Zero-norm signature vector detected, using random initialization")
        signature_vector = np.random.normal(0, 0.01, size=signature_vector.shape).astype(np.float32)
    
    logger.debug(f"Validated signature vector: shape={signature_vector.shape}, norm={np.linalg.norm(signature_vector):.4f}")
    return signature_vector

def make_suppression_direction(signature_vector: np.ndarray, target_hidden_size: int, 
                              config: CapsuleConfig) -> torch.Tensor:
    """
    Create a properly dimensioned suppression direction with robust dimension handling.
    """
    signature_vector = validate_signature_vector(signature_vector, config)
    sig_dim = signature_vector.shape[0]
    
    logger.info(f"Creating suppression direction: sig_dim={sig_dim}, target_dim={target_hidden_size}")
    
    if sig_dim == target_hidden_size:
        direction = torch.tensor(signature_vector, dtype=torch.float32, device=config.device)
        logger.debug("Perfect dimension match, using signature as-is")
    elif config.allow_dimension_projection:
        logger.info(f"Dimension mismatch: {sig_dim} -> {target_hidden_size}, applying projection")
        if sig_dim > target_hidden_size:
            direction = torch.tensor(signature_vector[:target_hidden_size], dtype=torch.float32, device=config.device)
            logger.warning(f"Truncated signature from {sig_dim} to {target_hidden_size} dimensions")
        else:
            if config.use_learnable_projection and sig_dim * 2 < target_hidden_size:
                padded = np.zeros(target_hidden_size, dtype=np.float32)
                indices = np.linspace(0, target_hidden_size-1, sig_dim, dtype=int)
                padded[indices] = signature_vector
                direction = torch.tensor(padded, dtype=torch.float32, device=config.device)
                logger.info(f"Applied interpolation padding from {sig_dim} to {target_hidden_size}")
            else:
                padded = np.zeros(target_hidden_size, dtype=np.float32)
                padded[:sig_dim] = signature_vector
                direction = torch.tensor(padded, dtype=torch.float32, device=config.device)
                logger.info(f"Applied zero padding from {sig_dim} to {target_hidden_size}")
    else:
        raise ValueError(
            f"Signature dimension {sig_dim} != target hidden size {target_hidden_size}. "
            f"Set allow_dimension_projection=True to enable automatic projection."
        )
    
    # Normalize the direction vector
    direction = F.normalize(direction, dim=0, eps=1e-8)
    logger.debug(f"Created suppression direction: final_dim={direction.shape[0]}, norm={direction.norm():.4f}")
    return direction

# ---------------------------------- Adapters ------------------------------------
class IA3Adapter(nn.Module):
    """IA³ Adapter for knowledge suppression (now with an activation-only path)"""
    
    def __init__(self, original_module: nn.Module, suppression_direction: torch.Tensor, 
                 scaling_factor: float = -1.0):
        super().__init__()
        self.original_module = original_module
        
        # Store suppression parameters
        self.register_buffer('suppression_direction', suppression_direction)
        self.suppression_strength = nn.Parameter(torch.tensor(scaling_factor))
        
        # Get module dimensions for validation
        if hasattr(original_module, 'weight'):
            self.hidden_dim = original_module.weight.shape[0]
        else:
            raise ValueError("Original module must have weight attribute")
            
        # Validate dimensions match
        if suppression_direction.shape[0] != self.hidden_dim:
            raise ValueError(
                f"Suppression direction dimension {suppression_direction.shape[0]} "
                f"does not match module hidden dimension {self.hidden_dim}"
            )
            
        logger.debug(f"IA3 adapter created: hidden_dim={self.hidden_dim}, "
                    f"direction_dim={suppression_direction.shape[0]}")
    
    def _apply_on_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Apply directional suppression directly on an activation tensor
        (no second forward through the original module).
        """
        try:
            output_hidden_dim = activation.shape[-1]
            direction = self.suppression_direction
            if direction.device != activation.device:
                direction = direction.to(activation.device)
            if direction.dtype != activation.dtype:
                direction = direction.to(activation.dtype)

            if output_hidden_dim != direction.shape[0]:
                logger.error("IA3 apply_on_activation: dimension mismatch "
                             f"{output_hidden_dim} vs {direction.shape[0]}; returning activation.")
                return activation

            # projection: [...,]
            proj = torch.matmul(activation, direction)
            # broadcast back to hidden via outer product
            if activation.dim() == 3:  # [batch, seq, hidden]
                comp = proj.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
            elif activation.dim() == 2:  # [batch, hidden]
                comp = proj.unsqueeze(-1) * direction.unsqueeze(0)
            else:
                logger.warning(f"IA3 apply_on_activation: unexpected activation shape {activation.shape}")
                return activation

            s = torch.clamp(self.suppression_strength, min=-5.0, max=0.0)  # safety clamp
            return activation + s * comp
        except Exception as e:
            logger.warning(f"IA3 apply_on_activation failed: {e}")
            return activation
        
    def forward(self, x):
        """
        Legacy path (kept): run the original module, then apply suppression.
        Prefer using _apply_on_activation on the module's output via a hook to avoid double-forward.
        """
        original_output = self.original_module(x)
        return self._apply_on_activation(original_output)

class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation adapter for knowledge suppression"""
    
    def __init__(self, original_module: nn.Module, rank: int = 8, alpha: float = 16, dropout: float = 0.1):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        
        if hasattr(original_module, 'weight'):
            in_features = original_module.weight.shape[1]
            out_features = original_module.weight.shape[0]
        else:
            raise ValueError("Original module must have weight attribute")
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        self.suppression_strength = nn.Parameter(torch.ones(1))
        
        logger.debug(f"LoRA adapter created: rank={rank}, in_features={in_features}, out_features={out_features}")
        
    def forward(self, x):
        original_output = self.original_module(x)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        suppressed_output = original_output + self.suppression_strength * lora_output
        return suppressed_output

# ---------------------------- Knowledge Capsule ---------------------------------
class KnowledgeSuppressionCapsule:
    """Knowledge suppression capsule using activation scaling and adapters"""
    
    def __init__(self, model: PreTrainedModel, subject: str, signature_data: Dict, config: CapsuleConfig):
        self.model = model
        self.subject = subject
        self.signature_data = signature_data
        self.config = config
        self.hooks = []
        self.adapters = {}
        self.is_active = False
        self.creation_status = "pending"
        self.error_message = None
        
        # Stage 1: Strict validation of signature data
        self._validate_signature_data()
        
        # Extract signature information with validation
        self.target_layer = signature_data.get("best_layer")
        self.effect_size = signature_data.get("effect_size", 0.0)
        
        # Extract signature vector with proper error handling
        signatures = signature_data.get("signatures", [])
        if not signatures or len(signatures) == 0:
            raise ValueError(f"No signature vectors found for subject {subject}")
        
        raw_signature = signatures[0]
        if isinstance(raw_signature, list):
            self.signature_vector = np.array(raw_signature, dtype=np.float32)
        elif isinstance(raw_signature, np.ndarray):
            self.signature_vector = raw_signature.astype(np.float32)
        else:
            raise ValueError(f"Invalid signature format for {subject}: {type(raw_signature)}")
        
        logger.info(f"Loaded signature for {subject}: shape={self.signature_vector.shape}, "
                   f"layer={self.target_layer}, effect_size={self.effect_size:.4f}")
        
        # Validate signature vector early
        try:
            self.signature_vector = validate_signature_vector(self.signature_vector, self.config)
        except ValueError as e:
            self.creation_status = "failed"
            self.error_message = f"Invalid signature vector: {e}"
            raise
        
    def _validate_signature_data(self):
        """Validate signature data structure"""
        required_fields = ["best_layer", "signatures"]
        for field in required_fields:
            if field not in self.signature_data:
                raise ValueError(f"Missing required field '{field}' in signature data for {self.subject}")
    
    def setup_adapter(self):
        """Setup the adapter for the target layer with robust dimension handling"""
        if hasattr(self, 'adapter'):
            logger.info(f"Adapter already exists for {self.subject}")
            return
            
        # Find target module with better error handling
        target_module = None
        target_module_name = None
        
        # Prefer MLP projections
        for name, module in self.model.named_modules():
            if (f"layers.{self.target_layer}.mlp" in name and 
                isinstance(module, nn.Linear) and
                ("up_proj" in name or "gate_proj" in name or "c_fc" in name)):
                target_module = module
                target_module_name = name
                break
                    
        if target_module is None:
            # Fallback: any Linear in layer
            for name, module in self.model.named_modules():
                if (f"layers.{self.target_layer}" in name and isinstance(module, nn.Linear)):
                    target_module = module
                    target_module_name = name
                    logger.warning(f"Using fallback module {name} for {self.subject}")
                    break
                    
        if target_module is None:
            raise ValueError(f"Could not find any Linear module in layer {self.target_layer} for {self.subject}")
            
        self.target_module_name = target_module_name
        
        # Get target hidden dimension from the module
        target_hidden_size = target_module.weight.shape[0]
        logger.info(f"Target module {target_module_name}: weight shape={target_module.weight.shape}, "
                   f"hidden_size={target_hidden_size}")
        
        # Create suppression direction with validation
        try:
            suppression_direction = make_suppression_direction(
                self.signature_vector, 
                target_hidden_size,
                self.config
            )
        except ValueError as e:
            self.creation_status = "failed"
            self.error_message = f"Failed to create suppression direction: {e}"
            raise ValueError(f"Failed to create suppression direction for {self.subject}: {e}")
        
        # Create adapter based on config
        try:
            if self.config.adapter_type == "lora":
                self.adapter = LoRAAdapter(
                    target_module, 
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
            else:  # IA³
                self.adapter = IA3Adapter(
                    target_module,
                    suppression_direction,
                    scaling_factor=self.config.scaling_factor_init
                )
            
            self.adapter = self.adapter.to(self.config.device)
            self.creation_status = "success"
            logger.info(f"Successfully setup {self.config.adapter_type} adapter for {self.subject} at {target_module_name}")
            
        except Exception as e:
            self.creation_status = "failed"
            self.error_message = f"Adapter creation failed: {e}"
            logger.error(f"Failed to create adapter for {self.subject}: {e}")
            raise
        
    def activate(self):
        """Activate the suppression capsule"""
        if self.is_active or not hasattr(self, 'adapter'):
            return self
            
        # Hook that applies suppression to the *output* (avoids double-forward)
        def suppression_hook(module, inputs, output):
            try:
                if isinstance(output, tuple):
                    main = output[0]
                else:
                    main = output

                # Ensure adapter buffers are on same device/dtype as output
                if hasattr(self.adapter, 'suppression_direction'):
                    if self.adapter.suppression_direction.device != main.device:
                        self.adapter.suppression_direction = self.adapter.suppression_direction.to(main.device)
                    if self.adapter.suppression_direction.dtype != main.dtype:
                        self.adapter.suppression_direction = self.adapter.suppression_direction.to(main.dtype)

                if isinstance(self.adapter, IA3Adapter):
                    suppressed = self.adapter._apply_on_activation(main)
                else:
                    # For LoRAAdapter path, we cannot apply on activation directly; return original output
                    suppressed = main

                if isinstance(output, tuple):
                    return (suppressed,) + tuple(output[1:])
                return suppressed

            except Exception as e:
                logger.warning(f"Hook error for {self.subject}: {e}")
                return output
            
        # Find and hook the target module
        for name, module in self.model.named_modules():
            if name == self.target_module_name:
                hook = module.register_forward_hook(suppression_hook)
                self.hooks.append(hook)
                break
        
        self.is_active = True
        logger.info(f"Activated suppression capsule for {self.subject}")
        return self
    
    def deactivate(self):
        """Deactivate the suppression capsule"""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.is_active = False
        logger.info(f"Deactivated suppression capsule for {self.subject}")
        return self

# ---------------------------- Forger (hardened load) -----------------------------
class CapsuleForger:
    """Main class for creating and managing knowledge suppression capsules"""
    
    def __init__(self, config: CapsuleConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.capsules = {}
        self.stats = {
            "total_subjects": 0,
            "successful_capsules": 0,
            "failed_capsules": 0,
            "failures": []
        }

    def _safe_load_model(self):
        """
        Load model on CPU first (device_map='cpu') to avoid CUDA allocator warm-up
        that can call torch.cuda.mem_get_info() and assert. Then move to CUDA if healthy.
        """
        logger.info(f"[Forger] Loading model from {self.config.model_dir} on CPU to avoid CUDA warm-up...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32  # change after move if CUDA + fp16
        )
        tok = AutoTokenizer.from_pretrained(self.config.model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Move to chosen device
        if self.config.device == "cuda":
            try:
                dtype = torch.float16 if self.config.use_half_precision else torch.float32
                logger.info(f"[Forger] Moving model to CUDA (dtype={dtype})...")
                model.to(dtype=dtype, device="cuda")
            except Exception as e:
                logger.warning(f"[Forger] Failed to move model to CUDA ({e}). Staying on CPU.")
                self.config.device = "cpu"
                self.config.use_half_precision = False
                model.to("cpu")

        return model, tok
        
    def load_model_and_signatures(self):
        """Load model and signature data (hardened)"""
        logger.info(f"Preparing model & tokenizer")
        self.model, self.tokenizer = self._safe_load_model()
        
        logger.info(f"Loading signatures from {self.config.signatures_file}")
        if not os.path.exists(self.config.signatures_file):
            raise FileNotFoundError(f"Signatures file not found: {self.config.signatures_file}")
        
        signatures = compress_pickle.load(self.config.signatures_file)
        logger.info(f"Loaded {len(signatures)} signature subjects")
        
        return signatures
    
    def create_capsules(self, signatures: Dict[str, Any]):
        """Create suppression capsules for all subjects"""
        self.stats["total_subjects"] = len(signatures)
        
        for subject, signature_data in tqdm(signatures.items(), desc="Creating capsules"):
            try:
                logger.info(f"Creating capsule for subject: {subject}")
                
                # Create capsule
                capsule = KnowledgeSuppressionCapsule(
                    self.model, subject, signature_data, self.config
                )
                
                # Setup adapter & (optionally) activate
                capsule.setup_adapter()
                capsule.activate()
                
                # Store capsule
                self.capsules[subject] = capsule
                self.stats["successful_capsules"] += 1
                
                logger.info(f"Successfully created capsule for {subject}")
                
            except Exception as e:
                self.stats["failed_capsules"] += 1
                error_info = {
                    "subject": subject,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                self.stats["failures"].append(error_info)
                logger.error(f"Failed to create capsule for {subject}: {e}")
                continue
    
    def export_capsule(self, capsule: KnowledgeSuppressionCapsule) -> Path:
        """Export individual capsule with evaluation results"""
        eval_results = {"status": "not_evaluated"}  # Simplified for now
        
        capsule_data = {
            "subject": capsule.subject,
            "target_layer": capsule.target_layer,
            "target_module_name": getattr(capsule, 'target_module_name', None),
            "signature_vector": capsule.signature_vector.tolist(),
            "effect_size": capsule.effect_size,
            "adapter_type": self.config.adapter_type,
            "config": {
                "scaling_factor_init": self.config.scaling_factor_init,
                "adapter_type": self.config.adapter_type,
                "lora_rank": self.config.lora_rank if self.config.adapter_type == "lora" else None,
                "lora_alpha": self.config.lora_alpha if self.config.adapter_type == "lora" else None,
            },
            "evaluation_results": eval_results,
            "creation_status": capsule.creation_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save adapter state dict
        if hasattr(capsule, 'adapter'):
            capsule_data["adapter_state_dict"] = {
                k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in capsule.adapter.state_dict().items()
            }
            
        capsule_filename = f"{capsule.subject.replace(' ', '_').replace('(', '').replace(')', '')}_capsule.pkl.gz"
        capsule_path = self.config.output_dir / capsule_filename
        
        compress_pickle.dump(capsule_data, capsule_path, compression="gzip")
        
        logger.info(f"Exported capsule for {capsule.subject} to {capsule_path}")
        return capsule_path

# ------------------------------------ Runner ------------------------------------
def run_module_d():
    """Main function to run Module D"""
    logger.info("=" * 60)
    logger.info("Starting KIF Module D: Knowledge Suppression Capsule Forger (FIXED, Hardened Loader)")
    logger.info("=" * 60)
    
    # Create configuration (auto device selection)
    config = CapsuleConfig()
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create forger
    forger = CapsuleForger(config)
    
    try:
        # Load model and signatures
        signatures = forger.load_model_and_signatures()
        
        # Create & activate capsules
        forger.create_capsules(signatures)
        
        # Export successful capsules
        for subject, capsule in forger.capsules.items():
            if capsule.creation_status == "success":
                forger.export_capsule(capsule)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Module D Summary:")
        logger.info(f"Total subjects: {forger.stats['total_subjects']}")
        logger.info(f"Successful capsules: {forger.stats['successful_capsules']}")
        logger.info(f"Failed capsules: {forger.stats['failed_capsules']}")
        
        if forger.stats["failures"]:
            logger.info("Failures:")
            for failure in forger.stats["failures"]:
                logger.info(f"  - {failure['subject']}: {failure['error_type']} - {failure['error']}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Module D failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_module_d()
