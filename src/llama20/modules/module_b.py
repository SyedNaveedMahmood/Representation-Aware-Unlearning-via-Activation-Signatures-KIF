# Module B: Activation Probing (Hooks-in-Parallel + Batching)
# -----------------------------------------------------------
# One forward pass per batch captures ALL target layers via hooks, then saves
# per-prompt, per-layer activations to disk. This removes the O(num_layers)
# forward-pass bottleneck and fully utilizes the GPU with batching.

import os
import re
import gc
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Optional advanced libs with safe fallbacks
try:
    import compress_pickle  # for fast compressed dumps
    ADVANCED_ANALYSIS_AVAILABLE = True
except Exception:
    ADVANCED_ANALYSIS_AVAILABLE = False
    import pickle
    import gzip

    def _cp_dump(data, filename, compression="gzip", compresslevel=3, **kwargs):
        if compression != "gzip":
            raise ValueError("Only gzip compression supported in fallback.")
        with gzip.open(filename, "wb", compresslevel=compresslevel) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _cp_load(filename):
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)

    compress_pickle = type("compress_pickle_fallback", (), {"dump": staticmethod(_cp_dump), "load": staticmethod(_cp_load)})

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("kif_probe.log")]
)
logger = logging.getLogger("KIF-ModuleB")

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
@dataclass
class ProbeConfig:
    """Configuration for activation probing via parallel hooks."""

    # IO
    model_dir: str = "outputs/model"
    prompts_file: str = "outputs/datasets/prompts.jsonl"
    output_dir: Path = Path("outputs/activations_subject_span_mlpblock")

    # What to capture
    layers: List[int] = field(default_factory=lambda: list(range(32)))  # adapt to your model depth

    # v1 invariant
    target_kind: str = "mlp_block"
    activation_source: str = "mlp_block"
    capture_scope: str = "subject_span_mean"

    # Keep only for backward compatibility / logging.
    # Do not use this for discovery in v1.
    targets: List[str] = field(default_factory=lambda: ["mlp"])       # capture MLPs by default

    # Subject-span behavior
    skip_missing_subject: bool = True

    # Performance
    batch_size: int = 32
    max_length: int = 128
    use_half_precision: bool = True              # fp16 for model weights (if supported)
    save_dtype_fp16: bool = True                 # store activations as float16 on disk

    # Device & memory
    device_map: str = "auto"                    # let HF shard model
    cleanup_every_batches: int = 10              # empty CUDA cache periodically

    # Storage
    compression_level: int = 3                   # gzip compress level for dumps

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        (self.output_dir / "mlp_block").mkdir(parents=True, exist_ok=True)

        if self.target_kind != "mlp_block":
            raise ValueError("KIF-SubjectSpan-MLPBlock-v1 requires target_kind='mlp_block'.")

        if self.activation_source != "mlp_block":
            raise ValueError("KIF-SubjectSpan-MLPBlock-v1 requires activation_source='mlp_block'.")

        allowed_scopes = {"subject_span_mean"}
        if self.capture_scope not in allowed_scopes:
            raise ValueError(
                "KIF-SubjectSpan-MLPBlock-v1 only supports "
                "capture_scope='subject_span_mean'."
            )
# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def get_primary_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def to_numpy_for_saving(t: torch.Tensor, fp16: bool = True) -> np.ndarray:
    if t.is_cuda:
        t = t.detach().to("cpu")
    if fp16 and t.dtype in (torch.float32, torch.float64):
        t = t.half()
    # ensure contiguous for pickle speed
    t = t.contiguous()
    return t.numpy().astype(np.float16 if fp16 else np.float32)


# -----------------------------------------------------------
# Parallel Hook Collector
# -----------------------------------------------------------
class ParallelActivationCollector:
    """Attach hooks to all target modules once and capture outputs in a single pass."""

    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, config: ProbeConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = get_primary_device(model)

        # module name -> layer_idx
        self.module_to_layer: Dict[str, int] = {}
        # ordered list of target module names (deterministic iteration)
        self.target_module_names: List[str] = []

        # populated on each forward pass: module_name -> tensor(batch, 1, hidden)
        self.captured_activations: Dict[str, torch.Tensor] = {}

        # Set immediately before forward; used by hooks to pool without storing [B, T, H].
        self.current_subject_token_indices: List[List[int]] = []
        self.current_valid_prompt_mask: List[bool] = []
        self.hooks: List[Any] = []

        self._discover_targets()
        self._register_all_hooks()

    # ---- discovery ----
    def _discover_targets(self) -> None:
        names = []

        for name, module in self.model.named_modules():
            # Exact outer MLP block only:
            # matches "model.layers.12.mlp"
            # does NOT match "model.layers.12.mlp.gate_proj"
            # does NOT match "model.layers.12.mlp.up_proj"
            # does NOT match "model.layers.12.mlp.down_proj"
            m = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", name)
            if not m:
                continue

            layer_idx = int(m.group(1))
            if layer_idx in self.config.layers:
                names.append((layer_idx, name))

        names.sort(key=lambda x: (x[0], x[1]))

        self.target_module_names = [n for _, n in names]
        self.module_to_layer = {n: layer_idx for layer_idx, n in names}

        if not self.target_module_names:
            raise RuntimeError(
                "No exact outer MLP block modules found. "
                "Expected names like 'model.layers.12.mlp'."
            )

        # Optional sanity warning: one exact MLP block per requested layer.
        found_layers = set(self.module_to_layer.values())
        requested_layers = set(self.config.layers)
        missing_layers = sorted(requested_layers - found_layers)
        if missing_layers:
            logger.warning(f"No exact MLP block found for requested layers: {missing_layers}")

        logger.info(
            f"Target modules: {len(self.target_module_names)} exact MLP blocks "
            f"across {len(found_layers)} layers"
        )
        logger.info(f"First few targets: {self.target_module_names[:5]}")

    # ---- hooks ----
    def _make_hook(self, module_name: str):
        def hook_fn(module, inputs, output):
            # For LlamaMLP and similar, output is a tensor.
            out = output[0] if isinstance(output, tuple) else output

            if self.config.capture_scope != "subject_span_mean":
                raise RuntimeError(
                    "This Module B v1 collector only supports subject_span_mean."
                )

            if not self.current_subject_token_indices:
                raise RuntimeError(
                    "Subject-token indices are not set before forward pass."
                )

            # out shape: [B, T, H]
            # pooled shape: [B, 1, H]
            pooled_rows = []
            hidden_dim = out.shape[-1]

            for b, token_ids in enumerate(self.current_subject_token_indices):
                if not token_ids:
                    # Invalid rows are not saved later.
                    pooled_rows.append(out.new_zeros((1, hidden_dim)))
                    continue

                idx = torch.tensor(token_ids, device=out.device, dtype=torch.long)
                pooled = out[b].index_select(0, idx).mean(dim=0, keepdim=True)
                pooled_rows.append(pooled)

            self.captured_activations[module_name] = torch.stack(
                pooled_rows,
                dim=0,
            ).detach()

        return hook_fn

    def _register_all_hooks(self) -> None:
        named = dict(self.model.named_modules())
        for module_name in self.target_module_names:
            mod = named.get(module_name, None)
            if mod is None:
                logger.warning(f"Module not found during hook registration: {module_name}")
                continue
            self.hooks.append(mod.register_forward_hook(self._make_hook(module_name)))
        logger.info(f"Registered {len(self.hooks)} forward hooks")

    def remove_hooks(self) -> None:
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
        logger.info("Removed all hooks")

    # ---- capture ----
    def _tokenize_batch(self, texts: List[str]) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[int, int]]]]:
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                "Subject-span capture requires a fast tokenizer with offset mappings. "
                "Load tokenizer with use_fast=True."
            )

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
        )

        # Keep offsets on CPU for span logic.
        offsets = enc.pop("offset_mapping").cpu().tolist()

        # Do not pass offset_mapping to the model.
        model_inputs = {k: v.to(self.device) for k, v in enc.items()}

        return model_inputs, offsets


    @torch.inference_mode()
    def collect_batch(
        self,
        prompts: List[str],
        prompt_ids: List[str],
        subjects: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run one forward pass and save per-prompt, per-layer subject-span MLP-block activations."""
        assert len(prompts) == len(prompt_ids) == len(subjects)

        self.captured_activations.clear()

        inputs, offsets = self._tokenize_batch(prompts)

        # Precompute subject token spans.
        subject_token_indices: List[List[int]] = []
        valid_prompt_mask: List[bool] = []

        for prompt, pid, subject, off in zip(prompts, prompt_ids, subjects, offsets):
            token_ids = self._subject_token_indices(prompt, subject, off)
            subject_token_indices.append(token_ids)

            is_valid = len(token_ids) > 0
            valid_prompt_mask.append(is_valid)

            if not is_valid:
                msg = (
                    f"Subject span not found for prompt_id={pid}; "
                    f"subject={subject!r}"
                )

                if self.config.skip_missing_subject:
                    logger.warning(f"Skipping. {msg}")
                else:
                    raise ValueError(msg)
        
        self.current_subject_token_indices = subject_token_indices
        self.current_valid_prompt_mask = valid_prompt_mask

        if not any(valid_prompt_mask):
            logger.warning("Skipping entire batch: no valid subject spans found.")
            self.current_subject_token_indices = []
            self.current_valid_prompt_mask = []
            return defaultdict(list)

        try:
            try:
                _ = self.model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )
            except TypeError:
                _ = self.model(**inputs)
        finally:
            self.current_subject_token_indices = []
            self.current_valid_prompt_mask = []

        saved: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for module_name, tensor in self.captured_activations.items():
            layer_idx = self.module_to_layer.get(module_name)
            if layer_idx is None:
                continue

            B = tensor.shape[0]

            for b in range(B):
                if not valid_prompt_mask[b]:
                    continue

                pid = prompt_ids[b]
                subject = subjects[b]
                token_ids = subject_token_indices[b]

                # Hook already reduced this to [1, H].
                selected = tensor[b]

                data_np = to_numpy_for_saving(
                    selected,
                    fp16=self.config.save_dtype_fp16,
                )

                # Shape should be [1, hidden_size] for v1.
                if data_np.ndim != 2 or data_np.shape[0] != 1:
                    raise RuntimeError(
                        f"Expected [1, H] subject-span activation, got shape={data_np.shape} "
                        f"for prompt_id={pid}, module={module_name}"
                    )

                filename = f"{pid}_layer{layer_idx}_mlpblock_subject_span_mean.pkl.gz"
                path = self.config.output_dir / "mlp_block" / filename

                compress_pickle.dump(
                    data_np,
                    path,
                    compression="gzip",
                    compresslevel=self.config.compression_level,
                )

                record = {
                    "path": str(path),
                    "layer": layer_idx,
                    "target_module_name": module_name,
                    "activation_source": self.config.activation_source,
                    "target_kind": self.config.target_kind,
                    "token_scope": self.config.capture_scope,
                    "subject": subject,
                    "subject_token_indices": token_ids,
                    "selected_shape": list(data_np.shape),
                    "feature_dim": int(data_np.shape[-1]),
                }

                saved[pid].append(record)

                del data_np
                del selected

            del tensor

        self.captured_activations.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return saved

    def _find_subject_char_span(self, prompt: str, subject: str) -> Tuple[int, int] | None:
        if not subject:
            return None

        start = prompt.find(subject)

        # Case-insensitive fallback, still exact string.
        if start < 0:
            start = prompt.lower().find(subject.lower())

        if start < 0:
            return None

        return start, start + len(subject)


    def _subject_token_indices(
        self,
        prompt: str,
        subject: str,
        offsets: List[Tuple[int, int]],
    ) -> List[int]:
        char_span = self._find_subject_char_span(prompt, subject)
        if char_span is None:
            return []

        char_start, char_end = char_span
        token_ids: List[int] = []

        for tok_idx, pair in enumerate(offsets):
            tok_start, tok_end = int(pair[0]), int(pair[1])

            # Skip padding/special tokens.
            if tok_end <= tok_start:
                continue

            # Token overlaps subject char span.
            if tok_end > char_start and tok_start < char_end:
                token_ids.append(tok_idx)

        return token_ids

# -----------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------
class OptimizedProbeRobot:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self._load_prompts()
        self._load_model()
        self.collector = ParallelActivationCollector(self.model, self.tokenizer, self.config)

    # ---- IO ----
    def _load_prompts(self) -> None:
        with open(self.config.prompts_file, "r", encoding="utf-8") as f:
            self.prompts: List[Dict[str, Any]] = [json.loads(line) for line in f]
        if not self.prompts:
            raise RuntimeError("No prompts found in prompts_file.")
        uniq = len({p.get("triple_id") for p in self.prompts})
        logger.info(f"Loaded {len(self.prompts)} prompts across {uniq} triples")

    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            # fallback padding
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Loading model from: {self.config.model_dir}")
        # free caches before big load
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_dir,
            device_map=self.config.device_map,
            torch_dtype=torch.float16 if self.config.use_half_precision else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        if torch.cuda.is_available():
            dev = get_primary_device(self.model)
            logger.info(f"Model loaded on device: {dev}; CUDA device count: {torch.cuda.device_count()}")
        else:
            logger.warning("CUDA not available; running on CPU.")

    # ---- collection ----
    def collect_all(self) -> Dict[str, List[Dict[str, Any]]]:
        saved_all: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        B = self.config.batch_size
        total = len(self.prompts)

        for start in tqdm(range(0, total, B), desc="Collecting activations"):
            batch = self.prompts[start:start + B]

            texts = [p["prompt"] for p in batch]
            ids = [p["id"] for p in batch]
            subjects = [p["subject"] for p in batch]

            try:
                saved = self.collector.collect_batch(texts, ids, subjects)
                for pid, records in saved.items():
                    saved_all[pid].extend(records)
            except Exception as e:
                logger.error(f"Batch {start}-{start+len(batch)} failed: {e}", exc_info=True)
                continue

            if torch.cuda.is_available() and ((start // max(B, 1) + 1) % self.config.cleanup_every_batches == 0):
                torch.cuda.empty_cache()

        return saved_all

    # ---- reports ----
    def _write_index(self, saved_records: Dict[str, List[Dict[str, Any]]]) -> None:
        prompt_by_id = {p["id"]: p for p in self.prompts}

        index = {
            "schema_version": "kif_subject_span_mlpblock_v1",
            "config": {
                "layers": self.config.layers,
                "targets": self.config.targets,
                "target_kind": self.config.target_kind,
                "activation_source": self.config.activation_source,
                "model": self.config.model_dir,
                "batch_size": self.config.batch_size,
                "capture_scope": self.config.capture_scope,
                "fp16_model": self.config.use_half_precision,
                "fp16_saves": self.config.save_dtype_fp16,
            },
            "prompts": {
                pid: {
                    "paths": [r["path"] for r in records],
                    "records": records,
                    "triple_id": prompt_by_id[pid].get("triple_id"),
                    "subject": prompt_by_id[pid]["subject"],
                    "activation_count": len(records),
                }
                for pid, records in saved_records.items()
            },
            "summary": {
                "total_prompts": len(saved_records),
                "total_files": sum(len(v) for v in saved_records.values()),
                "activation_source": self.config.activation_source,
                "token_scope": self.config.capture_scope,
                "target_kind": self.config.target_kind,
            },
        }

        path = self.config.output_dir / "activation_index.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Wrote activation index to {path}")

    def _write_report(self, saved_records: Dict[str, List[Dict[str, Any]]]) -> None:
        total_size = 0
        layer_counts: Dict[int, int] = defaultdict(int)

        for records in saved_records.values():
            for record in records:
                p = Path(record["path"])
                try:
                    total_size += os.path.getsize(p)
                    layer_counts[int(record["layer"])] += 1
                except OSError:
                    pass

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.config.model_dir,
            "activation_source": self.config.activation_source,
            "token_scope": self.config.capture_scope,
            "target_kind": self.config.target_kind,
            "results": {
                "prompts_processed": len(saved_records),
                "total_files": sum(len(v) for v in saved_records.values()),
                "storage_gb": total_size / (1024 ** 3),
                "files_per_layer": dict(layer_counts),
            },
        }

        path = self.config.output_dir / "collection_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Wrote collection report to {path}")

    def run(self) -> None:
        logger.info("=" * 64)
        logger.info("Starting Module B: Hooks-in-Parallel + Batching")
        logger.info("=" * 64)

        saved = self.collect_all()
        self._write_index(saved)
        self._write_report(saved)

        # tidy up hooks before exit
        self.collector.remove_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Module B completed successfully ✔")


# -----------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------

def run_module_b():
    # You can tweak defaults here for your box
    cfg = ProbeConfig(
        layers=list(range(32)),      # adapt to model depth (e.g., Llama-2-7B has 32)
        targets=["mlp"],
        batch_size=32,               # try 16/32/64 based on VRAM
        max_length=128,
        use_half_precision=True,     # fp16 weights if supported
        save_dtype_fp16=True,        # store activations as fp16
        device_map="auto",
        cleanup_every_batches=10,
        compression_level=3,
        capture_scope="subject_span_mean",       # or "last_token" to shrink storage massively
    )

    robot = OptimizedProbeRobot(cfg)
    robot.run()


if __name__ == "__main__":
    run_module_b()
