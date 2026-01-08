# Representation-Aware Unlearning via Activation Signatures (KIF Pipeline)

This repository contains a modular, end-to-end pipeline for selective unlearning in large language models using activation signatures, suppression capsules, and a self-healing LoRA loop.

## Requirements
- Python 3.10+
- CUDA-capable GPU recommended for Modules B/C/D/E/7/8
- Internet access required for Module A (Wikipedia/Wikidata)

## Install
```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

## Run the full pipeline
Run modules in order. Each module writes its outputs under `outputs/`.

```bash
llama20 module0   # quantize/load base model into outputs/model
llama20 module_a  # build dataset into outputs/datasets
llama20 module_b  # collect activations into outputs/activations
llama20 module_c  # mine signatures into outputs/signatures
llama20 module_d  # forge capsules into outputs/capsules
llama20 module_e  # run Hyper-Sentinel runtime (optional)
llama20 module7   # train forgetting adapter into outputs/global_adapters
ADAPTER_PATH=outputs/global_adapters/<adapter> llama20 module8  # evaluate
```

Run a sequence with the CLI dispatcher:
```bash
llama20 pipeline --modules module0 module_a module_b module_c module_d module7
```

## Self-healing loop (automatic)
Run training/evaluation cycles until forgetting holds for 10 consecutive rounds, capped at 3 days:
```bash
FORGET_SMR_THRESHOLD=0.05 \
FORGET_EL10_THRESHOLD=1.0 \
FORGET_CONFIRM_ROUNDS=10 \
FORGET_MAX_DAYS=3 \
llama20 loop
```

## Configuration (env vars)
These override defaults without editing code:
- `MODEL_DIR` - base model path (default: `outputs/model`)
- `ADAPTER_PATH` - adapter path for Module 8
- `CAPSULES_DIR` - capsules path (default: `outputs/capsules`)
- `PROMPTS_JSONL` - prompts file (default: `outputs/datasets/prompts.jsonl`)
- `OUT_DIR` - output directory for Module 8 (default: `outputs/eval_clean`)

Loop controls:
- `FORGET_SMR_THRESHOLD` (default 0.05)
- `FORGET_EL10_THRESHOLD` (default 1.0)
- `FORGET_CONFIRM_ROUNDS` (default 10)
- `FORGET_MAX_DAYS` (default 3)

## Outputs
Key artifacts by module:
- `outputs/model` - quantized base model
- `outputs/datasets` - triples + prompts
- `outputs/activations` - captured activations
- `outputs/signatures` - mined signature vectors
- `outputs/capsules` - suppression capsules
- `outputs/global_adapters` - LoRA adapters
- `outputs/eval_clean` - evaluation summaries

## Notes
- If you hit GPU OOM, lower batch sizes in the module configs.
- Module A is network-bound; retries and delays are built in.
- Module E is optional if you only need offline unlearning.
