# Llama 2.0 Pipeline (Notebook Split)

This repo extracts the original `llama_2.0.ipynb` into importable Python modules with a simple CLI dispatcher. The pipeline implements the research workflow described in `project_context.md`:

- Module 0: Enhanced Model Setup & Quantization
- Module A: Dataset Builder (Wikipedia/Wikidata prompts)
- Module B: Activation Probing
- Module C: Signature Mining (ROME-style)
- Module D: Knowledge Suppression Capsule Forger
- Module E: Hyper-Sentinel runtime/router
- Module 7: Qwen-Optimized Forgetting Trainer
- Module 8: Clean Evaluation (utility + forgetting + EL10 + Cohen’s d)

## Quickstart
1. Create env & install deps (CUDA required for most heavy steps):
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
   pip install --upgrade pip
   pip install -e .        # editable install; pulls dependencies from pyproject/requirements
   ```
2. Run a module (defaults match the notebook):
   ```bash
   llama20 module0          # quantize/load base model into outputs/model
   llama20 module_a         # build dataset into outputs/datasets
   llama20 module_b         # collect activations into outputs/activations
   llama20 module_c         # mine signatures into outputs/signatures
   llama20 module_d         # forge capsules into outputs/capsules
   llama20 module_e         # run Hyper-Sentinel runtime
   llama20 module7          # train forgetting adapter into outputs/global_adapters
   llama20 module8          # evaluate and write outputs/eval_clean
   ```
3. Run a sequence:
   ```bash
   llama20 pipeline --modules module0 module_a module_b module_c module_d
   ```
4. Auto-loop (train→eval until SMR+EL10 meet paper thresholds for 10 consecutive rounds, hard-capped at 3 days):
   ```bash
   FORGET_SMR_THRESHOLD=0.05 FORGET_EL10_THRESHOLD=1.0 FORGET_CONFIRM_ROUNDS=10 FORGET_MAX_DAYS=3 llama20 loop
   ```

## Repo Layout
- `src/llama20/modules/module*.py` — code from notebook cells, unchanged entrypoints (`run_module0`, `run_module_a`, …).
- `src/llama20/cli.py` — dispatcher (`python -m llama20.cli <module|pipeline>`).
- `project_context.md` — detailed pipeline description and tips.
- `requirements.txt` — runtime dependencies (mirrors `pyproject.toml`).
- `outputs/` — created at runtime (models, datasets, activations, signatures, capsules, adapters, evals).
- `llama20.loop` — orchestrator: runs Module 7 then Module 8 per round, stops after 10 consecutive rounds with SMR ≤ 0.05 and EL10 < 1 (or at 3 days).

## Notes & Tips
- Hardware: Modules B/C/D/E/7/8 expect GPU; Module 0 uses bitsandbytes 4-bit by default. Lower configs or switch to CPU in the configs inside each module if needed.
- Network: Module A hits Wikipedia/Wikidata; some steps may download models (tokenizers, SBERT, etc.).
- Data: Provide your own `subjects.txt` or let Module A create a default. Downstream modules read artifacts from `outputs/` produced by prior steps.
- Logging: Each module logs to its own file (e.g., `kif_setup.log`, `kif_dataset.log`, …).

## Best-Practice Checklist (adapted from paperswithcode/releasing-research-code)
- Reproduce: ship default configs and document environment (this README + `requirements.txt`).
- Clarity: one entrypoint per module, plus pipeline driver.
- Outputs: predictable `outputs/` tree; avoids touching the repo state.
- Licensing/Citation: add your chosen LICENSE and a `CITATION.cff` (not included yet).
- Testing: add smoke tests/CI when you define minimal CPU-friendly configs.

## Next Steps
- Add pinned CUDA/PyTorch versions that match your hardware stack.
- Add small toy data/configs for CPU smoke tests and wire into GitHub Actions.
- Publish a release tag or Zenodo DOI alongside the paper.
- Extend the loop driver with additional metrics or hyperparameter sweeps if you want automated tuning.
