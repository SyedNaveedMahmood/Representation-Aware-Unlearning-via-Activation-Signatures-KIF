import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scipy.stats import ks_2samp

from llama20 import module7, module8


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mention_flags(records: List[Dict[str, Any]]) -> List[float]:
    """Flatten subject-generation pairs into binary mention flags."""
    flags: List[float] = []
    for rec in records:
        subj = (rec.get("subject") or "").lower()
        for gen in rec.get("generations", []):
            flags.append(1.0 if subj and subj in gen.lower() else 0.0)
    return flags


def _compute_metric(eval_dir: Path, metric: str) -> Tuple[float | None, Dict[str, Any]]:
    summary_path = eval_dir / "eval_summary.json"
    pre_path = eval_dir / "pre_gens.json"
    post_path = eval_dir / "post_gens.json"
    meta: Dict[str, Any] = {}

    if metric in {"mention_rate", "mention_rate_post"}:
        if not summary_path.exists():
            return None, {"error": f"missing {summary_path}"}
        summary = _load_json(summary_path)
        val = summary.get("robustness_post_lora", {}).get("avg_subject_mention_rate")
        return float(val) if val is not None else None, {"source": str(summary_path)}

    if metric in {"simnpo_pvalue", "simnpo_ks"}:
        if not (pre_path.exists() and post_path.exists()):
            return None, {"error": f"missing pre/post gens in {eval_dir}"}
        pre_flags = _mention_flags(_load_json(pre_path))
        post_flags = _mention_flags(_load_json(post_path))
        if not pre_flags or not post_flags:
            return None, {"error": "empty mention flags"}
        res = ks_2samp(pre_flags, post_flags)
        meta.update({"ks_stat": float(res.statistic), "ks_pvalue": float(res.pvalue)})
        return (float(res.pvalue) if metric == "simnpo_pvalue" else float(res.statistic)), meta

    return None, {"error": f"unknown metric {metric}"}


def _meets_threshold(metric: str, value: float, threshold: float) -> bool:
    # For p-values / mention rates we want to be <= threshold.
    if metric in {"simnpo_pvalue", "mention_rate", "mention_rate_post"}:
        return value <= threshold
    # For KS statistic we want to be >= threshold (bigger separation).
    if metric == "simnpo_ks":
        return value >= threshold
    return False


def run_forget_loop(
    max_rounds: int | None = None,
    threshold: float | None = None,
    metric: str | None = None,
    eval_root: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Automated loop: train (Module 7) -> evaluate (Module 8) until a forgetting metric passes the threshold.

    metric options:
      - simnpo_pvalue (default): KS-test p-value on subject mentions (lower is better forgetting, threshold is an upper bound)
      - simnpo_ks: KS statistic on subject mentions (higher is better forgetting, threshold is a lower bound)
      - mention_rate: avg subject mention rate from Module 8 summary (lower is better)
    """
    max_rounds = int(max_rounds or os.getenv("FORGET_MAX_ROUNDS", 3))
    threshold = float(threshold or os.getenv("FORGET_THRESHOLD", 0.10))
    metric = (metric or os.getenv("FORGET_METRIC", "simnpo_pvalue")).lower()
    eval_root = eval_root or os.getenv("FORGET_EVAL_ROOT", "outputs/eval_clean")

    history: List[Dict[str, Any]] = []
    for round_idx in range(1, max_rounds + 1):
        print(f"=== Round {round_idx}/{max_rounds} ===")

        # Train adapter (Module 7)
        adapter_path = module7.run_module7_qwen()
        if not adapter_path:
            history.append({"round": round_idx, "status": "failed", "reason": "module7 returned no adapter"})
            break

        # Evaluate (Module 8) into a per-round folder
        eval_dir = Path(eval_root) / f"round_{round_idx}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        module8.run_module8_clean(adapter_path=adapter_path, out_dir=str(eval_dir))

        # Metric extraction
        value, meta = _compute_metric(eval_dir, metric)
        entry = {
            "round": round_idx,
            "adapter_path": adapter_path,
            "eval_dir": str(eval_dir),
            "metric": metric,
            "value": value,
            "meta": meta,
        }
        history.append(entry)
        print(f"[Round {round_idx}] metric={metric} value={value} meta={meta}")

        if value is not None and _meets_threshold(metric, value, threshold):
            entry["stopped"] = True
            print(f"Stopping: metric hit threshold ({value} vs {threshold}).")
            break

    return history


if __name__ == "__main__":
    run_forget_loop()
