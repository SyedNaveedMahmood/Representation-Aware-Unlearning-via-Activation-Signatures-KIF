"""
Llama 2.0 pipeline modules extracted from the original notebook.

Each module exposes a `run_module*` entrypoint mirroring the notebook cells.
See `llama20.cli` for a simple dispatcher.
"""

import importlib
from typing import Any

_MODULE_MAP = {
    "module0": "llama20.modules.module0",
    "module_a": "llama20.modules.module_a",
    "module_b": "llama20.modules.module_b",
    "module_c": "llama20.modules.module_c",
    "module_d": "llama20.modules.module_d",
    "module_e": "llama20.modules.module_e",
    "module7": "llama20.modules.module7",
    "module8": "llama20.modules.module8",
    "loop": "llama20.loop",
    "modules": "llama20.modules",
}

__all__ = [
    "modules",
    "module0",
    "module_a",
    "module_b",
    "module_c",
    "module_d",
    "module_e",
    "module7",
    "module8",
    "loop",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module wrappers to avoid heavy imports at package import time."""
    target = _MODULE_MAP.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return importlib.import_module(target)
