"""Backend factory. The runner imports `load_backend(cfg)` and never touches
HF / vLLM directly — see backends.base.Backend for the contract.

Lazy imports per backend so the harness doesn't require vllm installed for
HF runs (and vice versa)."""

from __future__ import annotations

import logging

from .base import Backend

logger = logging.getLogger(__name__)


def load_backend(cfg) -> Backend:
    """Dispatch on `cfg.runtime.backend` (defaults to 'hf' for back-compat
    with any pre-refactor config that doesn't set the field)."""
    runtime_cfg = cfg.get("runtime") or {}
    backend_name = runtime_cfg.get("backend", "hf")

    logger.info("Loading backend=%s", backend_name)

    if backend_name == "hf":
        from .hf import HFBackend
        return HFBackend(cfg.model)
    if backend_name == "vllm":
        from .vllm import VLLMBackend
        return VLLMBackend(cfg.model, runtime_cfg.get("vllm") or {})
    raise ValueError(
        f"Unknown runtime.backend={backend_name!r}; expected 'hf' or 'vllm'."
    )


__all__ = ["Backend", "load_backend"]
