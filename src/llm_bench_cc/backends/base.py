"""Backend protocol — the seam between the runner and a model runtime.

Each backend hides one inference stack (HF transformers, vLLM, ...) behind a
single `generate(image, prompt, gen_kwargs) -> (text, latency_ms)` call. The
runner stays runtime-agnostic; per-runtime quirks (vLLM's worker subprocess
model, HF's processor + chat template, etc.) live inside the backend.

Peak-VRAM is a backend concern because the measurement strategy differs:
HF lives in-process so `torch.cuda.max_memory_allocated()` is honest; vLLM v1
runs the model in a worker subprocess, so the parent's CUDA counters see
nothing and we poll `nvidia-smi` instead.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Backend(Protocol):
    """Minimal contract for a model runtime.

    Attributes are surfaced for `summary.json` provenance — the runner reads
    them straight into the output without knowing which runtime produced them.
    """

    name: str
    hf_id: str
    dtype: torch.dtype
    device: str
    quant_backend: str | None
    quant_mode: str | None

    def generate(
        self, image: Any, prompt: str, gen_kwargs: dict[str, Any]
    ) -> tuple[str, float]:
        """Run one multimodal generation. `image` is a PIL.Image (already
        resized by the runner). Returns (decoded_text, latency_ms)."""
        ...

    def reset_peak_vram(self) -> None:
        """Zero the per-task peak VRAM watermark."""
        ...

    def peak_vram_gb(self) -> float:
        """Peak VRAM observed since the most recent `reset_peak_vram()`, in GB."""
        ...
