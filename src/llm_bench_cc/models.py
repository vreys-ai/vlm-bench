from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

# Default for bnb llm_int8_skip_modules across all VLM variants — keep the vision
# tower and projector in their loading dtype so image tokens aren't degraded by
# quantization. Override per-variant via cfg.quant.skip_modules.
_DEFAULT_VLM_SKIP_MODULES = ["vision_tower", "multi_modal_projector"]


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    name: str
    hf_id: str
    dtype: torch.dtype
    device: torch.device
    quant_backend: str | None = None
    quant_mode: str | None = None


def _build_quant_config(quant_cfg):
    """Build a transformers quantization_config from our cfg.quant block.

    Lazy-imports bitsandbytes-related types so the base path doesn't require
    the `quant` extra to be installed.
    """
    backend = quant_cfg.get("backend")
    if backend != "bnb":
        raise ValueError(
            f"Unsupported quant backend {backend!r}; Tier A only supports 'bnb'."
        )

    from transformers import BitsAndBytesConfig

    skip_modules = quant_cfg.get("skip_modules") or _DEFAULT_VLM_SKIP_MODULES
    skip_modules = list(skip_modules)
    mode = quant_cfg.get("mode")

    if mode == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(quant_cfg.get("llm_int8_threshold", 6.0)),
            llm_int8_skip_modules=skip_modules,
        )
    if mode == "nf4":
        compute_dtype = getattr(
            torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(
                quant_cfg.get("bnb_4bit_use_double_quant", True)
            ),
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=skip_modules,
        )
    raise ValueError(f"Unsupported bnb mode {mode!r}; expected 'int8' or 'nf4'.")


def load_model(cfg) -> LoadedModel:
    dtype = getattr(torch, cfg.dtype)
    cache_dir = cfg.get("cache_dir", None)
    local_files_only = cfg.get("local_files_only", False)
    processor = AutoProcessor.from_pretrained(
        cfg.hf_id,
        trust_remote_code=cfg.get("trust_remote_code", False),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    quant_cfg = cfg.get("quant")
    quant_kwargs: dict[str, Any] = {}
    quant_backend: str | None = None
    quant_mode: str | None = None
    if quant_cfg is not None:
        quant_kwargs["quantization_config"] = _build_quant_config(quant_cfg)
        quant_backend = quant_cfg.get("backend")
        quant_mode = quant_cfg.get("mode")
        logger.info("Quantization enabled: backend=%s mode=%s", quant_backend, quant_mode)

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.hf_id,
        dtype=dtype,
        device_map=cfg.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=cfg.get("trust_remote_code", False),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        **quant_kwargs,
    )
    model.eval()
    device = next(model.parameters()).device
    return LoadedModel(
        model=model,
        processor=processor,
        name=cfg.name,
        hf_id=cfg.hf_id,
        dtype=dtype,
        device=device,
        quant_backend=quant_backend,
        quant_mode=quant_mode,
    )
