from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    name: str
    hf_id: str
    dtype: torch.dtype
    device: torch.device


def load_model(cfg) -> LoadedModel:
    dtype = getattr(torch, cfg.dtype)
    processor = AutoProcessor.from_pretrained(
        cfg.hf_id,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.hf_id,
        torch_dtype=dtype,
        device_map=cfg.device_map,
        low_cpu_mem_usage=True,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        trust_remote_code=cfg.get("trust_remote_code", False),
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
    )
