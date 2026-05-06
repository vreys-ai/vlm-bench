from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

# Subtree under which Linear modules ARE quantized. Everything else (vision_tower,
# audio_tower, embed_vision, embed_audio, lm_head, etc.) gets enumerated and
# passed to bnb as an explicit skip list — substring-pattern skips like
# "vision_tower" don't reliably stop bnb's recursion into deep submodules
# (their matching semantics differ across transformers versions), so we list
# every Linear path explicitly via a meta-load walk. Override the LLM subtree
# name per-variant via cfg.quant.llm_subtree.
_DEFAULT_LLM_SUBTREE = "language_model"


def _enumerate_non_llm_linear_paths(
    hf_id: str,
    *,
    llm_subtree: str = _DEFAULT_LLM_SUBTREE,
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> list[str]:
    """Meta-load the model (zero weights, zero VRAM) and return fully-qualified
    paths of every nn.Linear that is NOT under the LLM subtree. Use as
    bnb's llm_int8_skip_modules (Tier A) or llmcompressor's `ignore`
    (Tier B) so the matcher hits each Linear exactly."""
    from accelerate import init_empty_weights

    auto_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "cache_dir": cache_dir,
    }
    if local_files_only:
        auto_kwargs["local_files_only"] = True

    config = AutoConfig.from_pretrained(hf_id, **auto_kwargs)
    with init_empty_weights():
        meta_model = AutoModelForImageTextToText.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )

    # A Linear is "in the LLM subtree" iff `llm_subtree` appears as a full path
    # component anywhere in its name. Pad with dots so we don't false-match
    # something like "audio_language_model_proj".
    keep_marker = f".{llm_subtree}."
    skip: list[str] = []
    for name, module in meta_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            padded = f".{name}."
            if keep_marker not in padded:
                skip.append(name)

    del meta_model
    return skip


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


def _build_quant_config(quant_cfg, skip_modules: list[str]):
    """Build a transformers quantization_config from our cfg.quant block.

    Lazy-imports bitsandbytes-related types so the base path doesn't require
    the `quant` extra to be installed.

    Returns None for backends whose checkpoint already ships its own
    quantization config in `config.json` (e.g. FP8_BLOCK / W4A16 saved by
    llmcompressor as compressed-tensors). Those load via plain
    from_pretrained — passing a second config would conflict.
    """
    backend = quant_cfg.get("backend")
    if backend == "compressed_tensors":
        return None
    if backend != "bnb":
        raise ValueError(
            f"Unsupported quant backend {backend!r}; "
            f"expected 'bnb' or 'compressed_tensors'."
        )

    from transformers import BitsAndBytesConfig

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
    # The processor (tokenizer + image processor + audio processor + chat
    # template) can come from a different source than the weights. This
    # matters for compressed-tensors checkpoints saved by llmcompressor's
    # `model_free_ptq`: the safetensors + config.json land in `hf_id`, but
    # multimodal processor files (preprocessor_config.json, etc.) are not
    # always propagated from the source snapshot, and a missing file in the
    # local dir falls through to a Hub call with the local path as repo_id
    # → HFValidationError. Default `processor_id` to `hf_id` (so the bf16
    # base path is unchanged); override per-variant to the upstream Hub id.
    processor_id = cfg.get("processor_id") or cfg.hf_id
    processor = AutoProcessor.from_pretrained(
        processor_id,
        trust_remote_code=cfg.get("trust_remote_code", False),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    quant_cfg = cfg.get("quant")
    quant_kwargs: dict[str, Any] = {}
    quant_backend: str | None = None
    quant_mode: str | None = None
    if quant_cfg is not None:
        quant_backend = quant_cfg.get("backend")
        quant_mode = quant_cfg.get("mode")

        if quant_backend == "bnb":
            # bnb needs an explicit per-Linear skip list. If the variant doesn't
            # override, enumerate every Linear NOT under the LLM subtree via a
            # meta-load (zero VRAM).
            if quant_cfg.get("skip_modules"):
                skip_modules = list(quant_cfg.skip_modules)
                logger.info("Using %d explicit skip_modules from variant config", len(skip_modules))
            else:
                llm_subtree = quant_cfg.get("llm_subtree") or _DEFAULT_LLM_SUBTREE
                skip_modules = _enumerate_non_llm_linear_paths(
                    cfg.hf_id,
                    llm_subtree=llm_subtree,
                    trust_remote_code=cfg.get("trust_remote_code", False),
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
                logger.info(
                    "Auto-enumerated %d non-LLM Linear paths to skip (LLM subtree=%r)",
                    len(skip_modules), llm_subtree,
                )
            qc = _build_quant_config(quant_cfg, skip_modules)
            if qc is not None:
                quant_kwargs["quantization_config"] = qc
        elif quant_backend == "compressed_tensors":
            # The pre-quantized checkpoint at cfg.hf_id ships its own
            # compressed-tensors config; transformers + compressed-tensors
            # auto-instantiate it on from_pretrained. Nothing to inject here.
            pass
        else:
            raise ValueError(
                f"Unsupported quant backend {quant_backend!r}; "
                f"expected 'bnb' or 'compressed_tensors'."
            )

        logger.info("Quantization enabled: backend=%s mode=%s", quant_backend, quant_mode)
    else:
        # Loud signal so a misconfigured variant (e.g. bad Hydra defaults
        # composition) doesn't silently fall through to full precision.
        logger.info("No quantization configured (cfg.quant is None); loading full precision.")

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
