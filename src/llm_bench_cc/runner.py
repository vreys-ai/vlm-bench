from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from .backends import Backend, load_backend
from .composite import composite_score, retention_ratios
from .tasks.base import Prediction, Sample
from .tasks.registry import get_task
from .tracking import CarbonTracker, WandbRun

logger = logging.getLogger(__name__)


def _cleanup_dataset_cache(ds_cfg) -> None:
    # HF datasets writes to <cache_dir>/<hf_id with "/" replaced by "___">/.
    # Removing that subtree frees disk between tasks (esp. VQAv2's val split).
    cache_dir = ds_cfg.get("cache_dir") or os.path.expanduser("~/.cache/huggingface/datasets")
    slug = ds_cfg.hf_id.replace("/", "___")
    path = os.path.join(cache_dir, slug)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        logger.info("Removed dataset cache %s", path)


def _maybe_resize(image: Image.Image, max_side: int | None) -> Image.Image:
    if not max_side:
        return image
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _generate_one(
    backend: Backend,
    image,
    prompt: str,
    gen_kwargs: dict[str, Any],
    image_max_side: int | None,
) -> tuple[str, float]:
    image = _maybe_resize(image, image_max_side)
    return backend.generate(image, prompt, gen_kwargs)


def _run_task(backend, task_name, ds_cfg, n, seed, gen_kwargs, out_dir, carbon_cfg, run_name, runtime_cfg):
    task = get_task(task_name)
    samples: list[Sample] = task.load(n=n, seed=seed, ds_cfg=ds_cfg)
    if not samples:
        logger.warning("Task %s loaded 0 samples; skipping", task_name)
        return None

    carbon = CarbonTracker(
        project_name=f"{run_name}.{task_name}",
        output_dir=str(out_dir),
        cfg=OmegaConf.create({"carbon": carbon_cfg}),
        enabled=carbon_cfg.enabled,
    )

    backend.reset_peak_vram()

    image_max_side = runtime_cfg.get("image_max_side", None)
    empty_cache = bool(runtime_cfg.get("empty_cache_between_samples", False))

    carbon.start()
    preds: list[Prediction] = []
    latencies: list[float] = []
    for s in tqdm(samples, desc=task_name, leave=False):
        text, ms = _generate_one(backend, s.image, s.prompt, gen_kwargs, image_max_side)
        preds.append(Prediction(sample_id=s.sample_id, prediction=text, latency_ms=ms))
        latencies.append(ms)
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
    emissions = carbon.stop()

    peak_vram_gb = backend.peak_vram_gb()

    scores = task.score(samples, preds)
    primary_value = scores[task.primary_metric]

    metrics = {
        **scores,
        "primary_metric": task.primary_metric,
        "primary": primary_value,
        "latency_ms_p50": float(median(latencies)),
        "latency_ms_mean": float(mean(latencies)),
        "peak_vram_gb": float(peak_vram_gb),
        "energy_kwh": emissions.energy_consumed_kwh,
        "co2_grams": emissions.co2eq_grams,
        "n_samples": len(samples),
    }

    pred_path = Path(out_dir) / f"preds_{task_name}.jsonl"
    with pred_path.open("w") as f:
        for p in preds:
            f.write(json.dumps({
                "id": p.sample_id,
                "prediction": p.prediction,
                "latency_ms": p.latency_ms,
            }) + "\n")

    return metrics, pred_path


def run_eval(cfg: DictConfig) -> dict[str, Any]:
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    full_config = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Loading model %s (dtype=%s)", cfg.model.hf_id, cfg.model.dtype)
    backend = load_backend(cfg)

    # wandb.init() AFTER backend load: wandb's default console=auto mode on
    # Linux uses os.dup2 to redirect stdout/stderr to pipes its reader thread
    # drains. The vLLM backend spawns an EngineCore subprocess that inherits
    # those redirected fds; if wandb's reader stalls, the child fills the pipe
    # buffer (~64 KB) and deadlocks on its next write — symptom is vLLM hanging
    # mid-load right after the attention-backend log lines. Initializing wandb
    # after the worker is up means the child inherited the original tty fds
    # and is unaffected. HF backend is single-process so the order doesn't
    # matter for it; only vLLM is sensitive.
    wb = WandbRun(cfg, full_config) if cfg.wandb.mode != "disabled" else None

    base_gen_kwargs = dict(
        max_new_tokens=cfg.model.generation.max_new_tokens,
        do_sample=cfg.model.generation.do_sample,
        temperature=cfg.model.generation.temperature,
    )
    all_overrides = cfg.eval.get("generation_overrides") or {}
    runtime_cfg = cfg.get("runtime") or OmegaConf.create({})
    shared_cache_dir = cfg.eval.get("cache_dir")
    shared_local_only = cfg.eval.get("local_files_only", False)

    all_metrics: dict[str, Any] = {}
    pred_paths: list[Path] = []

    for task_name in cfg.eval.tasks:
        ds_cfg = cfg.eval.datasets[task_name]
        # Inherit eval-level cache_dir / local_files_only unless the dataset overrides.
        shared = {}
        if shared_cache_dir:
            shared["cache_dir"] = shared_cache_dir
        if shared_local_only:
            shared["local_files_only"] = shared_local_only
        if shared:
            ds_cfg = OmegaConf.merge(shared, ds_cfg)
        task_overrides = all_overrides.get(task_name) or {}
        task_gen_kwargs = {**base_gen_kwargs, **dict(task_overrides)}
        result = _run_task(
            backend=backend,
            task_name=task_name,
            ds_cfg=ds_cfg,
            n=cfg.eval.samples_per_task,
            seed=cfg.eval.seed,
            gen_kwargs=task_gen_kwargs,
            out_dir=out_dir,
            carbon_cfg=cfg.carbon,
            run_name=cfg.run_name,
            runtime_cfg=runtime_cfg,
        )
        if result is None:
            continue
        metrics, pred_path = result
        all_metrics[task_name] = metrics
        pred_paths.append(pred_path)
        if wb:
            wb.log_task(task_name, metrics)
        if runtime_cfg.get("cleanup_dataset_after_task", False):
            _cleanup_dataset_cache(ds_cfg)

    primaries = {t: m["primary"] for t, m in all_metrics.items()}

    is_baseline = cfg.model.name == "base" and cfg.baseline_path is None
    if is_baseline:
        composite = 1.0
        ratios = {t: 1.0 for t in primaries}
        baseline_payload = {
            "run_name": cfg.run_name,
            "model": backend.name,
            "hf_id": backend.hf_id,
            "dtype": str(backend.dtype),
            "device": str(backend.device),
            "quant_backend": backend.quant_backend,
            "quant_mode": backend.quant_mode,
            "primaries": primaries,
            "all_metrics": all_metrics,
        }
        with (out_dir / "baseline.json").open("w") as f:
            json.dump(baseline_payload, f, indent=2)
    else:
        if not cfg.baseline_path:
            raise ValueError(
                "Non-baseline run requires `baseline_path=<path-to-baseline.json>`"
            )
        with open(cfg.baseline_path) as f:
            baseline = json.load(f)
        ratios = retention_ratios(primaries, baseline["primaries"])
        composite = composite_score(primaries, baseline["primaries"])

    summary = {
        "run_name": cfg.run_name,
        "model": backend.name,
        "hf_id": backend.hf_id,
        "dtype": str(backend.dtype),
        "device": str(backend.device),
        "quant_backend": backend.quant_backend,
        "quant_mode": backend.quant_mode,
        "is_baseline": is_baseline,
        "composite": composite,
        "retention_ratios": ratios,
        "tasks": all_metrics,
        "total_energy_kwh": float(sum(m["energy_kwh"] for m in all_metrics.values())),
        "total_co2_grams": float(sum(m["co2_grams"] for m in all_metrics.values())),
        "total_peak_vram_gb": float(max(
            (m["peak_vram_gb"] for m in all_metrics.values()), default=0.0
        )),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    if wb:
        wb.log_summary(summary)
        wb.save_file(out_dir / "summary.json")
        if is_baseline:
            wb.save_file(out_dir / "baseline.json")
        for p in pred_paths:
            wb.save_file(p)
        wb.finish()

    logger.info("Composite: %.4f | energy: %.4f kWh | co2: %.2f g",
                composite, summary["total_energy_kwh"], summary["total_co2_grams"])

    return summary
