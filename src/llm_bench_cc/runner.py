from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .composite import composite_score, retention_ratios
from .models import LoadedModel, load_model
from .tasks.base import Prediction, Sample
from .tasks.registry import get_task
from .tracking import CarbonTracker, WandbRun

logger = logging.getLogger(__name__)


def _generate_one(
    loaded: LoadedModel,
    image,
    prompt: str,
    gen_kwargs: dict[str, Any],
) -> tuple[str, float]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = loaded.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(loaded.device)

    in_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = loaded.model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    text = loaded.processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
    return text, elapsed_ms


def _run_task(loaded, task_name, ds_cfg, n, seed, gen_kwargs, out_dir, carbon_cfg, run_name):
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    carbon.start()
    preds: list[Prediction] = []
    latencies: list[float] = []
    for s in tqdm(samples, desc=task_name, leave=False):
        text, ms = _generate_one(loaded, s.image, s.prompt, gen_kwargs)
        preds.append(Prediction(sample_id=s.sample_id, prediction=text, latency_ms=ms))
        latencies.append(ms)
    emissions = carbon.stop()

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available() else 0.0
    )

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
    wb = WandbRun(cfg, full_config) if cfg.wandb.mode != "disabled" else None

    logger.info("Loading model %s (dtype=%s)", cfg.model.hf_id, cfg.model.dtype)
    loaded = load_model(cfg.model)

    gen_kwargs = dict(
        max_new_tokens=cfg.model.generation.max_new_tokens,
        do_sample=cfg.model.generation.do_sample,
        temperature=cfg.model.generation.temperature,
    )

    all_metrics: dict[str, Any] = {}
    pred_paths: list[Path] = []

    for task_name in cfg.eval.tasks:
        ds_cfg = cfg.eval.datasets[task_name]
        result = _run_task(
            loaded=loaded,
            task_name=task_name,
            ds_cfg=ds_cfg,
            n=cfg.eval.samples_per_task,
            seed=cfg.eval.seed,
            gen_kwargs=gen_kwargs,
            out_dir=out_dir,
            carbon_cfg=cfg.carbon,
            run_name=cfg.run_name,
        )
        if result is None:
            continue
        metrics, pred_path = result
        all_metrics[task_name] = metrics
        pred_paths.append(pred_path)
        if wb:
            wb.log_task(task_name, metrics)

    primaries = {t: m["primary"] for t, m in all_metrics.items()}

    is_baseline = cfg.model.name == "base" and cfg.baseline_path is None
    if is_baseline:
        composite = 1.0
        ratios = {t: 1.0 for t in primaries}
        baseline_payload = {
            "run_name": cfg.run_name,
            "model": loaded.name,
            "hf_id": loaded.hf_id,
            "dtype": str(loaded.dtype),
            "device": str(loaded.device),
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
        "model": loaded.name,
        "hf_id": loaded.hf_id,
        "dtype": str(loaded.dtype),
        "device": str(loaded.device),
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
