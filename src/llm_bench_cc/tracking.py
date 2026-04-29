"""Thin wrappers around W&B and CodeCarbon. Both are optional — disabled modes return no-ops
so the runner stays tidy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------- W&B ----------

class WandbRun:
    def __init__(self, cfg, full_config_dict: dict[str, Any]):
        import wandb
        from omegaconf import OmegaConf

        mode = cfg.wandb.mode
        if mode == "disabled":
            self._wb = None
            return

        self._wb = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            mode=mode,
            tags=list(cfg.wandb.tags or []),
            config=full_config_dict,
            dir=cfg.output_dir,
        )

    def log_task(self, task_name: str, metrics: dict[str, Any]) -> None:
        if not self._wb:
            return
        flat = {f"{task_name}/{k}": v for k, v in metrics.items()}
        self._wb.log(flat)

    def log_summary(self, summary: dict[str, Any]) -> None:
        if not self._wb:
            return
        # flat top-level keys go into summary; tasks dict logged as artifact-friendly run summary
        self._wb.summary.update({
            "composite": summary["composite"],
            "total_energy_kwh": summary["total_energy_kwh"],
            "total_co2_grams": summary["total_co2_grams"],
        })
        self._wb.log({"composite": summary["composite"]})

    def save_file(self, path: str | Path) -> None:
        if not self._wb:
            return
        self._wb.save(str(path), policy="now")

    def finish(self) -> None:
        if self._wb:
            self._wb.finish()


# ---------- CodeCarbon ----------

@dataclass
class CarbonResult:
    co2eq_grams: float
    energy_consumed_kwh: float


class CarbonTracker:
    def __init__(self, project_name: str, output_dir: str, cfg, enabled: bool = True):
        self._tracker = None
        if not enabled:
            return
        try:
            from codecarbon import EmissionsTracker, OfflineEmissionsTracker
        except ImportError:
            logger.warning("codecarbon not installed; energy tracking disabled")
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        common_kwargs = dict(
            project_name=project_name,
            output_dir=output_dir,
            output_file=f"emissions_{project_name}.csv",
            log_level="warning",
            measure_power_secs=cfg.carbon.measure_power_secs,
            save_to_file=True,
            allow_multiple_runs=True,
        )
        country_iso_code = cfg.carbon.country_iso_code
        if country_iso_code:
            self._tracker = OfflineEmissionsTracker(
                country_iso_code=country_iso_code,
                **common_kwargs,
            )
        else:
            self._tracker = EmissionsTracker(**common_kwargs)

    def start(self) -> None:
        if self._tracker:
            self._tracker.start()

    def stop(self) -> CarbonResult:
        if not self._tracker:
            return CarbonResult(0.0, 0.0)
        kg = self._tracker.stop() or 0.0
        data = getattr(self._tracker, "final_emissions_data", None)
        kwh = float(getattr(data, "energy_consumed", 0.0) or 0.0)
        return CarbonResult(co2eq_grams=float(kg) * 1000.0, energy_consumed_kwh=kwh)
