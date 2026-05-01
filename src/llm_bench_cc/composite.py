"""Composite retention score: unweighted mean of per-task retention ratios, clipped to [0, 1]."""

from __future__ import annotations

import logging
from statistics import mean

logger = logging.getLogger(__name__)


def retention_ratios(
    candidate_primaries: dict[str, float],
    baseline_primaries: dict[str, float],
) -> dict[str, float]:
    out = {}
    missing: list[str] = []
    for task, base_v in baseline_primaries.items():
        cand_v = candidate_primaries.get(task)
        if cand_v is None:
            # Candidate didn't run this task (e.g., eval=standard vs a full baseline).
            # Compute composite over the intersection rather than penalize as 0.
            missing.append(task)
            continue
        if base_v <= 0:
            out[task] = 0.0
            continue
        out[task] = max(0.0, min(1.0, cand_v / base_v))
    if missing:
        logger.warning(
            "Candidate is missing %d task(s) present in baseline (%s); "
            "composite computed over %d shared task(s)",
            len(missing), ", ".join(missing), len(out),
        )
    return out


def composite_score(
    candidate_primaries: dict[str, float],
    baseline_primaries: dict[str, float],
) -> float:
    ratios = retention_ratios(candidate_primaries, baseline_primaries)
    return mean(ratios.values()) if ratios else 0.0
