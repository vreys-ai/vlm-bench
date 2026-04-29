"""Composite retention score: unweighted mean of per-task retention ratios, clipped to [0, 1]."""

from __future__ import annotations

from statistics import mean


def retention_ratios(
    candidate_primaries: dict[str, float],
    baseline_primaries: dict[str, float],
) -> dict[str, float]:
    out = {}
    for task, base_v in baseline_primaries.items():
        cand_v = candidate_primaries.get(task)
        if cand_v is None or base_v <= 0:
            out[task] = 0.0
            continue
        out[task] = max(0.0, min(1.0, cand_v / base_v))
    return out


def composite_score(
    candidate_primaries: dict[str, float],
    baseline_primaries: dict[str, float],
) -> float:
    ratios = retention_ratios(candidate_primaries, baseline_primaries)
    return mean(ratios.values()) if ratios else 0.0
