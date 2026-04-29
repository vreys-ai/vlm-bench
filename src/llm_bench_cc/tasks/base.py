from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL.Image import Image


@dataclass
class Sample:
    sample_id: str
    image: Image
    prompt: str
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    sample_id: str
    prediction: str
    latency_ms: float


class Task(ABC):
    """Each subclass owns its dataset adapter, prompt, and metric.

    The first key returned by score() is treated as the *primary* metric used in the
    composite retention ratio. All metrics live in [0, 1] so retention divides cleanly.
    """

    name: str
    primary_metric: str

    @abstractmethod
    def load(self, n: int, seed: int, ds_cfg: Any) -> list[Sample]:
        ...

    @abstractmethod
    def score(self, samples: list[Sample], preds: list[Prediction]) -> dict[str, float]:
        ...

    @staticmethod
    def _refs_map(samples: list[Sample]) -> dict[str, list[str]]:
        return {s.sample_id: s.references for s in samples}

    @staticmethod
    def _hyps_map(preds: list[Prediction]) -> dict[str, str]:
        return {p.sample_id: p.prediction for p in preds}
