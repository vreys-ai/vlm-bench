from __future__ import annotations

from .base import Task
from .caption import CaptionTask
from .chart import ChartTask
from .docvqa import DocVQATask
from .ocr import OCRTask
from .vqa import VQATask

TASK_REGISTRY: dict[str, type[Task]] = {
    "caption": CaptionTask,
    "ocr": OCRTask,
    "docvqa": DocVQATask,
    "vqa": VQATask,
    "chart": ChartTask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Known: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()
