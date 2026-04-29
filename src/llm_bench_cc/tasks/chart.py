from __future__ import annotations

from ..metrics import relaxed_accuracy
from .base import Prediction, Sample, Task


class ChartTask(Task):
    name = "chart"
    primary_metric = "relaxed_acc"

    def load(self, n, seed, ds_cfg):
        ds = self._load_split(ds_cfg, n, seed)
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            q = row.get("query") or row.get("question")
            answers = row.get("label") or row.get("answer") or row.get("answers")
            if isinstance(answers, str):
                answers = [answers]
            if not answers or img is None or not q:
                continue
            sid = str(row.get("id") or row.get("question_id") or i)
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=f"{q}\nAnswer with a short value.",
                references=[str(a) for a in answers if a is not None],
            ))
        return out

    def score(self, samples, preds):
        return {"relaxed_acc": relaxed_accuracy(self._refs_map(samples), self._hyps_map(preds))}
