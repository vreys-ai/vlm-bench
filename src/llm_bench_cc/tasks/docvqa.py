from __future__ import annotations

from ..metrics import anls
from .base import Prediction, Sample, Task


class DocVQATask(Task):
    name = "docvqa"
    primary_metric = "anls"

    def load(self, n, seed, ds_cfg):
        ds = self._load_split(ds_cfg, n, seed)
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            q = row.get("question")
            answers = row.get("answers") or row.get("answer")
            if isinstance(answers, str):
                answers = [answers]
            if not answers or img is None or not q:
                continue
            sid = str(row.get("questionId") or row.get("question_id") or row.get("id") or i)
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=f"{q}\nAnswer the question with a short word or phrase.",
                references=[str(a) for a in answers if a],
            ))
        return out

    def score(self, samples, preds):
        return {"anls": anls(self._refs_map(samples), self._hyps_map(preds))}
