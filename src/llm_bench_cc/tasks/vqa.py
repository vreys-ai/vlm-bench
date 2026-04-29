from __future__ import annotations

from datasets import load_dataset

from ..metrics import vqa_accuracy
from .base import Prediction, Sample, Task


class VQATask(Task):
    name = "vqa"
    primary_metric = "vqa_acc"

    def load(self, n, seed, ds_cfg):
        ds = load_dataset(ds_cfg.hf_id, split=ds_cfg.split)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            q = row.get("question")
            answers = row.get("answers") or row.get("answer")
            # VQAv2 mirrors often store answers as list of dicts {answer, answer_confidence, ...}
            if isinstance(answers, list) and answers and isinstance(answers[0], dict):
                answers = [a.get("answer") for a in answers if a.get("answer")]
            if isinstance(answers, str):
                answers = [answers]
            mc = row.get("multiple_choice_answer")
            if mc and (not answers or len(answers) == 0):
                answers = [mc]
            if not answers or img is None or not q:
                continue
            sid = str(row.get("question_id") or row.get("id") or i)
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=f"{q}\nAnswer the question with a short word or phrase.",
                references=[str(a) for a in answers if a],
            ))
        return out

    def score(self, samples, preds):
        return {"vqa_acc": vqa_accuracy(self._refs_map(samples), self._hyps_map(preds))}
