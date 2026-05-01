from __future__ import annotations

from ..metrics import anls
from .base import Prediction, Sample, Task


class OCRTask(Task):
    """OCRBench: OCR capability evaluated as VQA-style questions about text content.

    Standard metric is normalized exact match / ANLS — keeping ANLS for tolerance to
    minor formatting differences (capitalization, trailing punctuation).
    """
    name = "ocr"
    primary_metric = "anls"
    DEFAULT_PROMPT = "Answer the question based on the text shown in the image."

    def load(self, n, seed, ds_cfg):
        ds = self._load_split(ds_cfg, n, seed)
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            question = row.get("question") or self.DEFAULT_PROMPT
            answers = row.get("answer") or row.get("answers")
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                continue
            sid = str(row.get("question_id") or row.get("id") or i)
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=f"{question}\nAnswer the question with a short word or phrase.",
                references=[str(a) for a in answers if a],
                metadata={"category": row.get("dataset") or row.get("category")},
            ))
        return out

    def score(self, samples, preds):
        return {"anls": anls(self._refs_map(samples), self._hyps_map(preds))}
