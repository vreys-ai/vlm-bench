from __future__ import annotations

from ..metrics import bleu4
from .base import Prediction, Sample, Task


class CaptionTask(Task):
    name = "caption"
    primary_metric = "bleu4"
    PROMPT = "Describe this image in one short sentence."

    def load(self, n, seed, ds_cfg):
        ds = self._load_split(ds_cfg, n, seed)
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            refs = (
                row.get("caption")
                or row.get("captions")
                or row.get("sentences")
                or row.get("answer")        # lmms-lab/COCO-Caption stores refs here
                or row.get("references")
            )
            if isinstance(refs, str):
                refs = [refs]
            if isinstance(refs, list) and refs and isinstance(refs[0], dict):
                refs = [r.get("raw") or r.get("caption") or r.get("text") for r in refs]
            refs = [r for r in (refs or []) if r and r != "None"]
            if not refs or img is None:
                continue
            sid = str(
                row.get("cocoid")
                or row.get("question_id")   # lmms-lab/COCO-Caption uses filename as id
                or row.get("imgid")
                or row.get("id")
                or i
            )
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=self.PROMPT,
                references=refs,
            ))
        return out

    def score(self, samples, preds):
        return {"bleu4": bleu4(self._refs_map(samples), self._hyps_map(preds))}
