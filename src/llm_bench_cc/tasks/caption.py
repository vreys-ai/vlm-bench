from __future__ import annotations

from datasets import load_dataset

from ..metrics import bleu4
from .base import Prediction, Sample, Task


class CaptionTask(Task):
    name = "caption"
    primary_metric = "bleu4"
    PROMPT = "Describe this image in one short sentence."

    def load(self, n, seed, ds_cfg):
        ds = load_dataset(ds_cfg.hf_id, split=ds_cfg.split)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        out: list[Sample] = []
        for i, row in enumerate(ds):
            img = row.get("image")
            refs = (
                row.get("caption")
                or row.get("captions")
                or row.get("sentences")
                or row.get("references")
            )
            if isinstance(refs, str):
                refs = [refs]
            if isinstance(refs, list) and refs and isinstance(refs[0], dict):
                refs = [r.get("raw") or r.get("caption") or r.get("text") for r in refs]
            if not refs:
                continue
            sid = str(row.get("cocoid") or row.get("imgid") or row.get("id") or i)
            out.append(Sample(
                sample_id=sid,
                image=img.convert("RGB"),
                prompt=self.PROMPT,
                references=[r for r in refs if r],
            ))
        return out

    def score(self, samples, preds):
        return {"bleu4": bleu4(self._refs_map(samples), self._hyps_map(preds))}
