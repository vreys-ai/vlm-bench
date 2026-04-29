"""Metrics keyed off task type. Every metric returns a value in [0, 1] where higher is better,
so retention ratios composite cleanly."""

from __future__ import annotations

import re
import string
from statistics import mean

from rapidfuzz.distance import Levenshtein


# ---------- normalization ----------

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# ---------- caption: BLEU-4 ----------

def bleu4(refs_per_sample: dict[str, list[str]], hyps: dict[str, str]) -> float:
    """Corpus BLEU-4 with smoothing. NLTK is heavy at import — load lazily."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    smoothie = SmoothingFunction().method1

    list_of_refs, list_of_hyps = [], []
    for sid, hyp in hyps.items():
        refs = refs_per_sample.get(sid, [])
        if not refs:
            continue
        list_of_refs.append([r.split() for r in refs])
        list_of_hyps.append(hyp.split())
    if not list_of_hyps:
        return 0.0
    return float(corpus_bleu(
        list_of_refs, list_of_hyps,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    ))


# ---------- ocr: 1 - CER (character accuracy) ----------

def char_accuracy(refs_per_sample: dict[str, list[str]], hyps: dict[str, str]) -> float:
    """1 - CER, averaged across samples. Each sample picks its best reference."""
    from jiwer import cer
    scores = []
    for sid, hyp in hyps.items():
        refs = refs_per_sample.get(sid, [])
        if not refs:
            continue
        best = min(cer(r, hyp) for r in refs)  # lower is better
        scores.append(max(0.0, 1.0 - best))
    return mean(scores) if scores else 0.0


# ---------- docvqa: ANLS ----------

def anls(
    refs_per_sample: dict[str, list[str]],
    hyps: dict[str, str],
    threshold: float = 0.5,
) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA standard)."""
    scores = []
    for sid, hyp in hyps.items():
        refs = refs_per_sample.get(sid, [])
        if not refs:
            continue
        h = hyp.lower().strip()
        best = 0.0
        for r in refs:
            r = r.lower().strip()
            denom = max(len(h), len(r), 1)
            sim = 1.0 - (Levenshtein.distance(h, r) / denom)
            if sim < threshold:
                sim = 0.0
            best = max(best, sim)
        scores.append(best)
    return mean(scores) if scores else 0.0


# ---------- vqa: normalized exact match against any reference ----------

def vqa_accuracy(refs_per_sample: dict[str, list[str]], hyps: dict[str, str]) -> float:
    scores = []
    for sid, hyp in hyps.items():
        refs = refs_per_sample.get(sid, [])
        if not refs:
            continue
        h = normalize_answer(hyp)
        # standard VQAv2: min(matches/3, 1.0) when 10 human answers; with arbitrary refs
        # we degrade gracefully to a fraction of references matched, capped at 1.
        if len(refs) >= 3:
            matches = sum(1 for r in refs if normalize_answer(r) == h)
            scores.append(min(matches / 3.0, 1.0))
        else:
            scores.append(1.0 if any(normalize_answer(r) == h for r in refs) else 0.0)
    return mean(scores) if scores else 0.0


# ---------- chart: relaxed accuracy (numeric tolerance + EM) ----------

_NUM = re.compile(r"-?\d+\.?\d*")


def _maybe_number(s: str) -> float | None:
    m = _NUM.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def relaxed_accuracy(
    refs_per_sample: dict[str, list[str]],
    hyps: dict[str, str],
    rel_tol: float = 0.05,
) -> float:
    """ChartQA-style: 5% relative tolerance for numeric answers, EM otherwise."""
    scores = []
    for sid, hyp in hyps.items():
        refs = refs_per_sample.get(sid, [])
        if not refs:
            continue
        hit = False
        h_norm = normalize_answer(hyp)
        h_num = _maybe_number(hyp)
        for r in refs:
            r_num = _maybe_number(r)
            if h_num is not None and r_num is not None:
                if r_num == 0:
                    if abs(h_num) < rel_tol:
                        hit = True
                        break
                elif abs(h_num - r_num) / abs(r_num) <= rel_tol:
                    hit = True
                    break
            elif normalize_answer(r) == h_norm:
                hit = True
                break
        scores.append(1.0 if hit else 0.0)
    return mean(scores) if scores else 0.0
