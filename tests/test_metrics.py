"""Smoke tests for metrics. These lock the behavior our composite score depends on:
if these change, every prior baseline.json becomes incomparable."""

from __future__ import annotations

import math

import pytest

from llm_bench_cc.composite import composite_score, retention_ratios
from llm_bench_cc.metrics import (
    anls,
    bleu4,
    char_accuracy,
    normalize_answer,
    relaxed_accuracy,
    vqa_accuracy,
)


def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Cat.") == "cat"
    assert normalize_answer("  AN  apple  ") == "apple"


def test_anls_perfect_and_partial():
    refs = {"q1": ["Paris"], "q2": ["New York"]}
    assert anls(refs, {"q1": "paris", "q2": "new york"}) == pytest.approx(1.0)
    # one perfect, one zero (below threshold)
    score = anls(refs, {"q1": "paris", "q2": "tokyo"})
    assert score == pytest.approx(0.5)


def test_vqa_accuracy_fractional():
    # 3+ refs: VQAv2-style — match against 3 of 10 → 1.0
    refs = {"q1": ["red"] * 3 + ["blue"] * 7}
    assert vqa_accuracy(refs, {"q1": "red"}) == pytest.approx(1.0)
    refs = {"q1": ["red"] * 1 + ["blue"] * 9}
    assert vqa_accuracy(refs, {"q1": "red"}) == pytest.approx(1.0 / 3)


def test_relaxed_accuracy_numeric_tolerance():
    refs = {"a": ["100"], "b": ["50.0"], "c": ["yes"]}
    hyps = {"a": "98", "b": "60", "c": "yes."}
    # 98 within 5% of 100 → hit; 60 not within 5% of 50 → miss; "yes." → hit
    assert relaxed_accuracy(refs, hyps) == pytest.approx(2 / 3)


def test_char_accuracy_simple():
    refs = {"a": ["hello world"]}
    assert char_accuracy(refs, {"a": "hello world"}) == pytest.approx(1.0)
    s = char_accuracy(refs, {"a": "helo world"})  # 1 char off out of 11
    assert 0.85 < s < 1.0


def test_bleu4_runs_and_is_one_for_identical():
    refs = {"a": ["the quick brown fox jumps over the lazy dog"]}
    hyps = {"a": "the quick brown fox jumps over the lazy dog"}
    score = bleu4(refs, hyps)
    assert score > 0.9  # near-1 with smoothing


def test_composite_baseline_is_one():
    base = {"caption": 0.30, "ocr": 0.55, "vqa": 0.65}
    assert composite_score(base, base) == pytest.approx(1.0)


def test_composite_handles_partial():
    base = {"caption": 0.4, "ocr": 0.5}
    cand = {"caption": 0.32, "ocr": 0.45}  # 80% and 90%
    assert composite_score(cand, base) == pytest.approx(0.85)


def test_composite_clips_overshoot():
    base = {"caption": 0.4}
    cand = {"caption": 0.6}  # 150% — clipped to 1.0
    assert composite_score(cand, base) == pytest.approx(1.0)


def test_retention_ratios_drops_missing():
    base = {"caption": 0.4, "ocr": 0.5}
    cand = {"caption": 0.4}  # ocr missing — composite computed over intersection
    r = retention_ratios(cand, base)
    assert r == {"caption": pytest.approx(1.0)}


def test_composite_subset_candidate_against_full_baseline():
    # eval=standard (4 tasks) candidate vs eval=full (5 tasks) baseline:
    # composite is the mean over the 4 shared tasks, not penalized for the missing one.
    base = {"caption": 0.10, "ocr": 0.70, "docvqa": 0.80, "vqa": 0.55, "chart": 0.70}
    cand = {"caption": 0.10, "ocr": 0.70, "docvqa": 0.80, "chart": 0.70}  # vqa skipped
    assert composite_score(cand, base) == pytest.approx(1.0)
