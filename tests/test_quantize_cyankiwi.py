"""Unit tests for the pure helpers in scripts/quantize_cyankiwi.py.

These tests are intentionally narrow: they cover the three pure helpers
(failure-marker matching, calibration-column detection, recipe-payload
building) without importing llmcompressor or loading any model. The
end-to-end pipeline is verified by a smoke run in the implementation
plan, not here."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable as a top-level package for these tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import quantize_cyankiwi as qc  # noqa: E402


class TestIsKnownEVariantFailure:
    def test_matches_per_layer_input_gate(self):
        exc = AttributeError(
            "module 'GemmaDecoderLayer' has no attribute 'per_layer_input_gate' "
            "during fx tracing"
        )
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_num_kv_shared_layers(self):
        exc = RuntimeError(
            "Sequential pipeline state inconsistent: num_kv_shared_layers=18 "
            "but layer 24 has no preceding KV producer"
        )
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_fx_proxy(self):
        exc = RuntimeError("torch.fx Proxy object has no attribute __mul__")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_fx_substring(self):
        exc = RuntimeError("fx tracing failed at gemma4 PLE access")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_matches_sequential_pipeline(self):
        exc = RuntimeError("Sequential pipeline cannot dispatch hooked layer 18")
        assert qc._is_known_e_variant_failure(exc) is True

    def test_does_not_match_oom(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert qc._is_known_e_variant_failure(exc) is False

    def test_does_not_match_auth_failure(self):
        exc = RuntimeError("401 Unauthorized: token is invalid")
        assert qc._is_known_e_variant_failure(exc) is False

    def test_case_insensitive(self):
        exc = RuntimeError("Per_Layer_Input_Gate not found")
        assert qc._is_known_e_variant_failure(exc) is True
