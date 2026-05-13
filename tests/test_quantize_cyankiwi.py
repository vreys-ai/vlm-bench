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


class TestPickCalibrationColumn:
    def test_priority_instruction_first(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "instruction", "prompt"]
        )
        assert col == "instruction"
        assert needs_template is True

    def test_priority_prompt_over_text(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "prompt"]
        )
        assert col == "prompt"
        assert needs_template is True

    def test_fallback_text_no_template(self):
        col, needs_template = qc._pick_calibration_column(
            available_columns=["text", "metadata"]
        )
        assert col == "text"
        assert needs_template is False

    def test_no_match_raises_with_actual_columns(self):
        # The error formats `sorted(available_columns)`, so both names appear
        # but in alphabetical order — match each substring independently
        # rather than coupling the test to the sort order.
        with pytest.raises(ValueError) as excinfo:
            qc._pick_calibration_column(available_columns=["foo", "bar"])
        assert "foo" in str(excinfo.value)
        assert "bar" in str(excinfo.value)

    def test_no_match_error_lists_priority(self):
        with pytest.raises(ValueError, match=r"instruction.*prompt.*text"):
            qc._pick_calibration_column(available_columns=["foo"])


class TestBuildRecipePayload:
    def test_includes_all_required_fields(self):
        import argparse
        args = argparse.Namespace(
            model_id="google/gemma-4-E4B-it",
            calibration_dataset="garage-bAInd/Open-Platypus",
            calibration_split="train",
            num_calibration_samples=128,
            max_seq_length=2048,
            group_size=32,
            observer="mse",
            symmetric=False,
            gptq_block_size=128,
            gptq_dampening_frac=0.01,
            seed=42,
        )
        payload = qc._build_recipe_payload(
            args=args,
            entrypoint_used="llmcompressor.oneshot+GPTQModifier",
            git_sha="abc1234",
        )
        assert payload["model_id"] == "google/gemma-4-E4B-it"
        assert payload["entrypoint"] == "llmcompressor.oneshot+GPTQModifier"
        assert payload["git_sha"] == "abc1234"
        assert payload["scheme_kwargs"]["num_bits"] == 4
        assert payload["scheme_kwargs"]["group_size"] == 32
        assert payload["scheme_kwargs"]["observer"] == "mse"
        assert payload["scheme_kwargs"]["symmetric"] is False
        assert payload["ignore_patterns"] == qc.IGNORE
        assert payload["calibration"]["dataset"] == "garage-bAInd/Open-Platypus"
        assert payload["calibration"]["num_samples"] == 128
        assert payload["calibration"]["max_seq_length"] == 2048
        assert payload["calibration"]["seed"] == 42
        assert "timestamp_utc" in payload
        # Timestamp should be ISO-8601 with seconds precision (sortable).
        assert payload["timestamp_utc"].endswith("+00:00")
        assert "based_on" in payload
        assert "cyankiwi" in payload["based_on"]

    def test_reflects_cli_overrides(self):
        import argparse
        args = argparse.Namespace(
            model_id="google/gemma-4-E4B-it",
            calibration_dataset="HuggingFaceH4/ultrachat_200k",
            calibration_split="train_sft",
            num_calibration_samples=256,
            max_seq_length=1024,
            group_size=128,
            observer="minmax",
            symmetric=True,
            gptq_block_size=64,
            gptq_dampening_frac=0.05,
            seed=99,
        )
        payload = qc._build_recipe_payload(
            args=args,
            entrypoint_used="llmcompressor.oneshot+QuantizationModifier",
            git_sha="def5678",
        )
        assert payload["scheme_kwargs"]["group_size"] == 128
        assert payload["scheme_kwargs"]["observer"] == "minmax"
        assert payload["scheme_kwargs"]["symmetric"] is True
        assert payload["calibration"]["dataset"] == "HuggingFaceH4/ultrachat_200k"
        assert payload["calibration"]["split"] == "train_sft"
        assert payload["calibration"]["num_samples"] == 256
        assert payload["calibration"]["max_seq_length"] == 1024
        assert payload["calibration"]["seed"] == 99
        assert payload["entrypoint"] == "llmcompressor.oneshot+QuantizationModifier"
