# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Provenance record round-trip: every benchmark (paired or not) writes run_id,
model, provider, prompt_template, sample_count, and a powered flag derived
from the shared N_REQUIRED bar — and reads back complete.
"""

import yaml

from benchmarks.deepeval_suite import (
    BenchmarkResult, DeepEvalSuite, compute_paired_verdict, provenance_record,
    tally_paired, update_paired_stats_yaml, write_paired_artifact,
)
from benchmarks.paired_stats import N_REQUIRED


def test_provenance_record_fields():
    rec = provenance_record(run_id="gsm8k_x", model="m", provider="chutes",
                            prompt_template="cot-v1", sample_count=5)
    assert rec == {
        "run_id": "gsm8k_x", "model": "m", "provider": "chutes",
        "prompt_template": "cot-v1", "sample_count": 5, "powered": False,
    }


def test_provenance_record_powered_flag_tracks_bar_both_directions():
    below = provenance_record("r", "m", "p", "t", sample_count=N_REQUIRED - 1)
    at = provenance_record("r", "m", "p", "t", sample_count=N_REQUIRED)
    above = provenance_record("r", "m", "p", "t", sample_count=N_REQUIRED + 1)
    assert below["powered"] is False
    assert at["powered"] is True
    assert above["powered"] is True


def test_paired_stats_yaml_carries_provenance_round_trip(tmp_path):
    cases = [{"routed_correct": True, "direct_correct": False, "strategy": "math",
              "routed_error": None, "direct_error": None, "routed_model": "m"}]
    payload = {
        "timestamp": "2026-08-10T00:00:00", "benchmark": "GSM8K", "provider": "chutes",
        "model": "m", "prompt_template": "shared-template", "n": 1,
        "frozen_tolerance_arg": 0.05, "cases": cases, "tallies": tally_paired(cases),
        "verdict_record": compute_paired_verdict(cases, pinned_model="m", frozen_tolerance=0.05),
    }
    artifact = write_paired_artifact(payload, results_dir=str(tmp_path / "results"))
    stats_path = tmp_path / "STATS.yaml"
    update_paired_stats_yaml(payload, artifact, stats_path=str(stats_path))
    stats = yaml.safe_load(stats_path.read_text())["paired_gsm8k"]

    assert stats["run_id"] == f"paired_gsm8k_{payload['timestamp']}"
    assert stats["model"] == "m"
    assert stats["provider"] == "chutes"
    assert stats["prompt_template"] == "shared-template"
    assert stats["sample_count"] == 1
    assert stats["powered"] is False  # n_clean=1 < N_REQUIRED


def test_non_paired_stats_yaml_carries_provenance_round_trip(tmp_path):
    suite = DeepEvalSuite()
    suite.stats_path = tmp_path / "STATS.yaml"
    suite.results = [
        BenchmarkResult("GSM8K", N_REQUIRED, 50.0, 70.0, 20.0, "2026-08-10T00:00:00"),
        BenchmarkResult("TruthfulQA", 5, 60.0, 55.0, -5.0, "2026-08-10T00:00:00"),
    ]
    suite.update_stats_yaml(provider="chutes", prompt_template="cot-v1")
    stats = yaml.safe_load(suite.stats_path.read_text())["deepeval_benchmarks"]

    gsm8k = stats["benchmarks"]["GSM8K"]
    assert gsm8k["run_id"] == f"deepeval_benchmarks_GSM8K_{stats['last_run']}"
    assert gsm8k["provider"] == "chutes"
    assert gsm8k["prompt_template"] == "cot-v1"
    assert gsm8k["sample_count"] == N_REQUIRED
    assert gsm8k["powered"] is True  # n=100 >= N_REQUIRED

    truthfulqa = stats["benchmarks"]["TruthfulQA"]
    assert truthfulqa["sample_count"] == 5
    assert truthfulqa["powered"] is False  # n=5 < N_REQUIRED
