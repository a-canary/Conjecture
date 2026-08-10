# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Regression: STATS.yaml must not ship a live positive TruthfulQA delta.

The only adequately-powered TruthfulQA run (n=100) found decomposition
HURTS truthfulness (-13.0pp). Smaller runs (n=5, n=21) found the opposite
sign purely from underpowered noise. Those figures must stay marked
`status: superseded` — never re-surface as a live claim.
"""

from pathlib import Path

import yaml

STATS_PATH = Path(__file__).resolve().parent.parent / "STATS.yaml"


def _iter_truthfulqa_entries(stats):
    """Yield (path, entry_dict) for every TruthfulQA benchmark entry in STATS.yaml."""
    for section_name, section in stats.items():
        if not isinstance(section, dict):
            continue
        benchmarks = section.get("benchmarks")
        if isinstance(benchmarks, dict) and "TruthfulQA" in benchmarks:
            yield f"{section_name}.benchmarks.TruthfulQA", benchmarks["TruthfulQA"]
        for list_key in ("passing_20pp", "no_regression_benchmarks"):
            for item in section.get(list_key, []) or []:
                if isinstance(item, dict) and item.get("name") == "TruthfulQA":
                    yield f"{section_name}.{list_key}", item


def _as_float(delta):
    if isinstance(delta, str):
        return float(delta.rstrip("pp").lstrip("+"))
    return delta


def test_no_live_positive_truthfulqa_delta():
    stats = yaml.safe_load(STATS_PATH.read_text())
    live_positive = []
    for path, entry in _iter_truthfulqa_entries(stats):
        delta = _as_float(entry.get("delta"))
        if delta is None or delta <= 0:
            continue
        if entry.get("status") == "superseded":
            continue
        live_positive.append((path, delta))
    assert not live_positive, (
        f"live (non-superseded) positive TruthfulQA delta(s) shipped: {live_positive}"
    )


def test_truthfulqa_refutation_present_with_provenance():
    stats = yaml.safe_load(STATS_PATH.read_text())
    refutation = stats.get("truthfulqa_refutation")
    assert refutation is not None, "truthfulqa_refutation block missing from STATS.yaml"
    assert refutation["delta_pp"] == -13.0
    assert refutation["sample_count"] == 100
    assert refutation["source"] == "experiments/results/truthfulqa_20260306_194026.json"
    assert "CHALLENGED" in refutation["conclusion"]


def test_known_superseded_entries_are_marked():
    stats = yaml.safe_load(STATS_PATH.read_text())
    n5 = stats["o0008_quick"]["benchmarks"]["TruthfulQA"]
    assert n5["sample_count"] == 5
    assert n5["status"] == "superseded"

    n21 = next(
        item
        for item in stats["o0008_previous_blocked"]["gpt_oss_20b_results"]["passing_20pp"]
        if item["name"] == "TruthfulQA"
    )
    assert n21["samples"] == 21
    assert n21["status"] == "superseded"
