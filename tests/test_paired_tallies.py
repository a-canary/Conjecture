# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Tally logic and cache-bypass invariant for the paired routed-vs-direct GSM8K tracer."""

import inspect

from benchmarks.deepeval_suite import run_paired_gsm8k, tally_paired


def test_tally_paired_counts_and_strategy_breakdown():
    cases = [
        {"routed_correct": True, "direct_correct": False, "strategy": "math"},
        {"routed_correct": True, "direct_correct": True, "strategy": "math"},
        {"routed_correct": False, "direct_correct": True, "strategy": "recall"},
    ]
    t = tally_paired(cases)
    assert t["n"] == 3
    assert t["n_errored"] == 0
    assert t["n_clean"] == 3
    assert t["routed_correct"] == 2
    assert t["direct_correct"] == 2
    assert t["by_strategy"]["math"] == {"n": 2, "routed_correct": 2, "direct_correct": 1}
    assert t["by_strategy"]["recall"] == {"n": 1, "routed_correct": 0, "direct_correct": 1}


def test_tally_paired_empty():
    t = tally_paired([])
    assert t == {"n": 0, "n_errored": 0, "n_clean": 0,
                 "routed_correct": 0, "direct_correct": 0, "by_strategy": {}}


def test_errored_case_is_not_scored_as_a_miss():
    """An arm failure must be visible as an error, never absorbed as a regression."""
    cases = [
        {"routed_correct": True, "direct_correct": True, "strategy": "math",
         "routed_error": None, "direct_error": None},
        {"routed_correct": False, "direct_correct": False, "strategy": "math",
         "routed_error": "429 rate limited", "direct_error": None},
    ]
    t = tally_paired(cases)
    assert t["n"] == 2
    assert t["n_errored"] == 1
    assert t["n_clean"] == 1
    # the errored case contributes a False to both arms, which is exactly why
    # n_clean exists: 1/1 clean, not 1/2
    assert t["routed_correct"] == 1


def test_paired_run_never_touches_the_baseline_cache():
    """Pairing rests on both arms coming from the same run; a cached direct
    score from an earlier release would break it. Static check, so it holds
    without spending API calls."""
    src = inspect.getsource(run_paired_gsm8k)
    for forbidden in ("baseline_cache", "get_cached_baseline",
                      "set_cached_baseline", "load_baseline_cache",
                      "save_baseline_cache"):
        assert forbidden not in src, f"paired path references {forbidden}"


def test_paired_run_records_both_arms_for_every_case():
    """Pairing invariant: every case carries an outcome and an error slot for
    both arms, so no case can go half-measured and unnoticed."""
    src = inspect.getsource(run_paired_gsm8k)
    for key in ("routed_correct", "direct_correct", "routed_error",
                "direct_error", "strategy"):
        assert f'"{key}"' in src
    # case is appended once per golden, unconditionally, outside both arm handlers
    assert "cases.append(case)" in src


def test_compute_paired_verdict_wires_stats_into_suite():
    """Wiring check: the suite composes paired_stats end to end — delta, CI,
    tolerance, cowardice inputs, router accuracy, and a verdict with reasons."""
    from benchmarks.deepeval_suite import compute_paired_verdict

    cases = [
        {"routed_correct": True, "direct_correct": True, "strategy": "math",
         "routed_error": None, "direct_error": None},
        {"routed_correct": True, "direct_correct": False, "strategy": "math",
         "routed_error": None, "direct_error": None},
        {"routed_correct": False, "direct_correct": False, "strategy": "recall",
         "routed_error": None, "direct_error": None},
        {"routed_correct": False, "direct_correct": False, "strategy": "math",
         "routed_error": "429", "direct_error": None},
    ]
    r = compute_paired_verdict(cases)
    assert r["stats"]["n_clean"] == 3
    assert r["stats"]["delta"] is not None
    assert r["tolerance"]["tolerance"] is not None
    assert r["cowardice"]["non_direct_share"] == 1.0
    assert abs(r["router_accuracy"] - 2 / 3) < 1e-9
    # 3 clean cases < N_REQUIRED=100 — exploration-scale must read underpowered
    assert r["verdict"] == "underpowered"
    assert r["reasons"]


def test_compute_paired_verdict_no_clean_cases():
    from benchmarks.deepeval_suite import compute_paired_verdict

    r = compute_paired_verdict([
        {"routed_correct": False, "direct_correct": False, "strategy": "math",
         "routed_error": "boom", "direct_error": None}])
    assert r["stats"]["delta"] is None
    assert r["verdict"] == "underpowered"
