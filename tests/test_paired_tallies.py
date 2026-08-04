# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Tally logic and cache-bypass invariant for the paired routed-vs-direct GSM8K tracer."""

import inspect

from benchmarks.deepeval_suite import run_paired_benchmark, tally_paired


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
    src = inspect.getsource(run_paired_benchmark)
    for forbidden in ("baseline_cache", "get_cached_baseline",
                      "set_cached_baseline", "load_baseline_cache",
                      "save_baseline_cache"):
        assert forbidden not in src, f"paired path references {forbidden}"


def test_paired_run_records_both_arms_for_every_case():
    """Pairing invariant: every case carries an outcome and an error slot for
    both arms, so no case can go half-measured and unnoticed."""
    src = inspect.getsource(run_paired_benchmark)
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
    # "recall" is the cheap cot_lite path — it counts as direct-ish, not as
    # the router leaving the cheap path. 2 of 3 clean cases are off it.
    assert abs(r["cowardice"]["non_direct_share"] - 2 / 3) < 1e-9
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


def test_verdict_refused_when_arm_identities_do_not_match():
    """A verdict is a statement about ONE model — mismatched or unreported
    routed-arm identity must refuse, not emit."""
    from benchmarks.deepeval_suite import compute_paired_verdict

    base = {"routed_correct": True, "direct_correct": True, "strategy": "math",
            "routed_error": None, "direct_error": None}
    mismatched = compute_paired_verdict(
        [dict(base, routed_model="other/model")], pinned_model="pinned/model")
    assert mismatched["verdict"] == "refused-arm-mismatch"
    assert mismatched["arm_mismatch"]

    unreported = compute_paired_verdict(
        [dict(base, routed_model=None)], pinned_model="pinned/model")
    assert unreported["verdict"] == "refused-arm-mismatch"

    matched = compute_paired_verdict(
        [dict(base, routed_model="pinned/model")], pinned_model="pinned/model")
    assert matched["verdict"] != "refused-arm-mismatch"


def test_frozen_tolerance_overrides_derived():
    from benchmarks.deepeval_suite import compute_paired_verdict

    cases = [{"routed_correct": True, "direct_correct": False, "strategy": "math",
              "routed_error": None, "direct_error": None, "routed_model": "m"}]
    r = compute_paired_verdict(cases, pinned_model="m", frozen_tolerance=0.05)
    assert r["tolerance"]["tolerance"] == 0.05
    assert "frozen" in r["tolerance"]["basis"]


def test_paired_artifact_and_stats_sink_round_trip(tmp_path):
    """The run must survive the terminal session: JSON artifact + STATS.yaml
    entry with the fields needed to audit it."""
    import json

    import yaml

    from benchmarks.deepeval_suite import (
        compute_paired_verdict,
        tally_paired,
        update_paired_stats_yaml,
        write_paired_artifact,
    )

    cases = [{"routed_correct": True, "direct_correct": False, "strategy": "math",
              "routed_error": None, "direct_error": None, "routed_model": "m"}]
    payload = {
        "timestamp": "2026-08-04T00:00:00",
        "benchmark": "GSM8K",
        "provider": "test",
        "model": "m",
        "prompt_template": "shared-template",
        "n": 1,
        "frozen_tolerance_arg": 0.05,
        "cases": cases,
        "tallies": tally_paired(cases),
        "verdict_record": compute_paired_verdict(cases, pinned_model="m",
                                                 frozen_tolerance=0.05),
    }
    artifact = write_paired_artifact(payload, results_dir=str(tmp_path / "results"))
    with open(artifact) as f:
        loaded = json.load(f)
    assert loaded["model"] == "m"
    assert loaded["cases"][0]["routed_model"] == "m"
    assert loaded["verdict_record"]["verdict"] == "underpowered"

    stats_path = tmp_path / "STATS.yaml"
    update_paired_stats_yaml(payload, artifact, stats_path=str(stats_path))
    stats = yaml.safe_load(stats_path.read_text())["paired_gsm8k"]
    assert stats["artifact"] == artifact
    assert stats["model"] == "m"
    assert stats["tolerance"] == 0.05
    assert stats["verdict"] == "underpowered"


def _clean(strategy, routed, direct):
    return {"routed_correct": routed, "direct_correct": direct,
            "strategy": strategy, "routed_error": None, "direct_error": None}


def test_router_accuracy_uses_the_benchmarks_expected_strategies():
    """Widening invariant: ground truth is per-benchmark, not hardcoded "math".
    LogiQA cases classified "reasoning" are CORRECT routing; the pre-widening
    code scored them 0% accurate."""
    from benchmarks.deepeval_suite import compute_paired_verdict

    cases = [_clean("reasoning", True, False), _clean("reasoning", True, True),
             _clean("direct", False, True)]
    assert abs(compute_paired_verdict(cases, bench_key="logiqa")["router_accuracy"]
               - 2 / 3) < 1e-9
    # same cases read against GSM8K's ground truth ("math") score zero
    assert compute_paired_verdict(cases, bench_key="gsm8k")["router_accuracy"] == 0.0


def test_recall_class_benchmark_does_not_apply_the_cowardice_floor():
    """On TruthfulQA, staying on the cheap path IS correct (O-0009). An
    all-recall router must not be failed for cowardice — but must still be
    failed for a real regression."""
    from benchmarks.deepeval_suite import compute_paired_verdict
    from benchmarks.paired_stats import verdict

    stats = {"underpowered": False, "n_clean": 100, "n_required": 100,
             "ci_low": -0.01, "delta": 0.0}
    # all-direct router, no reasoning uplift: cowardice on a reasoning benchmark...
    assert verdict(stats, 0.05, 0.0, None, 1.0)["verdict"] == "fail-on-cowardice"
    # ...but correct behavior on a recall benchmark
    assert verdict(stats, 0.05, 0.0, None, 1.0,
                   apply_cowardice=False)["verdict"] == "pass"
    # the floor still bites on a recall benchmark
    regressed = dict(stats, ci_low=-0.20)
    assert verdict(regressed, 0.05, 0.0, None, 1.0,
                   apply_cowardice=False)["verdict"] == "fail-on-floor"

    r = compute_paired_verdict([_clean("recall", True, True)],
                               bench_key="truthfulqa")
    assert r["benchmark_class"] == "recall"


def test_gate_reduces_by_worst_case():
    from benchmarks.paired_stats import reduce_verdicts

    assert reduce_verdicts({"gsm8k": "pass", "logiqa": "pass"})["verdict"] == "pass"
    # one failing benchmark sinks the gate
    r = reduce_verdicts({"gsm8k": "pass", "logiqa": "fail-on-floor",
                         "truthfulqa": "underpowered"})
    assert r["verdict"] == "fail-on-floor"
    assert r["per_benchmark"]["gsm8k"] == "pass"
    # underpowered beats pass (an unmeasured benchmark is not a passing one)
    unmeasured = reduce_verdicts({"a": "pass", "b": "underpowered"})
    assert unmeasured["verdict"] == "underpowered"
    # a typo must never read as a pass
    assert reduce_verdicts({"a": "pass", "b": "paas"})["verdict"] == "paas"
    assert reduce_verdicts({})["verdict"] == "underpowered"


def test_release_gate_check_blocks_on_missing_stale_failing_or_ungated_model(tmp_path):
    """Cadence hook: absent or stale evidence blocks a release (UM-0500) —
    only a fresh pass on every pinned model lets it through."""
    from datetime import datetime, timedelta

    import yaml

    from benchmarks.deepeval_suite import check_release_gate
    from benchmarks.paired_stats import PINNED_SMALL_MODELS

    now = datetime(2026, 8, 4)
    p = tmp_path / "STATS.yaml"
    assert not check_release_gate(str(p), now=now)["ok"]  # file missing

    def write(gate):
        p.write_text(yaml.dump({"paired_gate": gate}))

    fresh = (now - timedelta(days=1)).isoformat()
    stale = (now - timedelta(days=200)).isoformat()
    write({"model_passes": {m: fresh for m in PINNED_SMALL_MODELS}})
    assert check_release_gate(str(p), now=now)["ok"]

    # staleness is per model: one fresh pass must not keep another alive
    write({"model_passes": {PINNED_SMALL_MODELS[0]: stale,
                            PINNED_SMALL_MODELS[1]: fresh}})
    r = check_release_gate(str(p), now=now)
    assert not r["ok"] and "stale" in r["reasons"][0]

    write({"model_passes": {PINNED_SMALL_MODELS[0]: fresh}})
    r = check_release_gate(str(p), now=now)
    assert not r["ok"] and "no passing gate on record" in r["reasons"][0]

    p.write_text(yaml.dump({"paired_gsm8k": {}}))
    assert not check_release_gate(str(p), now=now)["ok"]  # gate never run


def test_pinned_model_set_and_sample_count_are_declared_in_choices():
    """The gate is a statement about NAMED models at a PRE-REGISTERED n —
    both live in code and must be pinned in the decision record, or the gate
    can be silently widened after seeing results."""
    from pathlib import Path

    from benchmarks.paired_stats import N_REQUIRED, PINNED_SMALL_MODELS

    choices = (Path(__file__).parent.parent / "CHOICES.md").read_text()
    for model in PINNED_SMALL_MODELS:
        assert model in choices, f"{model} pinned in code but not in CHOICES.md"
    assert f"n>={N_REQUIRED}" in choices or f"n≥{N_REQUIRED}" in choices


def test_every_paired_benchmark_declares_its_gate_contract():
    from benchmarks.deepeval_suite import PAIRED_BENCHMARKS

    for key, spec in PAIRED_BENCHMARKS.items():
        assert spec["class"] in ("reasoning", "recall"), key
        assert spec["expected_strategies"], key
        assert callable(spec["load"]), key
    classes = {s["class"] for s in PAIRED_BENCHMARKS.values()}
    assert classes == {"reasoning", "recall"}, "gate must span both evidence classes"


def test_both_arms_share_one_prompt_and_record_served_model():
    """Static invariants: the routed arm is evaluated on the SAME prompt as the
    direct arm, and records the model each call reported serving."""
    src = inspect.getsource(run_paired_benchmark)
    assert "endpoint.evaluate(query=prompt" in src
    assert '"routed_model"' in src
