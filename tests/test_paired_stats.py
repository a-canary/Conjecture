# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmarks.paired_stats: paired delta, derived tolerance, verdict."""

from math import sqrt

import pytest

from benchmarks.paired_stats import (
    N_REQUIRED,
    Z_95,
    cowardice_metrics,
    non_inferiority_tolerance,
    paired_delta,
    verdict,
)


def case(routed, direct, strategy="direct", routed_error=None, direct_error=None):
    return {"routed_correct": routed, "direct_correct": direct,
            "strategy": strategy, "routed_error": routed_error,
            "direct_error": direct_error}


class TestPairedDelta:
    def test_point_delta_and_counts(self):
        cases = [case(True, False)] * 30 + [case(False, True)] * 10 + [case(True, True)] * 60
        s = paired_delta(cases)
        assert s["n"] == 100 and s["n_clean"] == 100
        assert s["delta"] == pytest.approx(0.20)
        assert not s["underpowered"]

    def test_ci_matches_normal_approx(self):
        cases = [case(True, False)] * 30 + [case(False, True)] * 10 + [case(True, True)] * 60
        s = paired_delta(cases)
        diffs = [1] * 30 + [-1] * 10 + [0] * 60
        mean = sum(diffs) / 100
        sd = sqrt(sum((d - mean) ** 2 for d in diffs) / 99)
        half = Z_95 * sd / sqrt(100)
        assert s["ci_low"] == pytest.approx(mean - half)
        assert s["ci_high"] == pytest.approx(mean + half)

    def test_errored_cases_excluded_from_stats(self):
        cases = [case(True, True)] * 5 + [case(False, False, routed_error="boom")] * 3
        s = paired_delta(cases)
        assert s["n"] == 8 and s["n_clean"] == 5
        assert s["delta"] == 0.0

    def test_underpowered_below_bar(self):
        s = paired_delta([case(True, False)] * 99)
        assert s["underpowered"]
        assert s["n_required"] == N_REQUIRED

    def test_empty_and_single(self):
        assert paired_delta([])["underpowered"]
        assert paired_delta([])["delta"] is None
        s = paired_delta([case(True, False)], n_required=1)
        assert s["delta"] == 1.0 and s["sd"] == 0.0
        assert s["ci_low"] == s["ci_high"] == 1.0


class TestNonInferiorityTolerance:
    def test_derived_from_sd_at_required_n(self):
        cases = [case(True, False)] * 30 + [case(False, True)] * 10 + [case(True, True)] * 60
        s = paired_delta(cases)
        t = non_inferiority_tolerance(s)
        assert t["tolerance"] == pytest.approx(Z_95 * s["sd"] / sqrt(N_REQUIRED))
        assert "n=100" in t["basis"]

    def test_no_clean_cases(self):
        t = non_inferiority_tolerance(paired_delta([]))
        assert t["tolerance"] is None


def powered_stats(delta=0.05, ci_low=0.01, ci_high=0.09):
    return {"n": 100, "n_clean": 100, "n_required": 100, "underpowered": False,
            "delta": delta, "ci_low": ci_low, "ci_high": ci_high, "sd": 0.2}


class TestCowardiceMetrics:
    def test_shares_and_uplift(self):
        cases = ([case(True, False, "math")] * 3 + [case(False, True, "math")] * 1
                 + [case(True, True, "direct")] * 4
                 + [case(True, True, "logic", routed_error="boom")] * 2)
        m = cowardice_metrics(cases)
        assert m["non_direct_share"] == pytest.approx(0.5)  # 4 math of 8 clean
        assert m["n_reasoning"] == 4
        assert m["reasoning_uplift"] == pytest.approx(0.5)

    def test_all_direct_and_empty(self):
        m = cowardice_metrics([case(True, True, "direct")] * 5)
        assert m["non_direct_share"] == 0.0
        assert m["reasoning_uplift"] is None
        assert cowardice_metrics([])["non_direct_share"] == 0.0

    def test_unmeasured_uplift_fails_cowardice(self):
        v = verdict(powered_stats(), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=None, router_accuracy=0.95)
        assert v["verdict"] == "fail-on-cowardice"
        assert any("unmeasured" in r for r in v["reasons"])


class TestVerdict:
    def test_underpowered_wins_over_everything(self):
        s = powered_stats()
        s.update(underpowered=True, n_clean=50)
        v = verdict(s, 0.05, 0.0, -1.0, 0.0)
        assert v["verdict"] == "underpowered"

    def test_fail_on_floor(self):
        v = verdict(powered_stats(ci_low=-0.10), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=0.1, router_accuracy=0.95)
        assert v["verdict"] == "fail-on-floor"

    def test_ci_low_inside_tolerance_passes_floor(self):
        v = verdict(powered_stats(ci_low=-0.04), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=0.1, router_accuracy=0.95)
        assert v["verdict"] == "pass"

    def test_all_direct_router_fails_cowardice(self):
        v = verdict(powered_stats(delta=0.0, ci_low=0.0, ci_high=0.0), tolerance=0.05,
                    non_direct_share=0.0, reasoning_uplift=0.0, router_accuracy=1.0)
        assert v["verdict"] == "fail-on-cowardice"
        assert any("non_direct_share" in r for r in v["reasons"])

    def test_zero_reasoning_uplift_fails(self):
        v = verdict(powered_stats(), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=0.0, router_accuracy=0.95)
        assert v["verdict"] == "fail-on-cowardice"

    def test_router_accuracy_below_floor_fails(self):
        v = verdict(powered_stats(), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=0.1, router_accuracy=0.85)
        assert v["verdict"] == "fail-on-cowardice"
        assert any("router_accuracy" in r for r in v["reasons"])

    def test_pass(self):
        v = verdict(powered_stats(), tolerance=0.05,
                    non_direct_share=0.4, reasoning_uplift=0.1, router_accuracy=0.95)
        assert v == {"verdict": "pass", "reasons": []}

    def test_floor_checked_before_cowardice(self):
        v = verdict(powered_stats(ci_low=-0.10), tolerance=0.05,
                    non_direct_share=0.0, reasoning_uplift=0.0, router_accuracy=0.0)
        assert v["verdict"] == "fail-on-floor"
