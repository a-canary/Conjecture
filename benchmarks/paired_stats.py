# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Paired statistics + two-sided verdict for the small-model do-no-harm gate.

Consumes per-case records from run_paired_gsm8k / tally_paired (deepeval_suite):
each case has routed_correct, direct_correct, strategy, routed_error, direct_error.

Statistics operate on clean cases only (neither arm errored) — an infra failure
is not a benchmark miss (UM-0500).

Validation bar N_REQUIRED=100 per STATISTICAL_REALITY_CHECK.md ("n=10-20 is for
exploration, n>=100 for validation"). Router accuracy floor 0.90 per the O-0009
gate (src/agent/task_router.py).
"""

from math import sqrt
from typing import Dict, List

N_REQUIRED = 100
Z_95 = 1.959963984540054
ROUTER_ACCURACY_FLOOR = 0.90


def paired_delta(cases: List[dict], n_required: int = N_REQUIRED) -> Dict:
    """Paired routed-minus-direct delta with 95% CI and underpowered determination.

    Per-case difference d_i = routed_correct - direct_correct in {-1, 0, +1};
    point delta = mean(d_i), CI = delta +/- z * sd/sqrt(n) (normal approx on
    paired differences). All values are proportions (pp/100).
    # ponytail: normal approx; exact McNemar interval if n_clean ever stays <30.
    """
    clean = [c for c in cases
             if not c.get("routed_error") and not c.get("direct_error")]
    n = len(clean)
    out = {
        "n": len(cases),
        "n_errored": len(cases) - n,
        "n_clean": n,
        "n_required": n_required,
        "underpowered": n < n_required,
        "delta": None,
        "ci_low": None,
        "ci_high": None,
        "sd": None,
    }
    if n == 0:
        return out
    diffs = [int(c["routed_correct"]) - int(c["direct_correct"]) for c in clean]
    delta = sum(diffs) / n
    var = sum((d - delta) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    sd = sqrt(var)
    half = Z_95 * sd / sqrt(n)
    out.update(delta=delta, sd=sd, ci_low=delta - half, ci_high=delta + half)
    return out


def non_inferiority_tolerance(stats: Dict) -> Dict:
    """Derive the non-inferiority tolerance from measured spread at n_required.

    Tolerance = the 95% CI half-width this run's measured sd would produce at
    the required sample count — the smallest regression the gate could resolve
    at the validation bar. Recorded with its basis so the threshold has a
    stated derivation rather than being picked by feel.

    Note: derived from the SAME run's sd, so at n_clean == n_required the floor
    reduces to delta >= 0. For a stable margin, freeze the tolerance from a
    reference run and pass that frozen value to verdict().
    """
    if stats["n_clean"] == 0 or stats["sd"] is None:
        return {"tolerance": None, "basis": "no clean cases; tolerance underivable"}
    half_at_required = Z_95 * stats["sd"] / sqrt(stats["n_required"])
    return {
        "tolerance": half_at_required,
        "basis": (f"95% CI half-width at n={stats['n_required']} "
                  f"with measured sd={stats['sd']:.4f}"),
    }


REASONING_STRATEGIES = frozenset({"math", "logic"})


def cowardice_metrics(cases: List[dict],
                      reasoning_strategies: frozenset = REASONING_STRATEGIES) -> Dict:
    """Derive the anti-cowardice inputs from per-case records.

    non_direct_share: fraction of clean cases routed to a non-"direct" strategy.
    reasoning_uplift: routed-minus-direct accuracy delta on the clean subset
    whose strategy is in reasoning_strategies (None if that subset is empty).
    """
    clean = [c for c in cases
             if not c.get("routed_error") and not c.get("direct_error")]
    if not clean:
        return {"non_direct_share": 0.0, "reasoning_uplift": None, "n_reasoning": 0}
    non_direct = sum(1 for c in clean if c["strategy"] != "direct")
    reasoning = [c for c in clean if c["strategy"] in reasoning_strategies]
    uplift = None
    if reasoning:
        uplift = (sum(int(c["routed_correct"]) - int(c["direct_correct"])
                      for c in reasoning) / len(reasoning))
    return {"non_direct_share": non_direct / len(clean),
            "reasoning_uplift": uplift,
            "n_reasoning": len(reasoning)}


def verdict(stats: Dict,
            tolerance: float,
            non_direct_share: float,
            reasoning_uplift: float,
            router_accuracy: float,
            router_accuracy_floor: float = ROUTER_ACCURACY_FLOOR) -> Dict:
    """Two-sided verdict: do-no-harm floor AND anti-cowardice floor AND router accuracy.

    Emits one of: underpowered / fail-on-floor / fail-on-cowardice / pass.

    - underpowered: below the validation bar — no verdict either way.
    - fail-on-floor (do-no-harm): CI lower bound below -tolerance.
    - fail-on-cowardice: router never leaves the direct path
      (non_direct_share <= 0), shows no uplift on the reasoning subset
      (reasoning_uplift <= 0), or misclassifies its way to safety
      (router_accuracy < floor). An all-direct router scores exactly equal to
      direct and would pass a bare floor by construction — this half makes the
      gate a measurement instead of a tautology.
    """
    reasons = []
    if stats["underpowered"] or stats["n_clean"] == 0:
        return {"verdict": "underpowered",
                "reasons": [f"n_clean={stats['n_clean']} < n_required={stats['n_required']}"]}
    if stats["ci_low"] < -tolerance:
        reasons.append(f"ci_low={stats['ci_low']:.4f} < -tolerance={-tolerance:.4f}")
        return {"verdict": "fail-on-floor", "reasons": reasons}
    if non_direct_share <= 0:
        reasons.append(f"non_direct_share={non_direct_share:.4f} <= 0 (all-direct router)")
    if reasoning_uplift is None:
        reasons.append("reasoning_uplift unmeasured (empty reasoning subset)")
    elif reasoning_uplift <= 0:
        reasons.append(f"reasoning_uplift={reasoning_uplift:.4f} <= 0")
    if router_accuracy < router_accuracy_floor:
        reasons.append(f"router_accuracy={router_accuracy:.4f} < {router_accuracy_floor}")
    if reasons:
        return {"verdict": "fail-on-cowardice", "reasons": reasons}
    return {"verdict": "pass", "reasons": []}
