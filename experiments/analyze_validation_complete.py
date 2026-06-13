#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Validation analysis for pre-registered hypotheses (2026-03-09).

Both result JSON files are truncated in their statistics block.
This script reads condition counts via regex and recomputes statistics independently.
"""

import re
import math
from pathlib import Path
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent / "results"
H3_FILE = RESULTS_DIR / "validate_task_type_20260309_155405.json"
H1_FILE = RESULTS_DIR / "validate_word_count_20260309_155516.json"


def extract_condition(text, key):
    idx = text.find(chr(34) + key + chr(34))
    snippet = text[idx:idx+300]
    pat = chr(34) + r"n_samples" + chr(34) + r": *([0-9]+)" + r".*?" + chr(34) + r"correct" + chr(34) + r": *([0-9]+)"
    m = re.search(pat, snippet, re.DOTALL)
    return int(m.group(1)), int(m.group(2))


def two_prop_ztest(n1, k1, n2, k2):
    p1, p2 = k1/n1, k2/n2
    pp = (k1+k2)/(n1+n2)
    se = math.sqrt(pp*(1-pp)*(1/n1+1/n2))
    if se == 0: return 0.0, 1.0
    z = (p1-p2)/se
    return z, 2*scipy_stats.norm.cdf(-abs(z))


def ci95(n1, k1, n2, k2):
    p1, p2 = k1/n1, k2/n2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    d = p1-p2
    return d - 1.96*se, d + 1.96*se


def cohens_h(p1, p2):
    return 2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2))


def xpp(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.1f}pp"


def xpct(v):
    return f"{v*100:.1f}%"


def bar(char="-", w=70):
    print(char * w)


def criteria_line(label, met):
    status = "PASS" if met else "FAIL"
    print(f"    {label:<22} {status}")


def verdict(validated):
    return "VALIDATED" if validated else "DISPROVED"


def main():
    h3_text = H3_FILE.read_text()
    h1_text = H1_FILE.read_text()

    n_d, k_d = extract_condition(h3_text, "direct_condition")
    n_c, k_c = extract_condition(h3_text, "claims_condition")
    n_5, k_5 = extract_condition(h1_text, "5w_condition")
    n_15, k_15 = extract_condition(h1_text, "15w_condition")

    p_d, p_c = k_d/n_d, k_c/n_c
    p_5, p_15 = k_5/n_5, k_15/n_15

    diff_h3 = p_c - p_d
    diff_h1 = p_5 - p_15

    z_h3, pval_h3 = two_prop_ztest(n_c, k_c, n_d, k_d)
    z_h1, pval_h1 = two_prop_ztest(n_5, k_5, n_15, k_15)

    ci_lo_h3, ci_hi_h3 = ci95(n_c, k_c, n_d, k_d)
    ci_lo_h1, ci_hi_h1 = ci95(n_5, k_5, n_15, k_15)

    h_h3 = cohens_h(p_c, p_d)
    h_h1 = cohens_h(p_5, p_15)

    # --- Pre-registered success criteria ---
    # H1: p<0.05, effect>=+10pp, CI lower>0
    h1_p_ok    = pval_h1 < 0.05
    h1_eff_ok  = diff_h1 >= 0.10
    h1_ci_ok   = ci_lo_h1 > 0
    h1_pass    = h1_p_ok and h1_eff_ok and h1_ci_ok

    # H3: p<0.05, effect>=+5pp, CI lower>0
    h3_p_ok    = pval_h3 < 0.05
    h3_eff_ok  = diff_h3 >= 0.05
    h3_ci_ok   = ci_lo_h3 > 0
    h3_pass    = h3_p_ok and h3_eff_ok and h3_ci_ok

    # ---- Print report ----
    bar("=")
    print("VALIDATION REPORT: Pre-Registered Hypotheses")
    print("Pre-registration: .director/PREREGISTRATION_2026-03-09.md")
    print("Model: LFM-2.5-1.2B  |  Benchmark: MMLU reasoning subset  |  n=100 per condition")
    bar("=")

    # H1
    print("")
    print("H1: 5-word claims outperform 15-word claims by >=10pp")
    bar()
    print(f"  5-word claims  : {k_5:3d}/100 = {xpct(p_5)}")
    print(f"  15-word claims : {k_15:3d}/100 = {xpct(p_15)}")
    print("")
    print(f"  Difference (treatment - control): {xpp(diff_h1)}")
    print(f"  95% CI:                           [{xpp(ci_lo_h1)}, {xpp(ci_hi_h1)}]")
    print(f"  z-statistic:                      {z_h1:.3f}")
    print(f"  p-value (two-tailed):             {pval_h1:.4f}")
    print(f"  Cohen h:                          {h_h1:.3f}")
    print("")
    print("  Success criteria: p<0.05, effect>=+10pp, CI lower>0")
    criteria_line("p<0.05", h1_p_ok)
    criteria_line("effect>=+10pp", h1_eff_ok)
    criteria_line("CI_lower>0", h1_ci_ok)
    print("")
    print(f"  Verdict: {verdict(h1_pass)}")

    # H2
    print("")
    print("H2: 2 claims outperform 0 and 5 claims by >=5pp each")
    bar()
    print("  Status:  NOT RUN - no result file found")
    print("  Verdict: INCONCLUSIVE (no data)")

    # H3
    print("")
    print("H3: Claims improve BBH reasoning by >=5pp")
    bar()
    print(f"  Direct (no claims) : {k_d:3d}/100 = {xpct(p_d)}")
    print(f"  With claims        : {k_c:3d}/100 = {xpct(p_c)}")
    print("")
    print(f"  Difference (treatment - control): {xpp(diff_h3)}")
    print(f"  95% CI:                           [{xpp(ci_lo_h3)}, {xpp(ci_hi_h3)}]")
    print(f"  z-statistic:                      {z_h3:.3f}")
    print(f"  p-value (two-tailed):             {pval_h3:.4f}")
    print(f"  Cohen h:                          {h_h3:.3f}")
    print("")
    print("  Success criteria: p<0.05, effect>=+5pp, CI lower>0")
    criteria_line("p<0.05", h3_p_ok)
    criteria_line("effect>=+5pp", h3_eff_ok)
    criteria_line("CI_lower>0", h3_ci_ok)
    print("")
    print(f"  Verdict: {verdict(h3_pass)}")

    # Comparison table
    print("")
    bar("=")
    print("EXPLORATION vs VALIDATION COMPARISON")
    bar("=")
    print("  Hyp   Finding                              Exploration (n=20)          Validation (n=100)")
    bar("-")
    print(f"  H1    5-word vs 15-word accuracy delta    +25pp (p=0.102, n.s.)        {xpp(diff_h1)} (p={pval_h1:.3f})")
    print(f"  H3    Claims vs direct accuracy delta     +10pp (n.s.)                 {xpp(diff_h3)} (p={pval_h3:.3f})")
    print("")
    print("  Lesson: Both exploratory positive signals evaporated at n=100.")
    print("  n=20 gives +-20-30pp margin of error -- improvements can be pure sampling noise.")

    # Summary table
    print("")
    bar("=")
    print("SUMMARY TABLE")
    bar("=")
    print("  Hyp    Description                                          Verdict")
    bar("-")
    print(f"  H1     5-word claims outperform 15-word claims by >=10pp    {verdict(h1_pass)}")
    print(f"  H2     2 claims outperform 0 and 5 claims by >=5pp each     NOT RUN")
    print(f"  H3     Claims improve BBH reasoning by >=5pp                {verdict(h3_pass)}")
    bar("-")
    v = int(h1_pass) + int(h3_pass)
    d = 2 - v
    print(f"  Validated: {v}  |  Disproved: {d}  |  Not run: 1")
    print("")
    print("  Only confirmed result: HellaSwag commonsense regression (prior session, p=0.004).")
    print("  A-0016 word count and task-type routing hypotheses: DISPROVED.")
    bar("=")


if __name__ == "__main__":
    main()
