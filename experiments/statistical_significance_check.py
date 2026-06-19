#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Statistical Significance Analysis

Check if our key findings are statistically significant or just noise.
Use proper hypothesis testing with p-values.
"""

import math
from scipy import stats


def binomial_proportion_test(n1, p1, n2, p2, test_name):
    """
    Test if two proportions are significantly different.

    H0: p1 = p2 (no difference)
    H1: p1 ≠ p2 (significant difference)
    """
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")

    # Calculate standard error
    if p1 == 1.0:
        p1 = 0.999  # Avoid division by zero
    if p2 == 1.0:
        p2 = 0.999
    if p1 == 0.0:
        p1 = 0.001
    if p2 == 0.0:
        p2 = 0.001

    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

    # Calculate z-score
    diff = p2 - p1
    if se > 0:
        z = diff / se
        p_value = 2 * stats.norm.cdf(-abs(z))
    else:
        z = float('inf') if diff != 0 else 0
        p_value = 0.0 if diff != 0 else 1.0

    # Calculate 95% confidence interval
    ci_margin = 1.96 * se
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin

    print(f"Sample sizes: n1={n1}, n2={n2}")
    print(f"Proportions: p1={p1:.3f}, p2={p2:.3f}")
    print(f"Difference: {diff*100:+.1f}pp")
    print(f"Standard error: {se:.4f}")
    print(f"Z-score: {z:.3f}")
    print(f"P-value: {p_value:.4f}")
    print(f"95% CI: [{ci_lower*100:.1f}pp, {ci_upper*100:.1f}pp]")

    # Interpretation
    if p_value < 0.001:
        verdict = "⭐⭐⭐ HIGHLY SIGNIFICANT"
    elif p_value < 0.01:
        verdict = "⭐⭐ VERY SIGNIFICANT"
    elif p_value < 0.05:
        verdict = "⭐ SIGNIFICANT"
    elif p_value < 0.10:
        verdict = "⚠️  MARGINALLY SIGNIFICANT"
    else:
        verdict = "❌ NOT SIGNIFICANT (could be noise)"

    print(f"\nVerdict: {verdict}")

    # Sample size adequacy
    margin_of_error = ci_margin * 100
    print(f"Margin of error: ±{margin_of_error:.1f}pp (95% confidence)")

    if margin_of_error > 20:
        print(f"⚠️  WARNING: Large margin of error! Need n≥{int(1/(0.05**2))} for ±10pp")
    elif margin_of_error > 10:
        print(f"⚠️  Moderate margin of error. Consider n≥{int(1/(0.05**2))} for ±10pp")
    else:
        print(f"✅ Acceptable margin of error")

    return {
        "test": test_name,
        "n1": n1,
        "n2": n2,
        "p1": p1,
        "p2": p2,
        "diff_pp": diff * 100,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "verdict": verdict,
        "margin_of_error": margin_of_error
    }


def main():
    print("="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Checking if our findings are statistically significant or just noise")
    print("="*80)

    results = []

    # 1. Word count optimization (CRITICAL FINDING)
    print("\n" + "="*80)
    print("PHASE 2: WORD COUNT OPTIMIZATION")
    print("="*80)

    # 5-word vs 15-word (claimed +25pp improvement)
    r1 = binomial_proportion_test(
        n1=20, p1=0.40,  # 15-word: 8/20 = 40%
        n2=20, p2=0.65,  # 5-word: 13/20 = 65%
        test_name="5-word vs 15-word claims"
    )
    results.append(r1)

    # 5-word vs 10-word
    r2 = binomial_proportion_test(
        n1=20, p1=0.45,  # 10-word: 9/20 = 45%
        n2=20, p2=0.65,  # 5-word: 13/20 = 65%
        test_name="5-word vs 10-word claims"
    )
    results.append(r2)

    # 2. Multi-benchmark validation
    print("\n" + "="*80)
    print("PHASE 3: MULTI-BENCHMARK VALIDATION")
    print("="*80)

    # HellaSwag regression (CATASTROPHIC)
    r3 = binomial_proportion_test(
        n1=20, p1=0.55,  # Baseline: 11/20 = 55%
        n2=20, p2=0.15,  # With claims: 3/20 = 15%
        test_name="HellaSwag: baseline vs 5-word claims"
    )
    results.append(r3)

    # TruthfulQA improvement
    r4 = binomial_proportion_test(
        n1=20, p1=0.55,  # Baseline: 11/20 = 55%
        n2=20, p2=0.65,  # With claims: 13/20 = 65%
        test_name="TruthfulQA: baseline vs 5-word claims"
    )
    results.append(r4)

    # BBH-causal regression
    r5 = binomial_proportion_test(
        n1=20, p1=0.45,  # Baseline: 9/20 = 45%
        n2=20, p2=0.25,  # With claims: 5/20 = 25%
        test_name="BBH-causal: baseline vs 5-word claims"
    )
    results.append(r5)

    # 3. Original Goldilocks discovery (Phase 1)
    print("\n" + "="*80)
    print("PHASE 1: ORIGINAL GOLDILOCKS DISCOVERY")
    print("="*80)

    # 1-3 claims vs 0 claims (claimed +10pp)
    r6 = binomial_proportion_test(
        n1=10, p1=0.90,  # 0 claims: 9/10 = 90%
        n2=10, p2=1.00,  # 2 claims: 10/10 = 100%
        test_name="BBH: 0 claims vs 2 claims (original discovery)"
    )
    results.append(r6)

    # 4. Claim selection optimization
    print("\n" + "="*80)
    print("PHASE 2: CLAIM SELECTION")
    print("="*80)

    # Keyword vs random (claimed +5pp)
    r7 = binomial_proportion_test(
        n1=20, p1=0.45,  # Random: 9/20 = 45%
        n2=20, p2=0.50,  # Keyword: 10/20 = 50%
        test_name="Claim selection: random vs keyword matching"
    )
    results.append(r7)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: STATISTICAL SIGNIFICANCE")
    print("="*80)
    print(f"{'Finding':<50} {'Diff':<10} {'P-value':<12} {'Significant?'}")
    print("-"*80)

    for r in results:
        sig_marker = "✅ YES" if r['significant'] else "❌ NO"
        print(f"{r['test']:<50} {r['diff_pp']:>+6.1f}pp   {r['p_value']:>8.4f}    {sig_marker}")

    print("="*80)

    # Overall assessment
    significant_count = sum(1 for r in results if r['significant'])
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Significant findings: {significant_count}/{len(results)} ({100*significant_count/len(results):.0f}%)")

    highly_sig = sum(1 for r in results if r['p_value'] < 0.01)
    print(f"  Highly significant (p<0.01): {highly_sig}/{len(results)}")

    # Warnings
    print(f"\n⚠️  WARNINGS:")
    print(f"  - Small sample sizes (n=10-20) yield large margins of error (±20-30pp)")
    print(f"  - Need n≥100 for reliable ±10pp confidence intervals")
    print(f"  - Some 'significant' results may be false positives")
    print(f"  - Recommend validation with larger samples before production")

    # Recommendations
    print(f"\n📊 RECOMMENDATIONS:")
    print(f"  1. Re-run word count optimization with n=50-100 (current: n=20)")
    print(f"  2. Re-run multi-benchmark with n=50-100 (current: n=20)")
    print(f"  3. Original Goldilocks needs n=50 minimum (current: n=10)")
    print(f"  4. Apply Bonferroni correction for multiple testing")
    print(f"  5. Use holdout validation set for final claims")


if __name__ == "__main__":
    main()
