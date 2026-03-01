#!/usr/bin/env python3
"""
Statistical Analysis Tools for Baseline Comparisons
Advanced statistical methods for comparing Conjecture vs baseline approaches
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, mannwhitneyu
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StatisticalTest:
    """Results of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    interpretation: str

@dataclass
class ABTestResult:
    """Results of A/B test statistical analysis"""
    comparison_name: str
    sample_size_a: int
    sample_size_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_difference: float
    statistical_tests: Dict[str, StatisticalTest]
    recommendation: str
    confidence: float

class StatisticalAnalyzer:
    """Advanced statistical analysis for baseline comparisons"""

    def __init__(self, alpha: float = 0.05, effect_size_threshold: float = 0.2):
        self.alpha = alpha
        self.effect_size_threshold = effect_size_threshold

    def calculate_cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d

    def calculate_hedges_g(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)"""
        cohens_d = self.calculate_cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        
        # Correction factor for small sample bias
        correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * correction_factor
        
        return hedges_g

    def calculate_confidence_interval(self, 
                                    mean_diff: float, 
                                    std_diff: float, 
                                    n: int, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference"""
        df = n - 1 if n > 1 else 1
        t_critical = stats.t.ppf((1 + confidence) / 2, df)
        
        standard_error = std_diff / np.sqrt(n) if n > 0 else 0
        margin_error = t_critical * standard_error
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return (ci_lower, ci_upper)

    def paired_t_test(self, 
                     group1: List[float], 
                     group2: List[float]) -> StatisticalTest:
        """Perform paired t-test for related samples"""
        if len(group1) != len(group2) or len(group1) < 2:
            return StatisticalTest(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation="Insufficient data for test"
            )
        
        try:
            statistic, p_value = ttest_rel(group1, group2)
            effect_size = self.calculate_hedges_g(group1, group2)
            
            # Calculate confidence interval for difference
            differences = [a - b for a, b in zip(group1, group2)]
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            ci = self.calculate_confidence_interval(mean_diff, std_diff, len(differences))
            
            is_significant = p_value < self.alpha
            effect_size_interpretation = self._interpret_effect_size(effect_size)
            
            interpretation = f"Paired t-test: t({len(group1)-1}) = {statistic:.3f}, p = {p_value:.4f}, "
            interpretation += f"Hedges' g = {effect_size:.3f} ({effect_size_interpretation}), "
            interpretation += f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]. "
            interpretation += "Significant difference" if is_significant else "No significant difference"
            
            return StatisticalTest(
                test_name="Paired t-test",
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                is_significant=is_significant,
                confidence_interval=ci,
                interpretation=interpretation
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation=f"Test failed: {str(e)}"
            )

    def independent_t_test(self, 
                         group1: List[float], 
                         group2: List[float]) -> StatisticalTest:
        """Perform independent t-test for unrelated samples"""
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalTest(
                test_name="Independent t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation="Insufficient data for test"
            )
        
        try:
            statistic, p_value = ttest_ind(group1, group2)
            effect_size = self.calculate_hedges_g(group1, group2)
            
            # Calculate confidence interval for difference
            mean_diff = np.mean(group1) - np.mean(group2)
            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                 (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                (len(group1) + len(group2) - 2))
            n_total = len(group1) + len(group2)
            ci = self.calculate_confidence_interval(mean_diff, pooled_std, n_total)
            
            is_significant = p_value < self.alpha
            effect_size_interpretation = self._interpret_effect_size(effect_size)
            
            interpretation = f"Independent t-test: t({n_total-2}) = {statistic:.3f}, p = {p_value:.4f}, "
            interpretation += f"Hedges' g = {effect_size:.3f} ({effect_size_interpretation}), "
            interpretation += f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]. "
            interpretation += "Significant difference" if is_significant else "No significant difference"
            
            return StatisticalTest(
                test_name="Independent t-test",
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                is_significant=is_significant,
                confidence_interval=ci,
                interpretation=interpretation
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name="Independent t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation=f"Test failed: {str(e)}"
            )

    def wilcoxon_signed_rank_test(self, 
                                group1: List[float], 
                                group2: List[float]) -> StatisticalTest:
        """Perform Wilcoxon signed-rank test for non-parametric paired data"""
        if len(group1) != len(group2) or len(group1) < 2:
            return StatisticalTest(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation="Insufficient data for test"
            )
        
        try:
            statistic, p_value = wilcoxon(group1, group2)
            
            # Calculate rank-biserial correlation as effect size
            n = len(group1)
            effect_size = (statistic - (n * (n + 1) / 4)) / (n * (n + 1) / 4)
            
            is_significant = p_value < self.alpha
            effect_size_interpretation = self._interpret_effect_size(abs(effect_size))
            
            interpretation = f"Wilcoxon signed-rank test: W = {statistic:.1f}, p = {p_value:.4f}, "
            interpretation += f"rank-biserial r = {effect_size:.3f} ({effect_size_interpretation}). "
            interpretation += "Significant difference" if is_significant else "No significant difference"
            
            return StatisticalTest(
                test_name="Wilcoxon signed-rank test",
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                is_significant=is_significant,
                confidence_interval=(0.0, 0.0),  # Not typically calculated for non-parametric tests
                interpretation=interpretation
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation=f"Test failed: {str(e)}"
            )

    def mann_whitney_u_test(self, 
                          group1: List[float], 
                          group2: List[float]) -> StatisticalTest:
        """Perform Mann-Whitney U test for non-parametric independent data"""
        if len(group1) < 2 or len(group2) < 2:
            return StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation="Insufficient data for test"
            )
        
        try:
            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Calculate rank-biserial correlation as effect size
            n1, n2 = len(group1), len(group2)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            
            is_significant = p_value < self.alpha
            effect_size_interpretation = self._interpret_effect_size(abs(effect_size))
            
            interpretation = f"Mann-Whitney U test: U = {statistic:.1f}, p = {p_value:.4f}, "
            interpretation += f"rank-biserial r = {effect_size:.3f} ({effect_size_interpretation}). "
            interpretation += "Significant difference" if is_significant else "No significant difference"
            
            return StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                is_significant=is_significant,
                confidence_interval=(0.0, 0.0),  # Not typically calculated for non-parametric tests
                interpretation=interpretation
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                interpretation=f"Test failed: {str(e)}"
            )

    def analyze_ab_test(self, 
                       group_a_scores: List[float], 
                       group_b_scores: List[float],
                       comparison_name: str,
                       is_paired: bool = True) -> ABTestResult:
        """Comprehensive A/B test analysis"""
        
        # Basic descriptive statistics
        n_a, n_b = len(group_a_scores), len(group_b_scores)
        mean_a, mean_b = np.mean(group_a_scores), np.mean(group_b_scores)
        std_a, std_b = np.std(group_a_scores, ddof=1), np.std(group_b_scores, ddof=1)
        mean_difference = mean_a - mean_b
        
        # Run appropriate statistical tests
        statistical_tests = {}
        
        if is_paired:
            # For paired data (same test cases compared across conditions)
            statistical_tests['paired_t'] = self.paired_t_test(group_a_scores, group_b_scores)
            statistical_tests['wilcoxon'] = self.wilcoxon_signed_rank_test(group_a_scores, group_b_scores)
        else:
            # For independent data
            statistical_tests['independent_t'] = self.independent_t_test(group_a_scores, group_b_scores)
            statistical_tests['mann_whitney'] = self.mann_whitney_u_test(group_a_scores, group_b_scores)
        
        # Determine recommendation based on tests
        significant_tests = [test for test in statistical_tests.values() if test.is_significant]
        
        if len(significant_tests) == 0:
            recommendation = "NO SIGNIFICANT DIFFERENCE - No evidence to prefer either approach"
            confidence = 0.0
        elif mean_difference > 0:
            recommendation = f"GROUP A PREFERRED - Significantly better performance ({mean_difference:+.3f})"
            confidence = np.mean([test.effect_size for test in significant_tests])
        else:
            recommendation = f"GROUP B PREFERRED - Significantly better performance ({mean_difference:+.3f})"
            confidence = np.mean([abs(test.effect_size) for test in significant_tests])
        
        return ABTestResult(
            comparison_name=comparison_name,
            sample_size_a=n_a,
            sample_size_b=n_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            mean_difference=mean_difference,
            statistical_tests=statistical_tests,
            recommendation=recommendation,
            confidence=min(1.0, abs(confidence))
        )

    def analyze_baseline_comparisons(self, 
                                   comparison_results: List[Dict[str, Any]]) -> Dict[str, ABTestResult]:
        """Analyze baseline comparison results with statistical tests"""
        
        # Organize results by baseline type and model
        baseline_results = {}
        
        for result in comparison_results:
            baseline_type = result.get('baseline_result', {}).get('metadata', {}).get('baseline_type', 'unknown')
            model_name = result.get('model_name', 'unknown')
            winner = result.get('winner', 'unknown')
            
            key = f"{baseline_type}_{model_name}"
            
            if key not in baseline_results:
                baseline_results[key] = []
            
            baseline_results[key].append(result)
        
        # Perform statistical analysis for each baseline type
        analyses = {}
        
        for key, results in baseline_results.items():
            if len(results) < 2:
                continue
            
            # Extract scores for each condition
            conjecture_scores = []
            baseline_scores = []
            
            for result in results:
                # Calculate overall score for each approach
                conj_evals = result.get('conjecture_evaluations', {})
                base_evals = result.get('baseline_evaluations', {})
                
                conj_score = self._calculate_overall_score(conj_evals)
                base_score = self._calculate_overall_score(base_evals)
                
                conjecture_scores.append(conj_score)
                baseline_scores.append(base_score)
            
            # Run A/B test analysis
            analysis = self.analyze_ab_test(
                group_a_scores=conjecture_scores,
                group_b_scores=baseline_scores,
                comparison_name=key,
                is_paired=True
            )
            
            analyses[key] = analysis
        
        return analyses

    def _calculate_overall_score(self, evaluations: Dict[str, Any]) -> float:
        """Calculate overall score from evaluation results"""
        if not evaluations:
            return 0.5
        
        scores = []
        for criterion, eval_result in evaluations.items():
            if hasattr(eval_result, 'score'):
                scores.append(eval_result.score)
            elif hasattr(eval_result, 'final_score'):
                scores.append(eval_result.final_score)
            elif isinstance(eval_result, dict):
                scores.append(eval_result.get('score', eval_result.get('final_score', 0.5)))
        
        return np.mean(scores) if scores else 0.5

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    def generate_statistical_summary(self, 
                                   analyses: Dict[str, ABTestResult]) -> str:
        """Generate comprehensive statistical summary report"""
        
        if not analyses:
            return "No statistical analyses available."
        
        summary = "# Statistical Analysis Summary\n\n"
        summary += f"Analysis performed at α = {self.alpha} significance level\n"
        summary += f"Effect size threshold: {self.effect_size_threshold}\n\n"
        
        # Overall summary
        total_comparisons = len(analyses)
        significant_results = sum(1 for a in analyses.values() 
                                if any(test.is_significant for test in a.statistical_tests.values()))
        
        summary += f"## Overview\n"
        summary += f"- Total comparisons: {total_comparisons}\n"
        summary += f"- Significant results: {significant_results} ({significant_results/total_comparisons*100:.1f}%)\n\n"
        
        # Detailed results by comparison
        for comparison, result in analyses.items():
            summary += f"## {comparison.replace('_', ' ').title()}\n\n"
            summary += f"**Sample Sizes**: Conjecture n={result.sample_size_a}, Baseline n={result.sample_size_b}\n"
            summary += f"**Performance**: Conjecture μ={result.mean_a:.3f}±{result.std_a:.3f}, "
            summary += f"Baseline μ={result.mean_b:.3f}±{result.std_b:.3f}\n"
            summary += f"**Mean Difference**: {result.mean_difference:+.3f}\n\n"
            
            summary += "### Statistical Tests\n"
            for test_name, test_result in result.statistical_tests.items():
                summary += f"- **{test_name}**: {test_result.interpretation}\n"
            
            summary += f"\n### Recommendation\n"
            summary += f"**{result.recommendation}** (Confidence: {result.confidence:.2f})\n\n"
            
            summary += "---\n\n"
        
        # Meta-analysis across all comparisons
        all_effects = []
        for result in analyses.values():
            for test in result.statistical_tests.values():
                if test.is_significant:
                    all_effects.append(abs(test.effect_size))
        
        if all_effects:
            mean_effect_size = np.mean(all_effects)
            median_effect_size = np.median(all_effects)
            
            summary += "## Meta-Analysis\n\n"
            summary += f"**Mean effect size** (significant results): {mean_effect_size:.3f}\n"
            summary += f"**Median effect size** (significant results): {median_effect_size:.3f}\n"
            
            if mean_effect_size > 0.5:
                summary += "- **Overall**: Strong evidence of Conjecture effectiveness\n"
            elif mean_effect_size > 0.2:
                summary += "- **Overall**: Moderate evidence of Conjecture effectiveness\n"
            else:
                summary += "- **Overall**: Limited evidence of Conjecture effectiveness\n"
        
        return summary

def main():
    """Example usage of statistical analyzer"""
    # Example data
    conjecture_scores = [0.8, 0.7, 0.9, 0.75, 0.85, 0.72, 0.88, 0.79]
    baseline_scores = [0.6, 0.65, 0.7, 0.58, 0.68, 0.62, 0.71, 0.64]
    
    analyzer = StatisticalAnalyzer()
    result = analyzer.analyze_ab_test(
        group_a_scores=conjecture_scores,
        group_b_scores=baseline_scores,
        comparison_name="example_comparison",
        is_paired=True
    )
    
    print(f"Comparison: {result.comparison_name}")
    print(f"Mean A (Conjecture): {result.mean_a:.3f}")
    print(f"Mean B (Baseline): {result.mean_b:.3f}")
    print(f"Recommendation: {result.recommendation}")
    print("\nStatistical tests:")
    for test_name, test_result in result.statistical_tests.items():
        print(f"  {test_name}: {test_result.interpretation}")

if __name__ == "__main__":
    main()