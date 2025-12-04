#!/usr/bin/env python3
"""
Statistical Analysis Tools for Conjecture Hypothesis Testing
Comprehensive statistical validation and analysis framework
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StatisticalTest:
    """Results of a statistical test"""

    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float]
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    assumptions: List[str]


@dataclass
class PowerAnalysis:
    """Statistical power analysis results"""

    sample_size: int
    effect_size: float
    alpha: float
    power: float
    recommendation: str


@dataclass
class ComparisonResult:
    """Results of comparing two conditions"""

    condition_a: str
    condition_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    percent_improvement: float
    statistical_test: StatisticalTest
    practical_significance: bool
    recommendation: str


class ConjectureStatisticalAnalyzer:
    """Comprehensive statistical analysis for Conjecture hypothesis testing"""

    def __init__(self, results_dir: str = "research/results"):
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path("research/analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        # Statistical parameters
        self.alpha_level = 0.05  # Significance level
        self.power_target = 0.8  # Desired statistical power
        self.effect_size_thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

    def analyze_hypothesis_results(
        self,
        hypothesis_id: str,
        test_results: List[Dict[str, Any]],
        success_criteria: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comprehensive analysis of hypothesis test results"""
        analysis = {
            "hypothesis_id": hypothesis_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "sample_size": len(test_results),
            "descriptive_stats": self._calculate_descriptive_stats(test_results),
            "statistical_tests": self._run_statistical_tests(
                test_results, success_criteria
            ),
            "power_analysis": self._calculate_power_analysis(
                test_results, success_criteria
            ),
            "effect_sizes": self._calculate_effect_sizes(test_results),
            "practical_significance": self._assess_practical_significance(
                test_results, success_criteria
            ),
            "assumptions_check": self._check_statistical_assumptions(test_results),
            "recommendations": self._generate_recommendations(
                test_results, success_criteria
            ),
        }

        return analysis

    def _calculate_descriptive_stats(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate descriptive statistics for test results"""
        if not test_results:
            return {}

        # Extract numeric metrics
        metrics = {}
        for result in test_results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)

        descriptive_stats = {}
        for metric, values in metrics.items():
            if values:
                descriptive_stats[metric] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                    "iqr": np.percentile(values, 75) - np.percentile(values, 25),
                    "cv": statistics.stdev(values) / statistics.mean(values)
                    if statistics.mean(values) != 0
                    else 0.0,  # Coefficient of variation
                }

        return descriptive_stats

    def _run_statistical_tests(
        self, test_results: List[Dict[str, Any]], success_criteria: Dict[str, Any]
    ) -> Dict[str, StatisticalTest]:
        """Run appropriate statistical tests based on hypothesis type"""
        tests = {}

        # Group results by condition (e.g., direct vs conjecture)
        conditions = self._group_by_condition(test_results)

        if len(conditions) >= 2:
            condition_names = list(conditions.keys())

            # Paired t-test for comparing two conditions
            if len(conditions) == 2:
                for metric in success_criteria.get("metrics", ["correctness"]):
                    if self._can_run_test(
                        conditions[condition_names[0]],
                        conditions[condition_names[1]],
                        metric,
                    ):
                        test_result = self._paired_t_test(
                            conditions[condition_names[0]],
                            conditions[condition_names[1]],
                            metric,
                            condition_names[0],
                            condition_names[1],
                        )
                        tests[f"{metric}_paired_t_test"] = test_result

            # ANOVA for comparing multiple conditions
            elif len(conditions) > 2:
                for metric in success_criteria.get("metrics", ["correctness"]):
                    if all(
                        self._can_run_test(cond, None, metric)
                        for cond in conditions.values()
                    ):
                        test_result = self._anova_test(conditions, metric)
                        tests[f"{metric}_anova"] = test_result

            # Chi-square test for categorical data
            for metric in success_criteria.get("categorical_metrics", []):
                test_result = self._chi_square_test(conditions, metric)
                tests[f"{metric}_chi_square"] = test_result

        # Single-sample t-test against threshold
        threshold = success_criteria.get("threshold")
        if threshold is not None:
            for metric in success_criteria.get("metrics", ["correctness"]):
                values = [r.get(metric, 0.0) for r in test_results if metric in r]
                if values:
                    test_result = self._one_sample_t_test(values, threshold, metric)
                    tests[f"{metric}_threshold_test"] = test_result

        return tests

    def _group_by_condition(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group test results by experimental condition"""
        conditions = {}

        for result in test_results:
            condition = result.get("approach", result.get("condition", "unknown"))
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append(result)

        return conditions

    def _can_run_test(
        self,
        condition_a: List[Dict[str, Any]],
        condition_b: Optional[List[Dict[str, Any]]],
        metric: str,
    ) -> bool:
        """Check if we can run statistical test on this metric"""
        values_a = [
            r.get(metric, None) for r in condition_a if r.get(metric) is not None
        ]

        if condition_b is not None:
            values_b = [
                r.get(metric, None) for r in condition_b if r.get(metric) is not None
            ]
            return len(values_a) >= 3 and len(values_b) >= 3
        else:
            return len(values_a) >= 3

    def _paired_t_test(
        self,
        condition_a: List[Dict[str, Any]],
        condition_b: List[Dict[str, Any]],
        metric: str,
        name_a: str,
        name_b: str,
    ) -> StatisticalTest:
        """Run paired t-test comparing two conditions"""
        # Extract paired data (assuming same test cases)
        values_a = [r.get(metric, 0.0) for r in condition_a if metric in r]
        values_b = [r.get(metric, 0.0) for r in condition_b if metric in r]

        # Ensure same length (use minimum)
        min_length = min(len(values_a), len(values_b))
        values_a = values_a[:min_length]
        values_b = values_b[:min_length]

        # Calculate differences
        differences = [a - b for a, b in zip(values_a, values_b)]

        # Paired t-test
        n = len(differences)
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences) if n > 1 else 0.0

        t_statistic = mean_diff / (std_diff / (n**0.5)) if std_diff > 0 else 0.0

        # Calculate p-value (two-tailed)
        df = n - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

        # Effect size (Cohen's d for paired samples)
        effect_size = mean_diff / std_diff if std_diff > 0 else 0.0

        # Confidence interval
        se = std_diff / (n**0.5)
        margin = stats.t.ppf(1 - self.alpha_level / 2, df) * se
        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin

        # Interpretation
        if p_value < self.alpha_level:
            if abs(effect_size) >= self.effect_size_thresholds["large"]:
                interpretation = f"Significant difference with large effect size ({name_a} > {name_b})"
            elif abs(effect_size) >= self.effect_size_thresholds["medium"]:
                interpretation = f"Significant difference with medium effect size ({name_a} > {name_b})"
            else:
                interpretation = f"Significant difference with small effect size ({name_a} > {name_b})"
        else:
            interpretation = f"No significant difference between {name_a} and {name_b}"

        return StatisticalTest(
            test_name="paired_t_test",
            statistic=t_statistic,
            p_value=p_value,
            critical_value=stats.t.ppf(1 - self.alpha_level / 2, df),
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            assumptions=[
                "Normal distribution of differences",
                "Independent observations",
            ],
        )

    def _anova_test(
        self, conditions: Dict[str, List[Dict[str, Any]]], metric: str
    ) -> StatisticalTest:
        """Run one-way ANOVA for comparing multiple conditions"""
        # Extract data for each condition
        group_data = []
        group_labels = []

        for condition_name, condition_results in conditions.items():
            values = [r.get(metric, 0.0) for r in condition_results if metric in r]
            group_data.extend(values)
            group_labels.extend([condition_name] * len(values))

        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(
            *[
                [r.get(metric, 0.0) for r in condition_results if metric in r]
                for condition_results in conditions.values()
            ]
        )

        # Effect size (eta-squared)
        total_sum_squares = sum(
            (x - statistics.mean(group_data)) ** 2 for x in group_data
        )
        between_group_sum_squares = sum(
            len(group) * (statistics.mean(group) - statistics.mean(group_data)) ** 2
            for group in [
                [r.get(metric, 0.0) for r in condition_results if metric in r]
                for condition_results in conditions.values()
            ]
        )
        eta_squared = (
            between_group_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else 0.0
        )

        # Interpretation
        if p_value < self.alpha_level:
            if eta_squared >= 0.14:  # Large effect
                interpretation = (
                    f"Significant differences between conditions with large effect size"
                )
            elif eta_squared >= 0.06:  # Medium effect
                interpretation = f"Significant differences between conditions with medium effect size"
            else:
                interpretation = (
                    f"Significant differences between conditions with small effect size"
                )
        else:
            interpretation = "No significant differences between conditions"

        return StatisticalTest(
            test_name="one_way_anova",
            statistic=f_statistic,
            p_value=p_value,
            critical_value=None,
            effect_size=eta_squared,
            confidence_interval=None,
            interpretation=interpretation,
            assumptions=[
                "Normal distribution",
                "Homogeneity of variances",
                "Independent observations",
            ],
        )

    def _chi_square_test(
        self, conditions: Dict[str, List[Dict[str, Any]]], metric: str
    ) -> StatisticalTest:
        """Run chi-square test for categorical data"""
        # Create contingency table
        categories = set()
        for condition_results in conditions.values():
            for result in condition_results:
                if metric in result:
                    categories.add(result[metric])

        categories = sorted(list(categories))
        contingency_table = []

        for condition_name, condition_results in conditions.items():
            row = []
            for category in categories:
                count = sum(1 for r in condition_results if r.get(metric) == category)
                row.append(count)
            contingency_table.append(row)

        # Chi-square test
        chi2_statistic, p_value, dof, expected = stats.chi2_contingency(
            contingency_table
        )

        # Effect size (Cramér's V)
        n = sum(sum(row) for row in contingency_table)
        min_dim = min(len(contingency_table), len(contingency_table[0]))
        cramers_v = (chi2_statistic / (n * (min_dim - 1))) ** 0.5 if n > 0 else 0.0

        # Interpretation
        if p_value < self.alpha_level:
            if cramers_v >= 0.25:  # Large effect
                interpretation = f"Significant association with large effect size"
            elif cramers_v >= 0.15:  # Medium effect
                interpretation = f"Significant association with medium effect size"
            else:
                interpretation = f"Significant association with small effect size"
        else:
            interpretation = "No significant association between conditions"

        return StatisticalTest(
            test_name="chi_square_test",
            statistic=chi2_statistic,
            p_value=p_value,
            critical_value=stats.chi2.ppf(1 - self.alpha_level, dof),
            effect_size=cramers_v,
            confidence_interval=None,
            interpretation=interpretation,
            assumptions=["Independent observations", "Expected frequencies > 5"],
        )

    def _one_sample_t_test(
        self, values: List[float], threshold: float, metric: str
    ) -> StatisticalTest:
        """Run one-sample t-test against threshold value"""
        if len(values) < 2:
            return StatisticalTest(
                test_name="one_sample_t_test",
                statistic=0.0,
                p_value=1.0,
                critical_value=None,
                effect_size=0.0,
                confidence_interval=None,
                interpretation="Insufficient data for test",
                assumptions=[],
            )

        sample_mean = statistics.mean(values)
        sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
        n = len(values)

        # One-sample t-test
        t_statistic = (
            (sample_mean - threshold) / (sample_std / (n**0.5))
            if sample_std > 0
            else 0.0
        )

        # Calculate p-value
        df = n - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

        # Effect size (Cohen's d for one sample)
        effect_size = (sample_mean - threshold) / sample_std if sample_std > 0 else 0.0

        # Confidence interval
        se = sample_std / (n**0.5)
        margin = stats.t.ppf(1 - self.alpha_level / 2, df) * se
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin

        # Interpretation
        if p_value < self.alpha_level:
            if sample_mean > threshold:
                interpretation = f"Significantly above threshold of {threshold}"
            else:
                interpretation = f"Significantly below threshold of {threshold}"
        else:
            interpretation = (
                f"Not significantly different from threshold of {threshold}"
            )

        return StatisticalTest(
            test_name="one_sample_t_test",
            statistic=t_statistic,
            p_value=p_value,
            critical_value=stats.t.ppf(1 - self.alpha_level / 2, df),
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            assumptions=["Normal distribution", "Random sampling"],
        )

    def _calculate_power_analysis(
        self, test_results: List[Dict[str, Any]], success_criteria: Dict[str, Any]
    ) -> PowerAnalysis:
        """Calculate statistical power analysis"""
        # Extract effect size and sample size
        effect_sizes = self._calculate_effect_sizes(test_results)
        sample_size = len(test_results)

        # Use the primary metric's effect size
        primary_metric = success_criteria.get("primary_metric", "correctness")
        effect_size = effect_sizes.get(primary_metric, {}).get("cohen_d", 0.0)

        # Calculate achieved power
        achieved_power = self._calculate_statistical_power(
            effect_size, sample_size, self.alpha_level
        )

        # Calculate required sample size for target power
        required_sample_size = self._calculate_required_sample_size(
            effect_size, self.power_target, self.alpha_level
        )

        # Recommendation
        if achieved_power >= self.power_target:
            recommendation = f"Adequate power ({achieved_power:.2f}) achieved with current sample size"
        elif required_sample_size <= sample_size * 2:
            recommendation = (
                f"Increase sample size to {required_sample_size} for target power"
            )
        else:
            recommendation = f"Large sample size required ({required_sample_size}) - consider effect size or design changes"

        return PowerAnalysis(
            sample_size=sample_size,
            effect_size=effect_size,
            alpha=self.alpha_level,
            power=achieved_power,
            recommendation=recommendation,
        )

    def _calculate_statistical_power(
        self, effect_size: float, sample_size: int, alpha: float
    ) -> float:
        """Calculate statistical power for given parameters"""
        # Use normal approximation for power calculation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * (sample_size**0.5) - z_alpha
        power = stats.norm.cdf(z_beta)

        return max(0.0, min(1.0, power))

    def _calculate_required_sample_size(
        self, effect_size: float, power: float, alpha: float
    ) -> int:
        """Calculate required sample size for desired power"""
        if effect_size == 0:
            return float("inf")

        # Use approximation formula
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        required_n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(required_n) + 1

    def _calculate_effect_sizes(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate various effect size measures"""
        effect_sizes = {}

        # Group by condition for comparison
        conditions = self._group_by_condition(test_results)

        if len(conditions) >= 2:
            condition_names = list(conditions.keys())

            for metric in set().union(*[list(r.keys()) for r in test_results]):
                if isinstance(test_results[0].get(metric), (int, float)):
                    # Cohen's d for two conditions
                    if len(conditions) == 2:
                        values_a = [
                            r.get(metric, 0.0)
                            for r in conditions[condition_names[0]]
                            if metric in r
                        ]
                        values_b = [
                            r.get(metric, 0.0)
                            for r in conditions[condition_names[1]]
                            if metric in r
                        ]

                        if len(values_a) >= 2 and len(values_b) >= 2:
                            pooled_std = (
                                (
                                    (len(values_a) - 1)
                                    * statistics.stdev(values_a) ** 2
                                    + (len(values_b) - 1)
                                    * statistics.stdev(values_b) ** 2
                                )
                                / (len(values_a) + len(values_b) - 2)
                            ) ** 0.5

                            mean_diff = statistics.mean(values_a) - statistics.mean(
                                values_b
                            )
                            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                            effect_sizes[metric] = effect_sizes.get(metric, {})
                            effect_sizes[metric]["cohens_d"] = cohens_d

                    # Glass's delta (using control group SD)
                    if len(conditions) == 2:
                        control_std = statistics.stdev(
                            [
                                r.get(metric, 0.0)
                                for r in conditions[condition_names[1]]
                                if metric in r
                            ]
                        )
                        if control_std > 0:
                            glass_delta = mean_diff / control_std
                            effect_sizes[metric]["glass_delta"] = glass_delta

        return effect_sizes

    def _assess_practical_significance(
        self, test_results: List[Dict[str, Any]], success_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess practical significance beyond statistical significance"""
        practical_assessment = {}

        # Check improvement thresholds
        improvement_threshold = success_criteria.get(
            "improvement_threshold", 0.1
        )  # 10% default

        conditions = self._group_by_condition(test_results)
        if len(conditions) >= 2:
            condition_names = list(conditions.keys())

            for metric in success_criteria.get("metrics", ["correctness"]):
                if self._can_run_test(
                    conditions[condition_names[0]],
                    conditions[condition_names[1]],
                    metric,
                ):
                    means = {}
                    for condition in condition_names:
                        values = [
                            r.get(metric, 0.0)
                            for r in conditions[condition]
                            if metric in r
                        ]
                        means[condition] = statistics.mean(values) if values else 0.0

                    # Calculate percent improvement
                    if len(means) >= 2:
                        baseline_mean = min(means.values())
                        for condition, mean in means.items():
                            if condition != baseline_mean:
                                percent_improvement = (
                                    (mean - baseline_mean) / baseline_mean * 100
                                )
                                practical_assessment[
                                    f"{metric}_{condition}_improvement"
                                ] = {
                                    "percent_improvement": percent_improvement,
                                    "practically_significant": percent_improvement
                                    >= improvement_threshold * 100,
                                    "meets_threshold": percent_improvement
                                    >= improvement_threshold * 100,
                                }

        return practical_assessment

    def _check_statistical_assumptions(
        self, test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check statistical assumptions for the tests"""
        assumptions_check = {
            "normality_tests": {},
            "homogeneity_tests": {},
            "independence_check": {},
            "outlier_analysis": {},
        }

        # Group by condition
        conditions = self._group_by_condition(test_results)

        for condition_name, condition_results in conditions.items():
            condition_assumptions = {}

            for metric in set().union(*[list(r.keys()) for r in condition_results]):
                if isinstance(condition_results[0].get(metric), (int, float)):
                    values = [
                        r.get(metric, 0.0) for r in condition_results if metric in r
                    ]

                    if len(values) >= 3:
                        # Normality test (Shapiro-Wilk)
                        shapiro_stat, shapiro_p = stats.shapiro(values)
                        condition_assumptions[f"{metric}_normality"] = {
                            "statistic": shapiro_stat,
                            "p_value": shapiro_p,
                            "is_normal": shapiro_p > 0.05,
                            "interpretation": "Normal distribution assumed"
                            if shapiro_p > 0.05
                            else "Normal distribution violated",
                        }

                        # Outlier analysis
                        q1 = np.percentile(values, 25)
                        q3 = np.percentile(values, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr

                        outliers = [
                            v for v in values if v < lower_bound or v > upper_bound
                        ]
                        condition_assumptions[f"{metric}_outliers"] = {
                            "count": len(outliers),
                            "percentage": len(outliers) / len(values) * 100,
                            "outlier_values": outliers,
                            "interpretation": "Low outlier rate"
                            if len(outliers) / len(values) < 0.1
                            else "High outlier rate",
                        }

            assumptions_check[condition_name] = condition_assumptions

        # Homogeneity of variances (Levene's test)
        for metric in set().union(*[list(r.keys()) for r in test_results]):
            if isinstance(test_results[0].get(metric), (int, float)):
                groups = []
                for condition_results in conditions.values():
                    values = [
                        r.get(metric, 0.0) for r in condition_results if metric in r
                    ]
                    if values:
                        groups.append(values)

                if len(groups) >= 2:
                    levene_stat, levene_p = stats.levene(*groups)
                    assumptions_check["homogeneity_tests"][f"{metric}_levene"] = {
                        "statistic": levene_stat,
                        "p_value": levene_p,
                        "equal_variances": levene_p > 0.05,
                        "interpretation": "Equal variances assumed"
                        if levene_p > 0.05
                        else "Equal variances violated",
                    }

        return assumptions_check

    def _generate_recommendations(
        self, test_results: List[Dict[str, Any]], success_criteria: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on statistical analysis"""
        recommendations = []

        # Sample size recommendations
        if len(test_results) < 20:
            recommendations.append(
                "Increase sample size to at least 20 for better statistical power"
            )
        elif len(test_results) < 50:
            recommendations.append(
                "Consider increasing sample size to 50+ for high-stakes hypotheses"
            )

        # Effect size recommendations
        effect_sizes = self._calculate_effect_sizes(test_results)
        primary_metric = success_criteria.get("primary_metric", "correctness")
        primary_effect_size = effect_sizes.get(primary_metric, {}).get("cohens_d", 0.0)

        if abs(primary_effect_size) < self.effect_size_thresholds["small"]:
            recommendations.append(
                "Effect size too small - consider experimental design changes"
            )
        elif abs(primary_effect_size) < self.effect_size_thresholds["medium"]:
            recommendations.append(
                "Small to medium effect size - larger sample size may help"
            )

        # Practical significance recommendations
        practical_assessment = self._assess_practical_significance(
            test_results, success_criteria
        )
        practically_significant = any(
            assessment.get("practically_significant", False)
            for assessment in practical_assessment.values()
            if isinstance(assessment, dict)
        )

        if not practically_significant:
            recommendations.append(
                "Results statistically significant but not practically significant - consider real-world impact"
            )

        # Assumptions recommendations
        assumptions = self._check_statistical_assumptions(test_results)
        violated_assumptions = []

        for condition, condition_assumptions in assumptions.items():
            if isinstance(condition_assumptions, dict):
                for test_name, test_result in condition_assumptions.items():
                    if isinstance(test_result, dict) and not test_result.get(
                        "is_normal", True
                    ):
                        violated_assumptions.append(
                            f"Normality assumption violated for {condition}"
                        )
                    if isinstance(test_result, dict) and not test_result.get(
                        "equal_variances", True
                    ):
                        violated_assumptions.append(
                            f"Equal variances assumption violated for {condition}"
                        )

        if violated_assumptions:
            recommendations.append(
                f"Address violated statistical assumptions: {', '.join(violated_assumptions)}"
            )

        return recommendations

    def generate_comprehensive_report(
        self, analysis_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive statistical analysis report"""
        report_lines = [
            "# Statistical Analysis Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Hypothesis: {analysis_results['hypothesis_id']}",
            "",
            "## Executive Summary",
            "",
            f"**Sample Size**: {analysis_results['sample_size']}",
            f"**Primary Findings**: {self._summarize_findings(analysis_results)}",
            "",
            "## Descriptive Statistics",
            "",
        ]

        # Add descriptive statistics
        descriptive_stats = analysis_results.get("descriptive_stats", {})
        for metric, stats in descriptive_stats.items():
            report_lines.extend(
                [
                    f"### {metric.replace('_', ' ').title()}",
                    f"- **Mean**: {stats['mean']:.3f}",
                    f"- **Std Dev**: {stats['std']:.3f}",
                    f"- **Range**: {stats['min']:.3f} to {stats['max']:.3f}",
                    f"- **CV**: {stats['cv']:.3f}",
                    "",
                ]
            )

        # Add statistical test results
        statistical_tests = analysis_results.get("statistical_tests", {})
        if statistical_tests:
            report_lines.extend(["## Statistical Test Results", ""])

            for test_name, test_result in statistical_tests.items():
                report_lines.extend(
                    [
                        f"### {test_name.replace('_', ' ').title()}",
                        f"- **Test Statistic**: {test_result.statistic:.3f}",
                        f"- **P-value**: {test_result.p_value:.4f}",
                        f"- **Effect Size**: {test_result.effect_size:.3f}",
                        f"- **Interpretation**: {test_result.interpretation}",
                        "",
                    ]
                )

        # Add power analysis
        power_analysis = analysis_results.get("power_analysis")
        if power_analysis:
            report_lines.extend(
                [
                    "## Power Analysis",
                    f"- **Achieved Power**: {power_analysis.power:.3f}",
                    f"- **Target Power**: {self.power_target:.3f}",
                    f"- **Required Sample Size**: {power_analysis.required_sample_size}",
                    f"- **Recommendation**: {power_analysis.recommendation}",
                    "",
                ]
            )

        # Add practical significance
        practical_sig = analysis_results.get("practical_significance", {})
        if practical_sig:
            report_lines.extend(["## Practical Significance", ""])

            for assessment_name, assessment in practical_sig.items():
                if isinstance(assessment, dict):
                    report_lines.extend(
                        [
                            f"### {assessment_name.replace('_', ' ').title()}",
                            f"- **Percent Improvement**: {assessment.get('percent_improvement', 0):.1f}%",
                            f"- **Practically Significant**: {assessment.get('practically_significant', False)}",
                            f"- **Meets Threshold**: {assessment.get('meets_threshold', False)}",
                            "",
                        ]
                    )

        # Add recommendations
        recommendations = analysis_results.get("recommendations", [])
        if recommendations:
            report_lines.extend(["## Recommendations", ""])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")

        report_content = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content

    def _summarize_findings(self, analysis_results: Dict[str, Any]) -> str:
        """Summarize key findings from analysis"""
        statistical_tests = analysis_results.get("statistical_tests", {})

        if not statistical_tests:
            return "No statistical tests conducted"

        # Count significant results
        significant_tests = 0
        total_tests = len(statistical_tests)

        for test_name, test_result in statistical_tests.items():
            if test_result.p_value < self.alpha_level:
                significant_tests += 1

        # Find primary result
        primary_metric = "correctness"  # Default
        primary_test = None
        primary_effect_size = 0.0

        for test_name, test_result in statistical_tests.items():
            if primary_metric in test_name.lower():
                primary_test = test_result
                primary_effect_size = abs(test_result.effect_size)
                break

        summary_parts = []

        if primary_test:
            if primary_test.p_value < self.alpha_level:
                if primary_effect_size >= self.effect_size_thresholds["large"]:
                    summary_parts.append(
                        "Statistically significant with large effect size"
                    )
                elif primary_effect_size >= self.effect_size_thresholds["medium"]:
                    summary_parts.append(
                        "Statistically significant with medium effect size"
                    )
                else:
                    summary_parts.append(
                        "Statistically significant with small effect size"
                    )
            else:
                summary_parts.append("Not statistically significant")

        summary_parts.append(f"({significant_tests}/{total_tests} tests significant)")

        return "; ".join(summary_parts)


def main():
    """Example usage of statistical analyzer"""
    analyzer = ConjectureStatisticalAnalyzer()

    # Example test results (would be loaded from actual experiments)
    example_results = [
        {
            "approach": "conjecture",
            "correctness": 0.75,
            "reasoning_quality": 0.68,
            "completeness": 0.72,
        },
        {
            "approach": "conjecture",
            "correctness": 0.78,
            "reasoning_quality": 0.71,
            "completeness": 0.74,
        },
        {
            "approach": "conjecture",
            "correctness": 0.77,
            "reasoning_quality": 0.69,
            "completeness": 0.73,
        },
        {
            "approach": "direct",
            "correctness": 0.65,
            "reasoning_quality": 0.62,
            "completeness": 0.68,
        },
        {
            "approach": "direct",
            "correctness": 0.67,
            "reasoning_quality": 0.64,
            "completeness": 0.70,
        },
        {
            "approach": "direct",
            "correctness": 0.66,
            "reasoning_quality": 0.63,
            "completeness": 0.69,
        },
    ]

    success_criteria = {
        "primary_metric": "correctness",
        "metrics": ["correctness", "reasoning_quality", "completeness"],
        "threshold": 0.7,
        "improvement_threshold": 0.1,
    }

    # Run analysis
    analysis = analyzer.analyze_hypothesis_results(
        "example_hypothesis", example_results, success_criteria
    )

    # Generate report
    report = analyzer.generate_comprehensive_report(analysis)

    # Save report
    output_path = Path("research/analysis/example_statistical_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Statistical analysis report saved to: {output_path}")
    print("\nKey Findings:")
    print(analyzer._summarize_findings(analysis))


if __name__ == "__main__":
    main()
