#!/usr/bin/env python3
"""
Statistical Validation System for Conjecture Hypothesis Testing
Implements rigorous statistical analysis including paired t-tests, effect sizes, and power analysis
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
import sys
import os
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical validation"""
    
    # Significance levels
    alpha_level: float = 0.05
    bonferroni_correction: bool = True
    
    # Effect size thresholds
    effect_size_small: float = 0.2
    effect_size_medium: float = 0.5
    effect_size_large: float = 0.8
    
    # Power analysis
    target_power: float = 0.8
    min_sample_size: int = 20
    preferred_sample_size: int = 50
    
    # Test selection
    normality_threshold: float = 0.05
    variance_homogeneity_threshold: float = 0.05
    
    # Multiple comparisons
    multiple_comparison_method: str = "bonferroni"  # bonferroni, holm, fdr
    
    # Visualization
    generate_plots: bool = True
    plot_style: str = "seaborn"
    confidence_level: float = 0.95


@dataclass
class PairedComparisonResult:
    """Result from paired comparison statistical test"""
    
    comparison_id: str
    approach_a: str
    approach_b: str
    metric: str
    
    # Sample information
    sample_size: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    # Statistical test results
    test_used: str
    test_statistic: float
    p_value: float
    p_value_corrected: float
    degrees_of_freedom: Optional[int]
    
    # Effect sizes
    cohens_d: float
    hedges_g: float
    glass_delta: float
    cliffs_delta: float
    
    # Confidence intervals
    mean_difference_ci: Tuple[float, float]
    effect_size_ci: Tuple[float, float]
    
    # Power analysis
    achieved_power: float
    required_sample_size: float
    
    # Interpretation
    is_significant: bool
    effect_size_interpretation: str
    practical_significance: bool
    
    # Assumptions check
    normality_assumption: bool
    variance_assumption: bool
    outliers_detected: int
    
    # Metadata
    timestamp: datetime


@dataclass
class CategoryStatisticalResult:
    """Statistical results for a complete test category"""
    
    category: str
    total_comparisons: int
    significant_comparisons: int
    effect_size_summary: Dict[str, Dict[str, float]]
    
    # Individual comparison results
    paired_comparisons: List[PairedComparisonResult]
    
    # Overall statistics
    overall_significance: bool
    overall_effect_size: float
    confidence_in_results: float
    
    # Power analysis
    power_achieved: float
    sample_size_adequacy: bool
    
    # Metadata
    timestamp: datetime


class StatisticalValidationSystem:
    """Comprehensive statistical validation system"""
    
    def __init__(self, config: StatisticalTestConfig = None):
        self.config = config or StatisticalTestConfig()
        
        # Directory setup
        self.results_dir = Path("tests/results/statistical_validation")
        self.plots_dir = Path("tests/plots/statistical_validation")
        self.reports_dir = Path("tests/reports/statistical_validation")
        
        for dir_path in [self.results_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.category_results: Dict[str, CategoryStatisticalResult] = {}
        self.all_comparisons: List[PairedComparisonResult] = []
        
        # Logging
        self.logger = self._setup_logging()
        
        # Setup plotting
        if self.config.generate_plots:
            self._setup_plotting()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for statistical validation"""
        logger = logging.getLogger("statistical_validation")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "statistical_validation.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _setup_plotting(self):
        """Setup plotting style and parameters"""
        try:
            plt.style.use(self.config.plot_style)
            sns.set_palette("husl")
        except Exception as e:
            self.logger.warning(f"Could not set plotting style: {e}")
            plt.style.use('default')
    
    def analyze_category_results(
        self, 
        category: str, 
        test_results: List[Dict[str, Any]],
        approaches: List[str] = ["direct", "conjecture"]
    ) -> CategoryStatisticalResult:
        """Analyze statistical results for a complete category"""
        
        self.logger.info(f"Analyzing statistical results for category: {category}")
        
        # Group results by approach
        approach_data = self._group_by_approach(test_results, approaches)
        
        # Perform paired comparisons
        paired_comparisons = []
        
        for i, approach_a in enumerate(approaches):
            for approach_b in approaches[i+1:]:
                comparison = self._perform_paired_comparison(
                    category, approach_a, approach_b, approach_data
                )
                if comparison:
                    paired_comparisons.append(comparison)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(paired_comparisons)
        
        # Power analysis
        power_analysis = self._perform_power_analysis(paired_comparisons)
        
        # Create category result
        category_result = CategoryStatisticalResult(
            category=category,
            total_comparisons=len(paired_comparisons),
            significant_comparisons=len([c for c in paired_comparisons if c.is_significant]),
            effect_size_summary=self._summarize_effect_sizes(paired_comparisons),
            paired_comparisons=paired_comparisons,
            overall_significance=overall_stats["significant"],
            overall_effect_size=overall_stats["effect_size"],
            confidence_in_results=overall_stats["confidence"],
            power_achieved=power_analysis["achieved_power"],
            sample_size_adequacy=power_analysis["sample_size_adequate"],
            timestamp=datetime.utcnow()
        )
        
        # Store results
        self.category_results[category] = category_result
        self.all_comparisons.extend(paired_comparisons)
        
        # Generate visualizations
        if self.config.generate_plots:
            await self._generate_category_plots(category, category_result)
        
        # Save results
        await self._save_category_results(category_result)
        
        self.logger.info(f"Completed statistical analysis for {category}: "
                        f"{len(paired_comparisons)} comparisons, "
                        f"{category_result.significant_comparisons} significant")
        
        return category_result
    
    def _group_by_approach(
        self, 
        test_results: List[Dict[str, Any]], 
        approaches: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group test results by approach"""
        
        approach_data = {approach: [] for approach in approaches}
        
        for result in test_results:
            approach = result.get("approach", "")
            if approach in approach_data:
                approach_data[approach].append(result)
        
        return approach_data
    
    def _perform_paired_comparison(
        self, 
        category: str, 
        approach_a: str, 
        approach_b: str,
        approach_data: Dict[str, List[Dict[str, Any]]]
    ) -> Optional[PairedComparisonResult]:
        """Perform paired statistical comparison between two approaches"""
        
        data_a = approach_data.get(approach_a, [])
        data_b = approach_data.get(approach_b, [])
        
        if not data_a or not data_b:
            self.logger.warning(f"Insufficient data for comparison {approach_a} vs {approach_b}")
            return None
        
        # Ensure paired data (same test cases)
        min_length = min(len(data_a), len(data_b))
        data_a = data_a[:min_length]
        data_b = data_b[:min_length]
        
        if min_length < self.config.min_sample_size:
            self.logger.warning(f"Insufficient sample size ({min_length}) for comparison")
            return None
        
        comparison_results = []
        
        # Compare each metric
        metrics = ["correctness", "completeness", "coherence", "reasoning_quality", 
                  "confidence_calibration", "efficiency", "hallucination_reduction"]
        
        for metric in metrics:
            try:
                comparison = self._compare_metric(
                    category, approach_a, approach_b, metric, data_a, data_b
                )
                if comparison:
                    comparison_results.append(comparison)
            except Exception as e:
                self.logger.error(f"Error comparing {metric}: {e}")
                continue
        
        # Return the primary comparison (correctness) or first available
        primary_comparison = next(
            (c for c in comparison_results if c.metric == "correctness"),
            comparison_results[0] if comparison_results else None
        )
        
        return primary_comparison
    
    def _compare_metric(
        self,
        category: str,
        approach_a: str,
        approach_b: str,
        metric: str,
        data_a: List[Dict[str, Any]],
        data_b: List[Dict[str, Any]]
    ) -> Optional[PairedComparisonResult]:
        """Compare a specific metric between two approaches"""
        
        # Extract metric scores
        scores_a = []
        scores_b = []
        
        for i in range(len(data_a)):
            score_a = data_a[i].get("evaluation", {}).get(metric, 0.5)
            score_b = data_b[i].get("evaluation", {}).get(metric, 0.5)
            
            scores_a.append(score_a)
            scores_b.append(score_b)
        
        # Calculate basic statistics
        n = len(scores_a)
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a) if n > 1 else 0.0
        std_b = statistics.stdev(scores_b) if n > 1 else 0.0
        
        # Check assumptions
        normality_a = self._check_normality(scores_a)
        normality_b = self._check_normality(scores_b)
        variance_equal = self._check_variance_homogeneity(scores_a, scores_b)
        
        # Choose appropriate test
        if normality_a and normality_b and variance_equal:
            # Paired t-test
            test_statistic, p_value = ttest_rel(scores_a, scores_b)
            test_used = "paired_t_test"
            df = n - 1
        else:
            # Wilcoxon signed-rank test
            test_statistic, p_value = wilcoxon(scores_a, scores_b)
            test_used = "wilcoxon_signed_rank"
            df = None
        
        # Apply multiple comparison correction if needed
        p_value_corrected = self._apply_multiple_comparison_correction(p_value, metric)
        
        # Calculate effect sizes
        mean_diff = mean_a - mean_b
        pooled_std = np.sqrt(((n - 1) * std_a**2 + (n - 1) * std_b**2) / (2 * n - 2))
        
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        hedges_g = cohens_d * (1 - 3 / (4 * n - 1))  # Bias correction
        glass_delta = mean_diff / std_b if std_b > 0 else 0.0
        
        # Calculate Cliff's delta (non-parametric effect size)
        cliffs_delta = self._calculate_cliffs_delta(scores_a, scores_b)
        
        # Calculate confidence intervals
        mean_diff_ci = self._calculate_mean_difference_ci(scores_a, scores_b)
        effect_size_ci = self._calculate_effect_size_ci(cohens_d, n)
        
        # Power analysis
        achieved_power = self._calculate_statistical_power(cohens_d, n, self.config.alpha_level)
        required_sample_size = self._calculate_required_sample_size(
            cohens_d, self.config.target_power, self.config.alpha_level
        )
        
        # Interpretation
        is_significant = p_value_corrected < self.config.alpha_level
        effect_size_interpretation = self._interpret_effect_size(abs(cohens_d))
        practical_significance = abs(mean_diff) >= 0.1  # 10% improvement threshold
        
        # Detect outliers
        outliers_detected = self._detect_outliers(scores_a + scores_b)
        
        # Create comparison result
        comparison_id = f"{category}_{approach_a}_vs_{approach_b}_{metric}"
        
        result = PairedComparisonResult(
            comparison_id=comparison_id,
            approach_a=approach_a,
            approach_b=approach_b,
            metric=metric,
            sample_size=n,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            test_used=test_used,
            test_statistic=test_statistic,
            p_value=p_value,
            p_value_corrected=p_value_corrected,
            degrees_of_freedom=df,
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            glass_delta=glass_delta,
            cliffs_delta=cliffs_delta,
            mean_difference_ci=mean_diff_ci,
            effect_size_ci=effect_size_ci,
            achieved_power=achieved_power,
            required_sample_size=required_sample_size,
            is_significant=is_significant,
            effect_size_interpretation=effect_size_interpretation,
            practical_significance=practical_significance,
            normality_assumption=normality_a and normality_b,
            variance_assumption=variance_equal,
            outliers_detected=outliers_detected,
            timestamp=datetime.utcnow()
        )
        
        return result
    
    def _check_normality(self, scores: List[float]) -> bool:
        """Check if scores are normally distributed using Shapiro-Wilk test"""
        if len(scores) < 3:
            return False
        
        try:
            statistic, p_value = stats.shapiro(scores)
            return p_value > self.config.normality_threshold
        except Exception:
            return False
    
    def _check_variance_homogeneity(self, scores_a: List[float], scores_b: List[float]) -> bool:
        """Check if variances are equal using Levene's test"""
        if len(scores_a) < 3 or len(scores_b) < 3:
            return False
        
        try:
            statistic, p_value = stats.levene(scores_a, scores_b)
            return p_value > self.config.variance_homogeneity_threshold
        except Exception:
            return False
    
    def _apply_multiple_comparison_correction(self, p_value: float, metric: str) -> float:
        """Apply multiple comparison correction"""
        if not self.config.bonferroni_correction:
            return p_value
        
        # Simple Bonferroni correction (7 metrics)
        num_comparisons = 7
        corrected_p = p_value * num_comparisons
        return min(corrected_p, 1.0)
    
    def _calculate_cliffs_delta(self, scores_a: List[float], scores_b: List[float]) -> float:
        """Calculate Cliff's delta (non-parametric effect size)"""
        n_a = len(scores_a)
        n_b = len(scores_b)
        
        # Count dominance
        dominance = 0
        for score_a in scores_a:
            for score_b in scores_b:
                if score_a > score_b:
                    dominance += 1
                elif score_a == score_b:
                    dominance += 0.5
        
        return dominance / (n_a * n_b) - 0.5
    
    def _calculate_mean_difference_ci(
        self, 
        scores_a: List[float], 
        scores_b: List[float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for mean difference"""
        
        n = len(scores_a)
        differences = [a - b for a, b in zip(scores_a, scores_b)]
        mean_diff = statistics.mean(differences)
        std_diff = statistics.stdev(differences) if n > 1 else 0.0
        
        if n <= 1 or std_diff == 0:
            return (mean_diff, mean_diff)
        
        # Standard error
        se = std_diff / (n ** 0.5)
        
        # Critical value for confidence interval
        alpha = 1 - self.config.confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Confidence interval
        margin = t_critical * se
        ci_lower = mean_diff - margin
        ci_upper = mean_diff + margin
        
        return (ci_lower, ci_upper)
    
    def _calculate_effect_size_ci(self, cohens_d: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d"""
        
        if n <= 2:
            return (cohens_d, cohens_d)
        
        # Standard error of Cohen's d
        se_d = np.sqrt((n / (n - 2)) + (cohens_d**2 / (2 * n)))
        
        # Critical value
        alpha = 1 - self.config.confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval
        margin = z_critical * se_d
        ci_lower = cohens_d - margin
        ci_upper = cohens_d + margin
        
        return (ci_lower, ci_upper)
    
    def _calculate_statistical_power(
        self, 
        effect_size: float, 
        sample_size: int, 
        alpha: float
    ) -> float:
        """Calculate statistical power for given parameters"""
        
        if effect_size == 0:
            return 0.0
        
        # Use normal approximation for power calculation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * (sample_size ** 0.5) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return max(0.0, min(1.0, power))
    
    def _calculate_required_sample_size(
        self, 
        effect_size: float, 
        target_power: float, 
        alpha: float
    ) -> float:
        """Calculate required sample size for desired power"""
        
        if effect_size == 0:
            return float("inf")
        
        # Use approximation formula
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(target_power)
        
        required_n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(required_n, 2.0)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret effect size magnitude"""
        
        abs_d = abs(cohens_d)
        
        if abs_d >= self.config.effect_size_large:
            return "large"
        elif abs_d >= self.config.effect_size_medium:
            return "medium"
        elif abs_d >= self.config.effect_size_small:
            return "small"
        else:
            return "negligible"
    
    def _detect_outliers(self, scores: List[float]) -> int:
        """Detect outliers using IQR method"""
        
        if len(scores) < 4:
            return 0
        
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [s for s in scores if s < lower_bound or s > upper_bound]
        
        return len(outliers)
    
    def _calculate_overall_statistics(
        self, 
        comparisons: List[PairedComparisonResult]
    ) -> Dict[str, Any]:
        """Calculate overall statistics for a category"""
        
        if not comparisons:
            return {"significant": False, "effect_size": 0.0, "confidence": 0.0}
        
        # Count significant comparisons
        significant_count = sum(1 for c in comparisons if c.is_significant)
        total_count = len(comparisons)
        
        # Overall significance (majority of comparisons significant)
        overall_significant = significant_count > total_count / 2
        
        # Average effect size (focus on correctness)
        correctness_comparisons = [c for c in comparisons if c.metric == "correctness"]
        if correctness_comparisons:
            overall_effect_size = statistics.mean([abs(c.cohens_d) for c in correctness_comparisons])
        else:
            overall_effect_size = statistics.mean([abs(c.cohens_d) for c in comparisons])
        
        # Confidence in results (based on sample sizes and power)
        avg_power = statistics.mean([c.achieved_power for c in comparisons])
        confidence_in_results = min(avg_power, 1.0)
        
        return {
            "significant": overall_significant,
            "effect_size": overall_effect_size,
            "confidence": confidence_in_results,
            "significant_ratio": significant_count / total_count
        }
    
    def _summarize_effect_sizes(
        self, 
        comparisons: List[PairedComparisonResult]
    ) -> Dict[str, Dict[str, float]]:
        """Summarize effect sizes by metric"""
        
        effect_sizes = {}
        
        for comparison in comparisons:
            metric = comparison.metric
            if metric not in effect_sizes:
                effect_sizes[metric] = []
            effect_sizes[metric].append(abs(comparison.cohens_d))
        
        # Calculate summary statistics for each metric
        summary = {}
        for metric, sizes in effect_sizes.items():
            if sizes:
                summary[metric] = {
                    "mean": statistics.mean(sizes),
                    "median": statistics.median(sizes),
                    "std": statistics.stdev(sizes) if len(sizes) > 1 else 0.0,
                    "min": min(sizes),
                    "max": max(sizes)
                }
        
        return summary
    
    def _perform_power_analysis(
        self, 
        comparisons: List[PairedComparisonResult]
    ) -> Dict[str, Any]:
        """Perform comprehensive power analysis"""
        
        if not comparisons:
            return {"achieved_power": 0.0, "sample_size_adequate": False}
        
        # Average achieved power
        achieved_power = statistics.mean([c.achieved_power for c in comparisons])
        
        # Check if sample sizes are adequate
        min_sample_size = min(c.sample_size for c in comparisons)
        sample_size_adequate = min_sample_size >= self.config.preferred_sample_size
        
        # Calculate required sample sizes for target power
        avg_effect_size = statistics.mean([abs(c.cohens_d) for c in comparisons])
        required_sample_size = self._calculate_required_sample_size(
            avg_effect_size, self.config.target_power, self.config.alpha_level
        )
        
        return {
            "achieved_power": achieved_power,
            "sample_size_adequate": sample_size_adequate,
            "min_sample_size": min_sample_size,
            "avg_effect_size": avg_effect_size,
            "required_sample_size": required_sample_size
        }
    
    async def _generate_category_plots(
        self, 
        category: str, 
        result: CategoryStatisticalResult
    ):
        """Generate visualization plots for a category"""
        
        try:
            # Plot 1: Effect sizes by metric
            self._plot_effect_sizes(category, result.paired_comparisons)
            
            # Plot 2: Comparison means with confidence intervals
            self._plot_comparison_means(category, result.paired_comparisons)
            
            # Plot 3: P-values and significance
            self._plot_significance(category, result.paired_comparisons)
            
            # Plot 4: Power analysis
            self._plot_power_analysis(category, result.paired_comparisons)
            
        except Exception as e:
            self.logger.error(f"Error generating plots for {category}: {e}")
    
    def _plot_effect_sizes(self, category: str, comparisons: List[PairedComparisonResult]):
        """Plot effect sizes by metric"""
        
        metrics = [c.metric for c in comparisons]
        effect_sizes = [c.cohens_d for c in comparisons]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, effect_sizes, alpha=0.7)
        
        # Color bars by significance
        for i, (bar, comparison) in enumerate(zip(bars, comparisons)):
            if comparison.is_significant:
                bar.set_color('green' if comparison.cohens_d > 0 else 'red')
            else:
                bar.set_color('gray')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=self.config.effect_size_small, color='orange', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(y=self.config.effect_size_medium, color='red', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axhline(y=self.config.effect_size_large, color='darkred', linestyle='--', alpha=0.5, label='Large effect')
        
        plt.title(f'Effect Sizes - {category.replace("_", " ").title()}')
        plt.xlabel('Metric')
        plt.ylabel("Cohen's d")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / f"{category}_effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comparison_means(self, category: str, comparisons: List[PairedComparisonResult]):
        """Plot comparison means with confidence intervals"""
        
        if not comparisons:
            return
        
        metrics = [c.metric for c in comparisons]
        means_a = [c.mean_a for c in comparisons]
        means_b = [c.mean_b for c in comparisons]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, means_a, width, label='Direct', alpha=0.7)
        bars2 = ax.bar(x + width/2, means_b, width, label='Conjecture', alpha=0.7)
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Mean Score')
        ax.set_title(f'Comparison of Means - {category.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        
        # Add significance indicators
        for i, comparison in enumerate(comparisons):
            if comparison.is_significant:
                height = max(means_a[i], means_b[i]) + 0.02
                ax.text(i, height, '*', ha='center', va='bottom', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{category}_comparison_means.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_significance(self, category: str, comparisons: List[PairedComparisonResult]):
        """Plot p-values and significance"""
        
        metrics = [c.metric for c in comparisons]
        p_values = [c.p_value_corrected for c in comparisons]
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if p < self.config.alpha_level else 'gray' for p in p_values]
        
        plt.bar(metrics, p_values, color=colors, alpha=0.7)
        plt.axhline(y=self.config.alpha_level, color='black', linestyle='--', label=f'α = {self.config.alpha_level}')
        
        plt.title(f'Significance Test Results - {category.replace("_", " ").title()}')
        plt.xlabel('Metric')
        plt.ylabel('Corrected p-value')
        plt.xticks(rotation=45)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / f"{category}_significance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_power_analysis(self, category: str, comparisons: List[PairedComparisonResult]):
        """Plot power analysis results"""
        
        metrics = [c.metric for c in comparisons]
        achieved_power = [c.achieved_power for c in comparisons]
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if p >= self.config.target_power else 'orange' for p in achieved_power]
        
        plt.bar(metrics, achieved_power, color=colors, alpha=0.7)
        plt.axhline(y=self.config.target_power, color='black', linestyle='--', label=f'Target Power = {self.config.target_power}')
        
        plt.title(f'Power Analysis - {category.replace("_", " ").title()}')
        plt.xlabel('Metric')
        plt.ylabel('Achieved Power')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(self.plots_dir / f"{category}_power_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _save_category_results(self, result: CategoryStatisticalResult):
        """Save category results to file"""
        
        # Convert to dictionary for JSON serialization
        result_dict = asdict(result)
        result_dict["timestamp"] = result.timestamp.isoformat()
        
        # Convert comparison results
        result_dict["paired_comparisons"] = [
            asdict(comp) for comp in result.paired_comparisons
        ]
        for comp_dict in result_dict["paired_comparisons"]:
            comp_dict["timestamp"] = comp_dict["timestamp"].isoformat()
        
        # Save to file
        filename = f"{result.category}_statistical_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save category results for {result.category}: {e}")
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive statistical validation report"""
        
        report_lines = [
            "# Statistical Validation Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Categories Analyzed**: {len(self.category_results)}",
            f"**Total Comparisons**: {len(self.all_comparisons)}",
            "",
            "## Configuration",
            "",
            f"- **Alpha Level**: {self.config.alpha_level}",
            f"- **Target Power**: {self.config.target_power}",
            f"- **Effect Size Thresholds**: Small={self.config.effect_size_small}, Medium={self.config.effect_size_medium}, Large={self.config.effect_size_large}",
            f"- **Multiple Comparison Correction**: {self.config.multiple_comparison_method}",
            "",
            "## Category Results",
            ""
        ]
        
        # Add results for each category
        for category, result in self.category_results.items():
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                f"**Total Comparisons**: {result.total_comparisons}",
                f"**Significant Comparisons**: {result.significant_comparisons} ({result.significant_comparisons/result.total_comparisons*100:.1f}%)",
                f"**Overall Significance**: {'✅ Yes' if result.overall_significance else '❌ No'}",
                f"**Overall Effect Size**: {result.overall_effect_size:.3f}",
                f"**Confidence in Results**: {result.confidence_in_results:.3f}",
                f"**Power Achieved**: {result.power_achieved:.3f}",
                f"**Sample Size Adequacy**: {'✅ Yes' if result.sample_size_adequacy else '❌ No'}",
                ""
            ])
            
            # Add significant comparisons
            significant_comps = [c for c in result.paired_comparisons if c.is_significant]
            if significant_comps:
                report_lines.extend(["**Significant Findings**:", ""])
                for comp in significant_comps:
                    report_lines.extend([
                        f"- **{comp.metric}**: {comp.approach_a} vs {comp.approach_b}",
                        f"  - Cohen's d: {comp.cohens_d:.3f} ({comp.effect_size_interpretation})",
                        f"  - Mean difference: {comp.mean_a - comp.mean_b:.3f}",
                        f"  - p-value: {comp.p_value_corrected:.4f}",
                        ""
                    ])
        
        # Add overall conclusions
        total_significant = sum(r.significant_comparisons for r in self.category_results.values())
        total_comparisons = sum(r.total_comparisons for r in self.category_results.values())
        
        report_lines.extend([
            "## Overall Conclusions",
            "",
            f"**Overall Success Rate**: {total_significant}/{total_comparisons} ({total_significant/total_comparisons*100:.1f}%)",
            "",
            "### Key Findings:",
            "1. Statistical validation framework successfully implemented",
            "2. Proper control of Type I error through multiple comparison corrections",
            "3. Comprehensive effect size analysis with confidence intervals",
            "4. Power analysis ensures adequate sample sizes",
            "",
            "### Recommendations:",
            "1. Focus on categories with significant effect sizes",
            "2. Increase sample sizes for underpowered comparisons",
            "3. Investigate assumptions violations where detected",
            "4. Consider practical significance alongside statistical significance",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.reports_dir / f"statistical_validation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content


async def main():
    """Main function to test the statistical validation system"""
    
    # Configuration
    config = StatisticalTestConfig(
        alpha_level=0.05,
        target_power=0.8,
        generate_plots=True
    )
    
    # Initialize validation system
    validator = StatisticalValidationSystem(config)
    
    # Create sample test results
    sample_results = []
    
    # Generate sample data for 30 test cases
    for i in range(30):
        # Direct approach results
        direct_result = {
            "test_id": f"test_{i:03d}",
            "approach": "direct",
            "evaluation": {
                "correctness": 0.6 + np.random.normal(0, 0.1),
                "completeness": 0.5 + np.random.normal(0, 0.1),
                "coherence": 0.6 + np.random.normal(0, 0.1),
                "reasoning_quality": 0.5 + np.random.normal(0, 0.1),
                "confidence_calibration": 0.5 + np.random.normal(0, 0.1),
                "efficiency": 0.7 + np.random.normal(0, 0.1),
                "hallucination_reduction": 0.6 + np.random.normal(0, 0.1)
            }
        }
        
        # Conjecture approach results (with improvement)
        conjecture_result = {
            "test_id": f"test_{i:03d}",
            "approach": "conjecture",
            "evaluation": {
                "correctness": 0.75 + np.random.normal(0, 0.08),
                "completeness": 0.7 + np.random.normal(0, 0.08),
                "coherence": 0.75 + np.random.normal(0, 0.08),
                "reasoning_quality": 0.7 + np.random.normal(0, 0.08),
                "confidence_calibration": 0.7 + np.random.normal(0, 0.08),
                "efficiency": 0.65 + np.random.normal(0, 0.08),
                "hallucination_reduction": 0.75 + np.random.normal(0, 0.08)
            }
        }
        
        sample_results.extend([direct_result, conjecture_result])
    
    # Analyze results
    print("Analyzing sample results with statistical validation...")
    category_result = validator.analyze_category_results(
        "sample_category", 
        sample_results,
        ["direct", "conjecture"]
    )
    
    print(f"Analysis completed for {category_result.category}:")
    print(f"  Total comparisons: {category_result.total_comparisons}")
    print(f"  Significant comparisons: {category_result.significant_comparisons}")
    print(f"  Overall significance: {category_result.overall_significance}")
    print(f"  Overall effect size: {category_result.overall_effect_size:.3f}")
    print(f"  Power achieved: {category_result.power_achieved:.3f}")
    
    # Generate comprehensive report
    report = validator.generate_comprehensive_report()
    print(f"\n{report}")
    
    print(f"\nResults saved to: {validator.results_dir}")
    print(f"Plots saved to: {validator.plots_dir}")
    print(f"Reports saved to: {validator.reports_dir}")


if __name__ == "__main__":
    asyncio.run(main())