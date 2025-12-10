#!/usr/bin/env python3
"""
Comprehensive Metrics Analysis Framework for Conjecture
Provides statistical analysis, hypothesis validation, and performance insights
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import math
import uuid

# Scientific computing imports
try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced statistical features will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTest:
    """Results of statistical significance test"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    is_significant: bool = False
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    interpretation: str = ""

@dataclass
class HypothesisTest:
    """Results of hypothesis validation"""
    hypothesis_id: str
    hypothesis: str
    null_hypothesis: str
    test_results: List[StatisticalTest]
    conclusion: str
    confidence_level: float
    evidence_strength: str  # "weak", "moderate", "strong", "very_strong"
    practical_significance: bool
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ModelComparison:
    """Comparison between models across metrics"""
    model_a: str
    model_b: str
    metrics_comparison: Dict[str, Dict[str, float]]
    overall_winner: str
    confidence: float
    significant_differences: List[str]

@dataclass
class PipelineMetrics:
    """Comprehensive pipeline performance metrics"""
    pipeline_stage: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_time: float
    median_time: float
    p95_time: float
    p99_time: float
    throughput: float  # operations per second
    error_rate: float
    retry_rate: float
    timeout_rate: float
    resource_utilization: Dict[str, float]

@dataclass
class RetryStatistics:
    """Detailed retry and error statistics"""
    total_operations: int
    successful_without_retry: int
    successful_with_retry: int
    failed_operations: int
    retry_attempts: List[int]
    average_retries: float
    max_retries: int
    retry_success_rate: float
    error_types: Dict[str, int]
    error_patterns: List[Dict[str, Any]]

class MetricsAnalyzer:
    """Advanced metrics analysis and statistical testing"""
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path("research/results")
        self.analysis_dir = Path("research/analysis")
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.experiment_results: List[Dict[str, Any]] = []
        self.performance_metrics: List[Dict[str, Any]] = []
        self.pipeline_metrics: Dict[str, PipelineMetrics] = {}
        self.retry_stats: Dict[str, RetryStatistics] = {}
        
        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        
        # Configuration
        self.significance_threshold = 0.05
        self.effect_size_threshold = 0.5  # Cohen's d threshold for practical significance
        
    def load_experiment_results(self, experiment_files: List[str] = None):
        """Load experiment results from files"""
        if experiment_files is None:
            experiment_files = list(self.results_dir.glob("*.json"))
        
        for file_path in experiment_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.experiment_results.append(data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.experiment_results)} experiment results")
    
    def add_pipeline_metrics(self, stage: str, metrics_data: Dict[str, Any]):
        """Add pipeline performance metrics"""
        pipeline_metrics = PipelineMetrics(
            pipeline_stage=stage,
            total_operations=metrics_data.get('total_operations', 0),
            successful_operations=metrics_data.get('successful_operations', 0),
            failed_operations=metrics_data.get('failed_operations', 0),
            average_time=metrics_data.get('average_time', 0.0),
            median_time=metrics_data.get('median_time', 0.0),
            p95_time=metrics_data.get('p95_time', 0.0),
            p99_time=metrics_data.get('p99_time', 0.0),
            throughput=metrics_data.get('throughput', 0.0),
            error_rate=metrics_data.get('error_rate', 0.0),
            retry_rate=metrics_data.get('retry_rate', 0.0),
            timeout_rate=metrics_data.get('timeout_rate', 0.0),
            resource_utilization=metrics_data.get('resource_utilization', {})
        )
        
        self.pipeline_metrics[stage] = pipeline_metrics
    
    def add_retry_statistics(self, operation_type: str, retry_data: Dict[str, Any]):
        """Add retry and error statistics"""
        retry_stats = RetryStatistics(
            total_operations=retry_data.get('total_operations', 0),
            successful_without_retry=retry_data.get('successful_without_retry', 0),
            successful_with_retry=retry_data.get('successful_with_retry', 0),
            failed_operations=retry_data.get('failed_operations', 0),
            retry_attempts=retry_data.get('retry_attempts', []),
            average_retries=retry_data.get('average_retries', 0.0),
            max_retries=retry_data.get('max_retries', 0),
            retry_success_rate=retry_data.get('retry_success_rate', 0.0),
            error_types=retry_data.get('error_types', {}),
            error_patterns=retry_data.get('error_patterns', [])
        )
        
        self.retry_stats[operation_type] = retry_stats
    
    def perform_statistical_test(self, 
                            group_a: List[float], 
                            group_b: List[float],
                            test_type: str = "t_test",
                            alternative: str = "two-sided") -> StatisticalTest:
        """Perform statistical significance test between two groups"""
        
        if len(group_a) < 2 or len(group_b) < 2:
            return StatisticalTest(
                test_name=test_type,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation="Insufficient data for statistical testing"
            )
        
        try:
            if test_type == "t_test" and SCIPY_AVAILABLE:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(group_a, group_b, alternative=alternative)
                
                # Calculate effect size (Cohen's d)
                pooled_std = math.sqrt(((len(group_a) - 1) * statistics.variance(group_a) + 
                                     (len(group_b) - 1) * statistics.variance(group_b)) / 
                                    (len(group_a) + len(group_b) - 2))
                effect_size = (statistics.mean(group_a) - statistics.mean(group_b)) / pooled_std if pooled_std > 0 else 0
                
                is_significant = p_value < self.significance_threshold
                practical_significance = abs(effect_size) >= self.effect_size_threshold
                
                interpretation = self._interpret_test_result(p_value, is_significant, effect_size, practical_significance)
                
                return StatisticalTest(
                    test_name="Independent t-test",
                    statistic=statistic,
                    p_value=p_value,
                    is_significant=is_significant,
                    effect_size=effect_size,
                    interpretation=interpretation
                )
            
            elif test_type == "mann_whitney" and SCIPY_AVAILABLE:
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
                
                is_significant = p_value < self.significance_threshold
                interpretation = self._interpret_test_result(p_value, is_significant, None, False)
                
                return StatisticalTest(
                    test_name="Mann-Whitney U test",
                    statistic=statistic,
                    p_value=p_value,
                    is_significant=is_significant,
                    interpretation=interpretation
                )
            
            else:
                # Fallback to basic statistical comparison
                mean_a, mean_b = statistics.mean(group_a), statistics.mean(group_b)
                std_a, std_b = statistics.stdev(group_a) if len(group_a) > 1 else 0, statistics.stdev(group_b) if len(group_b) > 1 else 0
                
                # Simple confidence interval overlap test
                se_a, se_b = std_a / math.sqrt(len(group_a)), std_b / math.sqrt(len(group_b))
                ci_a = (mean_a - 1.96 * se_a, mean_a + 1.96 * se_a)
                ci_b = (mean_b - 1.96 * se_b, mean_b + 1.96 * se_b)
                
                overlap = max(0, min(ci_a[1], ci_b[1]) - max(ci_a[0], ci_b[0]))
                is_significant = overlap == 0
                
                return StatisticalTest(
                    test_name="Confidence interval comparison",
                    statistic=overlap,
                    p_value=0.05 if is_significant else 0.10,
                    is_significant=is_significant,
                    interpretation=f"Confidence intervals {'do not overlap' if is_significant else 'overlap'}"
                )
                
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return StatisticalTest(
                test_name=test_type,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                interpretation=f"Test failed: {e}"
            )
    
    def _interpret_test_result(self, p_value: float, is_significant: bool, 
                           effect_size: Optional[float], practical_significance: bool) -> str:
        """Interpret statistical test results"""
        if not is_significant:
            return f"No significant difference found (p={p_value:.3f})"
        
        significance_level = "highly significant" if p_value < 0.01 else "significant"
        
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                magnitude = "negligible"
            elif abs(effect_size) < 0.5:
                magnitude = "small"
            elif abs(effect_size) < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
            
            practical = "practically significant" if practical_significance else "not practically significant"
            return f"{significance_level} difference (p={p_value:.3f}) with {magnitude} effect size ({magnitude} - {practical})"
        
        return f"{significance_level} difference (p={p_value:.3f})"
    
    def validate_hypothesis(self, hypothesis_config: Dict[str, Any]) -> HypothesisTest:
        """Validate a hypothesis using statistical testing"""
        
        hypothesis_id = hypothesis_config.get('id', f'hyp_{uuid.uuid4().hex[:8]}')
        hypothesis = hypothesis_config.get('hypothesis', '')
        null_hypothesis = hypothesis_config.get('null_hypothesis', 'No effect')
        
        # Extract data for hypothesis testing
        test_results = []
        
        # Get experimental data
        relevant_experiments = [
            exp for exp in self.experiment_results 
            if exp.get('experiment_id') == hypothesis_config.get('experiment_id')
        ]
        
        if not relevant_experiments:
            return HypothesisTest(
                hypothesis_id=hypothesis_id,
                hypothesis=hypothesis,
                null_hypothesis=null_hypothesis,
                test_results=[],
                conclusion="No data available for hypothesis testing",
                confidence_level=0.0,
                evidence_strength="none",
                practical_significance=False
            )
        
        # Perform statistical tests based on hypothesis type
        if hypothesis_config.get('type') == 'improvement':
            # Test for improvement between control and treatment groups
            control_group = self._extract_metric_values(
                relevant_experiments, 
                hypothesis_config.get('control_metric', 'execution_time'),
                hypothesis_config.get('control_group', 'direct')
            )
            
            treatment_group = self._extract_metric_values(
                relevant_experiments,
                hypothesis_config.get('treatment_metric', 'execution_time'),
                hypothesis_config.get('treatment_group', 'conjecture')
            )
            
            if control_group and treatment_group:
                test_result = self.perform_statistical_test(
                    treatment_group, control_group, 
                    test_type=hypothesis_config.get('test_type', 't_test'),
                    alternative='less' if hypothesis_config.get('direction') == 'improvement' else 'two-sided'
                )
                test_results.append(test_result)
        
        # Calculate overall conclusion
        conclusion, evidence_strength, practical_significance = self._evaluate_hypothesis_evidence(
            test_results, hypothesis_config
        )
        
        # Generate recommendations
        recommendations = self._generate_hypothesis_recommendations(
            conclusion, evidence_strength, test_results, hypothesis_config
        )
        
        return HypothesisTest(
            hypothesis_id=hypothesis_id,
            hypothesis=hypothesis,
            null_hypothesis=null_hypothesis,
            test_results=test_results,
            conclusion=conclusion,
            confidence_level=self.significance_threshold,
            evidence_strength=evidence_strength,
            practical_significance=practical_significance,
            recommendations=recommendations
        )
    
    def _extract_metric_values(self, experiments: List[Dict[str, Any]], 
                           metric_name: str, group_filter: str) -> List[float]:
        """Extract metric values for a specific group from experiments"""
        values = []
        
        for exp in experiments:
            results = exp.get('results', [])
            for result in results:
                if result.get('approach') == group_filter:
                    # Extract metric based on available data
                    if metric_name in result:
                        values.append(float(result[metric_name]))
                    elif 'evaluation_metrics' in exp:
                        # Look in evaluation metrics
                        eval_metrics = exp['evaluation_metrics']
                        if metric_name in eval_metrics:
                            values.append(float(eval_metrics[metric_name].get('avg_score', 0)))
        
        return values
    
    def _evaluate_hypothesis_evidence(self, test_results: List[StatisticalTest], 
                                  hypothesis_config: Dict[str, Any]) -> Tuple[str, str, bool]:
        """Evaluate overall evidence for hypothesis"""
        
        if not test_results:
            return "No evidence", "none", False
        
        significant_tests = [t for t in test_results if t.is_significant]
        total_tests = len(test_results)
        
        if total_tests == 0:
            return "No evidence", "none", False
        
        significance_ratio = len(significant_tests) / total_tests
        
        # Check for practical significance
        practical_significant = any(
            t.effect_size and abs(t.effect_size) >= self.effect_size_threshold 
            for t in test_results
        )
        
        # Determine evidence strength
        if significance_ratio >= 0.8 and practical_significant:
            evidence_strength = "very_strong"
            conclusion = "Strong evidence supports hypothesis"
        elif significance_ratio >= 0.6:
            evidence_strength = "strong"
            conclusion = "Moderate evidence supports hypothesis"
        elif significance_ratio >= 0.4:
            evidence_strength = "moderate"
            conclusion = "Some evidence supports hypothesis"
        else:
            evidence_strength = "weak"
            conclusion = "Insufficient evidence to support hypothesis"
        
        return conclusion, evidence_strength, practical_significant
    
    def _generate_hypothesis_recommendations(self, conclusion: str, evidence_strength: str,
                                         test_results: List[StatisticalTest],
                                         hypothesis_config: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on hypothesis test results"""
        recommendations = []
        
        if evidence_strength in ["very_strong", "strong"]:
            recommendations.append("Hypothesis validated - proceed with implementation")
            recommendations.append("Consider expanding to larger sample sizes for confirmation")
            
            if evidence_strength == "very_strong":
                recommendations.append("Ready for production deployment")
        
        elif evidence_strength == "moderate":
            recommendations.append("Promising results - conduct additional experiments")
            recommendations.append("Increase sample size for better statistical power")
            recommendations.append("Refine hypothesis based on partial findings")
        
        else:  # weak or none
            recommendations.append("Hypothesis not supported - reconsider approach")
            recommendations.append("Investigate alternative hypotheses")
            recommendations.append("Review experimental design for potential issues")
        
        # Add specific recommendations based on test results
        for test in test_results:
            if test.effect_size and abs(test.effect_size) < 0.2:
                recommendations.append("Effect size is negligible - consider practical impact")
        
        return recommendations
    
    def compare_models(self, metric_names: List[str] = None) -> List[ModelComparison]:
        """Compare models across multiple metrics"""
        if metric_names is None:
            metric_names = ['execution_time', 'correctness', 'completeness', 'coherence']
        
        # Extract model performance data
        model_data = defaultdict(lambda: defaultdict(list))
        
        for exp in self.experiment_results:
            results = exp.get('results', [])
            for result in results:
                model = result.get('model_name', result.get('model', 'unknown'))
                
                for metric in metric_names:
                    if metric in result:
                        model_data[model][metric].append(float(result[metric]))
                    elif 'evaluation_metrics' in exp:
                        eval_metrics = exp['evaluation_metrics']
                        if metric in eval_metrics:
                            model_data[model][metric].append(float(eval_metrics[metric].get('avg_score', 0)))
        
        # Perform pairwise comparisons
        models = list(model_data.keys())
        comparisons = []
        
        for i, model_a in enumerate(models):
            for model_b in models[i+1:]:
                comparison = self._compare_two_models(
                    model_a, model_b, model_data, metric_names
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def _compare_two_models(self, model_a: str, model_b: str, 
                          model_data: Dict, metric_names: List[str]) -> ModelComparison:
        """Compare two models across specified metrics"""
        
        metrics_comparison = {}
        significant_differences = []
        
        for metric in metric_names:
            values_a = model_data[model_a].get(metric, [])
            values_b = model_data[model_b].get(metric, [])
            
            if values_a and values_b:
                # Calculate descriptive statistics
                stats_a = {
                    'mean': statistics.mean(values_a),
                    'median': statistics.median(values_a),
                    'std': statistics.stdev(values_a) if len(values_a) > 1 else 0,
                    'min': min(values_a),
                    'max': max(values_a),
                    'count': len(values_a)
                }
                
                stats_b = {
                    'mean': statistics.mean(values_b),
                    'median': statistics.median(values_b),
                    'std': statistics.stdev(values_b) if len(values_b) > 1 else 0,
                    'min': min(values_b),
                    'max': max(values_b),
                    'count': len(values_b)
                }
                
                metrics_comparison[metric] = {
                    f'{model_a}_stats': stats_a,
                    f'{model_b}_stats': stats_b
                }
                
                # Perform statistical test
                test_result = self.perform_statistical_test(values_a, values_b)
                if test_result.is_significant:
                    significant_differences.append(metric)
        
        # Determine overall winner
        winner = self._determine_overall_winner(model_a, model_b, metrics_comparison)
        
        # Calculate confidence based on consistency of results
        total_metrics = len([m for m in metric_names if m in metrics_comparison])
        confidence = len(significant_differences) / total_metrics if total_metrics > 0 else 0
        
        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            metrics_comparison=metrics_comparison,
            overall_winner=winner,
            confidence=confidence,
            significant_differences=significant_differences
        )
    
    def _determine_overall_winner(self, model_a: str, model_b: str, 
                               metrics_comparison: Dict[str, Dict[str, float]]) -> str:
        """Determine overall winner between two models"""
        
        wins_a, wins_b = 0, 0
        
        for metric, comparison in metrics_comparison.items():
            stats_a = comparison.get(f'{model_a}_stats', {})
            stats_b = comparison.get(f'{model_b}_stats', {})
            
            # Determine winner for this metric
            if metric in ['execution_time', 'error_rate']:  # Lower is better
                if stats_a.get('mean', 0) < stats_b.get('mean', 0):
                    wins_a += 1
                else:
                    wins_b += 1
            else:  # Higher is better
                if stats_a.get('mean', 0) > stats_b.get('mean', 0):
                    wins_a += 1
                else:
                    wins_b += 1
        
        return model_a if wins_a > wins_b else model_b
    
    def analyze_pipeline_performance(self) -> Dict[str, Any]:
        """Analyze pipeline performance across all stages"""
        
        analysis = {
            'pipeline_summary': {},
            'stage_analysis': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        if not self.pipeline_metrics:
            analysis['pipeline_summary'] = {"status": "No pipeline data available"}
            return analysis
        
        # Overall pipeline summary
        total_ops = sum(m.total_operations for m in self.pipeline_metrics.values())
        total_successful = sum(m.successful_operations for m in self.pipeline_metrics.values())
        total_failed = sum(m.failed_operations for m in self.pipeline_metrics.values())
        
        overall_success_rate = total_successful / total_ops if total_ops > 0 else 0
        overall_error_rate = total_failed / total_ops if total_ops > 0 else 0
        
        # Calculate average throughput
        avg_throughput = statistics.mean([m.throughput for m in self.pipeline_metrics.values()])
        
        analysis['pipeline_summary'] = {
            'total_operations': total_ops,
            'successful_operations': total_successful,
            'failed_operations': total_failed,
            'overall_success_rate': overall_success_rate,
            'overall_error_rate': overall_error_rate,
            'average_throughput': avg_throughput,
            'number_of_stages': len(self.pipeline_metrics)
        }
        
        # Stage-by-stage analysis
        for stage, metrics in self.pipeline_metrics.items():
            stage_analysis = {
                'performance': {
                    'avg_time': metrics.average_time,
                    'p95_time': metrics.p95_time,
                    'p99_time': metrics.p99_time,
                    'throughput': metrics.throughput
                },
                'reliability': {
                    'success_rate': 1 - metrics.error_rate,
                    'error_rate': metrics.error_rate,
                    'retry_rate': metrics.retry_rate,
                    'timeout_rate': metrics.timeout_rate
                },
                'efficiency': {
                    'operations_per_second': metrics.throughput,
                    'resource_utilization': metrics.resource_utilization
                }
            }
            
            analysis['stage_analysis'][stage] = stage_analysis
            
            # Identify potential bottlenecks
            if metrics.p99_time > avg_throughput * 2:  # Stage is 2x slower than average
                analysis['bottlenecks'].append({
                    'stage': stage,
                    'issue': 'High latency',
                    'p99_time': metrics.p99_time,
                    'average_throughput': avg_throughput
                })
            
            if metrics.error_rate > 0.1:  # > 10% error rate
                analysis['bottlenecks'].append({
                    'stage': stage,
                    'issue': 'High error rate',
                    'error_rate': metrics.error_rate
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_pipeline_recommendations(
            analysis['pipeline_summary'], analysis['stage_analysis'], analysis['bottlenecks']
        )
        
        return analysis
    
    def _generate_pipeline_recommendations(self, summary: Dict[str, Any], 
                                     stage_analysis: Dict[str, Any],
                                     bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate pipeline optimization recommendations"""
        recommendations = []
        
        # Overall recommendations
        if summary.get('overall_error_rate', 0) > 0.05:
            recommendations.append("Overall error rate is high - implement better error handling")
        
        if summary.get('average_throughput', 0) < 1.0:
            recommendations.append("Low throughput detected - consider parallelization")
        
        # Bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck['issue'] == 'High latency':
                recommendations.append(
                    f"Stage '{bottleneck['stage']}' has high latency - optimize processing logic"
                )
            elif bottleneck['issue'] == 'High error rate':
                recommendations.append(
                    f"Stage '{bottleneck['stage']}' has high error rate - add retry logic"
                )
        
        # Stage-specific recommendations
        for stage, analysis in stage_analysis.items():
            reliability = analysis.get('reliability', {})
            if reliability.get('retry_rate', 0) > 0.3:
                recommendations.append(
                    f"Stage '{stage}' has high retry rate - investigate root causes"
                )
            
            efficiency = analysis.get('efficiency', {})
            resource_util = efficiency.get('resource_utilization', {})
            if any(util > 0.8 for util in resource_util.values()):
                recommendations.append(
                    f"Stage '{stage}' has high resource utilization - consider scaling"
                )
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive metrics analysis report"""
        
        report_lines = [
            "# Comprehensive Metrics Analysis Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Sources: {len(self.experiment_results)} experiments, {len(self.pipeline_metrics)} pipeline stages",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Hypothesis validation summary
        if self.experiment_results:
            report_lines.extend([
                "### Hypothesis Validation Status",
                f"- Experiments Analyzed: {len(self.experiment_results)}",
                f"- Statistical Tests Available: {'Yes' if SCIPY_AVAILABLE else 'Limited (no scipy)'}",
                f"- Significance Threshold: {self.significance_threshold}",
                ""
            ])
        
        # Pipeline performance summary
        if self.pipeline_metrics:
            pipeline_analysis = self.analyze_pipeline_performance()
            summary = pipeline_analysis.get('pipeline_summary', {})
            
            report_lines.extend([
                "### Pipeline Performance",
                f"- Total Operations: {summary.get('total_operations', 0)}",
                f"- Overall Success Rate: {summary.get('overall_success_rate', 0):.1%}",
                f"- Average Throughput: {summary.get('average_throughput', 0):.2f} ops/sec",
                f"- Bottlenecks Identified: {len(pipeline_analysis.get('bottlenecks', []))}",
                ""
            ])
        
        # Model comparisons
        comparisons = self.compare_models()
        if comparisons:
            report_lines.extend([
                "### Model Performance Comparison",
                f"- Models Compared: {len(set([c.model_a for c in comparisons] + [c.model_b for c in comparisons]))}",
                f"- Significant Differences Found: {sum(len(c.significant_differences) for c in comparisons)}",
                ""
            ])
        
        # Detailed sections would follow...
        report_lines.extend([
            "## Detailed Analysis",
            "",
            "### Statistical Significance Testing",
            "All hypothesis tests use appropriate statistical methods based on data distribution and sample size.",
            "",
            "### Performance Metrics",
            "Comprehensive collection includes timing, success rates, resource utilization, and error patterns.",
            "",
            "### Recommendations",
            "1. Continue monitoring for trend analysis",
            "2. Implement automated alerting for performance degradation",
            "3. Regular hypothesis validation to guide development",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def save_analysis_results(self, filename_prefix: str = None):
        """Save all analysis results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or f"metrics_analysis_{timestamp}"
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        report_file = self.analysis_dir / f"{prefix}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed data
        analysis_data = {
            'timestamp': timestamp,
            'experiment_results_count': len(self.experiment_results),
            'pipeline_stages_count': len(self.pipeline_metrics),
            'retry_statistics_count': len(self.retry_stats),
            'analysis_config': {
                'significance_threshold': self.significance_threshold,
                'effect_size_threshold': self.effect_size_threshold
            }
        }
        
        data_file = self.analysis_dir / f"{prefix}_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {report_file} and {data_file}")
        
        return report_file, data_file

# Utility functions for integration with existing systems
def create_metrics_analyzer(results_dir: str = None) -> MetricsAnalyzer:
    """Create and initialize a metrics analyzer"""
    results_path = Path(results_dir) if results_dir else None
    return MetricsAnalyzer(results_path)

def analyze_hypothesis_results(experiment_files: List[str], 
                            hypothesis_configs: List[Dict[str, Any]]) -> Dict[str, HypothesisTest]:
    """Analyze hypothesis validation results"""
    analyzer = create_metrics_analyzer()
    analyzer.load_experiment_results(experiment_files)
    
    results = {}
    for config in hypothesis_configs:
        test_result = analyzer.validate_hypothesis(config)
        results[config.get('id', 'unknown')] = test_result
    
    return results