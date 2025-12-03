#!/usr/bin/env python3
"""
Experiment Analysis and Reporting Tools
Comprehensive analysis tools for Conjecture research experiments
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment"""
    experiment_id: str
    experiment_name: str
    experiment_type: str
    total_tests: int
    successful_tests: int
    success_rate: float
    avg_execution_time: float
    avg_scores_by_metric: Dict[str, float]
    model_performance: Dict[str, Dict[str, float]]
    hypothesis_supported: bool
    confidence_level: float


@dataclass
class ComparisonResult:
    """Result of comparing two experiments or models"""
    comparison_type: str
    item_a: str
    item_b: str
    metric: str
    difference: float
    statistical_significance: float
    effect_size: float
    interpretation: str


class ExperimentAnalyzer:
    """Comprehensive experiment analysis and reporting"""
    
    def __init__(self, results_dir: str = "research/results", 
                 output_dir: str = "research/analysis"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_experiment_results(self, experiment_id: str = None) -> List[Dict[str, Any]]:
        """Load experiment results from files"""
        results = []
        
        if experiment_id:
            # Load specific experiment
            pattern = f"*{experiment_id}*.json"
        else:
            # Load all experiments
            pattern = "*.json"
        
        for file_path in self.results_dir.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def analyze_experiment(self, experiment_data: Dict[str, Any]) -> ExperimentSummary:
        """Analyze a single experiment"""
        config = experiment_data.get('experiment_config', {})
        test_results = experiment_data.get('test_results', [])
        evaluation_results = experiment_data.get('evaluation_results', [])
        summary = experiment_data.get('summary', {})
        
        # Basic statistics
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if not r.get('error')])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Execution time
        execution_times = [r.get('execution_time_seconds', 0) for r in test_results 
                          if not r.get('error')]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        # Average scores by metric
        avg_scores_by_metric = {}
        if evaluation_results:
            # Group by metric
            metrics = {}
            for eval_result in evaluation_results:
                metric = eval_result.get('metric', 'unknown')
                score = eval_result.get('score', 0)
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(score)
            
            for metric, scores in metrics.items():
                avg_scores_by_metric[metric] = statistics.mean(scores)
        
        # Model performance
        model_performance = {}
        if test_results:
            # Group by model
            model_results = {}
            for result in test_results:
                model = result.get('model_name', 'unknown')
                if model not in model_results:
                    model_results[model] = []
                model_results[model].append(result)
            
            for model, results in model_results.items():
                # Get evaluations for this model
                model_evals = [e for e in evaluation_results 
                             if e.get('test_result_id', '').endswith(f"_{model}")]
                
                if model_evals:
                    scores = [e.get('score', 0) for e in model_evals]
                    model_performance[model] = {
                        'avg_score': statistics.mean(scores),
                        'num_tests': len(results),
                        'success_rate': len([r for r in results if not r.get('error')]) / len(results)
                    }
                else:
                    model_performance[model] = {
                        'avg_score': 0,
                        'num_tests': len(results),
                        'success_rate': len([r for r in results if not r.get('error')]) / len(results)
                    }
        
        # Hypothesis evaluation
        hypothesis = config.get('hypothesis', '')
        hypothesis_supported = self._evaluate_hypothesis(hypothesis, avg_scores_by_metric, 
                                                       model_performance)
        confidence_level = self._calculate_confidence_level(evaluation_results)
        
        return ExperimentSummary(
            experiment_id=config.get('experiment_id', 'unknown'),
            experiment_name=config.get('name', 'Unknown'),
            experiment_type=config.get('experiment_type', 'unknown'),
            total_tests=total_tests,
            successful_tests=successful_tests,
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            avg_scores_by_metric=avg_scores_by_metric,
            model_performance=model_performance,
            hypothesis_supported=hypothesis_supported,
            confidence_level=confidence_level
        )
    
    def _evaluate_hypothesis(self, hypothesis: str, scores: Dict[str, float], 
                           model_performance: Dict[str, Dict[str, float]]) -> bool:
        """Evaluate if hypothesis is supported by results"""
        if not hypothesis:
            return False
        
        hypothesis_lower = hypothesis.lower()
        
        # Check for improvement claims
        if 'improvement' in hypothesis_lower or 'better' in hypothesis_lower:
            # Look for evidence of improvement
            if scores:
                avg_score = statistics.mean(scores.values())
                return avg_score > 0.7  # Threshold for "good" performance
        
        # Check for comparison claims
        if 'small model' in hypothesis_lower or 'tiny' in hypothesis_lower:
            # Check if small models performed well
            small_models = [m for m in model_performance.keys() 
                          if 'tiny' in m.lower() or 'small' in m.lower() or '9b' in m.lower()]
            if small_models:
                small_model_scores = [model_performance[m]['avg_score'] for m in small_models]
                return statistics.mean(small_model_scores) > 0.6
        
        # Default heuristic
        if scores:
            return statistics.mean(scores.values()) > 0.65
        
        return False
    
    def _calculate_confidence_level(self, evaluation_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence level in results"""
        if not evaluation_results:
            return 0.0
        
        # Use judge confidence as proxy
        confidences = [e.get('confidence', 0.5) for e in evaluation_results]
        return statistics.mean(confidences)
    
    def compare_models(self, experiment_data: Dict[str, Any]) -> List[ComparisonResult]:
        """Compare model performance within an experiment"""
        test_results = experiment_data.get('test_results', [])
        evaluation_results = experiment_data.get('evaluation_results', [])
        
        # Group results by model
        model_scores = {}
        for eval_result in evaluation_results:
            test_result_id = eval_result.get('test_result_id', '')
            metric = eval_result.get('metric', 'unknown')
            score = eval_result.get('score', 0)
            
            # Extract model name from test_result_id
            parts = test_result_id.split('_')
            if len(parts) >= 2:
                model_name = '_'.join(parts[1:])  # Rejoin after test case ID
                
                if model_name not in model_scores:
                    model_scores[model_name] = {}
                if metric not in model_scores[model_name]:
                    model_scores[model_name][metric] = []
                model_scores[model_name][metric].append(score)
        
        # Compare all pairs of models
        comparisons = []
        models = list(model_scores.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                
                # Compare on common metrics
                common_metrics = set(model_scores[model_a].keys()) & set(model_scores[model_b].keys())
                
                for metric in common_metrics:
                    scores_a = model_scores[model_a][metric]
                    scores_b = model_scores[model_b][metric]
                    
                    if len(scores_a) >= 3 and len(scores_b) >= 3:  # Minimum sample size
                        # Statistical test
                        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                        
                        # Effect size (Cohen's d)
                        pooled_std = statistics.sqrt(
                            ((len(scores_a) - 1) * statistics.variance(scores_a) + 
                             (len(scores_b) - 1) * statistics.variance(scores_b)) / 
                            (len(scores_a) + len(scores_b) - 2)
                        )
                        effect_size = (statistics.mean(scores_a) - statistics.mean(scores_b)) / pooled_std if pooled_std > 0 else 0
                        
                        # Interpretation
                        if p_value < 0.05:
                            if effect_size > 0.5:
                                interpretation = f"{model_a} significantly outperforms {model_b}"
                            elif effect_size < -0.5:
                                interpretation = f"{model_b} significantly outperforms {model_a}"
                            else:
                                interpretation = f"Statistically significant but small difference"
                        else:
                            interpretation = "No statistically significant difference"
                        
                        comparison = ComparisonResult(
                            comparison_type="model_comparison",
                            item_a=model_a,
                            item_b=model_b,
                            metric=metric,
                            difference=statistics.mean(scores_a) - statistics.mean(scores_b),
                            statistical_significance=p_value,
                            effect_size=effect_size,
                            interpretation=interpretation
                        )
                        comparisons.append(comparison)
        
        return comparisons
    
    def generate_visualizations(self, experiment_data: Dict[str, Any], 
                              experiment_summary: ExperimentSummary):
        """Generate visualizations for experiment results"""
        experiment_id = experiment_summary.experiment_id
        
        # Model performance comparison
        if experiment_summary.model_performance:
            models = list(experiment_summary.model_performance.keys())
            scores = [experiment_summary.model_performance[m]['avg_score'] for m in models]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, scores)
            plt.title(f'Model Performance - {experiment_summary.experiment_name}')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{experiment_id}_model_performance.png', dpi=300)
            plt.close()
        
        # Metric breakdown
        if experiment_summary.avg_scores_by_metric:
            metrics = list(experiment_summary.avg_scores_by_metric.keys())
            scores = list(experiment_summary.avg_scores_by_metric.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(metrics, scores)
            plt.title(f'Performance by Metric - {experiment_summary.experiment_name}')
            plt.ylabel('Average Score')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{experiment_id}_metric_breakdown.png', dpi=300)
            plt.close()
        
        # Success rate pie chart
        if experiment_summary.total_tests > 0:
            plt.figure(figsize=(8, 8))
            sizes = [experiment_summary.successful_tests, 
                    experiment_summary.total_tests - experiment_summary.successful_tests]
            labels = ['Successful', 'Failed']
            colors = ['#2ecc71', '#e74c3c']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Test Success Rate - {experiment_summary.experiment_name}')
            plt.axis('equal')
            
            plt.savefig(self.output_dir / f'{experiment_id}_success_rate.png', dpi=300)
            plt.close()
    
    def generate_comprehensive_report(self, experiment_ids: List[str] = None) -> str:
        """Generate comprehensive analysis report"""
        if experiment_ids:
            results = []
            for exp_id in experiment_ids:
                exp_results = self.load_experiment_results(exp_id)
                results.extend(exp_results)
        else:
            results = self.load_experiment_results()
        
        if not results:
            return "No experiment results found."
        
        report = []
        report.append("# Conjecture Research Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- Total experiments analyzed: {len(results)}")
        
        summaries = [self.analyze_experiment(result) for result in results]
        successful_experiments = [s for s in summaries if s.success_rate > 0.8]
        hypothesis_supported = [s for s in summaries if s.hypothesis_supported]
        
        report.append(f"- Successful experiments: {len(successful_experiments)}/{len(results)}")
        report.append(f"- Hypotheses supported: {len(hypothesis_supported)}/{len(results)}")
        report.append("")
        
        # Individual Experiment Analysis
        report.append("## Individual Experiment Analysis")
        report.append("")
        
        for summary in summaries:
            report.append(f"### {summary.experiment_name}")
            report.append(f"**Experiment ID:** {summary.experiment_id}")
            report.append(f"**Type:** {summary.experiment_type}")
            report.append(f"**Success Rate:** {summary.success_rate:.1%}")
            report.append(f"**Average Execution Time:** {summary.avg_execution_time:.2f}s")
            
            if summary.avg_scores_by_metric:
                report.append("**Performance by Metric:**")
                for metric, score in summary.avg_scores_by_metric.items():
                    report.append(f"- {metric}: {score:.3f}")
            
            if summary.model_performance:
                report.append("**Model Performance:**")
                sorted_models = sorted(summary.model_performance.items(), 
                                     key=lambda x: x[1]['avg_score'], reverse=True)
                for model, perf in sorted_models:
                    report.append(f"- {model}: {perf['avg_score']:.3f} ({perf['num_tests']} tests)")
            
            report.append(f"**Hypothesis Supported:** {'✅ Yes' if summary.hypothesis_supported else '❌ No'}")
            report.append(f"**Confidence Level:** {summary.confidence_level:.3f}")
            report.append("")
        
        # Cross-Experiment Analysis
        report.append("## Cross-Experiment Analysis")
        report.append("")
        
        # Model performance across all experiments
        all_model_performance = {}
        for summary in summaries:
            for model, perf in summary.model_performance.items():
                if model not in all_model_performance:
                    all_model_performance[model] = []
                all_model_performance[model].append(perf['avg_score'])
        
        if all_model_performance:
            report.append("### Overall Model Performance")
            for model, scores in all_model_performance.items():
                avg_score = statistics.mean(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0
                report.append(f"- {model}: {avg_score:.3f} ± {std_score:.3f}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if hypothesis_supported:
            best_hypotheses = sorted([(s.experiment_name, s.confidence_level) 
                                    for s in hypothesis_supported], 
                                   key=lambda x: x[1], reverse=True)
            report.append("### Validated Approaches")
            for name, confidence in best_hypotheses[:3]:
                report.append(f"- **{name}** (confidence: {confidence:.3f})")
            report.append("")
        
        if all_model_performance:
            best_models = sorted(all_model_performance.items(), 
                               key=lambda x: statistics.mean(x[1]), reverse=True)
            report.append("### Best Performing Models")
            for model, scores in best_models[:3]:
                avg_score = statistics.mean(scores)
                report.append(f"- **{model}** (average score: {avg_score:.3f})")
            report.append("")
        
        report.append("### Next Steps")
        report.append("1. Focus on validated hypotheses for further development")
        report.append("2. Investigate failed experiments to understand limitations")
        report.append("3. Expand test cases for successful approaches")
        report.append("4. Optimize model selection based on performance data")
        report.append("")
        
        return "\n".join(report)
    
    def save_analysis(self, experiment_data: Dict[str, Any]):
        """Save complete analysis for an experiment"""
        summary = self.analyze_experiment(experiment_data)
        comparisons = self.compare_models(experiment_data)
        
        # Generate visualizations
        self.generate_visualizations(experiment_data, summary)
        
        # Save summary
        summary_file = self.output_dir / f"{summary.experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        
        # Save comparisons
        if comparisons:
            comparison_file = self.output_dir / f"{summary.experiment_id}_comparisons.json"
            with open(comparison_file, 'w') as f:
                json.dump([asdict(c) for c in comparisons], f, indent=2)
        
        print(f"Analysis saved for {summary.experiment_name}")
        return summary


def main():
    """Main function to analyze experiments"""
    analyzer = ExperimentAnalyzer()
    
    # Load all experiments
    results = analyzer.load_experiment_results()
    print(f"Found {len(results)} experiment results")
    
    # Analyze each experiment
    summaries = []
    for result in results:
        summary = analyzer.save_analysis(result)
        summaries.append(summary)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Save report
    report_file = analyzer.output_dir / "comprehensive_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nComprehensive report saved to: {report_file}")
    print(f"\nSummary of analyzed experiments:")
    for summary in summaries:
        status = "✅" if summary.hypothesis_supported else "❌"
        print(f"  {status} {summary.experiment_name}: {summary.success_rate:.1%} success rate")


if __name__ == "__main__":
    main()