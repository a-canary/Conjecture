#!/usr/bin/env python3
"""
Comprehensive Test Suite for Metrics Collection and Analysis Framework
Tests all components of the metrics system including statistical analysis,
hypothesis validation, model comparison, and visualization
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monitoring import (
    PerformanceMonitor, 
    MetricsAnalyzer, 
    MetricsVisualizer,
    RetryTracker,
    RetryContext,
    create_metrics_analyzer,
    create_retry_tracker,
    create_visualizer
)

logger = logging.getLogger(__name__)


class ComprehensiveMetricsTest:
    """Comprehensive test suite for metrics framework"""
    
    def __init__(self):
        self.test_results = []
        self.output_dir = Path("tests/results/comprehensive_metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_models = ["model_a", "model_b", "model_c"]
        self.test_operations = ["llm_call", "database_query", "context_building"]
        self.test_duration_minutes = 2
        
    async def run_all_tests(self):
        """Run comprehensive tests of metrics framework"""
        print("🧪 COMPREHENSIVE METRICS FRAMEWORK TESTS")
        print("=" * 60)
        
        test_methods = [
            self.test_performance_monitoring,
            self.test_metrics_analysis,
            self.test_statistical_significance,
            self.test_hypothesis_validation,
            self.test_model_comparison,
            self.test_retry_tracking,
            self.test_visualization,
            self.test_integration
        ]
        
        for test_method in test_methods:
            try:
                print(f"\nRunning: {test_method.__name__}")
                result = await test_method()
                self.test_results.append({
                    'test_name': test_method.__name__,
                    'status': 'passed' if result['success'] else 'failed',
                    'duration': result['duration'],
                    'details': result['details']
                })
                
                status_marker = "[PASS]" if result['success'] else "[FAIL]"
                print(f"{status_marker} {test_method.__name__}: {result['summary']}")
                
            except Exception as e:
                error_msg = f"Test failed with exception: {e}"
                print(f"[ERROR] {test_method.__name__}: {error_msg}")
                self.test_results.append({
                    'test_name': test_method.__name__,
                    'status': 'error',
                    'duration': 0,
                    'details': {'error': str(e)}
                })
        
        # Generate comprehensive report
        await self.generate_test_report()
        
        # Print summary
        passed_tests = len([r for r in self.test_results if r['status'] == 'passed'])
        total_tests = len(self.test_results)
        
        print(f"\nTEST SUMMARY")
        print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("[SUCCESS] ALL TESTS PASSED - Metrics framework is fully functional!")
        else:
            print("[WARNING] Some tests failed - review implementation")
        
        return passed_tests == total_tests
    
    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring functionality"""
        start_time = time.time()
        
        try:
            # Initialize performance monitor
            monitor = PerformanceMonitor(
                max_history_size=1000,
                snapshot_interval=1,
                enable_system_monitoring=True
            )
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Simulate various operations
            for i in range(10):
                # Record timing metrics
                with monitor.timer("test_operation"):
                    await asyncio.sleep(0.01 + (i * 0.005))  # Variable timing
                
                # Record counters
                monitor.increment_counter("test_counter", 1, {"iteration": str(i)})
                
                # Record gauges
                monitor.set_gauge("test_gauge", i * 10, {"type": "test"})
                
                # Record cache performance
                if i % 2 == 0:
                    monitor.record_cache_hit("test_cache", "hit")
                else:
                    monitor.record_cache_miss("test_cache", "miss")
                
                # Record database operations
                monitor.record_database_operation("select", 0.005, 5)
                
                # Record LLM operations
                monitor.record_llm_operation("completion", 0.1, 50 + i * 5)
            
            # Wait for snapshots
            await asyncio.sleep(2)
            
            # Get performance summary
            summary = monitor.get_performance_summary(time_window_minutes=5)
            
            # Validate summary contains expected data
            required_keys = ['total_metrics', 'performance_stats', 'counters', 'gauges']
            missing_keys = [key for key in required_keys if key not in summary]
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Export metrics
            export_file = self.output_dir / "performance_test_export.json"
            monitor.export_metrics(str(export_file), time_window_hours=1)
            
            success = len(missing_keys) == 0 and export_file.exists()
            
            return {
                'success': success,
                'duration': time.time() - start_time,
                'summary': f"Performance monitoring {'✅' if success else '❌'}",
                'details': {
                    'summary_keys': list(summary.keys()),
                    'missing_keys': missing_keys,
                    'total_metrics': summary.get('total_metrics', 0),
                    'export_file_exists': export_file.exists()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Performance monitoring test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_metrics_analysis(self) -> Dict[str, Any]:
        """Test metrics analysis functionality"""
        start_time = time.time()
        
        try:
            # Create metrics analyzer
            analyzer = create_metrics_analyzer(str(self.output_dir))
            
            # Create sample experiment data
            sample_experiments = self._create_sample_experiment_data()
            
            # Load sample data
            for exp_data in sample_experiments:
                analyzer.experiment_results.append(exp_data)
            
            # Test statistical analysis
            group_a = [0.5, 0.6, 0.7, 0.8, 0.9]
            group_b = [0.4, 0.5, 0.6, 0.7, 0.8]
            
            statistical_test = analyzer.perform_statistical_test(
                group_a, group_b, test_type="t_test"
            )
            
            # Test hypothesis validation
            hypothesis_config = {
                'id': 'test_hyp_001',
                'hypothesis': 'Group A performs better than Group B',
                'null_hypothesis': 'No difference between groups',
                'type': 'improvement',
                'control_group': 'group_b',
                'treatment_group': 'group_a',
                'control_metric': 'performance',
                'treatment_metric': 'performance',
                'direction': 'improvement',
                'test_type': 't_test'
            }
            
            hypothesis_result = analyzer.validate_hypothesis(hypothesis_config)
            
            # Test model comparison
            model_comparisons = analyzer.compare_models(['execution_time', 'accuracy'])
            
            # Test pipeline analysis
            pipeline_data = self._create_sample_pipeline_data()
            for stage, metrics in pipeline_data.items():
                analyzer.add_pipeline_metrics(stage, metrics)
            
            pipeline_analysis = analyzer.analyze_pipeline_performance()
            
            # Test comprehensive report generation
            report = analyzer.generate_comprehensive_report()
            
            # Save analysis results
            report_file, data_file = analyzer.save_analysis_results("comprehensive_test")
            
            success = (
                statistical_test.is_significant is not None and
                hypothesis_result.evidence_strength != 'none' and
                len(model_comparisons) > 0 and
                len(pipeline_analysis) > 0 and
                report_file.exists() and
                data_file.exists()
            )
            
            return {
                'success': success,
                'duration': time.time() - start_time,
                'summary': f"Metrics analysis {'✅' if success else '❌'}",
                'details': {
                    'statistical_test_significant': statistical_test.is_significant,
                    'hypothesis_evidence_strength': hypothesis_result.evidence_strength,
                    'model_comparisons_count': len(model_comparisons),
                    'pipeline_stages_analyzed': len(pipeline_analysis),
                    'report_generated': report_file.exists(),
                    'data_saved': data_file.exists()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Metrics analysis test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_statistical_significance(self) -> Dict[str, Any]:
        """Test statistical significance functionality"""
        start_time = time.time()
        
        try:
            analyzer = create_metrics_analyzer()
            
            # Test different statistical tests
            test_cases = [
                {
                    'name': 't_test_normal',
                    'group_a': [1.0, 1.1, 1.2, 1.3, 1.4],
                    'group_b': [0.8, 0.9, 1.0, 1.1, 1.2],
                    'test_type': 't_test',
                    'expected_significant': True
                },
                {
                    'name': 'mann_whitney',
                    'group_a': [1.0, 1.5, 2.0, 2.5, 10.0],  # Outlier
                    'group_b': [1.1, 1.2, 1.3, 1.4, 1.5],
                    'test_type': 'mann_whitney',
                    'expected_significant': False  # Due to outlier
                },
                {
                    'name': 'no_difference',
                    'group_a': [1.0, 1.1, 1.2, 1.3, 1.4],
                    'group_b': [1.0, 1.1, 1.2, 1.3, 1.4],
                    'test_type': 't_test',
                    'expected_significant': False
                }
            ]
            
            test_results = []
            
            for test_case in test_cases:
                result = analyzer.perform_statistical_test(
                    test_case['group_a'],
                    test_case['group_b'],
                    test_type=test_case['test_type']
                )
                
                # Validate result
                success = (
                    result.test_name == test_case['test_type'] and
                    result.p_value is not None and
                    0 <= result.p_value <= 1.0
                )
                
                # Check if significance matches expectation
                if test_case['expected_significant']:
                    success = success and result.is_significant
                else:
                    success = success and not result.is_significant
                
                test_results.append({
                    'test_case': test_case['name'],
                    'success': success,
                    'p_value': result.p_value,
                    'is_significant': result.is_significant,
                    'effect_size': result.effect_size
                })
            
            all_passed = all(r['success'] for r in test_results)
            
            return {
                'success': all_passed,
                'duration': time.time() - start_time,
                'summary': f"Statistical significance tests {'✅' if all_passed else '❌'}",
                'details': {
                    'tests_run': len(test_results),
                    'tests_passed': len([r for r in test_results if r['success']]),
                    'test_results': test_results
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Statistical significance test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_hypothesis_validation(self) -> Dict[str, Any]:
        """Test hypothesis validation functionality"""
        start_time = time.time()
        
        try:
            analyzer = create_metrics_analyzer()
            
            # Create sample experiment results
            experiment_results = self._create_sample_experiment_data()
            for exp_data in experiment_results:
                analyzer.experiment_results.append(exp_data)
            
            # Test multiple hypothesis types
            hypothesis_tests = [
                {
                    'id': 'improvement_test',
                    'hypothesis': 'New approach improves performance by 20%',
                    'null_hypothesis': 'No performance improvement',
                    'type': 'improvement',
                    'experiment_id': 'exp_001',
                    'control_group': 'baseline',
                    'treatment_group': 'new_approach',
                    'expected_evidence': 'moderate'
                },
                {
                    'id': 'comparison_test',
                    'hypothesis': 'Model A outperforms Model B',
                    'null_hypothesis': 'No performance difference',
                    'type': 'comparison',
                    'experiment_id': 'exp_002',
                    'control_group': 'model_b',
                    'treatment_group': 'model_a',
                    'expected_evidence': 'strong'
                }
            ]
            
            validation_results = []
            
            for hyp_config in hypothesis_tests:
                result = analyzer.validate_hypothesis(hyp_config)
                
                # Validate result structure
                success = (
                    result.hypothesis_id == hyp_config['id'] and
                    result.conclusion != "" and
                    result.evidence_strength in ['weak', 'moderate', 'strong', 'very_strong']
                )
                
                validation_results.append({
                    'hypothesis_id': hyp_config['id'],
                    'success': success,
                    'evidence_strength': result.evidence_strength,
                    'practical_significance': result.practical_significance,
                    'recommendations_count': len(result.recommendations)
                })
            
            all_passed = all(r['success'] for r in validation_results)
            
            return {
                'success': all_passed,
                'duration': time.time() - start_time,
                'summary': f"Hypothesis validation {'✅' if all_passed else '❌'}",
                'details': {
                    'hypotheses_tested': len(hypothesis_tests),
                    'validations_passed': len([r for r in validation_results if r['success']]),
                    'validation_results': validation_results
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Hypothesis validation test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_model_comparison(self) -> Dict[str, Any]:
        """Test model comparison functionality"""
        start_time = time.time()
        
        try:
            analyzer = create_metrics_analyzer()
            
            # Create sample model data
            model_data = self._create_sample_model_data()
            for exp_data in model_data:
                analyzer.experiment_results.append(exp_data)
            
            # Test model comparison
            comparisons = analyzer.compare_models(['accuracy', 'speed', 'efficiency'])
            
            # Validate comparison results
            success = len(comparisons) > 0
            
            for comparison in comparisons:
                if not hasattr(comparison, 'model_a') or not hasattr(comparison, 'model_b'):
                    success = False
                    break
                
                if not hasattr(comparison, 'overall_winner'):
                    success = False
                    break
            
            return {
                'success': success,
                'duration': time.time() - start_time,
                'summary': f"Model comparison {'✅' if success else '❌'}",
                'details': {
                    'comparisons_generated': len(comparisons),
                    'comparisons_valid': success,
                    'models_compared': len(set([c.model_a for c in comparisons] + [c.model_b for c in comparisons]))
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Model comparison test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_retry_tracking(self) -> Dict[str, Any]:
        """Test retry tracking functionality"""
        start_time = time.time()
        
        try:
            # Create retry tracker
            retry_tracker = create_retry_tracker(max_events=1000)
            
            # Start background analysis
            retry_tracker.start_background_analysis()
            
            # Simulate various operations with retries
            operations = [
                {'type': 'llm_call', 'success_rate': 0.7, 'max_retries': 2},
                {'type': 'database_query', 'success_rate': 0.9, 'max_retries': 1},
                {'type': 'api_request', 'success_rate': 0.5, 'max_retries': 3}
            ]
            
            for i, op_config in enumerate(operations):
                # Simulate multiple operations of each type
                for j in range(5):
                    with RetryContext(retry_tracker, op_config['type'], max_attempts=op_config['max_retries']) as ctx:
                        # Simulate operation with potential retries
                        success_probability = op_config['success_rate'] * (1 + j * 0.1)  # Decreasing success
                        
                        if j < op_config['max_retries'] and (hash(f"{i}{j}") % 10 > success_probability * 10):
                            # Simulate failure
                            ctx.record_failure(f"SimulatedError_{j}", f"Operation {j} failed")
                        else:
                            # Simulate success
                            ctx.record_success()
                            break
            
            # Wait for background analysis
            await asyncio.sleep(1)
            
            # Get retry statistics
            retry_stats = retry_tracker.get_retry_statistics()
            real_time_metrics = retry_tracker.get_real_time_metrics()
            
            # Test error pattern analysis
            error_patterns = retry_tracker.analyze_error_patterns()
            
            # Generate retry report
            retry_report = retry_tracker.generate_retry_report()
            
            # Stop background analysis
            retry_tracker.stop_background_analysis()
            
            # Validate results
            success = (
                len(retry_stats) > 0 and
                'error_rates' in real_time_metrics and
                len(error_patterns) >= 0 and
                len(retry_report) > 0
            )
            
            return {
                'success': success,
                'duration': time.time() - start_time,
                'summary': f"Retry tracking {'✅' if success else '❌'}",
                'details': {
                    'retry_statistics_count': len(retry_stats),
                    'real_time_metrics_available': 'error_rates' in real_time_metrics,
                    'error_patterns_analyzed': len(error_patterns),
                    'report_generated': len(retry_report) > 0
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Retry tracking test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_visualization(self) -> Dict[str, Any]:
        """Test visualization functionality"""
        start_time = time.time()
        
        try:
            # Create visualizer
            visualizer = create_visualizer(str(self.output_dir))
            
            # Test different chart types
            test_data = {
                'timeline_data': self._create_sample_timeline_data(),
                'model_comparison': self._create_sample_comparison_data(),
                'response_times': self._create_sample_distribution_data(),
                'summary': {
                    'total_operations': 1000,
                    'success_rate': 0.85,
                    'average_response_time': 0.25
                }
            }
            
            chart_paths = []
            
            # Test timeline chart
            if 'timeline_data' in test_data:
                timeline_path = visualizer.create_performance_timeline(test_data['timeline_data'])
                chart_paths.append(timeline_path)
            
            # Test model comparison chart
            if 'model_comparison' in test_data:
                comparison_path = visualizer.create_model_comparison_chart(test_data['model_comparison'])
                chart_paths.append(comparison_path)
            
            # Test distribution chart
            if 'response_times' in test_data:
                dist_path = visualizer.create_distribution_chart(test_data['response_times'])
                chart_paths.append(dist_path)
            
            # Test dashboard
            dashboard_path = visualizer.create_dashboard(test_data)
            chart_paths.append(dashboard_path)
            
            # Validate charts were created
            charts_created = len([p for p in chart_paths if p and Path(p).exists()])
            
            return {
                'success': charts_created >= 3,  # At least 3 charts should be created
                'duration': time.time() - start_time,
                'summary': f"Visualization {'✅' if charts_created >= 3 else '❌'}",
                'details': {
                    'charts_attempted': len(chart_paths),
                    'charts_created': charts_created,
                    'chart_paths': chart_paths
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Visualization test failed: {e}",
                'details': {'error': str(e)}
            }
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration between all metrics components"""
        start_time = time.time()
        
        try:
            # Create all components
            monitor = PerformanceMonitor()
            analyzer = create_metrics_analyzer(str(self.output_dir))
            retry_tracker = create_retry_tracker()
            visualizer = create_visualizer(str(self.output_dir))
            
            # Start all systems
            monitor.start_monitoring()
            retry_tracker.start_background_analysis()
            
            # Simulate integrated workflow
            for i in range(5):
                operation_id = f"integrated_op_{i}"
                
                # Start operation tracking
                with RetryContext(retry_tracker, 'integrated_test', max_attempts=2) as retry_ctx:
                    
                    # Monitor performance
                    with monitor.timer("integrated_operation"):
                        await asyncio.sleep(0.02)
                    
                    # Record metrics
                    monitor.record_llm_operation("test", 0.02, 10)
                    monitor.increment_counter("integrated_ops", 1)
                    
                    # Simulate success/failure
                    if i % 4 != 0:  # 80% success rate
                        retry_ctx.record_success()
                    else:
                        retry_ctx.record_failure("IntegrationTestError", f"Operation {i} failed")
                
                # Add to analyzer
                analyzer.experiment_results.append({
                    'experiment_id': f'integration_test_{i}',
                    'results': [{
                        'model_name': f'test_model_{i % 3}',
                        'approach': 'integrated',
                        'execution_time': 0.02,
                        'success': i % 4 != 0,
                        'value': 0.8 + (i * 0.05)
                    }]
                })
            
            # Wait for data collection
            await asyncio.sleep(1)
            
            # Generate integrated analysis
            model_comparisons = analyzer.compare_models()
            pipeline_analysis = analyzer.analyze_pipeline_performance()
            
            # Create visualizations
            viz_data = {
                'timeline_data': self._create_sample_timeline_data(),
                'model_comparison': self._create_sample_comparison_data(),
                'summary': {
                    'total_operations': 5,
                    'success_rate': 0.8
                }
            }
            
            dashboard_path = visualizer.create_dashboard(viz_data)
            
            # Stop all systems
            monitor.stop_monitoring()
            retry_tracker.stop_background_analysis()
            
            # Validate integration
            success = (
                len(model_comparisons) > 0 and
                len(pipeline_analysis) > 0 and
                dashboard_path and Path(dashboard_path).exists()
            )
            
            return {
                'success': success,
                'duration': time.time() - start_time,
                'summary': f"Integration test {'✅' if success else '❌'}",
                'details': {
                    'components_integrated': 4,
                    'model_comparisons_generated': len(model_comparisons),
                    'pipeline_analysis_completed': len(pipeline_analysis) > 0,
                    'dashboard_created': dashboard_path is not None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'summary': f"Integration test failed: {e}",
                'details': {'error': str(e)}
            }
    
    def _create_sample_experiment_data(self) -> List[Dict[str, Any]]:
        """Create sample experiment data for testing"""
        experiments = []
        
        for i in range(3):
            exp_data = {
                'experiment_id': f'sample_exp_{i}',
                'results': [],
                'evaluation_metrics': {
                    'correctness': {'avg_score': 0.7 + (i * 0.1)},
                    'completeness': {'avg_score': 0.6 + (i * 0.15)},
                    'efficiency': {'avg_score': 0.8 + (i * 0.05)}
                }
            }
            
            # Add results for different models and approaches
            for model in self.test_models:
                for approach in ['conjecture', 'direct']:
                    exp_data['results'].append({
                        'model_name': model,
                        'approach': approach,
                        'execution_time': 0.5 + (i * 0.1) - (0.1 if approach == 'conjecture' else 0),
                        'success': True,
                        'value': 0.6 + (i * 0.1) + (0.1 if approach == 'conjecture' else 0)
                    })
            
            experiments.append(exp_data)
        
        return experiments
    
    def _create_sample_timeline_data(self) -> List[Dict[str, Any]]:
        """Create sample timeline data"""
        timeline_data = []
        base_time = datetime.utcnow()
        
        for i in range(20):
            timestamp = base_time + timedelta(minutes=i * 5)
            timeline_data.append({
                'timestamp': timestamp.isoformat(),
                'value': 0.2 + (i * 0.01) + (0.05 if i % 3 == 0 else 0),
                'model': self.test_models[i % len(self.test_models)]
            })
        
        return timeline_data
    
    def _create_sample_comparison_data(self) -> Dict[str, Dict[str, float]]:
        """Create sample model comparison data"""
        comparison_data = {}
        
        for model in self.test_models:
            comparison_data[model] = {
                'accuracy': 0.7 + (hash(model) % 10) * 0.03,
                'speed': 0.5 + (hash(model) % 8) * 0.05,
                'efficiency': 0.8 + (hash(model) % 6) * 0.02
            }
        
        return comparison_data
    
    def _create_sample_distribution_data(self) -> List[float]:
        """Create sample distribution data"""
        import random
        
        # Generate slightly skewed distribution
        data = []
        for i in range(100):
            base_value = 0.2
            noise = random.gauss(0, 0.05)
            data.append(max(0.05, base_value + noise))
        
        return data
    
    def _create_sample_pipeline_data(self) -> Dict[str, Dict[str, Any]]:
        """Create sample pipeline data"""
        pipeline_data = {}
        
        stages = ['context_building', 'llm_processing', 'response_parsing', 'result_storage']
        
        for i, stage in enumerate(stages):
            pipeline_data[stage] = {
                'total_operations': 50 + (i * 10),
                'successful_operations': 45 + (i * 8),
                'failed_operations': 5 + (i * 2),
                'average_time': 0.1 + (i * 0.05),
                'p95_time': 0.15 + (i * 0.08),
                'p99_time': 0.2 + (i * 0.1),
                'throughput': 10 - (i * 2),
                'error_rate': 0.1 + (i * 0.02),
                'retry_rate': 0.05 + (i * 0.01),
                'timeout_rate': 0.02 + (i * 0.005),
                'resource_utilization': {
                    'cpu_percent': 50 + (i * 10),
                    'memory_percent': 60 + (i * 5)
                }
            }
        
        return pipeline_data
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        report_lines = [
            "# Comprehensive Metrics Framework Test Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Tests: {len(self.test_results)}",
            "",
            "## Test Results Summary",
            "",
            "| Test Name | Status | Duration (s) | Summary |",
            "|-----------|--------|---------------|--------|"
        ]
        
        for result in self.test_results:
            status_emoji = "✅" if result['status'] == 'passed' else "❌"
            report_lines.append(
                f"| {result['test_name']} | {status_emoji} {result['status']} | "
                f"{result['duration']:.2f} | {result['summary']} |"
            )
        
        # Add detailed results
        report_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        for result in self.test_results:
            report_lines.extend([
                f"### {result['test_name']}",
                f"**Status**: {result['status']}",
                f"**Duration**: {result['duration']:.2f}s",
                f"**Summary**: {result['summary']}",
                f"**Details**: {json.dumps(result['details'], indent=2)}",
                ""
            ])
        
        # Overall assessment
        passed_tests = len([r for r in self.test_results if r['status'] == 'passed'])
        total_tests = len(self.test_results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        report_lines.extend([
            "## Overall Assessment",
            "",
            f"- **Tests Passed**: {passed_tests}/{total_tests} ({pass_rate:.1%})",
            f"- **Framework Status**: {'✅ Fully Functional' if pass_rate >= 0.9 else '⚠️ Needs Attention'}",
            "",
            "## Recommendations",
            ""
        ])
        
        if pass_rate < 1.0:
            failed_tests = [r for r in self.test_results if r['status'] != 'passed']
            for failed_test in failed_tests:
                report_lines.extend([
                    f"- **Fix {failed_test['test_name']}**: Review implementation details",
                    f"  - Error: {failed_test['details'].get('error', 'Unknown error')}",
                    ""
                ])
        else:
            report_lines.extend([
                "- ✅ All components working correctly",
                "- ✅ Ready for production use",
                "- ✅ Comprehensive metrics collection and analysis functional"
            ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "comprehensive_metrics_test_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nComprehensive test report saved to: {report_file}")


async def main():
    """Main function to run comprehensive metrics tests"""
    print("Starting Comprehensive Metrics Framework Tests")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_suite = ComprehensiveMetricsTest()
    success = await test_suite.run_all_tests()
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)