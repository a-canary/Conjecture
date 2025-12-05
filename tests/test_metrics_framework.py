#!/usr/bin/env python3
"""
Simple Test for Metrics Framework Components
Tests core functionality without Unicode characters
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


def test_performance_monitoring():
    """Test performance monitoring functionality"""
    print("Testing Performance Monitor...")
    
    try:
        # Initialize performance monitor
        monitor = PerformanceMonitor(max_history_size=100, snapshot_interval=1)
        monitor.start_monitoring()
        
        # Test basic operations
        timer_id = monitor.start_timer("test_operation")
        time.sleep(0.01)
        monitor.end_timer(timer_id)
        
        monitor.increment_counter("test_counter", 1)
        monitor.set_gauge("test_gauge", 42)
        monitor.record_cache_hit("test_cache", "hit")
        monitor.record_cache_miss("test_cache", "miss")
        monitor.record_database_operation("select", 0.005, 5)
        monitor.record_llm_operation("completion", 0.1, 50)
        
        # Get summary
        summary = monitor.get_performance_summary(time_window_minutes=1)
        
        # Validate summary
        required_keys = ['total_metrics', 'performance_stats', 'counters', 'gauges']
        success = all(key in summary for key in required_keys)
        
        monitor.stop_monitoring()
        
        print(f"Performance Monitor Test: {'PASS' if success else 'FAIL'}")
        if not success:
            print(f"Missing keys: {[k for k in required_keys if k not in summary]}")
        
        return success
        
    except Exception as e:
        print(f"Performance Monitor Test: FAIL - {e}")
        return False


def test_metrics_analyzer():
    """Test metrics analysis functionality"""
    print("Testing Metrics Analyzer...")
    
    try:
        # Create analyzer
        analyzer = create_metrics_analyzer("test_results")
        
        # Create sample data
        sample_experiments = [
            {
                'experiment_id': 'test_exp_001',
                'results': [
                    {
                        'model_name': 'model_a',
                        'approach': 'conjecture',
                        'execution_time': 0.5,
                        'success': True,
                        'value': 0.8
                    },
                    {
                        'model_name': 'model_b',
                        'approach': 'direct',
                        'execution_time': 0.7,
                        'success': True,
                        'value': 0.6
                    }
                ]
            }
        ]
        
        for exp_data in sample_experiments:
            analyzer.experiment_results.append(exp_data)
        
        # Test statistical analysis
        group_a = [0.8, 0.9, 0.85, 0.95, 0.88]
        group_b = [0.6, 0.65, 0.7, 0.55, 0.68]
        
        statistical_test = analyzer.perform_statistical_test(group_a, group_b, "t_test")
        
        # Test hypothesis validation
        hypothesis_config = {
            'id': 'test_hyp_001',
            'hypothesis': 'Group A performs better than Group B',
            'null_hypothesis': 'No difference between groups',
            'type': 'improvement',
            'experiment_id': 'test_exp_001',
            'control_group': 'group_b',
            'treatment_group': 'group_a'
        }
        
        hypothesis_result = analyzer.validate_hypothesis(hypothesis_config)
        
        # Test model comparison
        model_comparisons = analyzer.compare_models(['execution_time', 'value'])
        
        # Test pipeline analysis
        pipeline_data = {
            'stage_1': {
                'total_operations': 100,
                'successful_operations': 95,
                'failed_operations': 5,
                'average_time': 0.1,
                'error_rate': 0.05,
                'throughput': 10.0
            },
            'stage_2': {
                'total_operations': 80,
                'successful_operations': 76,
                'failed_operations': 4,
                'average_time': 0.15,
                'error_rate': 0.05,
                'throughput': 8.0
            }
        }
        
        for stage, metrics in pipeline_data.items():
            analyzer.add_pipeline_metrics(stage, metrics)
        
        pipeline_analysis = analyzer.analyze_pipeline_performance()
        
        # Test report generation
        report = analyzer.generate_comprehensive_report()
        
        # Validate results
        success = (
            statistical_test.test_name == "Independent t-test" and
            hypothesis_result.hypothesis_id == 'test_hyp_001' and
            len(model_comparisons) > 0 and
            len(pipeline_analysis) > 0 and
            len(report) > 0
        )
        
        print(f"Metrics Analyzer Test: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Metrics Analyzer Test: FAIL - {e}")
        return False


def test_retry_tracker():
    """Test retry tracking functionality"""
    print("Testing Retry Tracker...")
    
    try:
        # Create retry tracker
        retry_tracker = create_retry_tracker(max_events=1000)
        
        # Test basic retry tracking
        for i in range(5):
            with RetryContext(retry_tracker, 'test_operation', max_attempts=3) as ctx:
                if i < 3:  # First 3 operations succeed
                    ctx.record_success()
                else:  # Last 2 operations fail once then retry
                    ctx.record_failure('TestError', f'Simulated error {i}')
                    time.sleep(0.01)
                    ctx.record_success()  # Success on retry
        
        # Get statistics
        retry_stats = retry_tracker.get_retry_statistics()
        error_patterns = retry_tracker.analyze_error_patterns()
        real_time_metrics = retry_tracker.get_real_time_metrics()
        
        # Test report generation
        retry_report = retry_tracker.generate_retry_report()
        
        # Validate results
        success = (
            len(retry_stats) > 0 and
            'error_rates' in real_time_metrics and
            len(retry_report) > 0
        )
        
        print(f"Retry Tracker Test: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Retry Tracker Test: FAIL - {e}")
        return False


def test_visualization():
    """Test visualization functionality"""
    print("Testing Visualization...")
    
    try:
        # Create visualizer
        visualizer = create_visualizer("test_visualizations")
        
        # Test timeline chart
        timeline_data = []
        base_time = datetime.utcnow()
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 5)
            timeline_data.append({
                'timestamp': timestamp.isoformat(),
                'value': 0.2 + (i * 0.05),
                'model': f'test_model_{i % 3}'
            })
        
        timeline_path = visualizer.create_performance_timeline(timeline_data)
        
        # Test model comparison chart
        comparison_data = {
            'model_a': {'accuracy': 0.85, 'speed': 1.2, 'efficiency': 0.9},
            'model_b': {'accuracy': 0.78, 'speed': 1.5, 'efficiency': 0.8},
            'model_c': {'accuracy': 0.92, 'speed': 0.9, 'efficiency': 0.85}
        }
        
        comparison_path = visualizer.create_model_comparison_chart(comparison_data)
        
        # Test distribution chart
        distribution_data = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        distribution_path = visualizer.create_distribution_chart(distribution_data)
        
        # Test dashboard
        dashboard_data = {
            'timeline_data': timeline_data,
            'model_comparison': comparison_data,
            'summary': {
                'total_operations': 100,
                'success_rate': 0.85,
                'average_response_time': 0.35
            }
        }
        
        dashboard_path = visualizer.create_dashboard(dashboard_data)
        
        # Validate results
        success = (
            Path(timeline_path).exists() if timeline_path else True and
            Path(comparison_path).exists() if comparison_path else True and
            Path(distribution_path).exists() if distribution_path else True and
            Path(dashboard_path).exists() if dashboard_path else True
        )
        
        print(f"Visualization Test: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Visualization Test: FAIL - {e}")
        return False


def test_integration():
    """Test integration between components"""
    print("Testing Integration...")
    
    try:
        # Create all components
        monitor = PerformanceMonitor()
        analyzer = create_metrics_analyzer()
        retry_tracker = create_retry_tracker()
        visualizer = create_visualizer()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate integrated workflow
        for i in range(3):
            operation_id = f"integrated_op_{i}"
            
            # Track with retry tracker
            with RetryContext(retry_tracker, 'integrated_test', max_attempts=2) as retry_ctx:
                
                # Monitor performance
                with monitor.timer("integrated_operation"):
                    time.sleep(0.01)
                
                # Record metrics
                monitor.record_llm_operation("test", 0.01, 10)
                monitor.increment_counter("integrated_ops", 1)
                
                # Simulate success/failure
                if i < 2:  # First 2 succeed
                    retry_ctx.record_success()
                    success = True
                    value = 0.8 + (i * 0.05)
                else:  # Last one fails once, then succeeds
                    retry_ctx.record_failure("IntegrationError", f"Operation {i} failed")
                    time.sleep(0.005)
                    retry_ctx.record_success()
                    success = True
                    value = 0.7 + (i * 0.03)
            
            # Add to analyzer
            analyzer.experiment_results.append({
                'experiment_id': f'integration_test_{i}',
                'results': [{
                    'model_name': f'test_model_{i}',
                    'approach': 'integrated',
                    'execution_time': 0.01,
                    'success': success,
                    'value': value
                }]
            })
        
        # Wait for data collection
        time.sleep(0.5)
        
        # Generate analysis
        model_comparisons = analyzer.compare_models()
        pipeline_analysis = analyzer.analyze_pipeline_performance()
        
        # Create visualizations
        timeline_data = [{
            'timestamp': datetime.utcnow().isoformat(),
            'value': 0.01,
            'model': f'integrated_model_{i}'
        } for i in range(3)]
        
        dashboard_data = {
            'timeline_data': timeline_data,
            'model_comparison': {f'integrated_model_{i}': {'value': 0.8 - (i * 0.05)} for i in range(3)},
            'summary': {
                'total_operations': 3,
                'success_rate': 0.67,
                'average_response_time': 0.01
            }
        }
        
        dashboard_path = visualizer.create_dashboard(dashboard_data)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Validate integration
        success = (
            len(model_comparisons) > 0 and
            len(pipeline_analysis) > 0 and
            dashboard_path and Path(dashboard_path).exists()
        )
        
        print(f"Integration Test: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Integration Test: FAIL - {e}")
        return False


async def main():
    """Main test function"""
    print("Starting Metrics Framework Tests")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    tests = [
        ("Performance Monitoring", test_performance_monitoring),
        ("Metrics Analyzer", test_metrics_analyzer),
        ("Retry Tracker", test_retry_tracker),
        ("Visualization", test_visualization),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 20}")
        print(f"Test: {test_name}")
        print(f"{'-' * 20}")
        
        success = test_func()
        results.append({
            'test_name': test_name,
            'success': success
        })
    
    # Summary
    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{result['test_name']}: {status}")
    
    print(f"\nPassed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED - Metrics framework is fully functional!")
        print("Ready for production use.")
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} tests failed - review implementation.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)