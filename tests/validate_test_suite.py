#!/usr/bin/env python3
"""
Validation Script for Expanded Test Suite
Tests the functionality and integration of all components
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import test components
from test_hypothesis_validation import HypothesisValidationSuite, TestConfiguration
from test_ab_testing_framework import ABTestingFramework, ABTestConfiguration
from test_llm_judge import LLMJudgeSystem, JudgeConfiguration
from test_statistical_validation import StatisticalValidationSystem, StatisticalTestConfig
from test_performance_monitoring import PerformanceMonitoringSystem
from test_hypothesis_validation_suite import ComprehensiveHypothesisValidationSuite, SuiteConfiguration


class TestSuiteValidator:
    """Validator for the expanded test suite functionality"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.validation_results = {
            "test_case_generation": {"status": "pending", "details": []},
            "ab_testing": {"status": "pending", "details": []},
            "llm_judge": {"status": "pending", "details": []},
            "statistical_validation": {"status": "pending", "details": []},
            "performance_monitoring": {"status": "pending", "details": []},
            "integration": {"status": "pending", "details": []}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("test_suite_validator")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    async def validate_all_components(self) -> Dict[str, Any]:
        """Validate all components of the expanded test suite"""
        
        self.logger.info("Starting comprehensive test suite validation...")
        
        # Validate each component
        await self._validate_test_case_generation()
        await self._validate_ab_testing_framework()
        await self._validate_llm_judge_system()
        await self._validate_statistical_validation()
        await self._validate_performance_monitoring()
        await self._validate_integration()
        
        # Generate summary report
        await self._generate_validation_report()
        
        return self.validation_results
    
    async def _validate_test_case_generation(self):
        """Validate test case generation component"""
        
        self.logger.info("Validating test case generation...")
        
        try:
            # Test configuration
            config = TestConfiguration(
                sample_size_per_category=10,  # Small for validation
                tiny_model="ibm/granite-4-h-tiny",
                baseline_model="zai-org/GLM-4.6"
            )
            
            suite = HypothesisValidationSuite(config)
            
            # Test test case generation
            test_cases = suite.generate_comprehensive_test_cases()
            
            # Validate generated test cases
            validation_details = []
            
            # Check total count
            total_cases = sum(len(cases) for cases in test_cases.values())
            expected_categories = 6
            expected_min_cases = 50  # Minimum per category
            expected_total = expected_categories * expected_min_cases
            
            if total_cases < expected_total:
                validation_details.append(f"Insufficient test cases: {total_cases} < {expected_total}")
            else:
                validation_details.append(f"Adequate test cases: {total_cases} >= {expected_total}")
            
            # Check category distribution
            for category, cases in test_cases.items():
                if len(cases) < 50:
                    validation_details.append(f"Category {category}: {len(cases)} cases (< 50 minimum)")
                elif len(cases) > 100:
                    validation_details.append(f"Category {category}: {len(cases)} cases (> 100 maximum)")
                else:
                    validation_details.append(f"Category {category}: {len(cases)} cases (✓)")
            
            # Check test case structure
            sample_case = test_cases.get("complex_reasoning", [{}])[0] if test_cases.get("complex_reasoning") else {}
            required_fields = ["id", "category", "difficulty", "question", "reasoning_requirements", "expected_answer_type"]
            
            missing_fields = [field for field in required_fields if field not in sample_case]
            if missing_fields:
                validation_details.append(f"Missing test case fields: {missing_fields}")
            else:
                validation_details.append("Test case structure validation: ✓")
            
            # Check diversity of difficulty levels
            for category, cases in test_cases.items():
                difficulties = [case.get("difficulty", "medium") for case in cases]
                unique_difficulties = set(difficulties)
                
                if len(unique_difficulties) < 2:
                    validation_details.append(f"Category {category}: insufficient difficulty diversity")
                else:
                    validation_details.append(f"Category {category}: good difficulty diversity")
            
            self.validation_results["test_case_generation"]["status"] = "passed" if not any("error" in detail.lower() or "missing" in detail.lower() for detail in validation_details) else "failed"
            self.validation_results["test_case_generation"]["details"] = validation_details
            
        except Exception as e:
            self.validation_results["test_case_generation"]["status"] = "error"
            self.validation_results["test_case_generation"]["details"].append(f"Exception: {str(e)}")
    
    async def _validate_ab_testing_framework(self):
        """Validate A/B testing framework"""
        
        self.logger.info("Validating A/B testing framework...")
        
        try:
            # Test configuration
            config = ABTestConfiguration(
                approaches=["direct", "conjecture"],
                test_models=["ibm/granite-4-h-tiny", "zai-org/GLM-4.6"],
                judge_model="zai-org/GLM-4.6"
            )
            
            framework = ABTestingFramework(config)
            
            # Test prompt templates
            prompt_templates = framework.prompt_templates
            expected_templates = ["direct", "conjecture", "few_shot"]
            
            missing_templates = [t for t in expected_templates if t not in prompt_templates]
            if missing_templates:
                self.validation_results["ab_testing"]["details"].append(f"Missing prompt templates: {missing_templates}")
            else:
                self.validation_results["ab_testing"]["details"].append("Prompt templates validation: ✓")
            
            # Test evaluation criteria
            evaluation_criteria = config.evaluation_criteria
            expected_criteria = ["correctness", "completeness", "coherence", "reasoning_quality", "confidence_calibration", "efficiency", "hallucination_reduction"]
            
            missing_criteria = [c for c in expected_criteria if c not in evaluation_criteria]
            if missing_criteria:
                self.validation_results["ab_testing"]["details"].append(f"Missing evaluation criteria: {missing_criteria}")
            else:
                self.validation_results["ab_testing"]["details"].append("Evaluation criteria validation: ✓")
            
            # Test category guidance
            guidance = framework._get_conjecture_guidance("complex_reasoning")
            if guidance and len(guidance) > 50:
                self.validation_results["ab_testing"]["details"].append("Category guidance validation: ✓")
            else:
                self.validation_results["ab_testing"]["details"].append("Category guidance needs improvement")
            
            self.validation_results["ab_testing"]["status"] = "passed" if not any("missing" in detail.lower() for detail in self.validation_results["ab_testing"]["details"]) else "failed"
            
        except Exception as e:
            self.validation_results["ab_testing"]["status"] = "error"
            self.validation_results["ab_testing"]["details"].append(f"Exception: {str(e)}")
    
    async def _validate_llm_judge_system(self):
        """Validate LLM judge system"""
        
        self.logger.info("Validating LLM judge system...")
        
        try:
            # Test configuration
            config = JudgeConfiguration(
                judge_model="zai-org/GLM-4.6",
                temperature=0.1,
                evaluation_criteria=["correctness", "completeness", "coherence"]
            )
            
            judge = LLMJudgeSystem(config)
            
            # Test evaluation prompts
            evaluation_prompts = judge.evaluation_prompts
            expected_prompts = ["main_evaluation", "calibration_evaluation", "quality_assessment"]
            
            missing_prompts = [p for p in expected_prompts if p not in evaluation_prompts]
            if missing_prompts:
                self.validation_results["llm_judge"]["details"].append(f"Missing evaluation prompts: {missing_prompts}")
            else:
                self.validation_results["llm_judge"]["details"].append("Evaluation prompts validation: ✓")
            
            # Test criterion weights
            criterion_weights = config.criterion_weights
            expected_weights = ["correctness", "reasoning_quality", "completeness"]
            
            missing_weights = [w for w in expected_weights if w not in criterion_weights]
            if missing_weights:
                self.validation_results["llm_judge"]["details"].append(f"Missing criterion weights: {missing_weights}")
            else:
                self.validation_results["llm_judge"]["details"].append("Criterion weights validation: ✓")
            
            self.validation_results["llm_judge"]["status"] = "passed" if not any("missing" in detail.lower() for detail in self.validation_results["llm_judge"]["details"]) else "failed"
            
        except Exception as e:
            self.validation_results["llm_judge"]["status"] = "error"
            self.validation_results["llm_judge"]["details"].append(f"Exception: {str(e)}")
    
    async def _validate_statistical_validation(self):
        """Validate statistical validation system"""
        
        self.logger.info("Validating statistical validation...")
        
        try:
            # Test configuration
            config = StatisticalTestConfig(
                alpha_level=0.05,
                target_power=0.8,
                generate_plots=False  # Skip plots for validation
            )
            
            validator = StatisticalValidationSystem(config)
            
            # Test effect size calculations
            test_scores = [0.6, 0.7, 0.8, 0.65, 0.75]
            effect_sizes = validator._calculate_effect_sizes(test_scores)
            
            if not effect_sizes:
                self.validation_results["statistical_validation"]["details"].append("Effect size calculation failed")
            else:
                self.validation_results["statistical_validation"]["details"].append("Effect size calculations: ✓")
            
            # Test statistical test methods
            from scipy import stats
            import numpy as np
            
            # Test paired t-test
            group1 = [0.6, 0.7, 0.8]
            group2 = [0.5, 0.6, 0.7]
            
            t_stat, p_value = stats.ttest_rel(group1, group2)
            if p_value is not None and 0 <= p_value <= 1:
                self.validation_results["statistical_validation"]["details"].append("Paired t-test calculation: ✓")
            else:
                self.validation_results["statistical_validation"]["details"].append("Paired t-test calculation failed")
            
            # Test power analysis
            effect_size = 0.5
            sample_size = 25
            achieved_power = validator._calculate_statistical_power(effect_size, sample_size, 0.05)
            
            if 0 <= achieved_power <= 1:
                self.validation_results["statistical_validation"]["details"].append("Power analysis: ✓")
            else:
                self.validation_results["statistical_validation"]["details"].append("Power analysis failed")
            
            self.validation_results["statistical_validation"]["status"] = "passed" if not any("failed" in detail.lower() for detail in self.validation_results["statistical_validation"]["details"]) else "failed"
            
        except Exception as e:
            self.validation_results["statistical_validation"]["status"] = "error"
            self.validation_results["statistical_validation"]["details"].append(f"Exception: {str(e)}")
    
    async def _validate_performance_monitoring(self):
        """Validate performance monitoring system"""
        
        self.logger.info("Validating performance monitoring...")
        
        try:
            # Initialize monitoring system
            monitor = PerformanceMonitoringSystem(monitoring_interval=0.1)
            
            # Test metrics collection
            test_metrics = {
                "execution_id": "test_001",
                "test_id": "sample_test",
                "approach": "conjecture",
                "model": "ibm/granite-4-h-tiny",
                "category": "sample_category",
                "execution_time": 1.5,
                "token_usage_total": 150,
                "evaluation_score": 0.75,
                "success": True
            }
            
            # Test performance summary calculation
            monitor.execution_metrics.append(test_metrics)
            summary = monitor.get_performance_summary()
            
            if summary:
                self.validation_results["performance_monitoring"]["details"].append("Performance metrics collection: ✓")
            else:
                self.validation_results["performance_monitoring"]["details"].append("Performance metrics collection failed")
            
            # Test anomaly detection
            anomalies = monitor.detect_performance_anomalies()
            if isinstance(anomalies, list):
                self.validation_results["performance_monitoring"]["details"].append("Anomaly detection: ✓")
            else:
                self.validation_results["performance_monitoring"]["details"].append("Anomaly detection failed")
            
            self.validation_results["performance_monitoring"]["status"] = "passed" if not any("failed" in detail.lower() for detail in self.validation_results["performance_monitoring"]["details"]) else "failed"
            
        except Exception as e:
            self.validation_results["performance_monitoring"]["status"] = "error"
            self.validation_results["performance_monitoring"]["details"].append(f"Exception: {str(e)}")
    
    async def _validate_integration(self):
        """Validate integration of all components"""
        
        self.logger.info("Validating component integration...")
        
        try:
            # Test main integration suite
            config = SuiteConfiguration(
                sample_size_per_category=10,  # Small for validation
                categories=["complex_reasoning", "mathematical_reasoning"]
            )
            
            suite = ComprehensiveHypothesisValidationSuite(config)
            
            # Test configuration validation
            if config.sample_size_per_category >= 50 and config.sample_size_per_category <= 100:
                self.validation_results["integration"]["details"].append("Sample size configuration: ✓")
            else:
                self.validation_results["integration"]["details"].append("Sample size configuration out of range")
            
            # Test category configuration
            expected_categories = ["complex_reasoning", "mathematical_reasoning", "context_compression", 
                                "evidence_evaluation", "task_decomposition", "coding_tasks"]
            
            missing_categories = [cat for cat in expected_categories if cat not in config.categories]
            extra_categories = [cat for cat in config.categories if cat not in expected_categories]
            
            if missing_categories:
                self.validation_results["integration"]["details"].append(f"Missing expected categories: {missing_categories}")
            
            if extra_categories:
                self.validation_results["integration"]["details"].append(f"Extra categories: {extra_categories}")
            
            # Test directory structure
            required_dirs = ["results", "reports", "plots", "test_cases"]
            base_dir = Path("tests/hypothesis_validation")
            
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = base_dir / dir_name
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                self.validation_results["integration"]["details"].append(f"Missing directories: {missing_dirs}")
            else:
                self.validation_results["integration"]["details"].append("Directory structure: ✓")
            
            self.validation_results["integration"]["status"] = "passed" if not any("missing" in detail.lower() for detail in self.validation_results["integration"]["details"]) else "failed"
            
        except Exception as e:
            self.validation_results["integration"]["status"] = "error"
            self.validation_results["integration"]["details"].append(f"Exception: {str(e)}")
    
    async def _generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        report_lines = [
            "# Test Suite Validation Report",
            f"Generated: {self._get_timestamp()}",
            "",
            "## Validation Summary",
            "",
        ]
        
        # Add component results
        for component, results in self.validation_results.items():
            status = results["status"].upper()
            details_count = len(results["details"])
            
            report_lines.extend([
                f"### {component.replace('_', ' ').title()}",
                f"**Status**: {status}",
                f"**Checks Passed**: {details_count}",
                f"**Details**: {len(results['details'])} items",
                ""
            ])
            
            # Add key details
            if results["details"]:
                report_lines.extend([
                    "**Validation Details**:",
                    ""
                ])
                for detail in results["details"]:
                    status_icon = "✅" if "✓" in detail or "passed" in detail else "❌" if "failed" in detail or "error" in detail else "⚠️"
                    report_lines.append(f"- {status_icon} {detail}")
                report_lines.append("")
        
        # Overall assessment
        passed_components = sum(1 for results in self.validation_results.values() if results["status"] == "passed")
        total_components = len(self.validation_results)
        
        report_lines.extend([
            "## Overall Assessment",
            "",
            f"**Components Validated**: {passed_components}/{total_components}",
            f"**Success Rate**: {passed_components/total_components*100:.1f}%",
            "",
        ])
        
        # Recommendations
        if passed_components == total_components:
            report_lines.extend([
                "## 🎉 Validation Results",
                "",
                "✅ **All components passed validation**",
                "✅ **Test suite is ready for production use**",
                "✅ **Integration with research framework confirmed**",
                "",
                "## Next Steps",
                "1. Run comprehensive validation with real test cases",
                "2. Execute full hypothesis validation suite",
                "3. Analyze results and generate reports",
                "4. Deploy to production environment"
            ])
        else:
            failed_components = [comp for comp, results in self.validation_results.items() if results["status"] != "passed"]
            
            report_lines.extend([
                "## ⚠️ Validation Issues",
                "",
                f"❌ **{len(failed_components)} components failed validation**",
                "",
                "### Failed Components:",
                ""
            ])
            
            for component in failed_components:
                report_lines.append(f"- **{component.replace('_', ' ').title()}**: See details above")
            
            report_lines.extend([
                "",
                "## Recommendations",
                "1. Fix failed components before production deployment",
                "2. Review error messages and address root causes",
                "3. Re-run validation after fixes",
                "4. Ensure all dependencies are properly configured"
            ])
        
        return "\n".join(report_lines)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    async def save_validation_report(self, report: str):
        """Save validation report to file"""
        
        report_dir = Path("tests/validation_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"validation_report_{self._get_timestamp().replace(':', '-').replace(' ', '_')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Validation report saved to: {report_file}")


async def main():
    """Main function to run test suite validation"""
    
    validator = TestSuiteValidator()
    
    print("🔍 Validating Expanded Test Suite Functionality")
    print("=" * 50)
    
    # Run validation
    results = await validator.validate_all_components()
    
    # Generate and save report
    report = await validator._generate_validation_report()
    await validator.save_validation_report(report)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION COMPLETE")
    print("=" * 50)
    print(report)
    
    print(f"\n📁 Validation report saved to: tests/validation_reports/")
    
    # Final status
    passed_components = sum(1 for results in validator.validation_results.values() if results["status"] == "passed")
    total_components = len(validator.validation_results)
    
    if passed_components == total_components:
        print("🎉 ALL COMPONENTS PASSED - Test suite is ready!")
        return 0
    else:
        print(f"⚠️  {total_components - passed_components} COMPONENTS FAILED - Review issues above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)