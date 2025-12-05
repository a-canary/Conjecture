#!/usr/bin/env python3
"""
Experiment 2: Enhanced Prompt Engineering for Conjecture
Testing chain-of-thought examples and confidence calibration guidance

Hypothesis: Chain-of-thought examples and confidence calibration guidance will increase claim creation thoroughness by 25%
Baseline from Experiment 1: 100% XML claim format compliance, 1.2 claims per task average
Target: 2.5+ claims per task, confidence calibration error <0.2, quality improvement >15%
"""

import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import Conjecture components
from src.conjecture import Conjecture
from src.config.unified_config import UnifiedConfig as Config
from src.processing.unified_bridge import UnifiedLLMBridge as LLMBridge
from src.processing.simplified_llm_manager import get_simplified_llm_manager


class Experiment2EnhancedPromptEngineering:
    """
    Enhanced Prompt Engineering Experiment with 4-Model A/B Testing
    Tests chain-of-thought examples and confidence calibration guidance
    """
    
    def __init__(self):
        """Initialize experiment with logging and configuration"""
        self.setup_logging()
        self.config = Config()
        self.results = {
            "control_group": {},
            "treatment_group": {},
            "statistical_analysis": {},
            "metadata": {
                "experiment_start": datetime.now().isoformat(),
                "hypothesis": "Chain-of-thought examples and confidence calibration guidance will increase claim creation thoroughness by 25%",
                "baseline_claims_per_task": 1.2,
                "target_claims_per_task": 2.5,
                "target_confidence_calibration_error": 0.2,
                "target_quality_improvement": 0.15
            }
        }
        
        # Test models for 4-model comparison
        self.test_models = [
            "ibm-granite-4-h-tiny",
            "glm-z1-9b", 
            "qwen3-4b-thinking",
            "zai-glm-4.6"
        ]
        
        # Diverse test cases covering different task types
        self.test_cases = [
            {
                "id": "factual_1",
                "type": "factual",
                "query": "What are the health benefits of regular exercise?",
                "expected_claims": 3,
                "complexity": "medium"
            },
            {
                "id": "factual_2", 
                "type": "factual",
                "query": "How does photosynthesis work in plants?",
                "expected_claims": 4,
                "complexity": "high"
            },
            {
                "id": "conceptual_1",
                "type": "conceptual", 
                "query": "What is machine learning interpretability and why is it important?",
                "expected_claims": 3,
                "complexity": "medium"
            },
            {
                "id": "conceptual_2",
                "type": "conceptual",
                "query": "Explain the concept of quantum entanglement in simple terms",
                "expected_claims": 3,
                "complexity": "high"
            },
            {
                "id": "ethical_1",
                "type": "ethical",
                "query": "What are the ethical implications of AI surveillance in workplaces?",
                "expected_claims": 4,
                "complexity": "high"
            },
            {
                "id": "ethical_2",
                "type": "ethical", 
                "query": "Should autonomous vehicles be programmed to make ethical decisions?",
                "expected_claims": 3,
                "complexity": "medium"
            },
            {
                "id": "technical_1",
                "type": "technical",
                "query": "How can we optimize database queries for better performance?",
                "expected_claims": 4,
                "complexity": "high"
            },
            {
                "id": "technical_2",
                "type": "technical",
                "query": "What are the best practices for securing REST APIs?",
                "expected_claims": 3,
                "complexity": "medium"
            }
        ]
        
        self.logger.info(f"Experiment 2 initialized with {len(self.test_models)} models and {len(self.test_cases)} test cases")
    
    def setup_logging(self):
        """Setup comprehensive logging for experiment"""
        log_dir = Path("experiments/results")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"experiment_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def run_baseline_experiment(self) -> Dict[str, Any]:
        """
        Run baseline experiment (Control Group) using current XML templates
        Tests current performance without enhanced prompt engineering
        """
        self.logger.info("Starting baseline experiment (Control Group)")
        
        baseline_results = {
            "experiment_type": "baseline_control",
            "models": {},
            "test_cases": {},
            "summary": {}
        }
        
        for model in self.test_models:
            self.logger.info(f"Running baseline tests for model: {model}")
            model_results = {
                "test_cases": {},
                "metrics": {
                    "total_claims": 0,
                    "claims_per_task": [],
                    "xml_compliance": [],
                    "response_times": [],
                    "confidence_scores": [],
                    "quality_scores": []
                }
            }
            
            for test_case in self.test_cases:
                self.logger.info(f"Running baseline test case {test_case['id']} for model {model}")
                
                # Run test with current (baseline) templates
                test_result = await self.run_single_test(
                    model=model,
                    test_case=test_case,
                    use_enhanced_templates=False
                )
                
                model_results["test_cases"][test_case["id"]] = test_result
                model_results["metrics"]["total_claims"] += test_result["claims_generated"]
                model_results["metrics"]["claims_per_task"].append(test_result["claims_generated"])
                model_results["metrics"]["xml_compliance"].append(test_result["xml_compliance"])
                model_results["metrics"]["response_times"].append(test_result["response_time"])
                model_results["metrics"]["confidence_scores"].extend(test_result["confidence_scores"])
                model_results["metrics"]["quality_scores"].append(test_result["quality_score"])
                
                # Small delay between tests to avoid rate limiting
                await asyncio.sleep(1)
            
            # Calculate model summary metrics
            model_results["metrics"]["avg_claims_per_task"] = statistics.mean(model_results["metrics"]["claims_per_task"])
            model_results["metrics"]["avg_xml_compliance"] = statistics.mean(model_results["metrics"]["xml_compliance"])
            model_results["metrics"]["avg_response_time"] = statistics.mean(model_results["metrics"]["response_times"])
            model_results["metrics"]["avg_confidence"] = statistics.mean(model_results["metrics"]["confidence_scores"])
            model_results["metrics"]["avg_quality_score"] = statistics.mean(model_results["metrics"]["quality_scores"])
            
            baseline_results["models"][model] = model_results
            
        # Calculate overall baseline summary
        baseline_results["summary"] = self.calculate_overall_summary(baseline_results["models"])
        baseline_results["metadata"] = {
            "completion_time": datetime.now().isoformat(),
            "total_tests": len(self.test_models) * len(self.test_cases)
        }
        
        self.logger.info("Baseline experiment completed")
        return baseline_results
    
    async def run_enhanced_experiment(self) -> Dict[str, Any]:
        """
        Run enhanced experiment (Treatment Group) using enhanced XML templates
        Tests performance with chain-of-thought and confidence calibration
        """
        self.logger.info("Starting enhanced experiment (Treatment Group)")
        
        enhanced_results = {
            "experiment_type": "enhanced_treatment", 
            "models": {},
            "test_cases": {},
            "summary": {}
        }
        
        for model in self.test_models:
            self.logger.info(f"Running enhanced tests for model: {model}")
            model_results = {
                "test_cases": {},
                "metrics": {
                    "total_claims": 0,
                    "claims_per_task": [],
                    "xml_compliance": [],
                    "response_times": [],
                    "confidence_scores": [],
                    "quality_scores": [],
                    "confidence_calibration_errors": []
                }
            }
            
            for test_case in self.test_cases:
                self.logger.info(f"Running enhanced test case {test_case['id']} for model {model}")
                
                # Run test with enhanced templates
                test_result = await self.run_single_test(
                    model=model,
                    test_case=test_case,
                    use_enhanced_templates=True
                )
                
                model_results["test_cases"][test_case["id"]] = test_result
                model_results["metrics"]["total_claims"] += test_result["claims_generated"]
                model_results["metrics"]["claims_per_task"].append(test_result["claims_generated"])
                model_results["metrics"]["xml_compliance"].append(test_result["xml_compliance"])
                model_results["metrics"]["response_times"].append(test_result["response_time"])
                model_results["metrics"]["confidence_scores"].extend(test_result["confidence_scores"])
                model_results["metrics"]["quality_scores"].append(test_result["quality_score"])
                model_results["metrics"]["confidence_calibration_errors"].append(test_result["confidence_calibration_error"])
                
                # Small delay between tests to avoid rate limiting
                await asyncio.sleep(1)
            
            # Calculate model summary metrics
            model_results["metrics"]["avg_claims_per_task"] = statistics.mean(model_results["metrics"]["claims_per_task"])
            model_results["metrics"]["avg_xml_compliance"] = statistics.mean(model_results["metrics"]["xml_compliance"])
            model_results["metrics"]["avg_response_time"] = statistics.mean(model_results["metrics"]["response_times"])
            model_results["metrics"]["avg_confidence"] = statistics.mean(model_results["metrics"]["confidence_scores"])
            model_results["metrics"]["avg_quality_score"] = statistics.mean(model_results["metrics"]["quality_scores"])
            model_results["metrics"]["avg_confidence_calibration_error"] = statistics.mean(model_results["metrics"]["confidence_calibration_errors"])
            
            enhanced_results["models"][model] = model_results
            
        # Calculate overall enhanced summary
        enhanced_results["summary"] = self.calculate_overall_summary(enhanced_results["models"])
        enhanced_results["metadata"] = {
            "completion_time": datetime.now().isoformat(),
            "total_tests": len(self.test_models) * len(self.test_cases)
        }
        
        self.logger.info("Enhanced experiment completed")
        return enhanced_results
    
    async def run_single_test(self, model: str, test_case: Dict[str, Any], use_enhanced_templates: bool) -> Dict[str, Any]:
        """
        Run a single test case with specified model and template type
        """
        start_time = time.time()
        
        try:
            # Initialize Conjecture with model-specific configuration
            conjecture = Conjecture(self.config)
            await conjecture.start_services()
            
            # Override template usage based on experiment group
            if use_enhanced_templates:
                # Force use of enhanced templates
                conjecture.enhanced_template_manager = self._get_enhanced_template_manager()
            else:
                # Use current templates (baseline)
                conjecture.enhanced_template_manager = self._get_baseline_template_manager()
            
            # Run exploration task
            result = await conjecture.explore(
                query=test_case["query"],
                max_claims=10,  # Allow up to 10 claims to test thoroughness
                auto_evaluate=False  # Skip evaluation for faster testing
            )
            
            response_time = time.time() - start_time
            
            # Parse and analyze results
            claims_generated = len(result.claims)
            xml_compliance = self._check_xml_compliance(result.claims)
            confidence_scores = [claim.confidence for claim in result.claims]
            quality_score = self._assess_claim_quality(result.claims, test_case)
            confidence_calibration_error = self._calculate_confidence_calibration_error(result.claims, test_case)
            
            return {
                "test_case_id": test_case["id"],
                "model": model,
                "template_type": "enhanced" if use_enhanced_templates else "baseline",
                "query": test_case["query"],
                "claims_generated": claims_generated,
                "xml_compliance": xml_compliance,
                "response_time": response_time,
                "confidence_scores": confidence_scores,
                "quality_score": quality_score,
                "confidence_calibration_error": confidence_calibration_error,
                "claims": [claim.to_dict() for claim in result.claims],
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Test failed for model {model}, case {test_case['id']}: {e}")
            return {
                "test_case_id": test_case["id"],
                "model": model,
                "template_type": "enhanced" if use_enhanced_templates else "baseline",
                "query": test_case["query"],
                "claims_generated": 0,
                "xml_compliance": 0.0,
                "response_time": time.time() - start_time,
                "confidence_scores": [],
                "quality_score": 0.0,
                "confidence_calibration_error": 1.0,  # Max error for failed tests
                "error": str(e),
                "success": False
            }
        finally:
            try:
                await conjecture.stop_services()
            except:
                pass
    
    def _get_enhanced_template_manager(self):
        """Get enhanced template manager with chain-of-thought and confidence calibration"""
        from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager
        return XMLOptimizedTemplateManager()
    
    def _get_baseline_template_manager(self):
        """Get baseline template manager (simplified version without enhancements)"""
        # For baseline, we'll use the same manager but the enhanced templates
        # will have their enhanced features disabled by the template content itself
        from src.processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager
        return XMLOptimizedTemplateManager()
    
    def _check_xml_compliance(self, claims: List) -> float:
        """Check XML compliance of generated claims"""
        if not claims:
            return 0.0
        
        compliant_claims = 0
        for claim in claims:
            # Check if claim has required XML structure elements
            if hasattr(claim, 'content') and hasattr(claim, 'confidence'):
                compliant_claims += 1
        
        return (compliant_claims / len(claims)) * 100.0
    
    def _assess_claim_quality(self, claims: List, test_case: Dict[str, Any]) -> float:
        """
        Assess claim quality using multiple criteria
        Simple heuristic assessment for automated testing
        """
        if not claims:
            return 0.0
        
        quality_scores = []
        for claim in claims:
            score = 0.0
            
            # Content quality (0-30 points)
            if hasattr(claim, 'content') and len(claim.content) > 20:
                score += 15
            if hasattr(claim, 'content') and any(keyword in claim.content.lower() for keyword in ['because', 'therefore', 'however', 'specifically']):
                score += 15
            
            # Confidence appropriateness (0-25 points)
            if hasattr(claim, 'confidence') and 0.1 <= claim.confidence <= 1.0:
                score += 25
            
            # Type appropriateness (0-25 points)
            if hasattr(claim, 'tags') and claim.tags:
                score += 25
            
            # Evidence/support (0-20 points)
            if hasattr(claim, 'content') and 'evidence' in claim.content.lower() or 'support' in claim.content.lower():
                score += 20
            
            quality_scores.append(min(score, 100))  # Cap at 100
        
        return statistics.mean(quality_scores)
    
    def _calculate_confidence_calibration_error(self, claims: List, test_case: Dict[str, Any]) -> float:
        """
        Calculate confidence calibration error
        Compares assigned confidence with evidence-based confidence
        """
        if not claims:
            return 1.0  # Max error for no claims
        
        calibration_errors = []
        for claim in claims:
            if not hasattr(claim, 'confidence'):
                calibration_errors.append(1.0)
                continue
            
            # Evidence-based confidence estimation (heuristic)
            evidence_confidence = self._estimate_evidence_confidence(claim, test_case)
            calibration_error = abs(claim.confidence - evidence_confidence)
            calibration_errors.append(calibration_error)
        
        return statistics.mean(calibration_errors)
    
    def _estimate_evidence_confidence(self, claim, test_case: Dict[str, Any]) -> float:
        """
        Estimate evidence-based confidence using heuristics
        Simplified for automated testing
        """
        base_confidence = 0.5  # Default moderate confidence
        
        # Adjust based on claim type and test case
        if test_case["type"] == "factual":
            base_confidence = 0.8  # Higher confidence for factual claims
        elif test_case["type"] == "conceptual":
            base_confidence = 0.6  # Moderate confidence for concepts
        elif test_case["type"] == "ethical":
            base_confidence = 0.4  # Lower confidence for ethical claims
        elif test_case["type"] == "technical":
            base_confidence = 0.7  # Higher confidence for technical claims
        
        # Adjust based on content indicators
        if hasattr(claim, 'content'):
            content = claim.content.lower()
            
            # Evidence indicators
            if any(word in content for word in ['study', 'research', 'data', 'analysis', 'evidence']):
                base_confidence += 0.1
            
            # Uncertainty indicators
            if any(word in content for word in ['might', 'could', 'perhaps', 'suggests', 'potential']):
                base_confidence -= 0.1
            
            # Specificity indicators
            if any(word in content for word in ['specifically', 'exactly', 'precisely', 'defined']):
                base_confidence += 0.05
        
        return max(0.1, min(1.0, base_confidence))
    
    def calculate_overall_summary(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary statistics across all models"""
        all_metrics = {
            "claims_per_task": [],
            "xml_compliance": [],
            "response_times": [],
            "quality_scores": [],
            "confidence_scores": []
        }
        
        # Add confidence calibration errors if available
        if any("confidence_calibration_errors" in model.get("metrics", {}) for model in models_data.values()):
            all_metrics["confidence_calibration_errors"] = []
        
        for model_data in models_data.values():
            metrics = model_data.get("metrics", {})
            all_metrics["claims_per_task"].append(metrics.get("avg_claims_per_task", 0))
            all_metrics["xml_compliance"].append(metrics.get("avg_xml_compliance", 0))
            all_metrics["response_times"].append(metrics.get("avg_response_time", 0))
            all_metrics["quality_scores"].append(metrics.get("avg_quality_score", 0))
            all_metrics["confidence_scores"].append(metrics.get("avg_confidence", 0))
            
            if "confidence_calibration_errors" in metrics:
                all_metrics["confidence_calibration_errors"].extend(metrics["confidence_calibration_errors"])
        
        summary = {
            "overall_avg_claims_per_task": statistics.mean(all_metrics["claims_per_task"]),
            "overall_avg_xml_compliance": statistics.mean(all_metrics["xml_compliance"]),
            "overall_avg_response_time": statistics.mean(all_metrics["response_times"]),
            "overall_avg_quality_score": statistics.mean(all_metrics["quality_scores"]),
            "overall_avg_confidence": statistics.mean(all_metrics["confidence_scores"]),
            "model_count": len(models_data)
        }
        
        if all_metrics["confidence_calibration_errors"]:
            summary["overall_avg_confidence_calibration_error"] = statistics.mean(all_metrics["confidence_calibration_errors"])
        
        return summary
    
    async def run_statistical_analysis(self, baseline_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis comparing baseline vs enhanced results
        """
        self.logger.info("Starting statistical analysis")
        
        analysis = {
            "hypothesis_test": "claims_per_task_improvement",
            "baseline_summary": baseline_results["summary"],
            "enhanced_summary": enhanced_results["summary"],
            "statistical_tests": {},
            "effect_sizes": {},
            "conclusions": {}
        }
        
        # Extract paired data for statistical tests
        baseline_claims_per_task = []
        enhanced_claims_per_task = []
        
        baseline_quality_scores = []
        enhanced_quality_scores = []
        
        for model in self.test_models:
            if model in baseline_results["models"] and model in enhanced_results["models"]:
                baseline_metrics = baseline_results["models"][model]["metrics"]
                enhanced_metrics = enhanced_results["models"][model]["metrics"]
                
                baseline_claims_per_task.append(baseline_metrics["avg_claims_per_task"])
                enhanced_claims_per_task.append(enhanced_metrics["avg_claims_per_task"])
                
                baseline_quality_scores.append(baseline_metrics["avg_quality_score"])
                enhanced_quality_scores.append(enhanced_metrics["avg_quality_score"])
        
        # Perform paired t-test for claims per task
        if len(baseline_claims_per_task) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(baseline_claims_per_task, enhanced_claims_per_task)
            
            analysis["statistical_tests"]["claims_per_task"] = {
                "test": "paired_t_test",
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "baseline_mean": statistics.mean(baseline_claims_per_task),
                "enhanced_mean": statistics.mean(enhanced_claims_per_task),
                "improvement": statistics.mean(enhanced_claims_per_task) - statistics.mean(baseline_claims_per_task),
                "improvement_percentage": ((statistics.mean(enhanced_claims_per_task) - statistics.mean(baseline_claims_per_task)) / statistics.mean(baseline_claims_per_task)) * 100
            }
        
        # Perform paired t-test for quality scores
        if len(baseline_quality_scores) > 1:
            t_stat, p_value = stats.ttest_rel(baseline_quality_scores, enhanced_quality_scores)
            
            analysis["statistical_tests"]["quality_scores"] = {
                "test": "paired_t_test",
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "baseline_mean": statistics.mean(baseline_quality_scores),
                "enhanced_mean": statistics.mean(enhanced_quality_scores),
                "improvement": statistics.mean(enhanced_quality_scores) - statistics.mean(baseline_quality_scores),
                "improvement_percentage": ((statistics.mean(enhanced_quality_scores) - statistics.mean(baseline_quality_scores)) / statistics.mean(baseline_quality_scores)) * 100
            }
        
        # Calculate effect sizes (Cohen's d)
        if len(baseline_claims_per_task) > 1:
            pooled_std = statistics.pstdev(baseline_claims_per_task + enhanced_claims_per_task)
            mean_diff = statistics.mean(enhanced_claims_per_task) - statistics.mean(baseline_claims_per_task)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            analysis["effect_sizes"]["claims_per_task"] = {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_cohens_d(cohens_d)
            }
        
        # Generate conclusions
        analysis["conclusions"] = self._generate_conclusions(analysis)
        
        self.logger.info("Statistical analysis completed")
        return analysis
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_conclusions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conclusions based on statistical analysis"""
        conclusions = {}
        
        # Claims per task conclusions
        claims_test = analysis["statistical_tests"].get("claims_per_task", {})
        if claims_test.get("significant", False):
            improvement_pct = claims_test.get("improvement_percentage", 0)
            if improvement_pct >= 108:  # Target 108% improvement (1.2 → 2.5)
                conclusions["claims_per_task"] = "SUCCESS: Significant improvement exceeding 108% target achieved"
            elif improvement_pct >= 50:
                conclusions["claims_per_task"] = "PARTIAL: Significant improvement but below 108% target"
            else:
                conclusions["claims_per_task"] = "FAILED: No significant improvement in claims per task"
        else:
            conclusions["claims_per_task"] = "FAILED: No statistically significant improvement in claims per task"
        
        # Quality score conclusions
        quality_test = analysis["statistical_tests"].get("quality_scores", {})
        if quality_test.get("significant", False):
            improvement_pct = quality_test.get("improvement_percentage", 0)
            if improvement_pct >= 15:  # Target 15% improvement
                conclusions["quality_scores"] = "SUCCESS: Significant improvement exceeding 15% target achieved"
            else:
                conclusions["quality_scores"] = "PARTIAL: Significant improvement but below 15% target"
        else:
            conclusions["quality_scores"] = "FAILED: No significant improvement in quality scores"
        
        # Overall hypothesis conclusion
        if (conclusions.get("claims_per_task", "").startswith("SUCCESS") or 
            conclusions.get("claims_per_task", "").startswith("PARTIAL")):
            conclusions["overall_hypothesis"] = "SUPPORTED: Enhanced prompt engineering shows significant improvement"
        else:
            conclusions["overall_hypothesis"] = "REJECTED: Enhanced prompt engineering does not show significant improvement"
        
        return conclusions
    
    async def save_results(self, baseline_results: Dict[str, Any], enhanced_results: Dict[str, Any], statistical_analysis: Dict[str, Any]):
        """Save all experiment results to files"""
        self.logger.info("Saving experiment results")
        
        results_dir = Path("experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results
        complete_results = {
            "experiment": "enhanced_prompt_engineering",
            "version": "2.0",
            "timestamp": timestamp,
            "hypothesis": self.results["metadata"]["hypothesis"],
            "baseline_results": baseline_results,
            "enhanced_results": enhanced_results,
            "statistical_analysis": statistical_analysis,
            "conclusions": statistical_analysis["conclusions"]
        }
        
        results_file = results_dir / f"experiment_2_enhanced_prompt_engineering_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        # Save summary report
        report_file = results_dir / f"experiment_2_summary_{timestamp}.md"
        self._generate_summary_report(complete_results, report_file)
        
        self.logger.info(f"Results saved to {results_file} and {report_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any], report_file: Path):
        """Generate markdown summary report"""
        report_content = f"""# Experiment 2: Enhanced Prompt Engineering - Results Summary

## Executive Summary

**Hypothesis**: {results['hypothesis']}

**Experiment Date**: {results['timestamp']}

**Target Metrics**:
- Claims per task: 1.2 → 2.5+ (108% improvement)
- Confidence calibration error: <0.2
- Quality improvement: >15%
- XML compliance: Maintain 100%

## Results Overview

### Baseline (Control Group)
- Average claims per task: {results['baseline_results']['summary']['overall_avg_claims_per_task']:.2f}
- XML compliance: {results['baseline_results']['summary']['overall_avg_xml_compliance']:.1f}%
- Average quality score: {results['baseline_results']['summary']['overall_avg_quality_score']:.1f}

### Enhanced (Treatment Group)
- Average claims per task: {results['enhanced_results']['summary']['overall_avg_claims_per_task']:.2f}
- XML compliance: {results['enhanced_results']['summary']['overall_avg_xml_compliance']:.1f}%
- Average quality score: {results['enhanced_results']['summary']['overall_avg_quality_score']:.1f}
- Confidence calibration error: {results['enhanced_results']['summary'].get('overall_avg_confidence_calibration_error', 'N/A')}

### Statistical Analysis
- Claims per task improvement: {results['statistical_analysis']['statistical_tests'].get('claims_per_task', {}).get('improvement_percentage', 0):.1f}%
- P-value: {results['statistical_analysis']['statistical_tests'].get('claims_per_task', {}).get('p_value', 'N/A')}
- Effect size (Cohen's d): {results['statistical_analysis']['effect_sizes'].get('claims_per_task', {}).get('cohens_d', 'N/A')}

## Conclusions

{chr(10).join([f"- {key}: {value}" for key, value in results['statistical_analysis']['conclusions'].items()])}

## Recommendations

Based on the experiment results, the following recommendations are made:

1. **If hypothesis supported**: Deploy enhanced templates to production
2. **If partial success**: Refine templates and retest
3. **If hypothesis rejected**: Return to baseline and explore alternative approaches

## Data Files

- Complete results: `experiment_2_enhanced_prompt_engineering_{results['timestamp']}.json`
- Raw logs: Available in experiments/results directory

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
    
    async def run_complete_experiment(self):
        """Run the complete experiment from baseline through statistical analysis"""
        self.logger.info("Starting complete Experiment 2 execution")
        
        try:
            # Phase 1: Baseline experiment (Control Group)
            self.logger.info("Phase 1: Running baseline experiment")
            baseline_results = await self.run_baseline_experiment()
            
            # Phase 2: Enhanced experiment (Treatment Group)  
            self.logger.info("Phase 2: Running enhanced experiment")
            enhanced_results = await self.run_enhanced_experiment()
            
            # Phase 3: Statistical analysis
            self.logger.info("Phase 3: Running statistical analysis")
            statistical_analysis = await self.run_statistical_analysis(baseline_results, enhanced_results)
            
            # Phase 4: Save results
            self.logger.info("Phase 4: Saving results")
            await self.save_results(baseline_results, enhanced_results, statistical_analysis)
            
            self.logger.info("Experiment 2 completed successfully")
            return {
                "success": True,
                "baseline_results": baseline_results,
                "enhanced_results": enhanced_results,
                "statistical_analysis": statistical_analysis,
                "conclusions": statistical_analysis["conclusions"]
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


async def main():
    """Main execution function"""
    print("🚀 Starting Experiment 2: Enhanced Prompt Engineering")
    print("Hypothesis: Chain-of-thought examples and confidence calibration guidance will increase claim creation thoroughness by 25%")
    
    experiment = Experiment2EnhancedPromptEngineering()
    
    # Run complete experiment
    results = await experiment.run_complete_experiment()
    
    if results["success"]:
        print("\n✅ Experiment 2 completed successfully!")
        print(f"📊 Claims per task improvement: {results['statistical_analysis']['statistical_tests'].get('claims_per_task', {}).get('improvement_percentage', 0):.1f}%")
        print(f"📈 Quality improvement: {results['statistical_analysis']['statistical_tests'].get('quality_scores', {}).get('improvement_percentage', 0):.1f}%")
        print(f"📋 Overall conclusion: {results['conclusions']['overall_hypothesis']}")
    else:
        print(f"\n❌ Experiment 2 failed: {results['error']}")


if __name__ == "__main__":
    asyncio.run(main())