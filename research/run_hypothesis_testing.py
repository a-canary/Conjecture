#!/usr/bin/env python3
"""
Conjecture Hypothesis Testing Orchestrator
Main execution script for comprehensive hypothesis testing with iteration loop
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.common import ProviderConfig
from hypothesis_testing_framework import ConjectureTestingFramework, HypothesisTier
from enhanced_test_generator import ConjectureTestCaseGenerator
from statistical_analyzer import ConjectureStatisticalAnalyzer


class HypothesisTestingOrchestrator:
    """Main orchestrator for Conjecture hypothesis testing"""

    def __init__(self):
        self.framework = ConjectureTestingFramework()
        self.test_generator = ConjectureTestCaseGenerator()
        self.statistical_analyzer = ConjectureStatisticalAnalyzer()

    async def setup_providers(self, config_path: str = None) -> bool:
        """Setup LLM providers for testing"""
        if config_path and Path(config_path).exists():
            # Load from config file
            with open(config_path, "r") as f:
                config_data = json.load(f)

            providers = []
            for provider_config in config_data.get("providers", []):
                providers.append(
                    ProviderConfig(
                        url=provider_config["url"],
                        api_key=provider_config.get("api_key", ""),
                        model=provider_config["model"],
                    )
                )
        else:
            # Default providers
            providers = [
                ProviderConfig(
                    url="http://localhost:1234",  # LM Studio
                    api_key="",
                    model="ibm/granite-4-h-tiny",
                ),
                ProviderConfig(
                    url="http://localhost:1234",  # LM Studio
                    api_key="",
                    model="glm-z1-9b-0414",
                ),
                ProviderConfig(
                    url="https://llm.chutes.ai/v1",  # Chutes
                    api_key="your-api-key",
                    model="zai-org/GLM-4.6",
                ),
            ]

        return await self.framework.initialize(providers)

    async def run_complete_testing_cycle(
        self,
        config_path: str = None,
        hypothesis_tiers: list = None,
        max_iterations: int = 5,
        generate_new_tests: bool = True,
    ) -> dict:
        """Run complete hypothesis testing cycle"""
        print("🚀 Starting Conjecture Hypothesis Testing Cycle")
        print("=" * 60)

        # Step 1: Setup providers
        print("📡 Step 1: Setting up LLM providers...")
        if not await self.setup_providers(config_path):
            print("❌ Failed to setup providers")
            return {"status": "failed", "error": "provider_setup_failed"}

        print("✅ Providers setup complete")

        # Step 2: Generate test cases (if requested)
        if generate_new_tests:
            print("🧪 Step 2: Generating comprehensive test cases...")
            categories = [
                "task_decomposition",
                "relevant_context",
                "cost_efficiency",
                "model_parity",
                "claims_reasoning",
                "end_to_end_pipeline",
            ]

            generated_cases = self.test_generator.generate_test_cases(categories)
            print(
                f"✅ Generated {sum(len(cases) for cases in generated_cases.values())} test cases"
            )

        # Step 3: Select hypotheses to test
        print("🎯 Step 3: Selecting hypotheses for testing...")

        if hypothesis_tiers is None:
            # Test core technical hypotheses first
            hypothesis_ids = [
                "central_hypothesis",
                "task_decomposition",
                "cost_efficiency",
                "model_parity",
                "claims_reasoning",
                "end_to_end_pipeline",
            ]
        else:
            # Filter by tier
            hypothesis_ids = []
            for hypo_id, config in self.framework.hypotheses.items():
                if config.tier in hypothesis_tiers:
                    hypothesis_ids.append(hypo_id)

        print(f"📍 Selected {len(hypothesis_ids)} hypotheses for testing")

        # Step 4: Run hypothesis testing with iteration loop
        print("🔄 Step 4: Running hypothesis testing with iteration loop...")

        results = await self.framework.run_hypothesis_testing_cycle(
            hypothesis_ids=hypothesis_ids, max_iterations=max_iterations
        )

        # Step 5: Generate comprehensive analysis
        print("📊 Step 5: Generating comprehensive analysis...")

        analysis_results = {}
        for hypothesis_id, result in results.items():
            # Load test results for this hypothesis
            test_results = self._load_test_results_for_hypothesis(hypothesis_id)

            if test_results:
                success_criteria = self._get_success_criteria_for_hypothesis(
                    hypothesis_id
                )
                analysis = self.statistical_analyzer.analyze_hypothesis_results(
                    hypothesis_id, test_results, success_criteria
                )
                analysis_results[hypothesis_id] = analysis

        # Step 6: Generate final report
        print("📋 Step 6: Generating final comprehensive report...")

        final_report = await self._generate_final_report(results, analysis_results)

        # Save final report
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = Path(
            f"research/analysis/conjecture_hypothesis_final_report_{timestamp}.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        print(f"✅ Final report saved to: {report_path}")

        # Step 7: Display progress dashboard
        print("📈 Step 7: Displaying progress dashboard...")

        dashboard = self.framework.get_progress_dashboard()
        self._display_progress_dashboard(dashboard)

        return {
            "status": "completed",
            "hypotheses_tested": len(results),
            "results": results,
            "analysis": analysis_results,
            "final_report_path": str(report_path),
            "progress_dashboard": dashboard,
        }

    def _load_test_results_for_hypothesis(self, hypothesis_id: str) -> list:
        """Load test results for a specific hypothesis"""
        results_dir = Path("research/results")

        # Look for hypothesis result files
        hypothesis_files = list(results_dir.glob(f"hypothesis_{hypothesis_id}_*.json"))

        if not hypothesis_files:
            return []

        # Load the most recent file
        latest_file = max(hypothesis_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract test results from iteration history
            test_results = []

            # Load iteration files
            iteration_files = list(
                results_dir.glob(f"iteration_{hypothesis_id}_*.json")
            )
            for iter_file in sorted(iteration_files, key=lambda f: f.stat().st_mtime):
                try:
                    with open(iter_file, "r", encoding="utf-8") as f:
                        iter_data = json.load(f)

                    if "test_results" in iter_data:
                        test_results.extend(iter_data["test_results"])
                except Exception as e:
                    print(f"Warning: Could not load iteration file {iter_file}: {e}")

            return test_results

        except Exception as e:
            print(
                f"Warning: Could not load hypothesis results for {hypothesis_id}: {e}"
            )
            return []

    def _get_success_criteria_for_hypothesis(self, hypothesis_id: str) -> dict:
        """Get success criteria for a specific hypothesis"""
        criteria_map = {
            "central_hypothesis": {
                "primary_metric": "correctness",
                "metrics": ["correctness", "reasoning_quality", "completeness"],
                "threshold": 0.7,
                "improvement_threshold": 0.2,  # 20% improvement
                "statistical_threshold": 0.05,
            },
            "task_decomposition": {
                "primary_metric": "correctness",
                "metrics": ["correctness", "reasoning_quality", "completeness"],
                "threshold": 0.7,
                "improvement_threshold": 0.2,
                "statistical_threshold": 0.05,
            },
            "cost_efficiency": {
                "primary_metric": "efficiency",
                "metrics": ["efficiency", "correctness"],
                "threshold": 0.6,
                "improvement_threshold": 0.15,  # 15% cost reduction
                "statistical_threshold": 0.05,
            },
            "model_parity": {
                "primary_metric": "correctness",
                "metrics": ["correctness", "reasoning_quality"],
                "threshold": 0.65,
                "improvement_threshold": 0.1,  # ≤10% performance gap
                "statistical_threshold": 0.05,
            },
            "claims_reasoning": {
                "primary_metric": "correctness",
                "metrics": [
                    "correctness",
                    "confidence_calibration",
                    "reasoning_quality",
                ],
                "threshold": 0.65,
                "improvement_threshold": 0.15,  # 15% improvement
                "statistical_threshold": 0.05,
            },
            "end_to_end_pipeline": {
                "primary_metric": "correctness",
                "metrics": ["correctness", "completeness", "reasoning_quality"],
                "threshold": 0.7,
                "improvement_threshold": 0.25,  # 25% improvement
                "statistical_threshold": 0.05,
            },
        }

        return criteria_map.get(
            hypothesis_id,
            {
                "primary_metric": "correctness",
                "metrics": ["correctness"],
                "threshold": 0.7,
                "improvement_threshold": 0.1,
                "statistical_threshold": 0.05,
            },
        )

    async def _generate_final_report(
        self, hypothesis_results: dict, analysis_results: dict
    ) -> str:
        """Generate final comprehensive report"""
        report_lines = [
            "# Conjecture Hypothesis Testing - Final Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"This report presents the comprehensive results of testing Conjecture's core hypotheses through systematic experimentation and statistical validation.",
            "",
            f"**Total Hypotheses Tested**: {len(hypothesis_results)}",
            "",
            "## Testing Methodology",
            "",
            "### Framework Design",
            "- **5-Phase Testing Cycle**: Baseline Establishment → Initial Testing → Analysis & Refinement → Iteration Testing → Consolidation & Validation",
            "- **Statistical Validation**: Paired t-tests, ANOVA, chi-square tests with α=0.05",
            "- **Effect Size Analysis**: Cohen's d, eta-squared, and Cramér's V",
            "- **Power Analysis**: Target 80% statistical power with sample size validation",
            "- **Iteration Loop**: Automatic refinement based on performance thresholds",
            "",
            "### Evaluation Rubrics",
            "- **Correctness** (weight: 1.5, threshold: 0.70): Factual accuracy",
            "- **Reasoning Quality** (weight: 1.2, threshold: 0.65): Logical reasoning quality",
            "- **Completeness** (weight: 1.0, threshold: 0.75): Coverage of requirements",
            "- **Coherence** (weight: 1.0, threshold: 0.70): Logical flow and consistency",
            "- **Confidence Calibration** (weight: 1.0, threshold: 0.60): Confidence vs accuracy alignment",
            "- **Efficiency** (weight: 0.5, threshold: 0.60): Token/time efficiency",
            "- **Hallucination Reduction** (weight: 1.3, threshold: 0.80): Factual grounding improvement",
            "",
            "## Hypothesis Testing Results",
            "",
        ]

        # Group results by tier
        tier_results = {}
        for hypothesis_id, result in hypothesis_results.items():
            if hypothesis_id in self.framework.hypotheses:
                tier = self.framework.hypotheses[hypothesis_id].tier.value
                if tier not in tier_results:
                    tier_results[tier] = []
                tier_results[tier].append((hypothesis_id, result))

        # Add tier-by-tier results
        for tier, hypotheses in tier_results.items():
            report_lines.extend([f"### {tier.replace('_', ' ').title()}", ""])

            for hypothesis_id, result in hypotheses:
                status_emoji = {
                    "proven": "✅",
                    "partially_proven": "🟡",
                    "disproven": "❌",
                    "in_progress": "🔄",
                    "untested": "❓",
                }.get(result.status.value, "❓")

                report_lines.extend(
                    [
                        f"#### {status_emoji} {hypothesis_id.replace('_', ' ').title()}",
                        f"**Status**: {result.status.value}",
                        f"**Confidence**: {result.confidence:.2f}",
                        f"**Evidence Strength**: {result.evidence_strength.value}",
                        f"**Primary Score**: {result.primary_metric_score:.3f}",
                        f"**Statistical Significance**: p={result.statistical_significance:.4f}",
                        f"**Effect Size**: {result.effect_size:.3f}",
                        f"**Iterations Completed**: {result.iterations_completed}",
                        "",
                    ]
                )

                # Add analysis if available
                if hypothesis_id in analysis_results:
                    analysis = analysis_results[hypothesis_id]
                    report_lines.extend(
                        [
                            "**Statistical Analysis**:",
                            f"- Sample Size: {analysis.get('sample_size', 'N/A')}",
                            f"- Power Achieved: {analysis.get('power_analysis', {}).get('power', 'N/A'):.3f}",
                            f"- Practical Significance: {self._assess_practical_sig(analysis)}",
                            "",
                        ]
                    )

        # Add overall summary
        proven_count = len(
            [r for r in hypothesis_results.values() if r.status.value == "proven"]
        )
        partially_proven_count = len(
            [
                r
                for r in hypothesis_results.values()
                if r.status.value == "partially_proven"
            ]
        )
        total_count = len(hypothesis_results)

        report_lines.extend(
            [
                "## Overall Summary",
                "",
                f"**Proven Hypotheses**: {proven_count}/{total_count} ({proven_count / total_count * 100:.1f}%)",
                f"**Partially Proven**: {partially_proven_count}/{total_count} ({partially_proven_count / total_count * 100:.1f}%)",
                f"**Overall Success Rate**: {(proven_count + 0.5 * partially_proven_count) / total_count * 100:.1f}%",
                "",
                "## Key Findings",
                "",
            ]
        )

        # Add key findings based on results
        key_findings = self._extract_key_findings(hypothesis_results, analysis_results)
        for finding in key_findings:
            report_lines.extend([f"- {finding}", ""])

        # Add recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### For Proven Hypotheses",
                "- **Immediate Integration**: Incorporate proven methodologies into production Conjecture implementation",
                "- **Documentation**: Create best practice guides and implementation patterns",
                "- **Scaling**: Extend validation to broader test cases and real-world scenarios",
                "",
                "### For Partially Proven Hypotheses",
                "- **Targeted Refinement**: Focus on specific failure modes and edge cases",
                "- **Additional Iterations**: Conduct 2-3 more testing cycles with refined approaches",
                "- **Hybrid Approaches**: Consider combining proven and unproven elements",
                "",
                "### For Disproven Hypotheses",
                "- **Strategic Pivot**: Re-evaluate fundamental assumptions and design principles",
                "- **Alternative Approaches**: Explore different methodological directions",
                "- **Documentation**: Record lessons learned to avoid repeated failures",
                "",
                "### Next Steps",
                "- **Production Readiness**: Focus on integrating proven hypotheses into core system",
                "- **Extended Validation**: Test proven approaches with larger, more diverse datasets",
                "- **Continuous Testing**: Establish ongoing validation framework for future improvements",
                "",
                "## Technical Implementation Status",
                "",
                "The following implementation status is recommended based on testing results:",
                "",
            ]
        )

        # Add implementation recommendations
        impl_recommendations = self._generate_implementation_recommendations(
            hypothesis_results
        )
        for rec in impl_recommendations:
            report_lines.extend([f"- {rec}", ""])

        report_lines.extend(
            [
                "---",
                f"*Report generated by Conjecture Hypothesis Testing Framework v1.0*",
                f"*Timestamp: {datetime.utcnow().isoformat()}*",
            ]
        )

        return "\n".join(report_lines)

    def _assess_practical_significance(self, analysis: dict) -> str:
        """Assess practical significance from analysis"""
        practical_sig = analysis.get("practical_significance", {})

        significant_count = 0
        total_assessments = 0

        for assessment_name, assessment in practical_sig.items():
            if isinstance(assessment, dict) and assessment.get(
                "practically_significant", False
            ):
                significant_count += 1
            total_assessments += 1

        if total_assessments > 0:
            percentage = significant_count / total_assessments * 100
            if percentage >= 75:
                return "High practical significance"
            elif percentage >= 50:
                return "Moderate practical significance"
            else:
                return "Low practical significance"

        return "Insufficient data"

    def _extract_key_findings(
        self, hypothesis_results: dict, analysis_results: dict
    ) -> list:
        """Extract key findings from results"""
        findings = []

        # Analyze core technical hypotheses
        core_hypotheses = [
            "central_hypothesis",
            "task_decomposition",
            "cost_efficiency",
            "model_parity",
            "claims_reasoning",
            "end_to_end_pipeline",
        ]

        proven_core = []
        for hypo_id in core_hypotheses:
            if (
                hypo_id in hypothesis_results
                and hypothesis_results[hypo_id].status.value == "proven"
            ):
                proven_core.append(hypo_id)

        if len(proven_core) >= 4:  # At least 4 out of 6
            findings.append(
                "Core technical framework shows strong validation with majority of key hypotheses proven"
            )
        elif len(proven_core) >= 2:
            findings.append(
                "Core technical framework shows partial validation with some key hypotheses proven"
            )
        else:
            findings.append(
                "Core technical framework requires significant refinement - few key hypotheses proven"
            )

        # Analyze statistical rigor
        total_analyses = len(analysis_results)
        rigorous_analyses = 0

        for analysis in analysis_results.values():
            if analysis.get("sample_size", 0) >= 20:
                rigorous_analyses += 1

        if rigorous_analyses >= total_analyses * 0.8:
            findings.append(
                "Statistical validation demonstrates rigorous methodology with adequate sample sizes"
            )
        else:
            findings.append(
                "Statistical validation limited by small sample sizes - requires larger datasets"
            )

        # Analyze iteration effectiveness
        total_iterations = sum(
            result.iterations_completed for result in hypothesis_results.values()
        )
        avg_iterations = (
            total_iterations / len(hypothesis_results) if hypothesis_results else 0
        )

        if avg_iterations <= 3:
            findings.append(
                "Iteration loop proves effective with rapid convergence to conclusions"
            )
        elif avg_iterations <= 5:
            findings.append(
                "Iteration loop shows moderate effectiveness with reasonable convergence"
            )
        else:
            findings.append(
                "Iteration loop shows limited effectiveness requiring many iterations"
            )

        return findings

    def _generate_implementation_recommendations(
        self, hypothesis_results: dict
    ) -> list:
        """Generate implementation recommendations based on results"""
        recommendations = []

        # Check central hypothesis
        if "central_hypothesis" in hypothesis_results:
            result = hypothesis_results["central_hypothesis"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Implement full Conjecture methodology with relevant context and in-context learning examples"
                )
                recommendations.append(
                    "✅ Deploy claims-based representation with confidence scoring and evidence linking"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Implement partial Conjecture methodology focusing on successful components"
                )
                recommendations.append(
                    "🟡 Refine context relevance and in-context learning example generation"
                )
            else:
                recommendations.append(
                    "❌ Reconsider central hypothesis - fundamental issues with claims-based approach"
                )

        # Check task decomposition
        if "task_decomposition" in hypothesis_results:
            result = hypothesis_results["task_decomposition"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Implement automatic task decomposition for complex problems"
                )
                recommendations.append(
                    "✅ Add step-by-step breakdown capabilities to Conjecture pipeline"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Implement selective task decomposition for specific problem types"
                )
            else:
                recommendations.append(
                    "❌ Maintain current approach - task decomposition not showing benefit"
                )

        # Check cost efficiency
        if "cost_efficiency" in hypothesis_results:
            result = hypothesis_results["cost_efficiency"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Implement multi-claim evaluation with session persistence"
                )
                recommendations.append(
                    "✅ Optimize for smaller context windows and cumulative token savings"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Implement partial cost optimization with selective claim persistence"
                )
            else:
                recommendations.append(
                    "❌ Focus on single-turn efficiency - multi-claim approach not cost-effective"
                )

        # Check model parity
        if "model_parity" in hypothesis_results:
            result = hypothesis_results["model_parity"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Deploy Conjecture prompting as primary method for tiny models"
                )
                recommendations.append(
                    "✅ Position small models with Conjecture as competitive alternative to large models"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Deploy Conjecture prompting for specific domains where parity achieved"
                )
            else:
                recommendations.append(
                    "❌ Maintain standard prompting for small models - Conjecture not providing parity"
                )

        # Check claims reasoning
        if "claims_reasoning" in hypothesis_results:
            result = hypothesis_results["claims_reasoning"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Implement claims-based reasoning with confidence scoring"
                )
                recommendations.append(
                    "✅ Add evidence evaluation and conflict resolution capabilities"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Implement partial claims-based reasoning for specific evidence types"
                )
            else:
                recommendations.append(
                    "❌ Maintain standard reasoning - claims-based approach not showing improvement"
                )

        # Check end-to-end pipeline
        if "end_to_end_pipeline" in hypothesis_results:
            result = hypothesis_results["end_to_end_pipeline"]
            if result.status.value == "proven":
                recommendations.append(
                    "✅ Implement full Conjecture pipeline with all components integrated"
                )
                recommendations.append(
                    "✅ Deploy comprehensive research and analysis capabilities"
                )
            elif result.status.value == "partially_proven":
                recommendations.append(
                    "🟡 Implement partial pipeline focusing on successful components"
                )
            else:
                recommendations.append(
                    "❌ Maintain modular approach - full pipeline not showing benefit"
                )

        return recommendations

    def _display_progress_dashboard(self, dashboard: dict):
        """Display progress dashboard in terminal"""
        print("\n" + "=" * 70)
        print("📊 CONJECTURE HYPOTHESIS TESTING PROGRESS DASHBOARD")
        print("=" * 70)

        print(
            f"📈 Overall Progress: {dashboard.get('completion_rate', 0):.1%} complete"
        )
        print(
            f"✅ Proven: {dashboard.get('status_breakdown', {}).get('proven', 0)} hypotheses"
        )
        print(
            f"🟡 Partially Proven: {dashboard.get('status_breakdown', {}).get('partially_proven', 0)} hypotheses"
        )
        print(
            f"❌ Disproven: {dashboard.get('status_breakdown', {}).get('disproven', 0)} hypotheses"
        )
        print(
            f"🔄 In Progress: {dashboard.get('status_breakdown', {}).get('in_progress', 0)} hypotheses"
        )

        print("\n📊 Tier Breakdown:")
        tier_breakdown = dashboard.get("tier_breakdown", {})
        for tier, count in tier_breakdown.items():
            tier_name = tier.replace("_", " ").title()
            print(f"  {tier_name}: {count} hypotheses")

        if dashboard.get("active_hypotheses"):
            print(
                f"\n🔄 Currently Testing: {', '.join(dashboard['active_hypotheses'])}"
            )

        print("=" * 70)


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Conjecture Hypothesis Testing Orchestrator"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=[t.value for t in HypothesisTier],
        help="Hypothesis tiers to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum iterations per hypothesis (default: 5)",
    )
    parser.add_argument(
        "--no-new-tests", action="store_true", help="Skip test case generation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with minimal iterations and analysis",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = HypothesisTestingOrchestrator()

    # Configure parameters
    max_iterations = 2 if args.quick else args.iterations
    generate_new_tests = not args.no_new_tests

    print("🚀 Conjecture Hypothesis Testing Orchestrator")
    print(f"📝 Configuration: {args.config or 'default'}")
    print(f"🎯 Tiers: {args.tiers or 'all'}")
    print(f"🔄 Max Iterations: {max_iterations}")
    print(f"🧪 Generate Tests: {generate_new_tests}")
    print()

    # Run complete testing cycle
    results = await orchestrator.run_complete_testing_cycle(
        config_path=args.config,
        hypothesis_tiers=args.tiers,
        max_iterations=max_iterations,
        generate_new_tests=generate_new_tests,
    )

    if results["status"] == "completed":
        print("\n🎉 Hypothesis testing cycle completed successfully!")
        print(f"📋 Final report: {results['final_report_path']}")
    else:
        print(
            f"\n❌ Hypothesis testing failed: {results.get('error', 'unknown_error')}"
        )

    return results


if __name__ == "__main__":
    asyncio.run(main())
