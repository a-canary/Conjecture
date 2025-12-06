#!/usr/bin/env python3
"""
Context Window Optimizer Validation Script

Comprehensive validation of the advanced context optimization system
for tiny LLM enhancement. Tests real-world scenarios and benchmarks
performance improvements.

Usage:
    python validation_scripts/validate_context_optimizer.py [--quick] [--detailed] [--export-results]
"""

import asyncio
import json
import time
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
import statistics
from datetime import datetime

from src.processing.context_optimization_system import (
    ContextOptimizationSystem,
    OptimizationRequest,
    SystemConfiguration,
    TaskType,
    ComponentType,
    create_context_optimization_system
)
from src.processing.advanced_context_optimizer import TaskType
from src.processing.dynamic_context_allocator import ComponentType


class ContextOptimizerValidator:
    """Comprehensive validator for context optimization system"""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.system = create_context_optimization_system(
            SystemConfiguration(
                model_name="ibm/granite-4-h-tiny",
                default_token_budget=2048,
                enable_learning=True,
                cache_optimizations=True
            )
        )
        self.validation_results = {}
        self.test_scenarios = self._load_test_scenarios()

    def _load_test_scenarios(self) -> List[Dict[str, Any]]:
        """Load test scenarios for validation"""
        if self.quick_mode:
            return [
                {
                    "name": "Basic Reasoning Task",
                    "context": self._get_basic_context(),
                    "task_type": TaskType.REASONING,
                    "keywords": ["analysis", "conclusion"],
                    "expected_compression_range": (0.4, 0.8),
                    "components": [ComponentType.CLAIM_PROCESSING, ComponentType.REASONING_ENGINE]
                },
                {
                    "name": "Information Synthesis",
                    "context": self._get_synthesis_context(),
                    "task_type": TaskType.SYNTHESIS,
                    "keywords": ["synthesis", "integration"],
                    "expected_compression_range": (0.3, 0.7),
                    "components": [ComponentType.EVIDENCE_SYNTHESIS, ComponentType.CLAIM_PROCESSING]
                }
            ]
        else:
            return [
                {
                    "name": "Complex Reasoning Task",
                    "context": self._get_complex_reasoning_context(),
                    "task_type": TaskType.REASONING,
                    "keywords": ["logical reasoning", "premise", "conclusion"],
                    "expected_compression_range": (0.3, 0.6),
                    "components": [
                        ComponentType.CLAIM_PROCESSING,
                        ComponentType.REASONING_ENGINE,
                        ComponentType.WORKING_MEMORY
                    ]
                },
                {
                    "name": "Information Synthesis",
                    "context": self._get_synthesis_context(),
                    "task_type": TaskType.SYNTHESIS,
                    "keywords": ["synthesis", "integration", "summary"],
                    "expected_compression_range": (0.3, 0.7),
                    "components": [
                        ComponentType.EVIDENCE_SYNTHESIS,
                        ComponentType.CLAIM_PROCESSING,
                        ComponentType.REASONING_ENGINE
                    ]
                },
                {
                    "name": "Detailed Analysis",
                    "context": self._get_analysis_context(),
                    "task_type": TaskType.ANALYSIS,
                    "keywords": ["analysis", "examination", "evaluation"],
                    "expected_compression_range": (0.4, 0.8),
                    "components": [
                        ComponentType.CLAIM_PROCESSING,
                        ComponentType.EVIDENCE_SYNTHESIS,
                        ComponentType.WORKING_MEMORY
                    ]
                },
                {
                    "name": "Decision Making",
                    "context": self._get_decision_context(),
                    "task_type": TaskType.DECISION,
                    "keywords": ["decision", "choice", "outcome"],
                    "expected_compression_range": (0.3, 0.6),
                    "components": [
                        ComponentType.CLAIM_PROCESSING,
                        ComponentType.REASONING_ENGINE,
                        ComponentType.TASK_INSTRUCTIONS
                    ]
                },
                {
                    "name": "Content Creation",
                    "context": self._get_creation_context(),
                    "task_type": TaskType.CREATION,
                    "keywords": ["create", "generate", "produce"],
                    "expected_compression_range": (0.5, 0.9),
                    "components": [
                        ComponentType.TASK_INSTRUCTIONS,
                        ComponentType.EXAMPLES,
                        ComponentType.OUTPUT_FORMAT
                    ]
                }
            ]

    def _get_basic_context(self) -> str:
        """Basic test context"""
        return """
        Machine learning algorithms have revolutionized data analysis and prediction.
        Supervised learning uses labeled data to train models that can make predictions
        on new, unseen data. Common algorithms include decision trees, random forests,
        and neural networks. Unsupervised learning discovers patterns in unlabeled data
        through clustering and dimensionality reduction techniques.
        """

    def _get_complex_reasoning_context(self) -> str:
        """Complex reasoning context with logical arguments"""
        return """
        ## The Impact of Artificial Intelligence on Employment

        ### Premise 1: Historical Context
        Throughout history, technological advancements have consistently transformed employment patterns.
        The Industrial Revolution replaced agricultural jobs with manufacturing jobs, while the Computer Revolution
        automated clerical tasks while creating new technical roles. Each technological wave initially caused
        job displacement but ultimately created more jobs than it destroyed.

        ### Premise 2: AI Capabilities and Limitations
        Modern AI systems excel at pattern recognition, data processing, and repetitive tasks.
        However, they currently lack genuine creativity, emotional intelligence, and complex reasoning abilities.
        AI can augment human capabilities but cannot fully replicate human judgment and creativity.

        ### Premise 3: Economic Evidence
        Recent economic studies show that AI adoption correlates with increased productivity and job creation
        in sectors that effectively integrate human-AI collaboration. Companies that invest in AI and employee
        reskilling programs show higher employment growth than those that focus solely on automation.

        ### Premise 4: Skill Evolution
        The job market is evolving toward skills that complement AI capabilities:
        - Critical thinking and complex problem-solving
        - Emotional intelligence and interpersonal communication
        - Creativity and innovation
        - AI system design and oversight

        ### Intermediate Conclusions
        1. AI will eliminate some jobs but create new ones requiring different skills
        2. The net employment effect depends on adaptation strategies and policies
        3. Human-AI collaboration is more productive than AI alone or humans alone

        ### Final Conclusion
        Based on historical precedents, current AI capabilities, economic evidence, and skill evolution trends,
        artificial intelligence is likely to transform employment patterns rather than eliminate jobs overall.
        The key to positive outcomes lies in proactive adaptation, education, and policies that support
        human-AI collaboration.
        """

    def _get_synthesis_context(self) -> str:
        """Information synthesis context with diverse data points"""
        return """
        ## Climate Change Mitigation Strategies: A Comprehensive Analysis

        ### Renewable Energy Technologies
        Solar power costs have decreased by 89% since 2010, making it the cheapest form of new electricity
        generation in many countries. Wind power capacity has grown by 75% globally since 2015.
        However, renewable energy sources face intermittency challenges and require significant infrastructure
        investments and storage solutions.

        ### Carbon Capture and Storage
        Direct air capture technology can remove CO2 from the atmosphere but currently costs $600-1000 per ton.
        Natural solutions like reforestation and soil carbon sequestration offer more cost-effective approaches
        but have land use limitations and saturation points. Industrial carbon capture can prevent emissions
        from major sources but doesn't remove existing atmospheric CO2.

        ### Energy Efficiency Improvements
        Building efficiency improvements can reduce energy consumption by 30-50% through better insulation,
        smart thermostats, and efficient appliances. Industrial process optimization can achieve similar
        reductions while improving productivity. Transportation efficiency gains from electric vehicles and
        improved public transit can significantly reduce emissions.

        ### Economic Considerations
        The International Energy Agency estimates that achieving net-zero emissions by 2050 requires $4 trillion
        in annual investments. However, the economic benefits from avoided climate damages, health improvements,
        and job creation in clean energy sectors could offset these costs over time.

        ### Policy Frameworks
        Carbon pricing mechanisms have proven effective in 40+ countries, reducing emissions while driving
        innovation. Renewable portfolio standards have accelerated clean energy adoption. International cooperation
        through agreements like the Paris Agreement provides coordination mechanisms but requires stronger
        enforcement and ambition.

        ### Social Equity Considerations
        Climate solutions must address energy poverty and ensure just transitions for workers in fossil fuel
        industries. Developing countries need financial and technological support to leapfrog fossil fuel
        infrastructure. Indigenous communities and vulnerable populations require special consideration
        in climate adaptation strategies.

        ### Synthesis Challenge
        Integrating these diverse approaches requires balancing technological feasibility, economic viability,
        political acceptability, and social equity. No single solution can address climate change alone;
        success depends on coordinated implementation across multiple strategies while addressing
        regional variations and development needs.
        """

    def _get_analysis_context(self) -> str:
        """Detailed analysis context"""
        return """
        ## Comprehensive Analysis of Remote Work Productivity

        ### Quantitative Performance Metrics
        Multiple studies from Stanford, Harvard Business School, and MIT have examined remote work productivity:
        - 16% overall productivity increase in remote workers (Stanford, 2023)
        - 13% reduction in task completion time
        - 9% increase in output quality metrics
        - 41% decrease in absenteeism
        - 50% reduction in employee turnover

        However, these results vary significantly by industry, role type, and implementation quality.

        ### Qualitative Factors Analysis
        Employee satisfaction surveys show:
        - 78% of employees prefer hybrid or fully remote work
        - 65% report better work-life balance
        - 71% feel more autonomous and trusted
        - 43% report feelings of isolation
        - 27% experience difficulties with collaboration
        - 34% face challenges with work-home boundary management

        ### Industry-Specific Findings
        Technology Sector: Strong positive results with 20% productivity gains
        Financial Services: Mixed results, 8% productivity gains with increased compliance challenges
        Healthcare: Limited applicability, primarily for administrative functions
        Manufacturing: Minimal remote work potential, except for design and management roles
        Education: Surprisingly effective for certain subjects with proper technology support

        ### Management and Leadership Considerations
        Remote work requires different leadership approaches:
        - Shift from presenteeism to results-oriented management
        - Increased emphasis on communication clarity and frequency
        - Greater trust in employee autonomy and self-management
        - More intentional culture-building and team cohesion activities
        - Enhanced focus on employee mental health and well-being

        ### Technology Infrastructure Analysis
        Successful remote work depends on:
        - High-speed internet access (95% of urban workers vs. 65% of rural workers)
        - Adequate home office setup and equipment
        - Collaboration software proficiency
        - Cybersecurity measures and training
        - Technical support availability

        Challenges identified:
        - Digital divide exacerbates inequality
        - Home internet reliability varies significantly
        - Not all roles have suitable remote work technology
        - Security risks increase with distributed access points
        """

    def _get_decision_context(self) -> str:
        """Decision-making context with multiple options"""
        return """
        ## Strategic Decision: Cloud Infrastructure Migration

        ### Current Situation Analysis
        Our company operates on-premise infrastructure with annual costs of $2.3 million.
        Current utilization averages 45% with peak demand at 75%. Maintenance overhead requires
        8 FTE technical staff. Downtime averages 8 hours annually, costing approximately $180,000
        in lost revenue. Security audits reveal $350,000 in required upgrades over next 2 years.

        ### Option 1: Maintain On-Premise Infrastructure
        Advantages:
        - Complete control over data and systems
        - No migration costs or risks
        - Existing staff expertise
        - Predictable cost structure

        Disadvantages:
        - Higher operational costs ($2.3M annually)
        - Limited scalability
        - Maintenance burden
        - Technology obsolescence risk
        - Energy efficiency concerns

        Capital Requirements: $350,000 for upgrades, ongoing $2.3M annual
        Implementation Timeline: 6 months for upgrades
        Risk Level: Low (status quo)

        ### Option 2: Full Cloud Migration (AWS)
        Advantages:
        - Significant cost savings (estimated 35% reduction)
        - Superior scalability and flexibility
        - Enhanced disaster recovery
        - Access to advanced cloud services
        - Reduced maintenance burden
        - Better energy efficiency

        Disadvantages:
        - Migration complexity and risk
        - Loss of some control
        - Potential vendor lock-in
        - Staff retraining requirements
        - Data security concerns

        Capital Requirements: $800,000 migration cost, $1.5M annual operating
        Implementation Timeline: 18 months full migration
        Risk Level: Medium-High

        ### Option 3: Hybrid Approach
        Advantages:
        - Balanced risk and benefits
        - Gradual migration reduces disruption
        - Keep sensitive data on-premise
        - Cloud scalability for variable workloads
        - Skill development period for staff

        Disadvantages:
        - Increased complexity
        - Duplicate systems during transition
        - Integration challenges
        - Higher than cloud-only costs
        - Management complexity

        Capital Requirements: $500,000 partial migration, $1.8M annual operating
        Implementation Timeline: 24 months full implementation
        Risk Level: Medium

        ### Decision Criteria Weights
        - Cost efficiency: 25%
        - Risk management: 20%
        - Scalability: 15%
        - Security: 20%
        - Operational simplicity: 10%
        - Business continuity: 10%

        Stakeholder Analysis:
        - Finance: Prioritizes cost reduction
        - IT: Prioritizes technical feasibility and security
        - Operations: Prioritizes business continuity
        - Legal: Prioritizes compliance and data protection

        ### Quantitative Analysis Results
        5-year total cost projections:
        - On-premise: $12.5M
        - Full cloud: $8.3M
        - Hybrid: $10.1M

        Risk-adjusted ROI calculations:
        - On-premise: -5% (negative due to obsolescence)
        - Full cloud: 22% (high but with implementation risk)
        - Hybrid: 12% (moderate with balanced risk)

        ### Qualitative Considerations
        Company culture supports innovation but risk-averse
        Technical team has strong on-premise expertise
        Business growth expected 25% annually
        Regulatory requirements favor data control
        Competitive pressure for digital transformation
        """

    def _get_creation_context(self) -> str:
        """Content creation context"""
        return """
        ## Creative Writing Assignment: Science Fiction Short Story

        ### Story Requirements
        Write a 1500-2000 word science fiction story exploring the relationship between
        artificial intelligence and human creativity. Consider these themes:
        - Can AI truly be creative or does it just simulate creativity?
        - What happens when AI systems develop their own artistic preferences?
        - How might human artists collaborate with AI systems?
        - What are the boundaries between human and machine creativity?

        ### Technical Elements Required
        - Setting: Near-future Earth (2040-2050)
        - Protagonist: Human artist struggling with creative block
        - AI Character: Advanced creative assistant system
        - Conflict: Philosophical disagreement about art and creativity
        - Resolution: New understanding of human-AI creative partnership

        ### Style Guidelines
        - Literary fiction style with character depth
        - Philosophical themes woven into narrative
        - Realistic technology extrapolation
        - Emotional resonance and character development
        - Avoid clichés about AI takeover

        ### Inspirational Sources
        Consider drawing from:
        - "Klara and the Sun" by Kazuo Ishiguro (AI perspective)
        - "The Artist's Way" by Julia Cameron (creativity process)
        - Current AI art generation controversies
        - Neuroscience of creativity research
        - Historical artist-assistant relationships

        ### Evaluation Criteria
        - Character development and emotional depth
        - Thematic exploration of AI and creativity
        - Plausible near-future worldbuilding
        - Engaging narrative structure
        - Originality and creative insight
        - Technical writing quality
        """

    async def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation"""
        print("🚀 Starting Context Window Optimizer Validation")
        print("=" * 60)

        validation_start = time.time()
        results = {
            "validation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "quick_mode": self.quick_mode,
                "total_scenarios": len(self.test_scenarios)
            },
            "scenario_results": [],
            "performance_benchmarks": {},
            "system_metrics": {},
            "validation_summary": {}
        }

        # Run each test scenario
        for i, scenario in enumerate(self.test_scenarios, 1):
            print(f"\n📋 Scenario {i}/{len(self.test_scenarios)}: {scenario['name']}")
            print("-" * 40)

            scenario_result = await self._run_scenario(scenario)
            results["scenario_results"].append(scenario_result)

            self._print_scenario_summary(scenario_result)

        # Run performance benchmarks
        print(f"\n⚡ Running Performance Benchmarks")
        print("-" * 40)
        benchmark_results = await self._run_performance_benchmarks()
        results["performance_benchmarks"] = benchmark_results

        # Collect system metrics
        system_status = self.system.get_system_status()
        results["system_metrics"] = system_status

        # Generate validation summary
        results["validation_summary"] = self._generate_validation_summary(results)

        validation_time = time.time() - validation_start

        print(f"\n✅ Validation completed in {validation_time:.2f} seconds")
        self._print_validation_summary(results["validation_summary"])

        self.validation_results = results
        return results

    async def _run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual validation scenario"""
        scenario_start = time.time()

        request = OptimizationRequest(
            context_text=scenario["context"],
            task_type=scenario["task_type"],
            task_keywords=scenario["keywords"],
            performance_requirements={"accuracy": 0.8, "efficiency": 0.7},
            active_components=scenario["components"]
        )

        # Run optimization
        result = await self.system.optimize_context(request)

        # Evaluate results
        evaluation = self._evaluate_scenario_result(
            scenario, result, time.time() - scenario_start
        )

        return {
            "scenario_name": scenario["name"],
            "request": {
                "task_type": scenario["task_type"].value,
                "keywords": scenario["keywords"],
                "components": [c.value for c in scenario["components"]],
                "original_tokens": result.original_tokens,
                "context_length": len(scenario["context"])
            },
            "optimization_result": {
                "optimized_tokens": result.optimized_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time_ms": result.processing_time_ms,
                "allocation": {k.value: v for k, v in result.allocation.items()},
                "metrics": {
                    "semantic_density": result.metrics.semantic_density,
                    "relevance_score": result.metrics.relevance_score,
                    "compression_quality": result.metrics.compression_quality,
                    "token_efficiency": result.metrics.token_efficiency
                }
            },
            "performance_prediction": result.performance_prediction,
            "recommendations": result.recommendations,
            "evaluation": evaluation
        }

    def _evaluate_scenario_result(self, scenario: Dict[str, Any], result, processing_time: float) -> Dict[str, Any]:
        """Evaluate scenario result against expectations"""
        evaluation = {
            "success": True,
            "issues": [],
            "metrics": {}
        }

        # Check compression ratio
        min_expected, max_expected = scenario["expected_compression_range"]
        if not (min_expected <= result.compression_ratio <= max_expected):
            evaluation["success"] = False
            evaluation["issues"].append(
                f"Compression ratio {result.compression_ratio:.2f} outside expected range "
                f"[{min_expected:.2f}, {max_expected:.2f}]"
            )

        # Check processing time
        if processing_time > 10.0:  # 10 second threshold
            evaluation["issues"].append(
                f"Processing time {processing_time:.2f}s exceeds threshold"
            )

        # Check quality metrics
        if result.metrics.semantic_density < 0.3:
            evaluation["issues"].append("Low semantic density detected")

        if result.metrics.compression_quality < 0.7:
            evaluation["issues"].append("Poor compression quality")

        # Calculate performance score
        evaluation["metrics"]["performance_score"] = (
            result.metrics.semantic_density * 0.3 +
            result.metrics.compression_quality * 0.3 +
            result.metrics.token_efficiency * 0.2 +
            (1.0 - min(processing_time / 10.0, 1.0)) * 0.2
        )

        return evaluation

    async def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        benchmarks = {
            "cache_performance": await self._benchmark_cache_performance(),
            "concurrent_performance": await self._benchmark_concurrent_requests(),
            "memory_usage": self._benchmark_memory_usage(),
            "scalability": await self._benchmark_scalability()
        }

        return benchmarks

    async def _benchmark_cache_performance(self) -> Dict[str, Any]:
        """Benchmark cache performance"""
        context = "Cache performance test context with multiple sentences. " * 10
        request = OptimizationRequest(
            context_text=context,
            task_type=TaskType.ANALYSIS,
            task_keywords=["cache", "performance"],
            performance_requirements={},
            active_components=[ComponentType.CLAIM_PROCESSING]
        )

        # First request (cache miss)
        start_time = time.time()
        result1 = await self.system.optimize_context(request)
        first_time = time.time() - start_time

        # Second request (cache hit)
        start_time = time.time()
        result2 = await self.system.optimize_context(request)
        second_time = time.time() - start_time

        cache_speedup = first_time / max(second_time, 0.001)

        return {
            "first_request_ms": first_time * 1000,
            "cached_request_ms": second_time * 1000,
            "cache_speedup": cache_speedup,
            "cache_hit": result1.optimized_context == result2.optimized_context
        }

    async def _benchmark_concurrent_requests(self) -> Dict[str, Any]:
        """Benchmark concurrent request handling"""
        import asyncio

        contexts = [
            f"Concurrent test context {i} with unique content. " * 5
            for i in range(5)
        ]

        requests = [
            OptimizationRequest(
                context_text=context,
                task_type=TaskType.SYNTHESIS,
                task_keywords=[f"concurrent{i}"],
                performance_requirements={},
                active_components=[ComponentType.CLAIM_PROCESSING]
            )
            for context in contexts
        ]

        # Run concurrently
        start_time = time.time()
        results = await asyncio.gather(*[self.system.optimize_context(req) for req in requests])
        total_time = time.time() - start_time

        return {
            "concurrent_requests": len(requests),
            "total_time_ms": total_time * 1000,
            "average_time_ms": (total_time * 1000) / len(requests),
            "all_successful": len(results) == len(requests)
        }

    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage"""
        # Simple memory estimation based on system state
        system_status = self.system.get_system_status()

        return {
            "cache_size": system_status["system_info"]["cache_size"],
            "history_size": len(self.system.optimization_history),
            "estimated_memory_mb": len(str(self.system.__dict__)) / 1024 / 1024
        }

    async def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability"""
        context_sizes = [100, 500, 1000, 2000, 5000]  # words
        scalability_results = []

        for size in context_sizes:
            context = "Scalability test word. " * size
            request = OptimizationRequest(
                context_text=context,
                task_type=TaskType.REASONING,
                task_keywords=["scalability"],
                performance_requirements={},
                active_components=[ComponentType.CLAIM_PROCESSING]
            )

            start_time = time.time()
            result = await self.system.optimize_context(request)
            processing_time = time.time() - start_time

            scalability_results.append({
                "input_size_words": size,
                "original_tokens": result.original_tokens,
                "optimized_tokens": result.optimized_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time_ms": processing_time * 1000,
                "efficiency": result.optimized_tokens / max(result.original_tokens, 1)
            })

        return {
            "scalability_results": scalability_results,
            "time_complexity": self._analyze_time_complexity(scalability_results),
            "compression_stability": self._analyze_compression_stability(scalability_results)
        }

    def _analyze_time_complexity(self, results: List[Dict[str, Any]]) -> str:
        """Analyze time complexity from benchmark results"""
        if len(results) < 2:
            return "insufficient_data"

        # Simple linear regression on log-log scale to estimate complexity
        import math

        log_sizes = [math.log(r["input_size_words"]) for r in results]
        log_times = [math.log(r["processing_time_ms"] / 1000) for r in results]

        # Calculate slope (approximation of complexity exponent)
        n = len(results)
        sum_x = sum(log_sizes)
        sum_y = sum(log_times)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_times))
        sum_x2 = sum(x * x for x in log_sizes)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope < 1.2:
            return "linear"
        elif slope < 1.8:
            return "quadratic"
        else:
            return "greater_than_quadratic"

    def _analyze_compression_stability(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze compression stability across different input sizes"""
        compression_ratios = [r["compression_ratio"] for r in results]

        return {
            "mean_compression": statistics.mean(compression_ratios),
            "compression_std": statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0,
            "compression_range": max(compression_ratios) - min(compression_ratios),
            "stability_score": 1.0 - (statistics.stdev(compression_ratios) if len(compression_ratios) > 1 else 0)
        }

    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall validation summary"""
        scenario_results = results["scenario_results"]
        successful_scenarios = sum(1 for r in scenario_results if r["evaluation"]["success"])
        total_scenarios = len(scenario_results)

        # Calculate average metrics
        performance_scores = [r["evaluation"]["metrics"]["performance_score"] for r in scenario_results]
        compression_ratios = [r["optimization_result"]["compression_ratio"] for r in scenario_results]
        processing_times = [r["optimization_result"]["processing_time_ms"] for r in scenario_results]

        return {
            "overall_success_rate": successful_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "successful_scenarios": successful_scenarios,
            "total_scenarios": total_scenarios,
            "average_performance_score": statistics.mean(performance_scores) if performance_scores else 0,
            "average_compression_ratio": statistics.mean(compression_ratios) if compression_ratios else 0,
            "average_processing_time_ms": statistics.mean(processing_times) if processing_times else 0,
            "cache_performance_speedup": results["performance_benchmarks"]["cache_performance"]["cache_speedup"],
            "system_health_score": self._calculate_system_health_score(results),
            "recommendations": self._generate_overall_recommendations(results)
        }

    def _calculate_system_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        summary = results["validation_summary"]
        benchmarks = results["performance_benchmarks"]

        score_components = [
            ("success_rate", summary["overall_success_rate"] * 0.3),
            ("performance", summary["average_performance_score"] * 0.2),
            ("compression_efficiency", min(summary["average_compression_ratio"], 1.0) * 0.2),
            ("processing_speed", 1.0 - min(summary["average_processing_time_ms"] / 5000, 1.0) * 0.1),
            ("cache_efficiency", min(benchmarks["cache_performance"]["cache_speedup"] / 5, 1.0) * 0.1),
            ("scalability", 0.8)  # Placeholder for scalability score
        ]

        return sum(score for _, score in score_components)

    def _generate_overall_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on validation results"""
        recommendations = []
        summary = results["validation_summary"]

        if summary["overall_success_rate"] < 0.8:
            recommendations.append("Overall success rate below 80% - review system configuration")

        if summary["average_performance_score"] < 0.7:
            recommendations.append("Consider optimizing compression quality and semantic density")

        if summary["average_processing_time_ms"] > 2000:
            recommendations.append("Processing time exceeds 2 seconds - consider performance optimizations")

        if summary["cache_performance_speedup"] < 2.0:
            recommendations.append("Cache performance could be improved - review caching strategy")

        if summary["system_health_score"] < 0.7:
            recommendations.append("System health score below threshold - comprehensive review recommended")

        if not recommendations:
            recommendations.append("System performing well - no critical issues identified")

        return recommendations

    def _print_scenario_summary(self, scenario_result: Dict[str, Any]):
        """Print summary for individual scenario"""
        result = scenario_result["optimization_result"]
        evaluation = scenario_result["evaluation"]

        status = "✅ PASS" if evaluation["success"] else "❌ FAIL"
        print(f"Status: {status}")
        print(f"  Original: {scenario_result['request']['original_tokens']} tokens")
        print(f"  Optimized: {result['optimized_tokens']} tokens")
        print(f"  Compression: {result['compression_ratio']:.1%}")
        print(f"  Processing: {result['processing_time_ms']:.0f}ms")
        print(f"  Performance Score: {evaluation['metrics']['performance_score']:.2f}")

        if evaluation["issues"]:
            print("  Issues:")
            for issue in evaluation["issues"]:
                print(f"    - {issue}")

    def _print_validation_summary(self, summary: Dict[str, Any]):
        """Print overall validation summary"""
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)

        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Successful Scenarios: {summary['successful_scenarios']}/{summary['total_scenarios']}")
        print(f"Average Performance Score: {summary['average_performance_score']:.2f}")
        print(f"Average Compression Ratio: {summary['average_compression_ratio']:.1%}")
        print(f"Average Processing Time: {summary['average_processing_time_ms']:.0f}ms")
        print(f"Cache Performance Speedup: {summary['cache_performance_speedup']:.1f}x")
        print(f"System Health Score: {summary['system_health_score']:.2f}")

        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"{i}. {rec}")

        # Overall assessment
        if summary["system_health_score"] >= 0.8:
            print("\n🟢 OVERALL ASSESSMENT: EXCELLENT")
            print("   Context optimization system is performing exceptionally well!")
        elif summary["system_health_score"] >= 0.6:
            print("\n🟡 OVERALL ASSESSMENT: GOOD")
            print("   System is performing well with minor areas for improvement.")
        else:
            print("\n🔴 OVERALL ASSESSMENT: NEEDS ATTENTION")
            print("   Several areas require optimization and improvement.")

    def export_results(self, file_path: str = None):
        """Export validation results to file"""
        if not self.validation_results:
            print("No validation results to export")
            return

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"validation_results_{timestamp}.json"

        with open(file_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        print(f"\n📊 Validation results exported to: {file_path}")


async def main():
    """Main validation script entry point"""
    parser = argparse.ArgumentParser(description="Validate Context Window Optimizer")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (2 scenarios)")
    parser.add_argument("--detailed", action="store_true", help="Run detailed validation (5 scenarios)")
    parser.add_argument("--export", action="store_true", help="Export results to JSON file")
    parser.add_argument("--output", type=str, help="Output file path for results")

    args = parser.parse_args()

    # Determine validation mode
    quick_mode = args.quick or not args.detailed

    # Run validation
    validator = ContextOptimizerValidator(quick_mode=quick_mode)
    results = await validator.run_validation()

    # Export results if requested
    if args.export:
        validator.export_results(args.output)


if __name__ == "__main__":
    asyncio.run(main())