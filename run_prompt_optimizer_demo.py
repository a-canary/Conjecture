#!/usr/bin/env python3
"""
Tiny LLM Prompt Engineering Optimizer Demo

Demonstrates the advanced prompt engineering optimization system
for enhancing tiny LLM performance on complex tasks.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from src.processing.llm_prompts.tiny_llm_optimizer import (
    TinyLLMPromptOptimizer,
    AdaptiveTemplateGenerator,
    TinyModelCapabilityProfiler,
    TaskDescription,
    TaskComplexity,
    OptimizationStrategy
)
from src.processing.llm_prompts.template_evolution import (
    TemplateEvolution,
    EvolutionConfig,
    TemplateGenome
)
from src.processing.llm_prompts.models import PromptTemplate
from tests.tiny_llm_optimizer_tests import run_tiny_llm_optimizer_tests


async def demonstrate_prompt_optimization():
    """Demonstrate prompt optimization capabilities"""

    print("\n" + "="*60)
    print("PROMPT ENGINEERING OPTIMIZER DEMONSTRATION")
    print("="*60)

    # Initialize optimizer
    optimizer = TinyLLMPromptOptimizer()

    # Create sample tasks for different complexity levels
    sample_tasks = [
        {
            "name": "Simple Extraction Task",
            "task_type": "extraction",
            "complexity": TaskComplexity.SIMPLE,
            "input": "The meeting is scheduled for December 15, 2024 at 2:00 PM EST.",
            "context": ["Meeting schedules", "Calendar information"],
            "model": "granite-tiny"
        },
        {
            "name": "Moderate Analysis Task",
            "task_type": "analysis",
            "complexity": TaskComplexity.MODERATE,
            "input": "Q3 sales increased by 15% while customer acquisition costs decreased by 8%.",
            "context": ["Sales reports", "Customer metrics", "Market analysis"],
            "model": "llama-3.2-1b"
        },
        {
            "name": "Complex Research Task",
            "task_type": "research",
            "complexity": TaskComplexity.COMPLEX,
            "input": "Analyze the long-term effects of AI automation on the creative industries.",
            "context": [
                "AI research papers",
                "Creative industry reports",
                "Economic impact studies",
                "Technology trends analysis",
                "Employment statistics",
                "Innovation patterns"
            ],
            "model": "phi-3-mini"
        }
    ]

    print(f"\nDemonstrating optimization on {len(sample_tasks)} tasks...\n")

    for i, task_data in enumerate(sample_tasks, 1):
        print(f"\n{'-'*40}")
        print(f"Task {i}: {task_data['name']}")
        print(f"Model: {task_data['model']}")
        print(f"Complexity: {task_data['complexity'].value}")
        print(f"{'-'*40}")

        # Create task description
        task = TaskDescription(
            task_type=task_data["task_type"],
            complexity=task_data["complexity"],
            required_inputs=[task_data["input"]],
            expected_output_format="structured",
            context_requirements=task_data["context"],
            token_budget=500 if task_data["complexity"] == TaskComplexity.SIMPLE else 1000
        )

        # Demonstrate different optimization strategies
        strategies = [
            OptimizationStrategy.MINIMAL_TOKENS,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.ADAPTIVE
        ]

        for strategy in strategies:
            print(f"\n  Strategy: {strategy.value}")

            start_time = time.time()

            try:
                # Optimize prompt
                optimized_prompt = await optimizer.optimize_prompt(
                    task=task,
                    context_items=task_data["context"],
                    model_name=task_data["model"],
                    optimization_strategy=strategy
                )

                optimization_time = (time.time() - start_time) * 1000

                # Display results
                prompt_length = len(optimized_prompt.optimized_prompt.split())
                print(f"    [SUCCESS] Optimization successful ({optimization_time:.1f}ms)")
                print(f"    [INFO] Prompt length: {prompt_length} tokens")
                print(f"    [STRATEGY] Strategy used: {optimized_prompt.optimization_strategy}")

                # Show preview of optimized prompt
                preview = optimized_prompt.optimized_prompt[:200] + "..." if len(optimized_prompt.optimized_prompt) > 200 else optimized_prompt.optimized_prompt
                print(f"    [PREVIEW] {preview}")

            except Exception as e:
                print(f"    [ERROR] Optimization failed: {e}")

        # Get recommendations for this task and model
        recommendations = optimizer.get_recommendations(task, task_data["model"])
        if recommendations:
            print(f"\n  [RECOMMENDATIONS]")
            for rec in recommendations[:3]:  # Show top 3
                print(f"    - {rec}")


async def demonstrate_template_evolution():
    """Demonstrate template evolution using genetic algorithms"""

    print("\n" + "="*60)
    print("TEMPLATE EVOLUTION DEMONSTRATION")
    print("="*60)

    # Create initial seed templates
    seed_templates = [
        PromptTemplate(
            id="seed_1",
            name="Basic Research Template",
            description="Basic template for research tasks",
            template_type="research",
            template_content="You are a research assistant. Research the following topic: {{user_input}}. Use the context: {{context}}. Provide a detailed response.",
            variables=[
                {"name": "user_input", "type": "string", "required": True, "description": "Topic to research"},
                {"name": "context", "type": "string", "required": False, "description": "Background context"}
            ]
        ),
        PromptTemplate(
            id="seed_2",
            name="Simple Analysis Template",
            description="Simple template for analysis tasks",
            template_type="analysis",
            template_content="Analyze this input: {{user_input}}. Consider the provided context and give insights.",
            variables=[
                {"name": "user_input", "type": "string", "required": True, "description": "Input to analyze"},
                {"name": "context", "type": "string", "required": False, "description": "Background context"}
            ]
        )
    ]

    # Configure evolution
    evolution_config = EvolutionConfig(
        population_size=10,  # Smaller for demo
        generations=5,       # Fewer generations for demo
        mutation_rate=0.2,
        crossover_rate=0.8
    )

    print(f"\nStarting template evolution with {len(seed_templates)} seed templates")
    print(f"Population size: {evolution_config.population_size}")
    print(f"Generations: {evolution_config.generations}")
    print(f"Mutation rate: {evolution_config.mutation_rate}")

    # Create evolution system
    evolution = TemplateEvolution(evolution_config)

    # Define fitness function (mock for demo)
    async def fitness_function(genome: TemplateGenome) -> float:
        """Mock fitness function for demonstration"""
        # In real implementation, this would test the template and score performance
        base_fitness = 0.5

        # Reward appropriate length
        content = ' '.join(genome.content_segments)
        length = len(content.split())
        if 50 <= length <= 200:  # Optimal range for tiny models
            base_fitness += 0.3

        # Reward structure
        if any(keyword in content.lower() for keyword in ['task', 'input', 'format']):
            base_fitness += 0.1

        # Reward clarity
        if 'example' in content.lower():
            base_fitness += 0.1

        return min(1.0, base_fitness + (genome.generation * 0.02))  # Improve over generations

    print("\nRunning evolution...")
    start_time = time.time()

    try:
        # Run evolution
        evolved_templates = await evolution.evolve_templates(
            initial_templates=seed_templates,
            fitness_function=fitness_function
        )

        evolution_time = (time.time() - start_time) * 1000

        print(f"\n✅ Evolution completed in {evolution_time:.1f}ms")
        print(f"📊 Generated {len(evolved_templates)} optimized templates")

        # Show best template
        best_template = evolution.get_best_template()
        if best_template:
            print(f"\n🏆 Best Evolved Template:")
            print(f"   ID: {best_template.id}")
            print(f"   Generation: {best_template.metadata.get('generation', 'Unknown')}")
            print(f"   Fitness: {best_template.metadata.get('fitness_score', 0):.3f}")
            print(f"   Length: {len(best_template.template_content.split())} tokens")
            print(f"   Content: {best_template.template_content[:200]}...")

        # Show evolution statistics
        evolution_report = evolution.get_evolution_report()
        print(f"\n📈 Evolution Statistics:")
        print(f"   Generations: {evolution_report['generations_completed']}")
        print(f"   Best Fitness: {evolution_report['best_fitness_achieved']:.3f}")
        print(f"   Improvement: {evolution_report['improvement_from_baseline']:.3f}")
        print(f"   Total Mutations: {evolution_report['total_mutations_applied']}")

    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")


async def demonstrate_model_profiling():
    """Demonstrate tiny model capability profiling"""

    print("\n" + "="*60)
    print("MODEL CAPABILITY PROFILING DEMONSTRATION")
    print("="*60)

    profiler = TinyModelCapabilityProfiler()

    # Get profiles for known models
    models_to_profile = ["granite-tiny", "llama-3.2-1b", "phi-3-mini"]

    print(f"\nProfiling {len(models_to_profile)} tiny models...")

    for model_name in models_to_profile:
        capabilities = profiler.get_model_capabilities(model_name)

        if capabilities:
            print(f"\n🤖 Model: {capabilities.model_name}")
            print(f"   Context Window: {capabilities.context_window} tokens")
            print(f"   Max Reasoning Depth: {capabilities.max_reasoning_depth}")
            print(f"   Preferred Structure: {capabilities.preferred_structure}")
            print(f"   Token Efficiency: {capabilities.token_efficiency:.1f} tokens/task")
            print(f"   Strengths: {', '.join(capabilities.known_strengths[:3])}")
            print(f"   Limitations: {', '.join(capabilities.known_limitations[:3])}")
            print(f"   Optimal Tasks: {', '.join(capabilities.optimal_task_types[:3])}")
        else:
            print(f"\n❌ No profile found for model: {model_name}")


async def demonstrate_performance_analysis():
    """Demonstrate performance analysis capabilities"""

    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS DEMONSTRATION")
    print("="*60)

    from src.processing.llm_prompts.tiny_llm_optimizer import PromptPerformanceAnalyzer
    from src.processing.llm_prompts.models import LLMResponse

    analyzer = PromptPerformanceAnalyzer()

    # Simulate performance data for a template
    template_id = "demo_template"

    print(f"\nSimulating performance analysis for template: {template_id}")

    # Simulate multiple responses
    test_responses = [
        {"success": True, "time": 1200, "tokens": 250, "quality": 0.8},
        {"success": True, "time": 1500, "tokens": 300, "quality": 0.7},
        {"success": True, "time": 1100, "tokens": 200, "quality": 0.9},
        {"success": False, "time": 3000, "tokens": 400, "quality": 0.3},
        {"success": True, "time": 1300, "tokens": 280, "quality": 0.8}
    ]

    performance_scores = []

    for i, response_data in enumerate(test_responses):
        # Create mock response
        mock_response = LLMResponse(
            content=f"Response {i+1}",
            model="granite-tiny",
            token_usage={
                "total_tokens": response_data["tokens"],
                "prompt_tokens": response_data["tokens"] // 2,
                "completion_tokens": response_data["tokens"] // 2
            },
            response_time_ms=response_data["time"]
        )

        # Create mock template
        mock_template = PromptTemplate(
            id=template_id,
            name="Demo Template",
            description="Template for performance demo",
            template_type="general",
            template_content="Demo content",
            variables=[]
        )

        # Analyze performance
        metrics = await analyzer.analyze_performance(
            prompt=mock_template,
            response=mock_response,
            task_success=response_data["success"],
            quality_metrics={"relevance": response_data["quality"]}
        )

        if metrics:
            performance_scores.append(metrics.performance_score)
            print(f"  Response {i+1}: {'✅' if response_data['success'] else '❌'} "
                  f"Time: {response_data['time']}ms, Tokens: {response_data['tokens']}, "
                  f"Score: {metrics.performance_score:.3f}")

    # Get performance trends
    trends = analyzer.get_performance_trends(template_id)
    if trends:
        print(f"\n📊 Performance Trends:")
        print(f"   Success Rate: {trends['recent_success_rate']:.1%}")
        print(f"   Average Score: {trends['recent_avg_score']:.3f}")
        print(f"   Total Uses: {trends['total_uses']}")
        print(f"   Trend: {trends['trend_direction']}")


async def run_complete_demo():
    """Run complete demonstration of prompt engineering optimizer"""

    print("\n" + "="*60)
    print("TINY LLM PROMPT ENGINEERING OPTIMIZER")
    print("Complete Demonstration")
    print("="*60)

    try:
        # 1. Demonstrate prompt optimization
        await demonstrate_prompt_optimization()

        # 2. Demonstrate template evolution
        await demonstrate_template_evolution()

        # 3. Demonstrate model profiling
        await demonstrate_model_profiling()

        # 4. Demonstrate performance analysis
        await demonstrate_performance_analysis()

        # 5. Run comprehensive tests
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUITE")
        print("="*60)

        test_results = await run_tiny_llm_optimizer_tests()

        # Final summary
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)

        overall_metrics = test_results.get("overall_metrics", {})
        success_rate = overall_metrics.get("success_rate", 0)

        if success_rate >= 0.8:
            print(f"🎉 EXCELLENT: System performing at {success_rate:.1%} success rate")
        elif success_rate >= 0.6:
            print(f"✅ GOOD: System performing at {success_rate:.1%} success rate")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: System at {success_rate:.1%} success rate")

        print(f"\nKey Achievements:")
        print(f"  • Advanced prompt optimization for tiny LLMs")
        print(f"  • Genetic algorithm-based template evolution")
        print(f"  • Model-specific capability profiling")
        print(f"  • Real-time performance analysis")
        print(f"  • Comprehensive testing framework")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Tiny LLM Prompt Engineering Optimizer Demo...")

    # Run the complete demonstration
    asyncio.run(run_complete_demo())