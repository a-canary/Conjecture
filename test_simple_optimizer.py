#!/usr/bin/env python3
"""
Simple Test for Tiny LLM Prompt Engineering Optimizer
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from processing.llm_prompts.tiny_llm_optimizer import (
    TinyLLMPromptOptimizer,
    AdaptiveTemplateGenerator,
    TinyModelCapabilityProfiler,
    TaskDescription,
    TaskComplexity,
    OptimizationStrategy
)


async def test_basic_functionality():
    """Test basic functionality of prompt optimizer"""

    print("="*50)
    print("TINY LLM PROMPT OPTIMIZER TEST")
    print("="*50)

    try:
        # 1. Test model profiler
        print("\n1. Testing Model Profiler...")
        profiler = TinyModelCapabilityProfiler()
        capabilities = profiler.get_model_capabilities("granite-tiny")

        if capabilities:
            print(f"   [OK] Found profile for {capabilities.model_name}")
            print(f"   Context window: {capabilities.context_window}")
            print(f"   Preferred structure: {capabilities.preferred_structure}")
        else:
            print("   [ERROR] No profile found for granite-tiny")

        # 2. Test template generator
        print("\n2. Testing Template Generator...")
        generator = AdaptiveTemplateGenerator()

        task = TaskDescription(
            task_type="extraction",
            complexity=TaskComplexity.SIMPLE,
            required_inputs=["Extract the date from this text."],
            expected_output_format="structured"
        )

        model_caps = profiler.get_model_capabilities("granite-tiny") or profiler.analyze_model_characteristics("granite-tiny", [])

        template = await generator.generate_template(
            task=task,
            model_capabilities=model_caps,
            optimization_strategy=OptimizationStrategy.MINIMAL_TOKENS
        )

        print(f"   [OK] Generated template: {template.name}")
        print(f"   Template length: {len(template.template_content.split())} tokens")
        print(f"   Variables: {len(template.variables)}")

        # 3. Test optimizer
        print("\n3. Testing Prompt Optimizer...")
        optimizer = TinyLLMPromptOptimizer()

        optimized_prompt = await optimizer.optimize_prompt(
            task=task,
            context_items=["Some context information"],
            model_name="granite-tiny",
            optimization_strategy=OptimizationStrategy.MINIMAL_TOKENS
        )

        print(f"   [OK] Optimization completed")
        print(f"   Strategy: {optimized_prompt.optimization_strategy}")
        print(f"   Prompt length: {len(optimized_prompt.optimized_prompt.split())} tokens")

        # 4. Test recommendations
        print("\n4. Testing Recommendations...")
        recommendations = optimizer.get_recommendations(task, "granite-tiny")
        print(f"   [OK] Generated {len(recommendations)} recommendations")
        for rec in recommendations:
            print(f"   - {rec}")

        # 5. Test statistics
        print("\n5. Testing Statistics...")
        stats = optimizer.get_optimization_stats()
        print(f"   [OK] Optimizations performed: {stats['optimizations_performed']}")

        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*50)

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_different_tasks():
    """Test optimizer with different task types"""

    print("\n" + "="*50)
    print("TESTING DIFFERENT TASK TYPES")
    print("="*50)

    optimizer = TinyLLMPromptOptimizer()
    profiler = TinyModelCapabilityProfiler()

    test_tasks = [
        {
            "name": "Simple Extraction",
            "type": "extraction",
            "complexity": TaskComplexity.SIMPLE,
            "input": "The meeting is on December 15, 2024.",
            "model": "granite-tiny"
        },
        {
            "name": "Moderate Analysis",
            "type": "analysis",
            "complexity": TaskComplexity.MODERATE,
            "input": "Sales increased by 15% while costs decreased by 8%.",
            "model": "llama-3.2-1b"
        },
        {
            "name": "Complex Research",
            "type": "research",
            "complexity": TaskComplexity.COMPLEX,
            "input": "Analyze the impact of AI on creative industries.",
            "model": "phi-3-mini"
        }
    ]

    for i, test_task in enumerate(test_tasks, 1):
        print(f"\n{i}. Testing {test_task['name']}...")

        try:
            task = TaskDescription(
                task_type=test_task["type"],
                complexity=test_task["complexity"],
                required_inputs=[test_task["input"]],
                expected_output_format="structured",
                token_budget=500 if test_task["complexity"] == TaskComplexity.SIMPLE else 1000
            )

            # Test different strategies
            for strategy in [OptimizationStrategy.MINIMAL_TOKENS, OptimizationStrategy.BALANCED]:
                optimized_prompt = await optimizer.optimize_prompt(
                    task=task,
                    context_items=["Test context"],
                    model_name=test_task["model"],
                    optimization_strategy=strategy
                )

                prompt_length = len(optimized_prompt.optimized_prompt.split())
                print(f"   {strategy.value}: {prompt_length} tokens")

            print(f"   [OK] {test_task['name']} completed")

        except Exception as e:
            print(f"   [ERROR] {test_task['name']} failed: {e}")


if __name__ == "__main__":
    print("Starting simple prompt optimizer test...")

    async def run_tests():
        success = await test_basic_functionality()

        if success:
            await test_with_different_tasks()

        print("\nTest suite completed.")

    asyncio.run(run_tests())