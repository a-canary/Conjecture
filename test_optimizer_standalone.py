#!/usr/bin/env python3
"""
Standalone Test for Tiny LLM Prompt Engineering Optimizer Core Components
Tests the core functionality without complex dependencies
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class OptimizationStrategy(str, Enum):
    """Prompt optimization strategies for tiny LLMs"""
    MINIMAL_TOKENS = "minimal_tokens"
    BALANCED = "balanced"
    MAX_STRUCTURE = "max_structure"
    ADAPTIVE = "adaptive"


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class TinyModelCapabilities:
    """Tiny model capability profile"""
    model_name: str
    context_window: int
    max_reasoning_depth: int
    preferred_structure: str
    token_efficiency: float
    known_strengths: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)


@dataclass
class TaskDescription:
    """Task description for optimization"""
    task_type: str
    complexity: TaskComplexity
    required_inputs: List[str]
    expected_output_format: str
    token_budget: Optional[int] = None


@dataclass
class OptimizedPrompt:
    """Represents an optimized prompt"""
    original_prompt: str
    optimized_prompt: str
    optimization_strategy: str
    token_reduction: int
    optimization_score: float
    changes_made: List[str]


class TinyModelCapabilityProfiler:
    """Simplified model capability profiler"""

    def __init__(self):
        """Initialize capability profiler"""
        self.model_profiles = {
            "granite-tiny": TinyModelCapabilities(
                model_name="granite-tiny",
                context_window=2048,
                max_reasoning_depth=3,
                preferred_structure="xml",
                token_efficiency=150.0,
                known_strengths=["pattern_matching", "structured_data"],
                known_limitations=["complex_reasoning", "long_context"]
            ),
            "llama-3.2-1b": TinyModelCapabilities(
                model_name="llama-3.2-1b",
                context_window=4096,
                max_reasoning_depth=4,
                preferred_structure="json",
                token_efficiency=120.0,
                known_strengths=["instruction_following", "analysis"],
                known_limitations=["multi_step_reasoning"]
            ),
            "phi-3-mini": TinyModelCapabilities(
                model_name="phi-3-mini",
                context_window=4096,
                max_reasoning_depth=4,
                preferred_structure="plain",
                token_efficiency=100.0,
                known_strengths=["conversation", "reasoning"],
                known_limitations=["very_structured_output"]
            )
        }

    def get_model_capabilities(self, model_name: str) -> Optional[TinyModelCapabilities]:
        """Get capabilities for a specific model"""
        return self.model_profiles.get(model_name)


class SimplePromptOptimizer:
    """Simplified prompt optimizer for testing"""

    def __init__(self):
        """Initialize prompt optimizer"""
        self.profiler = TinyModelCapabilityProfiler()
        self.stats = {
            "optimizations_performed": 0,
            "total_tokens_saved": 0
        }

    async def optimize_prompt(
        self,
        task: TaskDescription,
        context_items: List[str],
        model_name: str,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    ) -> OptimizedPrompt:
        """Optimize prompt for tiny LLM"""

        start_time = time.time()

        # Get model capabilities
        model_capabilities = self.profiler.get_model_capabilities(model_name)
        if not model_capabilities:
            raise ValueError(f"Unknown model: {model_name}")

        # Generate base prompt
        base_prompt = self._generate_base_prompt(task, context_items)

        # Apply optimization strategy
        optimized_content = self._apply_optimization_strategy(
            base_prompt, task, model_capabilities, optimization_strategy
        )

        # Calculate metrics
        original_tokens = len(base_prompt.split())
        optimized_tokens = len(optimized_content.split())
        token_reduction = original_tokens - optimized_tokens
        optimization_score = self._calculate_optimization_score(
            optimized_content, task, model_capabilities
        )

        optimization_time = (time.time() - start_time) * 1000

        # Create result
        result = OptimizedPrompt(
            original_prompt=base_prompt,
            optimized_prompt=optimized_content,
            optimization_strategy=optimization_strategy.value,
            token_reduction=token_reduction,
            optimization_score=optimization_score,
            changes_made=[
                f"Applied {optimization_strategy.value} strategy",
                f"Optimized for {model_capabilities.model_name}",
                f"Reduced by {token_reduction} tokens"
            ]
        )

        # Update stats
        self.stats["optimizations_performed"] += 1
        self.stats["total_tokens_saved"] += token_reduction

        return result

    def _generate_base_prompt(self, task: TaskDescription, context_items: List[str]) -> str:
        """Generate base prompt from task description"""

        base_prompt = f"""
You are an AI assistant.

TASK: {task.task_type.replace('_', ' ').title()}

INPUT: {task.required_inputs[0] if task.required_inputs else 'No input provided'}

CONTEXT:
"""

        # Add context
        for i, context in enumerate(context_items[:3]):  # Limit context
            base_prompt += f"{i+1}. {context}\n"

        base_prompt += "\nPlease provide a response."

        return base_prompt.strip()

    def _apply_optimization_strategy(
        self,
        prompt: str,
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities,
        strategy: OptimizationStrategy
    ) -> str:
        """Apply optimization strategy to prompt"""

        optimized = prompt

        if strategy == OptimizationStrategy.MINIMAL_TOKENS:
            # Minimize tokens
            optimized = self._minimize_tokens(optimized)
        elif strategy == OptimizationStrategy.BALANCED:
            # Balanced approach
            optimized = self._balance_prompt(optimized, task)
        elif strategy == OptimizationStrategy.ADAPTIVE:
            # Adaptive based on model and task
            optimized = self._adaptive_optimization(optimized, task, model_capabilities)

        # Apply model-specific optimizations
        optimized = self._apply_model_specific_optimizations(optimized, model_capabilities)

        return optimized

    def _minimize_tokens(self, prompt: str) -> str:
        """Minimize token usage"""
        # Remove redundant words
        minimizers = [
            (r'\bplease\b', ''),
            (r'\bkindly\b', ''),
            (r'\byou are\b', "You're"),
            (r'\byou have\b', "You've"),
            (r'\bwe need to\b', "Need to"),
            (r'\bit is important to\b', "Important to"),
        ]

        import re
        for pattern, replacement in minimizers:
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)

        # Remove excessive whitespace
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)

        return prompt.strip()

    def _balance_prompt(self, prompt: str, task: TaskDescription) -> str:
        """Apply balanced optimization"""
        # Moderate token reduction while maintaining clarity
        if task.complexity == TaskComplexity.SIMPLE:
            return self._minimize_tokens(prompt)
        else:
            # Keep more detail for complex tasks
            return prompt[:int(len(prompt) * 0.8)]  # Reduce by 20%

    def _adaptive_optimization(
        self,
        prompt: str,
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities
    ) -> str:
        """Apply adaptive optimization based on model and task"""

        if task.complexity == TaskComplexity.SIMPLE:
            return self._minimize_tokens(prompt)
        elif model_capabilities.preferred_structure == "xml":
            # Add XML structure for models that prefer it
            return self._add_xml_structure(prompt, task)
        else:
            return self._balance_prompt(prompt, task)

    def _apply_model_specific_optimizations(
        self,
        prompt: str,
        model_capabilities: TinyModelCapabilities
    ) -> str:
        """Apply model-specific optimizations"""

        if model_capabilities.preferred_structure == "xml":
            if "<" not in prompt:
                prompt = self._add_xml_structure(prompt, None)

        elif model_capabilities.preferred_structure == "json":
            if "{" not in prompt:
                prompt = self._add_json_structure(prompt)

        # Apply token limits
        max_tokens = model_capabilities.context_window // 4  # Leave room for response
        current_tokens = len(prompt.split())

        if current_tokens > max_tokens:
            # Truncate to fit
            words = prompt.split()
            prompt = " ".join(words[:max_tokens])

        return prompt

    def _add_xml_structure(self, prompt: str, task: Optional[TaskDescription]) -> str:
        """Add XML structure to prompt"""
        return f"""<task>
  <instruction>{prompt}</instruction>
</task>

<output_format>
  <answer>Your answer here</answer>
  <confidence>0.0-1.0</confidence>
</output_format>"""

    def _add_json_structure(self, prompt: str) -> str:
        """Add JSON structure to prompt"""
        return f"""{prompt}

RESPONSE FORMAT:
{{
  "answer": "Your answer here",
  "confidence": 0.0
}}"""

    def _calculate_optimization_score(
        self,
        prompt: str,
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities
    ) -> float:
        """Calculate optimization score"""

        score = 0.0

        # Length score (optimal length gets higher score)
        word_count = len(prompt.split())
        optimal_length = 200 if task.complexity == TaskComplexity.SIMPLE else 400
        length_score = max(0, 1.0 - abs(word_count - optimal_length) / optimal_length)
        score += length_score * 0.4

        # Structure score
        if model_capabilities.preferred_structure == "xml" and "<" in prompt:
            score += 0.3
        elif model_capabilities.preferred_structure == "json" and "{" in prompt:
            score += 0.3
        elif model_capabilities.preferred_structure == "plain":
            score += 0.3

        # Clarity score
        if "TASK:" in prompt and "INPUT:" in prompt:
            score += 0.3

        return min(1.0, score)

    def get_recommendations(self, task: TaskDescription, model_name: str) -> List[str]:
        """Get optimization recommendations"""
        model_capabilities = self.profiler.get_model_capabilities(model_name)
        recommendations = []

        if model_capabilities:
            if "complex_reasoning" in model_capabilities.known_limitations:
                recommendations.append("Break down complex tasks into simpler steps")

            if task.complexity == TaskComplexity.VERY_COMPLEX:
                recommendations.append("Consider chunked processing for very complex tasks")

            if model_capabilities.preferred_structure == "xml":
                recommendations.append("Use XML formatting for better model understanding")

        return recommendations

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.stats.copy()


async def test_prompt_optimizer():
    """Test the prompt optimizer"""

    print("="*60)
    print("PROMPT ENGINEERING OPTIMIZER TEST")
    print("="*60)

    optimizer = SimplePromptOptimizer()

    # Test tasks
    test_tasks = [
        {
            "name": "Simple Extraction",
            "task": TaskDescription(
                task_type="extraction",
                complexity=TaskComplexity.SIMPLE,
                required_inputs=["Extract the date from: The meeting is on December 15, 2024."],
                expected_output_format="structured"
            ),
            "context": ["Meeting scheduling context", "Date formats"],
            "model": "granite-tiny"
        },
        {
            "name": "Moderate Analysis",
            "task": TaskDescription(
                task_type="analysis",
                complexity=TaskComplexity.MODERATE,
                required_inputs=["Sales increased by 15% while costs decreased by 8%."],
                expected_output_format="analysis"
            ),
            "context": ["Sales data", "Cost analysis", "Market trends"],
            "model": "llama-3.2-1b"
        },
        {
            "name": "Complex Research",
            "task": TaskDescription(
                task_type="research",
                complexity=TaskComplexity.COMPLEX,
                required_inputs=["Analyze the impact of AI on creative industries."],
                expected_output_format="comprehensive"
            ),
            "context": [
                "AI research papers",
                "Creative industry reports",
                "Economic impact studies",
                "Technology trends"
            ],
            "model": "phi-3-mini"
        }
    ]

    all_results = []

    for i, test_case in enumerate(test_tasks, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Model: {test_case['model']}")
        print(f"   Complexity: {test_case['task'].complexity.value}")

        # Test different strategies
        strategies = [OptimizationStrategy.MINIMAL_TOKENS, OptimizationStrategy.ADAPTIVE]

        for strategy in strategies:
            try:
                print(f"\n   Strategy: {strategy.value}")

                result = await optimizer.optimize_prompt(
                    task=test_case['task'],
                    context_items=test_case['context'],
                    model_name=test_case['model'],
                    optimization_strategy=strategy
                )

                original_length = len(result.original_prompt.split())
                optimized_length = len(result.optimized_prompt.split())

                print(f"     [SUCCESS] Optimization completed")
                print(f"     Original: {original_length} tokens")
                print(f"     Optimized: {optimized_length} tokens")
                print(f"     Reduction: {result.token_reduction} tokens")
                print(f"     Score: {result.optimization_score:.3f}")

                # Show preview
                preview = result.optimized_prompt[:150] + "..." if len(result.optimized_prompt) > 150 else result.optimized_prompt
                print(f"     Preview: {preview}")

                all_results.append({
                    "task": test_case['name'],
                    "model": test_case['model'],
                    "strategy": strategy.value,
                    "success": True,
                    "score": result.optimization_score,
                    "token_reduction": result.token_reduction
                })

            except Exception as e:
                print(f"     [ERROR] {e}")
                all_results.append({
                    "task": test_case['name'],
                    "model": test_case['model'],
                    "strategy": strategy.value,
                    "success": False,
                    "score": 0,
                    "error": str(e)
                })

        # Get recommendations
        recommendations = optimizer.get_recommendations(test_case['task'], test_case['model'])
        if recommendations:
            print(f"\n   Recommendations:")
            for rec in recommendations:
                print(f"     - {rec}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    successful_tests = [r for r in all_results if r['success']]
    total_tests = len(all_results)

    print(f"Total tests: {total_tests}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Success rate: {len(successful_tests)/total_tests:.1%}")

    if successful_tests:
        avg_score = sum(r['score'] for r in successful_tests) / len(successful_tests)
        total_tokens_saved = sum(r['token_reduction'] for r in successful_tests)
        print(f"Average score: {avg_score:.3f}")
        print(f"Total tokens saved: {total_tokens_saved}")

    # Get optimizer stats
    stats = optimizer.get_optimization_stats()
    print(f"\nOptimizer Statistics:")
    print(f"  Optimizations performed: {stats['optimizations_performed']}")
    print(f"  Total tokens saved: {stats['total_tokens_saved']}")

    # Test model profiler
    print(f"\nModel Profiler Test:")
    profiler = TinyModelCapabilityProfiler()
    for model_name in ["granite-tiny", "llama-3.2-1b", "phi-3-mini"]:
        capabilities = profiler.get_model_capabilities(model_name)
        if capabilities:
            print(f"  {model_name}: {capabilities.context_window} context, prefers {capabilities.preferred_structure}")

    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

    return len(successful_tests) == total_tests


if __name__ == "__main__":
    print("Starting standalone prompt optimizer test...")

    async def run_test():
        success = await test_prompt_optimizer()
        if success:
            print("\nAll tests passed!")
        else:
            print("\nSome tests failed.")

    asyncio.run(run_test())