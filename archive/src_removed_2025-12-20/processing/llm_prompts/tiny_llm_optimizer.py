"""
Tiny LLM Prompt Engineering Optimizer

Advanced prompt engineering optimization system specifically designed for
enhancing tiny LLM performance on complex tasks.

This module provides adaptive template generation, dynamic prompt optimization,
and performance-based tuning for small language models.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import random
from collections import defaultdict, deque

from .models import (
    PromptTemplate, PromptTemplateType, PromptMetrics,
    OptimizedPrompt, LLMResponse, TokenUsage
)
from ..json_frontmatter_parser import JSONFrontmatterParser

class OptimizationStrategy(str, Enum):
    """Prompt optimization strategies for tiny LLMs"""
    MINIMAL_TOKENS = "minimal_tokens"
    BALANCED = "balanced"
    MAX_STRUCTURE = "max_structure"
    ADAPTIVE = "adaptive"
    CHUNKED = "chunked"

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
    preferred_structure: str  # xml, json, plain
    token_efficiency: float  # tokens per successful task
    known_strengths: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    optimal_task_types: List[str] = field(default_factory=list)

@dataclass
class TaskDescription:
    """Task description for optimization"""
    task_type: str
    complexity: TaskComplexity
    required_inputs: List[str]
    expected_output_format: str
    token_budget: Optional[int] = None
    context_requirements: List[str] = field(default_factory=list)
    performance_priority: str = "balanced"  # speed, quality, efficiency

@dataclass
class OptimizationMetrics:
    """Prompt optimization performance metrics"""
    original_tokens: int
    optimized_tokens: int
    token_reduction: float
    performance_score: float
    optimization_time_ms: int
    strategy_used: OptimizationStrategy
    model_specific_gains: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceHistory:
    """Performance history for adaptive optimization"""
    task_type: str
    model_name: str
    prompt_templates: List[str]
    success_rates: List[float]
    token_usage: List[int]
    response_times: List[int]
    timestamps: List[datetime]
    quality_scores: List[float]

class TinyModelCapabilityProfiler:
    """Profiles tiny model capabilities and limitations"""

    def __init__(self):
        """Initialize capability profiler"""
        self.model_profiles = {
            "granite-tiny": TinyModelCapabilities(
                model_name="granite-tiny",
                context_window=2048,
                max_reasoning_depth=3,
                preferred_structure="xml",
                token_efficiency=150.0,
                known_strengths=["pattern_matching", "structured_data", "xml_parsing"],
                known_limitations=["complex_reasoning", "long_context", "abstract_thinking"],
                optimal_task_types=["classification", "extraction", "structured_generation"]
            ),
            "llama-3.2-1b": TinyModelCapabilities(
                model_name="llama-3.2-1b",
                context_window=4096,
                max_reasoning_depth=4,
                preferred_structure="json",
                token_efficiency=120.0,
                known_strengths=["instruction_following", "code_generation", "analysis"],
                known_limitations=["multi_step_reasoning", "creative_tasks"],
                optimal_task_types=["analysis", "coding", "structured_output"]
            ),
            "phi-3-mini": TinyModelCapabilities(
                model_name="phi-3-mini",
                context_window=4096,
                max_reasoning_depth=4,
                preferred_structure="plain",
                token_efficiency=100.0,
                known_strengths=["conversation", "reasoning", "adaptability"],
                known_limitations=["very_structured_output", "complex_xml"],
                optimal_task_types=["reasoning", "conversation", "adaptive_tasks"]
            )
        }

    def get_model_capabilities(self, model_name: str) -> Optional[TinyModelCapabilities]:
        """Get capabilities for a specific model"""
        return self.model_profiles.get(model_name)

    def analyze_model_characteristics(
        self,
        model_name: str,
        sample_responses: List[LLMResponse]
    ) -> TinyModelCapabilities:
        """Analyze model characteristics from sample responses"""
        if model_name in self.model_profiles:
            return self.model_profiles[model_name]

        # Default profile for unknown models
        return TinyModelCapabilities(
            model_name=model_name,
            context_window=2048,
            max_reasoning_depth=3,
            preferred_structure="plain",
            token_efficiency=130.0,
            known_strengths=["basic_reasoning"],
            known_limitations=["complex_tasks"],
            optimal_task_types=["simple_tasks"]
        )

class AdaptiveTemplateGenerator:
    """Generates adaptive templates for tiny models"""

    def __init__(self):
        """Initialize adaptive template generator"""
        self.template_patterns = {
            "research": self._get_research_patterns(),
            "analysis": self._get_analysis_patterns(),
            "coding": self._get_coding_patterns(),
            "extraction": self._get_extraction_patterns(),
            "classification": self._get_classification_patterns()
        }

        self.structure_templates = {
            "minimal": self._get_minimal_structure(),
            "balanced": self._get_balanced_structure(),
            "detailed": self._get_detailed_structure()
        }

    async def generate_template(
        self,
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    ) -> PromptTemplate:
        """Generate task-specific template for tiny model"""

        # Select appropriate pattern based on task type and model
        pattern = self._select_pattern(
            task.task_type,
            task.complexity,
            model_capabilities,
            optimization_strategy
        )

        # Generate template content
        template_content = self._build_template_content(
            pattern,
            task,
            model_capabilities,
            optimization_strategy
        )

        # Create template with appropriate variables
        template = PromptTemplate(
            id=f"tiny_optimized_{task.task_type}_{int(time.time())}",
            name=f"Tiny Optimized {task.task_type.title()} Template",
            description=f"Optimized template for {task.task_type} on {model_capabilities.model_name}",
            template_type=self._map_task_to_template_type(task.task_type),
            template_content=template_content,
            variables=self._extract_template_variables(template_content),
            metadata={
                "optimization_strategy": optimization_strategy.value,
                "target_model": model_capabilities.model_name,
                "task_complexity": task.complexity.value,
                "token_budget": task.token_budget
            }
        )

        return template

    def _select_pattern(
        self,
        task_type: str,
        complexity: TaskComplexity,
        model_capabilities: TinyModelCapabilities,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Select appropriate template pattern"""

        patterns = self.template_patterns.get(task_type, self.template_patterns["research"])

        # Adjust pattern based on strategy
        if strategy == OptimizationStrategy.MINIMAL_TOKENS:
            return patterns["minimal"]
        elif strategy == OptimizationStrategy.MAX_STRUCTURE:
            return patterns["structured"]
        elif strategy == OptimizationStrategy.ADAPTIVE:
            # Choose based on model strengths and task complexity
            if complexity == TaskComplexity.SIMPLE:
                return patterns["minimal"]
            elif model_capabilities.preferred_structure == "xml":
                return patterns["structured"]
            else:
                return patterns["balanced"]
        else:
            return patterns["balanced"]

    def _build_template_content(
        self,
        pattern: Dict[str, Any],
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities,
        strategy: OptimizationStrategy
    ) -> str:
        """Build template content from pattern"""

        structure = self.structure_templates[pattern["structure"]]

        # Build content sections
        sections = []

        # Role definition (adapted for tiny models)
        role = self._create_tiny_model_role(task, model_capabilities)
        sections.append(role)

        # Task description (simplified for tiny models)
        task_desc = self._create_tiny_task_description(task, model_capabilities)
        sections.append(task_desc)

        # Structure template
        if structure.get("include_examples", True) and task.complexity != TaskComplexity.SIMPLE:
            examples = self._create_mini_examples(task, model_capabilities)
            sections.append(examples)

        # Output format (tailored to model preferences)
        output_format = self._create_tiny_output_format(task, model_capabilities)
        sections.append(output_format)

        # Combine sections
        content = "\n\n".join(sections)

        # Apply token optimization
        if strategy == OptimizationStrategy.MINIMAL_TOKENS:
            content = self._minimize_tokens(content)

        return content

    def _create_tiny_model_role(self, task: TaskDescription, capabilities: TinyModelCapabilities) -> str:
        """Create role description optimized for tiny models"""

        base_roles = {
            "research": "You are a research assistant that finds and analyzes information.",
            "analysis": "You are an analyst that examines data and provides insights.",
            "coding": "You are a programmer that writes clear, efficient code.",
            "extraction": "You extract specific information from text accurately.",
            "classification": "You categorize items based on given criteria."
        }

        role = base_roles.get(task.task_type, "You are a helpful assistant.")

        # Add model-specific instructions
        if capabilities.preferred_structure == "xml":
            role += " Use the provided XML structure for your response."
        elif capabilities.preferred_structure == "json":
            role += " Format your response as valid JSON."

        return role

    def _create_tiny_task_description(self, task: TaskDescription, capabilities: TinyModelCapabilities) -> str:
        """Create simplified task description for tiny models"""

        # Simplify task description for tiny models
        if task.complexity == TaskComplexity.SIMPLE:
            return f"TASK: {task.task_type.replace('_', ' ').title()}\nINPUT: {{user_input}}"
        else:
            sections = [
                "TASK:",
                f"Type: {task.task_type.replace('_', ' ').title()}",
                f"Input: {{user_input}}",
                "Context: {{context}}"
            ]
            return "\n".join(sections)

    def _create_mini_examples(self, task: TaskDescription, capabilities: TinyModelCapabilities) -> str:
        """Create minimal examples for tiny models"""

        # Provide 1-2 very concise examples
        examples = {
            "research": "Example: Input: 'What causes rain?' Output: Rain occurs when water vapor in clouds condenses into droplets heavy enough to fall.",
            "analysis": "Example: Input: Data shows sales increased. Output: Sales show upward trend indicating positive market response.",
            "coding": "Example: Input: 'Add two numbers' Output: def add(a, b): return a + b"
        }

        example = examples.get(task.task_type, "Follow the format carefully.")
        return f"EXAMPLE:\n{example}"

    def _create_tiny_output_format(self, task: TaskDescription, capabilities: TinyModelCapabilities) -> str:
        """Create output format optimized for tiny models"""

        if capabilities.preferred_structure == "xml":
            return """RESPONSE FORMAT:
<result>
  <answer>{{answer}}</answer>
  <confidence>{{confidence}}</confidence>
</result>"""
        elif capabilities.preferred_structure == "json":
            return """RESPONSE FORMAT:
{
  "answer": "{{answer}}",
  "confidence": {{confidence}}
}"""
        else:
            return "Provide your answer followed by a confidence score (0-1)."

    def _minimize_tokens(self, content: str) -> str:
        """Minimize token usage while preserving clarity"""

        # Remove redundant words
        minimizers = [
            (r'\bplease\b', ''),
            (r'\bkindly\b', ''),
            (r'\byou are\b', "You're"),
            (r'\byou have\b', "You've"),
            (r'\bwe need to\b', "Need to"),
            (r'\bit is important to\b', "Important to"),
        ]

        for pattern, replacement in minimizers:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        return content.strip()

    def _extract_template_variables(self, template_content: str) -> List[Any]:
        """Extract variables from template content"""
        # This would integrate with existing variable extraction logic
        from .models import PromptVariable

        # Find {{variable}} patterns
        variables = re.findall(r'\{\{(\w+)\}\}', template_content)

        prompt_variables = []
        for var in set(variables):  # Remove duplicates
            prompt_variables.append(PromptVariable(
                name=var,
                type="string",
                required=True,
                description=f"Variable: {var}"
            ))

        return prompt_variables

    def _map_task_to_template_type(self, task_type: str) -> PromptTemplateType:
        """Map task type to prompt template type"""
        mapping = {
            "research": PromptTemplateType.RESEARCH,
            "analysis": PromptTemplateType.ANALYSIS,
            "coding": PromptTemplateType.CODING,
            "extraction": PromptTemplateType.VALIDATION,
            "classification": PromptTemplateType.VALIDATION
        }
        return mapping.get(task_type, PromptTemplateType.GENERAL)

    def _get_research_patterns(self) -> Dict[str, Any]:
        """Get research task patterns"""
        return {
            "minimal": {
                "structure": "minimal",
                "components": ["role", "task", "output"]
            },
            "balanced": {
                "structure": "balanced",
                "components": ["role", "task", "context", "output"]
            },
            "structured": {
                "structure": "detailed",
                "components": ["role", "task", "context", "examples", "output"]
            }
        }

    def _get_analysis_patterns(self) -> Dict[str, Any]:
        """Get analysis task patterns"""
        return self._get_research_patterns()  # Similar structure

    def _get_coding_patterns(self) -> Dict[str, Any]:
        """Get coding task patterns"""
        return {
            "minimal": {
                "structure": "minimal",
                "components": ["role", "task", "examples", "output"]
            },
            "balanced": {
                "structure": "balanced",
                "components": ["role", "task", "requirements", "examples", "output"]
            },
            "structured": {
                "structure": "detailed",
                "components": ["role", "task", "requirements", "examples", "constraints", "output"]
            }
        }

    def _get_extraction_patterns(self) -> Dict[str, Any]:
        """Get extraction task patterns"""
        return self._get_research_patterns()  # Similar structure

    def _get_classification_patterns(self) -> Dict[str, Any]:
        """Get classification task patterns"""
        return {
            "minimal": {
                "structure": "minimal",
                "components": ["role", "task", "categories", "output"]
            },
            "balanced": {
                "structure": "balanced",
                "components": ["role", "task", "context", "categories", "examples", "output"]
            },
            "structured": {
                "structure": "detailed",
                "components": ["role", "task", "context", "categories", "examples", "criteria", "output"]
            }
        }

    def _get_minimal_structure(self) -> Dict[str, Any]:
        """Get minimal template structure"""
        return {
            "include_examples": False,
            "include_context": False,
            "include_constraints": False
        }

    def _get_balanced_structure(self) -> Dict[str, Any]:
        """Get balanced template structure"""
        return {
            "include_examples": True,
            "include_context": True,
            "include_constraints": False
        }

    def _get_detailed_structure(self) -> Dict[str, Any]:
        """Get detailed template structure"""
        return {
            "include_examples": True,
            "include_context": True,
            "include_constraints": True
        }

class PromptPerformanceAnalyzer:
    """Analyzes prompt performance and provides optimization feedback"""

    def __init__(self):
        """Initialize performance analyzer"""
        self.performance_history = defaultdict(list)
        self.baseline_metrics = {}

    async def analyze_performance(
        self,
        prompt: PromptTemplate,
        response: LLMResponse,
        task_success: bool,
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationMetrics:
        """Analyze prompt effectiveness and provide feedback"""

        # Calculate basic metrics
        prompt_tokens = prompt.template_content.split()  # Approximate token count
        response_tokens = response.content.split()

        # Calculate performance score
        performance_score = self._calculate_performance_score(
            task_success,
            response.response_time_ms,
            response.token_usage,
            quality_metrics
        )

        # Store performance data
        self._record_performance(
            prompt.id,
            task_success,
            performance_score,
            response.token_usage,
            response.response_time_ms
        )

        # Create optimization metrics
        metrics = OptimizationMetrics(
            original_tokens=len(prompt_tokens),
            optimized_tokens=len(prompt_tokens),  # Would be actual optimized count
            token_reduction=0.0,  # Would calculate actual reduction
            performance_score=performance_score,
            optimization_time_ms=0,  # Would track actual time
            strategy_used=OptimizationStrategy.ADAPTIVE
        )

        return metrics

    def _calculate_performance_score(
        self,
        success: bool,
        response_time: int,
        token_usage: Dict[str, int],
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate comprehensive performance score"""

        # Base success score
        success_score = 1.0 if success else 0.0

        # Response time score (lower is better)
        time_score = max(0, 1.0 - (response_time / 10000))  # Normalize to 10s max

        # Token efficiency score
        total_tokens = token_usage.get("total_tokens", 0)
        efficiency_score = max(0, 1.0 - (total_tokens / 1000))  # Normalize to 1000 tokens max

        # Quality metrics score
        quality_score = 0.0
        if quality_metrics:
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)

        # Weighted combination
        weights = {
            "success": 0.5,
            "time": 0.2,
            "efficiency": 0.2,
            "quality": 0.1
        }

        total_score = (
            success_score * weights["success"] +
            time_score * weights["time"] +
            efficiency_score * weights["efficiency"] +
            quality_score * weights["quality"]
        )

        return total_score

    def _record_performance(
        self,
        template_id: str,
        success: bool,
        performance_score: float,
        token_usage: Dict[str, int],
        response_time: int
    ) -> None:
        """Record performance data for future optimization"""

        performance_record = {
            "timestamp": datetime.utcnow(),
            "success": success,
            "performance_score": performance_score,
            "token_usage": token_usage,
            "response_time": response_time
        }

        self.performance_history[template_id].append(performance_record)

        # Keep only recent records (last 100)
        if len(self.performance_history[template_id]) > 100:
            self.performance_history[template_id] = self.performance_history[template_id][-100:]

    def get_performance_trends(self, template_id: str) -> Dict[str, Any]:
        """Get performance trends for a template"""

        records = self.performance_history.get(template_id, [])
        if not records:
            return {}

        # Calculate trends
        recent_records = records[-10:]  # Last 10 uses
        older_records = records[-20:-10] if len(records) >= 20 else []

        recent_success_rate = sum(r["success"] for r in recent_records) / len(recent_records)
        recent_avg_score = sum(r["performance_score"] for r in recent_records) / len(recent_records)

        trends = {
            "recent_success_rate": recent_success_rate,
            "recent_avg_score": recent_avg_score,
            "total_uses": len(records),
            "trend_direction": "improving" if len(recent_records) > 0 and len(older_records) > 0 else "stable"
        }

        if older_records:
            older_success_rate = sum(r["success"] for r in older_records) / len(older_records)
            older_avg_score = sum(r["performance_score"] for r in older_records) / len(older_records)

            trends["improvement"] = {
                "success_rate": recent_success_rate - older_success_rate,
                "performance_score": recent_avg_score - older_avg_score
            }

        return trends

class TinyLLMPromptOptimizer:
    """Main optimizer for tiny LLM prompt engineering"""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize tiny LLM prompt optimizer"""
        self.model_profiler = TinyModelCapabilityProfiler()
        self.template_generator = AdaptiveTemplateGenerator()
        self.performance_analyzer = PromptPerformanceAnalyzer()
        self.json_parser = JSONFrontmatterParser()

        # Optimization settings
        self.optimization_cache = {}
        self.performance_cache = {}

        # Statistics
        self.stats = {
            "optimizations_performed": 0,
            "total_time_saved_ms": 0,
            "tokens_saved": 0,
            "performance_improvements": []
        }

    async def optimize_prompt(
        self,
        task: TaskDescription,
        context_items: List[Any],
        model_name: str,
        performance_history: Optional[PerformanceHistory] = None,
        optimization_strategy: Optional[OptimizationStrategy] = None
    ) -> OptimizedPrompt:
        """Generate optimized prompt for tiny LLM"""

        start_time = time.time()

        # Get model capabilities
        model_capabilities = self.model_profiler.get_model_capabilities(model_name)
        if not model_capabilities:
            model_capabilities = self.model_profiler.analyze_model_characteristics(model_name, [])

        # Determine optimization strategy
        if optimization_strategy is None:
            optimization_strategy = self._determine_strategy(task, model_capabilities, performance_history)

        # Generate adaptive template
        template = await self.template_generator.generate_template(
            task,
            model_capabilities,
            optimization_strategy
        )

        # Prepare context
        optimized_context = self._optimize_context_for_tiny_model(
            context_items,
            task,
            model_capabilities
        )

        # Render template
        variables = {
            "user_input": task.required_inputs[0] if task.required_inputs else "",
            "context": optimized_context,
            **{f"input_{i}": inp for i, inp in enumerate(task.required_inputs)}
        }

        try:
            rendered_prompt = template.render(variables)
        except Exception as e:
            # Fallback rendering
            rendered_prompt = self._fallback_render(template, variables)

        # Calculate optimization metrics
        optimization_time = (time.time() - start_time) * 1000

        optimized_prompt = OptimizedPrompt(
            original_prompt=rendered_prompt,  # Would track original vs optimized
            optimized_prompt=rendered_prompt,
            optimization_strategy=optimization_strategy.value,
            token_reduction=0,  # Would calculate actual reduction
            optimization_score=0.0,  # Would calculate based on model
            changes_made=[f"Applied {optimization_strategy.value} strategy"]
        )

        # Update statistics
        self.stats["optimizations_performed"] += 1

        return optimized_prompt

    def _determine_strategy(
        self,
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities,
        performance_history: Optional[PerformanceHistory]
    ) -> OptimizationStrategy:
        """Determine optimal strategy based on task and model"""

        # Base strategy on task complexity
        if task.complexity == TaskComplexity.SIMPLE:
            return OptimizationStrategy.MINIMAL_TOKENS
        elif task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            return OptimizationStrategy.CHUNKED
        else:
            return OptimizationStrategy.BALANCED

    def _optimize_context_for_tiny_model(
        self,
        context_items: List[Any],
        task: TaskDescription,
        model_capabilities: TinyModelCapabilities
    ) -> str:
        """Optimize context for tiny model constraints"""

        if not context_items:
            return ""

        # Prioritize context based on task requirements
        prioritized_context = self._prioritize_context(context_items, task)

        # Limit context based on model capabilities
        max_context_items = min(len(prioritized_context), 3)  # Tiny models need less context
        limited_context = prioritized_context[:max_context_items]

        # Format context based on model preferences
        if model_capabilities.preferred_structure == "xml":
            context_str = self._format_context_xml(limited_context)
        elif model_capabilities.preferred_structure == "json":
            context_str = self._format_context_json(limited_context)
        else:
            context_str = self._format_context_plain(limited_context)

        return context_str

    def _prioritize_context(self, context_items: List[Any], task: TaskDescription) -> List[Any]:
        """Prioritize context items based on task requirements"""
        # Simple implementation - could be enhanced with relevance scoring
        return context_items[:5]  # Return top 5 items

    def _format_context_xml(self, context_items: List[Any]) -> str:
        """Format context as XML for tiny models"""
        xml_parts = ["<context>"]
        for i, item in enumerate(context_items[:3]):  # Limit for tiny models
            xml_parts.append(f"  <item_{i+1}>{str(item)[:200]}...</item_{i+1}>")
        xml_parts.append("</context>")
        return "\n".join(xml_parts)

    def _format_context_json(self, context_items: List[Any]) -> str:
        """Format context as JSON for tiny models"""
        context_data = {}
        for i, item in enumerate(context_items[:3]):  # Limit for tiny models
            context_data[f"item_{i+1}"] = str(item)[:200] + "..."
        return json.dumps(context_data, indent=2)

    def _format_context_plain(self, context_items: List[Any]) -> str:
        """Format context as plain text for tiny models"""
        context_parts = ["CONTEXT:"]
        for i, item in enumerate(context_items[:3]):  # Limit for tiny models
            context_parts.append(f"{i+1}. {str(item)[:200]}...")
        return "\n".join(context_parts)

    def _fallback_render(self, template: PromptTemplate, variables: Dict[str, Any]) -> str:
        """Fallback template rendering"""
        content = template.template_content
        for name, value in variables.items():
            content = content.replace(f"{{{{{name}}}}}", str(value))
        return content

    async def analyze_performance(
        self,
        prompt: OptimizedPrompt,
        response: LLMResponse,
        task_success: bool,
        quality_metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationMetrics:
        """Analyze prompt performance and provide optimization feedback"""

        # Create a temporary template for analysis
        temp_template = PromptTemplate(
            id="temp_analysis",
            name="Analysis Template",
            description="Temporary template for performance analysis",
            template_type=PromptTemplateType.GENERAL,
            template_content=prompt.optimized_prompt,
            variables=[]
        )

        metrics = await self.performance_analyzer.analyze_performance(
            temp_template,
            response,
            task_success,
            quality_metrics
        )

        # Update optimization score based on performance
        prompt.optimization_score = metrics.performance_score

        return metrics

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.stats.copy()

    def get_recommendations(
        self,
        task: TaskDescription,
        model_name: str
    ) -> List[str]:
        """Get optimization recommendations for task and model"""

        model_capabilities = self.model_profiler.get_model_capabilities(model_name)
        recommendations = []

        if model_capabilities:
            # Model-specific recommendations
            if "complex_reasoning" in model_capabilities.known_limitations:
                recommendations.append("Break down complex tasks into simpler steps")

            if model_capabilities.preferred_structure == "xml":
                recommendations.append("Use XML formatting for better model understanding")

            if task.complexity == TaskComplexity.VERY_COMPLEX:
                recommendations.append("Consider chunked processing for very complex tasks")
                recommendations.append("Reduce context to most essential information")

        # Task-specific recommendations
        if task.task_type == "research":
            recommendations.append("Focus on specific, answerable questions")
            recommendations.append("Provide clear evaluation criteria")

        return recommendations