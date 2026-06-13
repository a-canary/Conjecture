#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Domain-Specific Reasoning Enhancement Module

Integrates domain claim templates with the prompt system for optimized
performance on DROP (mathematical), ARC (scientific), and BBH (logical) benchmarks.

Key enhancement: Position primacy - templates placed at prompt START for +10pp effect.
Reference: NEXT.md findings from 2026-03-01 R&D session.
"""

from typing import Optional, Dict, Any
from enum import Enum

from src.agent.domain_claim_templates import (
    format_claims_for_prompt,
    get_benchmark_templates,
    get_domain_selector,
)


class BenchmarkType(Enum):
    """Benchmark types supported by domain reasoning enhancement"""

    DROP = "DROP"  # Mathematical reasoning
    ARC = "ARC"    # Scientific reasoning
    BBH = "BBH"    # Logical reasoning
    GENERAL = "General"


class DomainReasoningEnhancer:
    """Enhances prompts with domain-specific reasoning templates

    Implements position primacy optimization:
    - Templates placed at prompt START (+10pp improvement)
    - Selective template inclusion (confidence threshold 0.5 per findings)
    - Benchmark-specific claim selection
    """

    def __init__(self):
        self.selector = get_domain_selector()
        self.benchmark_domain_mapping = {
            BenchmarkType.DROP: "mathematical",
            BenchmarkType.ARC: "scientific",
            BenchmarkType.BBH: "logical",
            BenchmarkType.GENERAL: "general",
        }

    def enhance_prompt_with_domain_claims(
        self,
        prompt: str,
        domain: str,
        max_claims: int = 3,
        position: str = "start"
    ) -> str:
        """Enhance prompt with domain-specific claim templates

        Args:
            prompt: Original prompt to enhance
            domain: Domain key (mathematical, scientific, logical, general)
            max_claims: Maximum number of claim templates to include
            position: Where to place templates (start, middle, end) - start is optimal

        Returns:
            Enhanced prompt with domain templates injected
        """
        claim_text = format_claims_for_prompt(domain, max_claims)

        if not claim_text:
            return prompt

        if position == "start":
            # Position primacy: templates at START (+10pp per NEXT.md)
            return claim_text + "\n\n" + prompt
        elif position == "middle":
            # Less effective but still helps
            mid = len(prompt) // 2
            return prompt[:mid] + "\n\n" + claim_text + "\n\n" + prompt[mid:]
        else:  # end
            # Least effective
            return prompt + "\n\n" + claim_text

    def enhance_for_benchmark(
        self,
        prompt: str,
        benchmark: BenchmarkType,
        max_claims: int = 3
    ) -> str:
        """Enhance prompt optimized for a specific benchmark

        Args:
            prompt: Original prompt
            benchmark: Benchmark type (DROP, ARC, BBH, or GENERAL)
            max_claims: Maximum templates to include

        Returns:
            Benchmark-optimized enhanced prompt
        """
        domain = self.benchmark_domain_mapping.get(
            benchmark,
            self.benchmark_domain_mapping[BenchmarkType.GENERAL]
        )
        return self.enhance_prompt_with_domain_claims(
            prompt,
            domain,
            max_claims,
            position="start"  # Always use optimal position
        )

    def get_benchmark_context(self, benchmark: BenchmarkType) -> Dict[str, Any]:
        """Get context information for a specific benchmark

        Returns dict with:
        - domain: Domain name
        - description: Benchmark description
        - key_templates: Top claim templates
        """
        domain = self.benchmark_domain_mapping.get(
            benchmark,
            self.benchmark_domain_mapping[BenchmarkType.GENERAL]
        )

        templates = self.selector.get_templates_for_domain(domain, max_count=5)
        domain_data = self.selector.templates.get(domain, {})

        return {
            "benchmark": benchmark.value,
            "domain": domain,
            "domain_name": domain_data.get("domain_name", domain.capitalize()),
            "description": domain_data.get("description", ""),
            "key_templates": [t.to_dict() for t in templates],
            "template_count": len(templates)
        }

    def prepare_mathematical_reasoning_prompt(
        self,
        base_prompt: str,
        include_templates: bool = True
    ) -> str:
        """Prepare prompt optimized for DROP mathematical reasoning

        Args:
            base_prompt: Base system prompt
            include_templates: Whether to inject domain templates

        Returns:
            DROP-optimized prompt
        """
        if not include_templates:
            return base_prompt

        return self.enhance_for_benchmark(base_prompt, BenchmarkType.DROP, max_claims=3)

    def prepare_scientific_reasoning_prompt(
        self,
        base_prompt: str,
        include_templates: bool = True
    ) -> str:
        """Prepare prompt optimized for ARC scientific reasoning

        Args:
            base_prompt: Base system prompt
            include_templates: Whether to inject domain templates

        Returns:
            ARC-optimized prompt
        """
        if not include_templates:
            return base_prompt

        return self.enhance_for_benchmark(base_prompt, BenchmarkType.ARC, max_claims=3)

    def prepare_logical_reasoning_prompt(
        self,
        base_prompt: str,
        include_templates: bool = True
    ) -> str:
        """Prepare prompt optimized for BBH logical reasoning

        Args:
            base_prompt: Base system prompt
            include_templates: Whether to inject domain templates

        Returns:
            BBH-optimized prompt
        """
        if not include_templates:
            return base_prompt

        return self.enhance_for_benchmark(base_prompt, BenchmarkType.BBH, max_claims=3)


# Global instance for convenience
_global_enhancer = None


def get_enhancer() -> DomainReasoningEnhancer:
    """Get or create global domain reasoning enhancer instance"""
    global _global_enhancer
    if _global_enhancer is None:
        _global_enhancer = DomainReasoningEnhancer()
    return _global_enhancer


def enhance_for_benchmark(
    prompt: str,
    benchmark: str,
    max_claims: int = 3
) -> str:
    """Convenience function to enhance prompt for a specific benchmark

    Args:
        prompt: Original prompt
        benchmark: Benchmark name (DROP, ARC, BBH, or General)
        max_claims: Maximum templates to include

    Returns:
        Enhanced prompt
    """
    benchmark_type = BenchmarkType[benchmark.upper()] if benchmark.upper() in [
        b.name for b in BenchmarkType
    ] else BenchmarkType.GENERAL

    return get_enhancer().enhance_for_benchmark(prompt, benchmark_type, max_claims)


def get_benchmark_context(benchmark: str) -> Dict[str, Any]:
    """Convenience function to get benchmark context information"""
    benchmark_type = BenchmarkType[benchmark.upper()] if benchmark.upper() in [
        b.name for b in BenchmarkType
    ] else BenchmarkType.GENERAL

    return get_enhancer().get_benchmark_context(benchmark_type)
