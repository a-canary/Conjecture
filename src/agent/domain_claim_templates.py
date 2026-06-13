#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Domain-Specific Claim Templates for Enhanced Benchmark Performance

Provides domain-specific claim templates that prime the model for optimal
performance on mathematical (DROP), scientific (ARC), and logical (BBH) reasoning.

Each template is crafted based on domain-specific reasoning patterns and can be
injected into prompts at critical positions (per NEXT.md position primacy findings).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# Domain-Specific Claim Templates for priming model behavior
DOMAIN_CLAIM_TEMPLATES = {
    "mathematical": {
        "domain_name": "Mathematical Reasoning (DROP benchmark)",
        "description": "Templates for mathematical reasoning and word problem solving",
        "benchmark": "DROP",
        "templates": [
            "Arithmetic rule: Order of operations (PEMDAS/BODMAS) - Parentheses/Brackets, Exponents, Multiplication/Division left-to-right, Addition/Subtraction left-to-right",
            "Percentage calculation: To find X% of Y: (X/100) * Y. To find what % X is of Y: (X/Y) * 100",
            "Rate problems: Speed = Distance/Time, Distance = Speed*Time, Time = Distance/Speed",
            "Word problem strategy: Identify known values, identify what's being asked, set up equation, solve, verify units",
            "Estimation technique: Round numbers before calculating to check if final answer is reasonable",
            "Unit conversion: Always track units through calculations, convert to target units at appropriate step",
            "Algebraic principle: To solve equations, perform same operations on both sides to isolate variable",
            "Geometry basics: Rectangle area = length × width, Triangle area = (1/2) × base × height, Circle area = π × radius²",
            "Money and time problems: Keep track of starting values, changes, and final results. Watch for multi-step calculations",
            "Ratio problems: Ratio A:B means A/B relationship. Scale ratios proportionally by multiplying both by same factor"
        ]
    },
    "scientific": {
        "domain_name": "Scientific Reasoning (ARC benchmark)",
        "description": "Templates for scientific reasoning and experimental interpretation",
        "benchmark": "ARC",
        "templates": [
            "Scientific method: Observation → Hypothesis → Prediction → Experiment → Analysis → Conclusion",
            "Variables: Independent variable (what we change), Dependent variable (what we measure), Control variables (what we keep constant)",
            "Experimental design: Controls are essential - compare treatment vs no treatment to isolate cause",
            "Data interpretation: Look for patterns, trends, correlations. Distinguish correlation from causation",
            "Physical principles: Energy conservation - energy transforms but total remains constant in closed systems",
            "Chemical principle: Chemical reactions: Reactants → Products. Balance atoms on both sides",
            "Biological concept: Evolution via natural selection - organisms with beneficial traits survive and reproduce",
            "Measurement: Accuracy (closeness to true value) vs Precision (repeatability). Both matter in science",
            "Uncertainty: Larger samples reduce error. Outliers should be investigated for validity or removed if measurement error",
            "Evidence evaluation: Primary sources > secondary sources. Peer-reviewed > non-reviewed sources"
        ]
    },
    "logical": {
        "domain_name": "Logical Reasoning (BBH benchmark)",
        "description": "Templates for logical reasoning and puzzle solving",
        "benchmark": "BBH",
        "templates": [
            "Deductive reasoning: If premises are true and logic valid, conclusion must be true",
            "Premise-conclusion structure: Premises are facts/rules, conclusion follows necessarily from premises",
            "Logical operators: AND (both true), OR (at least one true), NOT (opposite), XOR (exactly one true)",
            "Quantifiers: All (every single one), Some (at least one), None (not a single one)",
            "Conditional logic: If P then Q means: P true → Q true, Q false → P false (contrapositive)",
            "Common logical fallacy: Affirming the consequent - 'If P then Q, Q is true, therefore P' is invalid",
            "Set theory: Subset (all elements in A are in B), Intersection (elements in both), Union (elements in either)",
            "Proof by contradiction: Assume statement false, derive contradiction, therefore statement must be true",
            "Necessary vs sufficient: Necessary (must have for something to occur), Sufficient (enough to make it occur)",
            "Analogical reasoning: If A:B :: C:?, find D where relationship mirrors A:B pattern"
        ]
    },
    "general": {
        "domain_name": "Cross-Domain Reasoning",
        "description": "General reasoning patterns applicable across domains",
        "benchmark": "General",
        "templates": [
            "Problem decomposition: Break complex problem into smaller parts, solve each, combine solutions",
            "Verification strategy: Check if answer satisfies original constraints and makes intuitive sense",
            "Alternative approach: If stuck, try different method - can validate answer or reveal errors",
            "Edge case analysis: Consider boundary conditions, extreme values, special cases",
            "Assumption documentation: State implicit assumptions - they may be wrong and cause errors",
            "Information relevance: Not all given information is necessary - identify what's essential",
            "Answer format: Check what form answer should take - number, word, equation, etc.",
            "Confidence calibration: Distinguish between high-certainty and uncertain conclusions"
        ]
    }
}


@dataclass
class ClaimTemplate:
    """Structured claim template for domain-specific reasoning priming"""

    domain: str
    content: str
    confidence: float = 0.85
    benchmark: str = "General"
    type: str = "template"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "content": self.content,
            "confidence": self.confidence,
            "benchmark": self.benchmark,
            "type": self.type
        }

    def __repr__(self) -> str:
        return f"ClaimTemplate(domain={self.domain}, benchmark={self.benchmark}, confidence={self.confidence})"


class DomainClaimSelector:
    """Selects and formats domain-specific claim templates based on problem type"""

    def __init__(self):
        self.templates = DOMAIN_CLAIM_TEMPLATES
        self.template_cache: Dict[str, List[ClaimTemplate]] = {}

    def get_templates_for_domain(self, domain: str, max_count: int = 5) -> List[ClaimTemplate]:
        """Get top claim templates for a specific domain"""
        domain_key = domain.lower().replace(" ", "_")

        if domain_key in self.template_cache:
            return self.template_cache[domain_key][:max_count]

        domain_data = self.templates.get(domain_key, self.templates["general"])
        benchmark = domain_data.get("benchmark", "General")

        claim_templates = [
            ClaimTemplate(
                domain=domain_key,
                content=template,
                benchmark=benchmark
            )
            for template in domain_data.get("templates", [])
        ]

        self.template_cache[domain_key] = claim_templates
        return claim_templates[:max_count]

    def format_claims_for_prompt(self, domain: str, max_count: int = 3) -> str:
        """Format domain claim templates as structured prompt hints with position primacy"""
        templates = self.get_templates_for_domain(domain, max_count)

        if not templates:
            return ""

        domain_data = self.templates.get(domain.lower().replace(" ", "_"), {})
        domain_name = domain_data.get("domain_name", "Domain Reasoning")

        # Format with position primacy - place at START of prompt for maximum attention
        formatted = f"\nKEY DOMAIN PATTERNS FOR {domain_name}:\n"
        for i, template in enumerate(templates, 1):
            formatted += f"• {template.content}\n"

        return formatted

    def get_all_benchmarks(self) -> Dict[str, str]:
        """Return mapping of domain keys to benchmark names"""
        return {
            key: data.get("benchmark", "General")
            for key, data in self.templates.items()
        }

    def select_by_benchmark(self, benchmark: str, max_count: int = 4) -> List[ClaimTemplate]:
        """Select claim templates by benchmark name"""
        templates = []
        for domain_key, domain_data in self.templates.items():
            if domain_data.get("benchmark") == benchmark:
                templates.extend([
                    ClaimTemplate(
                        domain=domain_key,
                        content=template,
                        benchmark=benchmark
                    )
                    for template in domain_data.get("templates", [])[:max_count]
                ])
        return templates


# Global instance for easy access
_global_selector = None


def get_domain_selector() -> DomainClaimSelector:
    """Get or create global domain claim selector instance"""
    global _global_selector
    if _global_selector is None:
        _global_selector = DomainClaimSelector()
    return _global_selector


def get_templates_for_domain(domain: str, max_count: int = 5) -> List[ClaimTemplate]:
    """Convenience function to get templates for a domain"""
    return get_domain_selector().get_templates_for_domain(domain, max_count)


def format_claims_for_prompt(domain: str, max_count: int = 3) -> str:
    """Convenience function to format domain claims for prompt injection"""
    return get_domain_selector().format_claims_for_prompt(domain, max_count)


def get_benchmark_templates(benchmark: str, max_count: int = 4) -> List[ClaimTemplate]:
    """Convenience function to get templates by benchmark name"""
    return get_domain_selector().select_by_benchmark(benchmark, max_count)
