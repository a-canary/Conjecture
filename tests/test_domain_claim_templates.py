#!/usr/bin/env python3
"""
Tests for Domain-Specific Claim Templates

Validates:
- Template loading and retrieval by domain
- Template formatting for prompt injection
- Benchmark-specific template selection
- Position primacy implementation
"""

import pytest
from src.agent.domain_claim_templates import (
    DOMAIN_CLAIM_TEMPLATES,
    ClaimTemplate,
    DomainClaimSelector,
    get_domain_selector,
    get_templates_for_domain,
    format_claims_for_prompt,
    get_benchmark_templates,
)
from src.agent.domain_reasoning_enhancement import (
    DomainReasoningEnhancer,
    BenchmarkType,
    get_enhancer,
    enhance_for_benchmark,
    get_benchmark_context,
)


class TestDomainClaimTemplates:
    """Test domain-specific claim template loading and management"""

    def test_templates_exist_for_all_domains(self):
        """Verify templates exist for all domain categories"""
        expected_domains = {"mathematical", "scientific", "logical", "general"}
        actual_domains = set(DOMAIN_CLAIM_TEMPLATES.keys())
        assert expected_domains == actual_domains

    def test_mathematical_templates_for_drop(self):
        """Verify mathematical templates exist for DROP benchmark"""
        math_templates = DOMAIN_CLAIM_TEMPLATES["mathematical"]
        assert math_templates["benchmark"] == "DROP"
        assert len(math_templates["templates"]) >= 8
        assert any("PEMDAS" in t for t in math_templates["templates"])
        assert any("percentage" in t for t in math_templates["templates"])

    def test_scientific_templates_for_arc(self):
        """Verify scientific templates exist for ARC benchmark"""
        sci_templates = DOMAIN_CLAIM_TEMPLATES["scientific"]
        assert sci_templates["benchmark"] == "ARC"
        assert len(sci_templates["templates"]) >= 8
        assert any("scientific method" in t.lower() for t in sci_templates["templates"])
        assert any("hypothesis" in t for t in sci_templates["templates"])

    def test_logical_templates_for_bbh(self):
        """Verify logical templates exist for BBH benchmark"""
        log_templates = DOMAIN_CLAIM_TEMPLATES["logical"]
        assert log_templates["benchmark"] == "BBH"
        assert len(log_templates["templates"]) >= 8
        assert any("deductive" in t.lower() for t in log_templates["templates"])
        assert any("contrapositive" in t for t in log_templates["templates"])

    def test_claim_template_creation(self):
        """Test ClaimTemplate dataclass creation and serialization"""
        template = ClaimTemplate(
            domain="mathematical",
            content="Test reasoning principle",
            confidence=0.9,
            benchmark="DROP"
        )

        assert template.domain == "mathematical"
        assert template.content == "Test reasoning principle"
        assert template.confidence == 0.9
        assert template.benchmark == "DROP"
        assert template.type == "template"

        # Test serialization
        template_dict = template.to_dict()
        assert template_dict["domain"] == "mathematical"
        assert template_dict["benchmark"] == "DROP"


class TestDomainClaimSelector:
    """Test the DomainClaimSelector functionality"""

    def test_selector_initialization(self):
        """Test DomainClaimSelector can be initialized"""
        selector = DomainClaimSelector()
        assert selector.templates is not None
        assert len(selector.templates) >= 3

    def test_get_templates_for_domain(self):
        """Test template retrieval for specific domain"""
        selector = DomainClaimSelector()

        math_templates = selector.get_templates_for_domain("mathematical", max_count=3)
        assert len(math_templates) == 3
        assert all(isinstance(t, ClaimTemplate) for t in math_templates)
        assert all(t.domain == "mathematical" for t in math_templates)

        sci_templates = selector.get_templates_for_domain("scientific", max_count=4)
        assert len(sci_templates) == 4

    def test_template_caching(self):
        """Test that templates are cached for performance"""
        selector = DomainClaimSelector()

        # First call - populates cache
        templates1 = selector.get_templates_for_domain("mathematical", max_count=5)
        assert "mathematical" in selector.template_cache

        # Second call - uses cache
        templates2 = selector.get_templates_for_domain("mathematical", max_count=5)
        assert templates1 == templates2

    def test_format_claims_for_prompt(self):
        """Test formatting of claim templates for prompt injection"""
        selector = DomainClaimSelector()
        formatted = selector.format_claims_for_prompt("mathematical", max_count=2)

        assert "KEY DOMAIN PATTERNS" in formatted
        assert "Mathematical Reasoning" in formatted
        assert formatted.count("•") == 2

    def test_get_all_benchmarks(self):
        """Test retrieval of benchmark mapping"""
        selector = DomainClaimSelector()
        benchmarks = selector.get_all_benchmarks()

        assert benchmarks["mathematical"] == "DROP"
        assert benchmarks["scientific"] == "ARC"
        assert benchmarks["logical"] == "BBH"
        assert benchmarks["general"] == "General"

    def test_select_by_benchmark(self):
        """Test template selection by benchmark name"""
        selector = DomainClaimSelector()

        drop_templates = selector.select_by_benchmark("DROP", max_count=3)
        assert len(drop_templates) > 0
        assert all(t.benchmark == "DROP" for t in drop_templates)

        arc_templates = selector.select_by_benchmark("ARC", max_count=3)
        assert len(arc_templates) > 0
        assert all(t.benchmark == "ARC" for t in arc_templates)


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_get_domain_selector(self):
        """Test global selector instance"""
        selector1 = get_domain_selector()
        selector2 = get_domain_selector()
        assert selector1 is selector2  # Same instance

    def test_get_templates_for_domain_convenience(self):
        """Test convenience function for getting templates"""
        templates = get_templates_for_domain("logical", max_count=3)
        assert len(templates) == 3
        assert all(t.domain == "logical" for t in templates)

    def test_format_claims_for_prompt_convenience(self):
        """Test convenience function for formatting claims"""
        formatted = format_claims_for_prompt("scientific", max_count=2)
        assert "KEY DOMAIN PATTERNS" in formatted
        assert "Scientific Reasoning" in formatted

    def test_get_benchmark_templates_convenience(self):
        """Test convenience function for benchmark-specific templates"""
        templates = get_benchmark_templates("DROP", max_count=3)
        assert len(templates) > 0
        assert all(t.benchmark == "DROP" for t in templates)


class TestDomainReasoningEnhancer:
    """Test the DomainReasoningEnhancer for prompt optimization"""

    def test_enhancer_initialization(self):
        """Test enhancer can be initialized"""
        enhancer = DomainReasoningEnhancer()
        assert enhancer.selector is not None
        assert enhancer.benchmark_domain_mapping is not None

    def test_enhance_prompt_with_domain_claims(self):
        """Test prompt enhancement with domain claims"""
        enhancer = DomainReasoningEnhancer()
        base_prompt = "Solve this mathematical problem: 5 + 3 = ?"

        enhanced = enhancer.enhance_prompt_with_domain_claims(
            base_prompt,
            "mathematical",
            max_claims=2,
            position="start"
        )

        # Templates should be at start
        assert enhanced.startswith("KEY DOMAIN PATTERNS")
        assert base_prompt in enhanced
        assert enhanced != base_prompt

    def test_enhance_for_drop_benchmark(self):
        """Test enhancement for DROP mathematical benchmark"""
        enhancer = DomainReasoningEnhancer()
        base_prompt = "Solve this word problem."

        enhanced = enhancer.enhance_for_benchmark(
            base_prompt,
            BenchmarkType.DROP,
            max_claims=3
        )

        assert enhanced != base_prompt
        assert "KEY DOMAIN PATTERNS" in enhanced
        assert "Mathematical Reasoning" in enhanced

    def test_enhance_for_arc_benchmark(self):
        """Test enhancement for ARC scientific benchmark"""
        enhancer = DomainReasoningEnhancer()
        base_prompt = "Analyze this scientific experiment."

        enhanced = enhancer.enhance_for_benchmark(
            base_prompt,
            BenchmarkType.ARC,
            max_claims=3
        )

        assert enhanced != base_prompt
        assert "Scientific Reasoning" in enhanced

    def test_enhance_for_bbh_benchmark(self):
        """Test enhancement for BBH logical benchmark"""
        enhancer = DomainReasoningEnhancer()
        base_prompt = "Solve this logic puzzle."

        enhanced = enhancer.enhance_for_benchmark(
            base_prompt,
            BenchmarkType.BBH,
            max_claims=3
        )

        assert enhanced != base_prompt
        assert "Logical Reasoning" in enhanced

    def test_get_benchmark_context(self):
        """Test retrieval of benchmark context information"""
        enhancer = DomainReasoningEnhancer()

        context = enhancer.get_benchmark_context(BenchmarkType.DROP)
        assert context["benchmark"] == "DROP"
        assert context["domain"] == "mathematical"
        assert "templates" in context["domain_name"].lower()
        assert len(context["key_templates"]) > 0

    def test_position_primacy_start(self):
        """Test that start position places templates at prompt beginning"""
        enhancer = DomainReasoningEnhancer()
        base = "Main problem text here"

        enhanced = enhancer.enhance_prompt_with_domain_claims(
            base,
            "mathematical",
            max_claims=1,
            position="start"
        )

        # Template should come before base text
        assert enhanced.find("KEY DOMAIN PATTERNS") < enhanced.find(base)

    def test_mathematical_reasoning_prompt(self):
        """Test mathematical reasoning prompt preparation"""
        enhancer = DomainReasoningEnhancer()
        base = "Solve: 2x + 3 = 7"

        enhanced = enhancer.prepare_mathematical_reasoning_prompt(
            base,
            include_templates=True
        )

        assert enhanced != base
        assert "KEY DOMAIN PATTERNS" in enhanced
        assert "DROP" in enhanced or "Mathematical" in enhanced

    def test_scientific_reasoning_prompt(self):
        """Test scientific reasoning prompt preparation"""
        enhancer = DomainReasoningEnhancer()
        base = "Analyze: What causes erosion?"

        enhanced = enhancer.prepare_scientific_reasoning_prompt(
            base,
            include_templates=True
        )

        assert enhanced != base
        assert "KEY DOMAIN PATTERNS" in enhanced
        assert "ARC" in enhanced or "Scientific" in enhanced

    def test_logical_reasoning_prompt(self):
        """Test logical reasoning prompt preparation"""
        enhancer = DomainReasoningEnhancer()
        base = "If A then B, prove B"

        enhanced = enhancer.prepare_logical_reasoning_prompt(
            base,
            include_templates=True
        )

        assert enhanced != base
        assert "KEY DOMAIN PATTERNS" in enhanced
        assert "BBH" in enhanced or "Logical" in enhanced

    def test_disable_templates(self):
        """Test that templates can be disabled"""
        enhancer = DomainReasoningEnhancer()
        base = "Test prompt"

        enhanced = enhancer.prepare_mathematical_reasoning_prompt(
            base,
            include_templates=False
        )

        assert enhanced == base  # Should be unchanged


class TestPromptEnhancementConvenience:
    """Test convenience functions for prompt enhancement"""

    def test_enhance_for_benchmark_drop(self):
        """Test enhance_for_benchmark convenience function for DROP"""
        base = "Mathematical problem"
        enhanced = enhance_for_benchmark(base, "DROP", max_claims=2)

        assert enhanced != base
        assert "KEY DOMAIN PATTERNS" in enhanced

    def test_enhance_for_benchmark_arc(self):
        """Test enhance_for_benchmark convenience function for ARC"""
        base = "Scientific problem"
        enhanced = enhance_for_benchmark(base, "ARC", max_claims=2)

        assert enhanced != base
        assert "KEY DOMAIN PATTERNS" in enhanced

    def test_get_benchmark_context_convenience(self):
        """Test get_benchmark_context convenience function"""
        context = get_benchmark_context("DROP")
        assert context["benchmark"] == "DROP"
        assert context["domain"] == "mathematical"

        context = get_benchmark_context("ARC")
        assert context["benchmark"] == "ARC"
        assert context["domain"] == "scientific"


class TestIntegrationWithPromptSystem:
    """Integration tests with the prompt system"""

    def test_templates_available_for_problem_types(self):
        """Verify templates are available for all Conjecture problem types"""
        from src.agent.prompt_system import ProblemType

        enhancer = DomainReasoningEnhancer()

        for problem_type in ProblemType:
            domain_mapping = {
                "MATHEMATICAL": "mathematical",
                "SCIENTIFIC": "scientific",
                "LOGICAL": "logical",
            }

            domain = domain_mapping.get(
                problem_type.name,
                "general"
            )

            formatted = format_claims_for_prompt(domain, max_count=2)
            # Should get some output for recognized domains
            if domain in ["mathematical", "scientific", "logical"]:
                assert formatted
                assert "KEY DOMAIN PATTERNS" in formatted
