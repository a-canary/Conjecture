# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Research-Optimized Claim Selector.

Validates all research-backed optimizations:
1. Windowing
2. Confidence gating
3. Semantic filtering
4. Category matching
5. Prompt position (primacy)
"""

import pytest
from src.process.research_optimized_selector import (
    ResearchOptimizedSelector,
    OptimizedClaim,
    create_optimized_selector
)


class TestResearchOptimizedSelector:
    """Test suite for optimized selector."""

    def test_create_selector(self):
        """Test basic selector creation."""
        selector = ResearchOptimizedSelector()
        assert selector.window_size == 20
        assert selector.confidence_threshold == 0.8
        assert selector.max_claims == 3

    def test_custom_config(self):
        """Test custom configuration."""
        selector = create_optimized_selector(
            window_size=10,
            confidence_threshold=0.9,
            max_claims=2
        )
        assert selector.window_size == 10
        assert selector.confidence_threshold == 0.9
        assert selector.max_claims == 2

    def test_add_claim(self):
        """Test claim addition."""
        selector = ResearchOptimizedSelector()
        selector.add_claim(
            content="Test claim",
            question="What is 2+2?",
            confidence=0.9,
            is_correct=True
        )
        assert len(selector.claims) == 1
        assert selector.claims[0].content == "Test claim"

    def test_windowing(self):
        """Test window limits recent claims."""
        selector = ResearchOptimizedSelector(window_size=5)

        # Add 10 claims
        for i in range(10):
            selector.add_claim(
                content=f"Claim {i}",
                question=f"Question {i}?",
                confidence=0.9,
                is_correct=True
            )

        windowed = selector._window_claims()
        assert len(windowed) == 5
        # Should be claims 5-9 (most recent)
        assert windowed[0].content == "Claim 5"
        assert windowed[-1].content == "Claim 9"

    def test_confidence_gating(self):
        """Test confidence threshold filtering."""
        selector = ResearchOptimizedSelector(confidence_threshold=0.8)

        # Add claims with varying confidence
        selector.add_claim("Low conf", "Q1?", confidence=0.5, is_correct=True)
        selector.add_claim("Med conf", "Q2?", confidence=0.7, is_correct=True)
        selector.add_claim("High conf", "Q3?", confidence=0.9, is_correct=True)
        selector.add_claim("Incorrect", "Q4?", confidence=0.9, is_correct=False)

        gated = selector._gate_claims(selector.claims)
        assert len(gated) == 1  # Only high conf + correct
        assert gated[0].content == "High conf"

    def test_category_detection(self):
        """Test automatic category detection."""
        selector = ResearchOptimizedSelector()

        assert selector._detect_category("Store sells 10 items at $5") == "sales"
        assert selector._detect_category("Car travels at 60 mph") == "distance"
        assert selector._detect_category("What is 25% of 100?") == "percentage"
        assert selector._detect_category("Divide 12 equally among 3") == "division"
        assert selector._detect_category("Calculate area of triangle") == "geometry"
        assert selector._detect_category("Random question here") == "general"

    def test_category_bonus(self):
        """Test same-category claims are preferred."""
        selector = ResearchOptimizedSelector()

        # Add sales claim
        selector.add_claim(
            content="Sales pattern",
            question="Store sells items",
            confidence=0.9,
            is_correct=True,
            category="sales"
        )

        # Add distance claim
        selector.add_claim(
            content="Distance pattern",
            question="Car travels",
            confidence=0.9,
            is_correct=True,
            category="distance"
        )

        # Query sales question
        selected = selector.select_claims("Store sells products", "sales")

        # Sales claim should rank higher due to category bonus
        if selected:
            top_claim = selected[0][0]
            assert top_claim.category == "sales"

    def test_max_claims_limit(self):
        """Test max claims returned is enforced."""
        selector = ResearchOptimizedSelector(max_claims=2)

        # Add 5 high-confidence correct claims
        for i in range(5):
            selector.add_claim(
                content=f"Claim {i}",
                question=f"Question {i}?",
                confidence=0.9,
                is_correct=True
            )

        selected = selector.select_claims("Test question?")
        assert len(selected) <= 2

    def test_build_prompt_primacy(self):
        """Test claims appear at START of prompt (primacy bias)."""
        selector = ResearchOptimizedSelector()

        selector.add_claim(
            content="Pattern: multiply then add",
            question="2*3+4?",
            confidence=0.9,
            is_correct=True
        )

        prompt = selector.build_prompt("What is 5*2+1?")

        # Claims should be at START
        assert prompt.startswith("KEY PATTERNS")
        # Problem should come after
        assert "Problem:" in prompt
        lines = prompt.split("\n")
        pattern_idx = next(i for i, l in enumerate(lines) if "PATTERNS" in l)
        problem_idx = next(i for i, l in enumerate(lines) if "Problem:" in l)
        assert pattern_idx < problem_idx

    def test_empty_claims(self):
        """Test handling when no claims match."""
        selector = ResearchOptimizedSelector()

        # Add only incorrect or low-confidence claims
        selector.add_claim("Bad", "Q?", confidence=0.5, is_correct=True)
        selector.add_claim("Wrong", "Q?", confidence=0.9, is_correct=False)

        selected = selector.select_claims("Test?")
        assert len(selected) == 0

        prompt = selector.build_prompt("Test?")
        assert "PATTERNS" not in prompt

    def test_embedding_similarity(self):
        """Test semantic similarity via embeddings."""
        selector = ResearchOptimizedSelector()

        # Embeddings should be similar for similar questions
        emb1 = selector._embed("Store sells 10 apples")
        emb2 = selector._embed("Store sells 20 oranges")
        emb3 = selector._embed("Car travels at speed")

        sim_same = selector._cosine_sim(emb1, emb2)
        sim_diff = selector._cosine_sim(emb1, emb3)

        # Same-topic should be more similar
        assert sim_same > sim_diff

    def test_get_stats(self):
        """Test statistics reporting."""
        selector = ResearchOptimizedSelector()

        for i in range(5):
            selector.add_claim(
                content=f"Claim {i}",
                question=f"Q{i}?",
                confidence=0.9 if i % 2 == 0 else 0.5,
                is_correct=True
            )

        stats = selector.get_stats()
        assert stats["total_claims"] == 5
        assert stats["gated_claims"] == 3  # Only conf >= 0.8


class TestIntegration:
    """Integration tests for full workflow."""

    def test_accumulation_workflow(self):
        """Test typical accumulation workflow."""
        selector = create_optimized_selector(
            window_size=10,
            confidence_threshold=0.8,
            max_claims=3
        )

        # Simulate solving problems
        problems = [
            ("Store sells 10 at $2", "20", "sales"),
            ("Car goes 60 mph for 2 hours", "120", "distance"),
            ("What is 50% of 200", "100", "percentage"),
            ("Store sells 5 at $3", "15", "sales"),
        ]

        for q, answer, cat in problems:
            # Build prompt with current memory
            prompt = selector.build_prompt(q, cat)

            # Simulate solving (always correct for test)
            selector.add_claim(
                content=f"{q[:30]}... → {answer}",
                question=q,
                confidence=0.9,
                is_correct=True,
                category=cat
            )

        stats = selector.get_stats()
        assert stats["total_claims"] == 4

        # Query a sales question - should prefer sales claims
        selected = selector.select_claims("Store has items", "sales")
        sales_claims = [c for c, _ in selected if c.category == "sales"]
        assert len(sales_claims) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
