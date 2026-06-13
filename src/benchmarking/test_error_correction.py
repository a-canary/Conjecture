# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Integration test for error correction prompts.

Demonstrates the error correction system working with different problem types.
No external dependencies or config issues.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.agent.error_correction_prompts import (
    get_error_correction_prompt,
    get_quick_error_correction,
    should_trigger_error_correction,
    ErrorCorrectionConfig,
    ProblemType,
)


def test_error_correction_mathematical():
    """Test error correction for mathematical problems."""
    problem = "If a rectangle has length 8 and width 5, what is its area?"
    initial_answer = "The area is 8 + 5 = 13 square units."

    correction = get_error_correction_prompt(
        problem, initial_answer, ProblemType.MATHEMATICAL
    )

    assert "CALCULATION REVIEW" in correction
    assert "PEMDAS/BODMAS" in correction
    assert problem in correction
    assert initial_answer in correction
    print("✓ Mathematical error correction test passed")


def test_error_correction_logical():
    """Test error correction for logical problems."""
    problem = "All humans are mortal. Socrates is human. Is Socrates mortal?"
    initial_answer = "Not necessarily."

    correction = get_error_correction_prompt(
        problem, initial_answer, ProblemType.LOGICAL
    )

    assert "PREMISE VERIFICATION" in correction
    assert "REASONING CHAIN" in correction
    assert problem in correction
    print("✓ Logical error correction test passed")


def test_error_correction_sequential():
    """Test error correction for sequential problems."""
    problem = "First prepare ingredients. Then mix. Finally bake."
    initial_answer = "Mix, prepare, bake."

    correction = get_error_correction_prompt(
        problem, initial_answer, ProblemType.SEQUENTIAL
    )

    assert "STEP ORDER" in correction
    assert "STEP COMPLETENESS" in correction
    assert "DEPENDENCY VERIFICATION" in correction
    print("✓ Sequential error correction test passed")


def test_quick_error_correction():
    """Test quick error correction reminders."""
    for problem_type in [
        ProblemType.MATHEMATICAL,
        ProblemType.LOGICAL,
        ProblemType.SEQUENTIAL,
        ProblemType.SCIENTIFIC,
        ProblemType.DECOMPOSITION,
        ProblemType.GENERAL,
    ]:
        quick = get_quick_error_correction(problem_type)
        assert len(quick) > 20
        assert len(quick) < 300
        print(f"✓ Quick correction for {problem_type.value}: {len(quick)} chars")


def test_error_correction_triggering():
    """Test error correction triggering logic."""
    # Should trigger below threshold
    assert should_trigger_error_correction(0.5, threshold=0.7) is True
    assert should_trigger_error_correction(0.65, threshold=0.7) is True

    # Should not trigger above threshold
    assert should_trigger_error_correction(0.75, threshold=0.7) is False
    assert should_trigger_error_correction(0.95, threshold=0.7) is False

    # With uncertainty markers
    assert should_trigger_error_correction(0.8, contains_uncertainty_markers=True) is True

    # With short response
    assert (
        should_trigger_error_correction(0.85, response_length=30) is True
    )

    print("✓ Error correction triggering tests passed")


def test_error_correction_config():
    """Test configuration system."""
    config = ErrorCorrectionConfig(
        enabled=True,
        confidence_threshold=0.75,
        target_domains=[ProblemType.MATHEMATICAL, ProblemType.LOGICAL],
        max_correction_attempts=2,
    )

    assert config.enabled is True
    assert config.confidence_threshold == 0.75
    assert len(config.target_domains) == 2
    assert config.max_correction_attempts == 2

    print("✓ Error correction configuration test passed")


def test_all_problem_types():
    """Ensure all problem types have error correction prompts."""
    problem = "Sample problem"
    response = "Sample response"

    for ptype in [
        ProblemType.MATHEMATICAL,
        ProblemType.LOGICAL,
        ProblemType.SCIENTIFIC,
        ProblemType.SEQUENTIAL,
        ProblemType.DECOMPOSITION,
        ProblemType.GENERAL,
    ]:
        correction = get_error_correction_prompt(problem, response, ptype)
        assert "ERROR CORRECTION STEP" in correction
        assert len(correction) > 100
        assert problem in correction
        assert response in correction
        print(f"✓ Error correction available for {ptype.value}")


if __name__ == "__main__":
    print("Running Error Correction Tests")
    print("=" * 60)
    print()

    test_error_correction_mathematical()
    test_error_correction_logical()
    test_error_correction_sequential()
    print()

    test_quick_error_correction()
    print()

    test_error_correction_triggering()
    test_error_correction_config()
    print()

    test_all_problem_types()
    print()

    print("=" * 60)
    print("All tests passed! Error correction system is working correctly.")
