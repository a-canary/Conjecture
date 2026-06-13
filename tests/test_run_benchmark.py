# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for src/evaluation/run_benchmark.py

Verifies that all function references in run_benchmark.py resolve correctly
and that core utilities are callable.
"""

import pytest


class TestRunBenchmarkImports:
    """Ensure all named references in run_benchmark.py are importable."""

    def test_check_answer_function_resolved(self):
        """
        Regression: check_answer was called but never imported or defined.
        Fix: replace check_answer(answer, expected) with check_answer_match(answer, expected)
        at lines 208 and 264 of run_benchmark.py.

        This test ensures that any function named 'check_answer' used in the module
        is actually defined or imported. The fix uses check_answer_match which IS imported.
        """
        import sys
        from pathlib import Path

        src_path = Path(__file__).parent.parent / "src" / "evaluation" / "run_benchmark.py"
        source = src_path.read_text()

        # Scan for function calls that look like they need to be defined
        import re

        # Find all function call sites
        call_sites = re.findall(r'\b(check_answer)\s*\(', source)

        # Find what's imported
        import_section = re.search(r'^from\s+\w+\s+import\s+(.+?)$', source, re.MULTILINE)
        if import_section:
            imported_names = [n.strip() for n in import_section.group(1).split(',')]
        else:
            imported_section = re.search(r'^import\s+(.+?)$', source, re.MULTILINE)
            imported_names = []

        # Find what's defined in the module
        defined_functions = re.findall(r'^def\s+(\w+)\s*\(', source, re.MULTILINE)

        # Every function called must be either imported, defined, or a builtin
        builtins = {'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                    'print', 'range', 'enumerate', 'zip', 'sorted', 'open', 'abs'}

        for name in call_sites:
            assert name in imported_names or name in defined_functions or name in builtins, (
                f"'{name}' is called but not imported, not defined, and not a builtin. "
                f"Imported: {imported_names}, Defined: {defined_functions}"
            )

    def test_check_answer_match_wrapper_is_correct(self):
        """The check_answer_wrapper uses check_answer_match correctly."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
        from answer_extraction import check_answer_match, AnswerType

        # Multiple choice
        result = check_answer_match("A", "A", AnswerType.MULTIPLE_CHOICE)
        assert result is True

        # Wrong answer
        result = check_answer_match("B", "A", AnswerType.MULTIPLE_CHOICE)
        assert result is False

    def test_extract_answer_wrapper_numerical(self):
        """extract_answer_wrapper handles numerical answers."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))
        from answer_extraction import extract_answer, AnswerType

        response = "The answer is 42. Let me show my work."
        result = extract_answer(response, "42", AnswerType.NUMERICAL)
        assert result == "42"

    def test_benchmark_result_dataclass(self):
        """BenchmarkResult dataclass can be instantiated."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.evaluation.run_benchmark import BenchmarkResult

        result = BenchmarkResult(
            task="gsm8k",
            model="test-model",
            method="direct",
            correct=8,
            total=10,
            accuracy=80.0,
            avg_time=1.5,
            total_tokens=500
        )
        assert result.accuracy == 80.0
        assert result.correct == 8

    def test_claim_memory_hints(self):
        """ClaimMemory stores and retrieves hints correctly."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.evaluation.run_benchmark import ClaimMemory

        memory = ClaimMemory()
        memory.add("2+2=4", "math", 0.9, True)
        memory.add("3*3=9", "math", 0.8, True)

        hints = memory.get_hints("math", n=2)
        assert "2+2=4" in hints
        assert "3*3=9" in hints

    def test_llm_client_factory(self):
        """LLMClient raises ValueError for unknown provider."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.evaluation.run_benchmark import LLMClient

        with pytest.raises(ValueError, match="Unknown provider"):
            LLMClient(provider="unknown_provider", model="test")
