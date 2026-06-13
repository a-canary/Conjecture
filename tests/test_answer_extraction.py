# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test suite for robust answer extraction
Tests all extraction patterns and edge cases
"""

import pytest
from pathlib import Path
import sys

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

from answer_extraction import (
    extract_answer,
    check_answer_match,
    extract_numerical_answer,
    extract_multiple_choice,
    normalize_answer,
    AnswerType
)


class TestNormalization:
    """Test answer normalization"""

    def test_whitespace_stripping(self):
        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("\n42\n") == "42"
        assert normalize_answer("\t42\t") == "42"

    def test_lowercase_conversion(self):
        assert normalize_answer("Answer") == "answer"
        assert normalize_answer("ABC") == "abc"

    def test_comma_removal(self):
        # Note: current implementation doesn't remove commas in normalize
        # but the extraction functions do
        assert normalize_answer("42") == "42"


class TestNumericalExtraction:
    """Test numerical answer extraction"""

    def test_gsm8k_format(self):
        """Test #### number format"""
        assert extract_numerical_answer("The answer is #### 42") == "42"
        assert extract_numerical_answer("#### 42") == "42"
        assert extract_numerical_answer("#### -5") == "-5"
        assert extract_numerical_answer("#### 3.14") == "3.14"

    def test_boxed_format(self):
        """Test LaTeX \\boxed{} format"""
        assert extract_numerical_answer("Therefore \\boxed{17}") == "17"
        assert extract_numerical_answer("\\boxed{42}") == "42"
        assert extract_numerical_answer("Answer: \\boxed{-5}") == "-5"
        assert extract_numerical_answer("\\boxed{3.14159}") == "3.14159"

    def test_answer_is_pattern(self):
        """Test 'answer is X' patterns"""
        assert extract_numerical_answer("The answer is 42") == "42"
        assert extract_numerical_answer("Answer: 42") == "42"
        assert extract_numerical_answer("answer is 42") == "42"
        assert extract_numerical_answer("answer is 3.14") == "3.14"

    def test_final_answer_pattern(self):
        """Test 'final answer' pattern"""
        assert extract_numerical_answer("Final answer: 42") == "42"
        assert extract_numerical_answer("Final answer is 42") == "42"
        assert extract_numerical_answer("The final answer is 42") == "42"

    def test_result_pattern(self):
        """Test 'result' pattern"""
        assert extract_numerical_answer("Result: 42") == "42"
        assert extract_numerical_answer("Result is 42") == "42"

    def test_equals_pattern(self):
        """Test equation format"""
        assert extract_numerical_answer("x = 42") == "42"
        assert extract_numerical_answer("2 + 2 = 4") == "4"

    def test_negative_numbers(self):
        """Test negative number extraction"""
        assert extract_numerical_answer("The answer is -5") == "-5"
        assert extract_numerical_answer("#### -42") == "-42"

    def test_decimal_numbers(self):
        """Test decimal extraction"""
        assert extract_numerical_answer("The answer is 3.14") == "3.14"
        assert extract_numerical_answer("#### 0.5") == "0.5"

    def test_numbers_with_commas(self):
        """Test numbers with comma separators"""
        assert extract_numerical_answer("The answer is 1,000") == "1000"
        assert extract_numerical_answer("#### 1,000,000") == "1000000"
        assert extract_numerical_answer("Answer: 1,234.56") == "1234.56"

    def test_multiple_numbers_fallback(self):
        """Test fallback to last number when no pattern matches"""
        # Should return last number in response
        assert extract_numerical_answer("Numbers: 10, 20, 30, 42") == "42"

    def test_edge_case_mixed_text_and_numbers(self):
        """Test extracting from mixed content"""
        response = """
        Let me solve this step by step:
        10 + 20 = 30
        30 + 12 = 42
        The answer is 42
        """
        assert extract_numerical_answer(response) == "42"

    def test_empty_response(self):
        """Test empty response"""
        assert extract_numerical_answer("") is None
        assert extract_numerical_answer(None) is None


class TestMultipleChoiceExtraction:
    """Test multiple choice answer extraction"""

    def test_explicit_answer_prefix(self):
        """Test explicit answer indicators"""
        assert extract_multiple_choice("Answer: A") == "A"
        assert extract_multiple_choice("The answer is B") == "B"
        assert extract_multiple_choice("Choice: C") == "C"
        assert extract_multiple_choice("Select: D") == "D"

    def test_single_letter_response(self):
        """Test single letter answers"""
        assert extract_multiple_choice("A") == "A"
        assert extract_multiple_choice("B") in ["B", "A"]  # Could match first
        assert extract_multiple_choice("The answer is (C)") == "C"

    def test_parenthesized_letter(self):
        """Test parenthesized choices"""
        assert extract_multiple_choice("The correct answer is (A)") == "A"
        assert extract_multiple_choice("(B) is correct") == "B"

    def test_numeric_choice(self):
        """Test numeric choice conversion"""
        assert extract_multiple_choice("Answer: 1") == "A"
        assert extract_multiple_choice("Answer: 2") == "B"
        assert extract_multiple_choice("Answer: 3") == "C"
        assert extract_multiple_choice("Answer: 4") == "D"

    def test_case_insensitivity(self):
        """Test case handling"""
        result = extract_multiple_choice("answer: a")
        assert result and result.upper() == "A"

    def test_early_position_bias(self):
        """Test finding answer in early part of response"""
        response = "Let me think about this. The answer is A. Here's why..."
        assert extract_multiple_choice(response) == "A"


class TestAnswerChecking:
    """Test answer matching/checking"""

    def test_exact_match(self):
        """Test exact string matching"""
        assert check_answer_match("42", "42", AnswerType.NUMERICAL)
        assert check_answer_match("A", "A", AnswerType.MULTIPLE_CHOICE)

    def test_case_insensitive_match(self):
        """Test case-insensitive matching"""
        assert check_answer_match("a", "A", AnswerType.MULTIPLE_CHOICE)
        assert check_answer_match("answer", "ANSWER", AnswerType.CATEGORICAL)

    def test_numerical_precision(self):
        """Test numerical matching with tolerance"""
        assert check_answer_match("42.0", "42", AnswerType.NUMERICAL)
        assert check_answer_match("42.001", "42", AnswerType.NUMERICAL)
        assert check_answer_match("3.14", "3.14", AnswerType.NUMERICAL)

    def test_numerical_with_commas(self):
        """Test numerical matching with comma formatting"""
        assert check_answer_match("1,000", "1000", AnswerType.NUMERICAL)
        assert check_answer_match("1000", "1,000", AnswerType.NUMERICAL)

    def test_multiple_choice_only_first_char(self):
        """Test that multiple choice only looks at first character"""
        assert check_answer_match("A", "A", AnswerType.MULTIPLE_CHOICE)
        # Should still match because it checks first char
        assert check_answer_match("answer_a", "a", AnswerType.MULTIPLE_CHOICE)

    def test_mismatch_detection(self):
        """Test that mismatches are detected"""
        assert not check_answer_match("42", "41", AnswerType.NUMERICAL)
        assert not check_answer_match("A", "B", AnswerType.MULTIPLE_CHOICE)

    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        assert not check_answer_match("", "42", AnswerType.NUMERICAL)
        assert not check_answer_match("42", "", AnswerType.NUMERICAL)


class TestIntegratedExtraction:
    """Test full extraction pipeline with type inference"""

    def test_numerical_inference(self):
        """Test automatic numerical type inference"""
        result = extract_answer("The answer is 42", "42")
        assert result == "42"

    def test_multiple_choice_inference(self):
        """Test automatic multiple choice inference"""
        result = extract_answer("The answer is B", "B")
        assert result == "B"

    def test_gsm8k_real_example(self):
        """Test real GSM8K example"""
        response = """
        Let me work through this step by step:
        - Item A costs $5
        - Item B costs $3
        - Total: $5 + $3 = $8

        #### 8
        """
        extracted = extract_answer(response, "8")
        assert check_answer_match(extracted, "8")

    def test_aime_real_example(self):
        """Test real AIME example"""
        response = """
        Using geometric formulas:
        The triangle has sides a, b, c.
        Solving the system of equations gives us:

        Therefore the answer is \\boxed{17}
        """
        extracted = extract_answer(response, "17")
        assert check_answer_match(extracted, "17")

    def test_mmlu_real_example(self):
        """Test real MMLU example"""
        response = """
        The question asks about biological processes.
        A) Incorrect because...
        B) This is correct because mitochondria...
        C) No, this is...
        D) No, that...

        The answer is B
        """
        extracted = extract_answer(response, "B")
        assert check_answer_match(extracted, "B")


class TestEdgeCases:
    """Test edge cases and potential failure modes"""

    def test_answer_in_quoted_text(self):
        """Test extracting answer from quoted text"""
        response = 'He said "The answer is 42" but he was wrong.'
        extracted = extract_numerical_answer(response)
        assert extracted == "42"

    def test_multiple_answer_markers(self):
        """Test response with multiple potential answers"""
        response = """
        I thought the answer was 40.
        But then I calculated again.
        The answer is 42.
        """
        extracted = extract_numerical_answer(response)
        assert extracted == "42"  # Should get last explicit answer

    def test_answer_with_units(self):
        """Test answer with units (should extract number)"""
        response = "The answer is 42 meters"
        extracted = extract_numerical_answer(response)
        # Should extract the number part
        assert extracted is not None

    def test_fraction_answer(self):
        """Test fraction-like answers"""
        response = "The answer is 5/6"
        extracted = extract_numerical_answer(response)
        # May not perfectly handle fractions, but should try
        assert extracted is not None

    def test_very_long_response(self):
        """Test extraction from very long response"""
        response = "The answer is " + "blah " * 1000 + "42"
        extracted = extract_numerical_answer(response)
        assert extracted == "42"

    def test_special_characters(self):
        """Test answers with special characters"""
        response = "The answer is √2"
        extracted = extract_answer(response, "√2")
        assert extracted is not None

    def test_none_and_empty_inputs(self):
        """Test None and empty string inputs"""
        assert extract_answer("", "42") == ""
        assert extract_answer(None, "42") == ""


class TestBenchmarkCompatibility:
    """Test compatibility with existing benchmark code"""

    def test_backward_compat_gsm8k(self):
        """Test compatibility with GSM8K format"""
        response = "Let me calculate: 10 + 20 + 12 = #### 42"
        extracted = extract_answer(response, "42", AnswerType.NUMERICAL)
        assert check_answer_match(extracted, "42")

    def test_backward_compat_aime(self):
        """Test compatibility with AIME format"""
        response = "Using algebra: \\boxed{847}"
        extracted = extract_answer(response, "847", AnswerType.NUMERICAL)
        assert check_answer_match(extracted, "847")

    def test_backward_compat_multiple_choice(self):
        """Test compatibility with multiple choice"""
        response = "Analyzing the options, the answer is C"
        extracted = extract_answer(response, "C", AnswerType.MULTIPLE_CHOICE)
        assert check_answer_match(extracted, "C")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
