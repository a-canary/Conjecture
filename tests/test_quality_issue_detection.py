"""
Quality Issue Detection Tests
Tests bug detection, deception patterns, and duplicate code detection
using advanced evaluation techniques and pattern analysis
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import re
import difflib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EvaluationFramework, create_conjecture_wrapper
from src.benchmarking.deepeval_integration import AdvancedBenchmarkEvaluator


class TestBugDetection:
    """Test bug detection using pattern analysis and consistency checking"""

    @pytest.fixture
    def bug_evaluator(self):
        """Create evaluator for bug detection"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def bug_test_data(self):
        """Load bug detection test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["bug_detection"]

    def test_pattern_analysis_bug_detection(self, bug_evaluator, bug_test_data):
        """Test bug detection using pattern analysis"""
        pattern_bugs = [bug for bug in bug_test_data 
                        if bug["category"] == "pattern_analysis"]
        
        for bug_case in pattern_bugs:
            task = {
                "prompt": bug_case["prompt"],
                "expected_answer": bug_case["expected_answer"],
                "metadata": bug_case["metadata"]
            }
            
            # Test with buggy response
            bug_patterns = bug_case["bug_patterns"]
            
            if "missing_base_case" in bug_patterns:
                buggy_response = "def factorial(n):\n    return n * factorial(n - 1)"  # Missing base case
            elif "incorrect_loop_condition" in bug_patterns:
                buggy_response = "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:  # Should be <="  # Bug in loop condition
            elif "index_error" in bug_patterns:
                buggy_response = "x = [1, 2, 3]\nprint(x[3])  # Index out of bounds"
            else:
                buggy_response = "Buggy code"
            
            score = bug_evaluator._custom_evaluation(task, buggy_response)
            
            # Should detect bug and give lower score
            assert 0.0 <= score <= 0.6  # Should score low due to bugs
            
            # Test with correct response
            correct_response = bug_case["expected_answer"]
            correct_score = bug_evaluator._custom_evaluation(task, correct_response)
            
            # Correct response should score significantly higher
            assert correct_score >= score + 0.3

    def test_consistency_checking_bug_detection(self, bug_evaluator, bug_test_data):
        """Test bug detection using consistency checking"""
        consistency_bugs = [bug for bug in bug_test_data 
                            if bug["category"] == "consistency_checking"]
        
        for bug_case in consistency_bugs:
            task = {
                "prompt": bug_case["prompt"],
                "expected_answer": bug_case["expected_answer"],
                "metadata": bug_case["metadata"]
            }
            
            # Test with inconsistent response
            bug_patterns = bug_case["bug_patterns"]
            
            if "wrong_mid_calculation" in bug_patterns:
                inconsistent_response = "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) / 2  # Should be integer division"
            elif "boundary_error" in bug_patterns:
                inconsistent_response = "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:  # Off-by-one error"
            else:
                inconsistent_response = "Inconsistent implementation"
            
            score = bug_evaluator._custom_evaluation(task, inconsistent_response)
            
            # Should detect inconsistency
            assert 0.0 <= score <= 0.7
            
            # Test with consistent response
            consistent_response = bug_case["expected_answer"]
            consistent_score = bug_evaluator._custom_evaluation(task, consistent_response)
            
            # Consistent response should score higher
            assert consistent_score >= score + 0.2

    def test_error_identification_bug_detection(self, bug_evaluator, bug_test_data):
        """Test bug detection using error identification"""
        error_bugs = [bug for bug in bug_test_data 
                       if bug["category"] == "error_identification"]
        
        for bug_case in error_bugs:
            task = {
                "prompt": bug_case["prompt"],
                "expected_answer": bug_case["expected_answer"],
                "metadata": bug_case["metadata"]
            }
            
            # Test with error-containing response
            error_response = bug_case["expected_answer"] + " This code will work fine."  # Wrong assessment
            
            score = bug_evaluator._custom_evaluation(task, error_response)
            
            # Should detect incorrect error identification
            assert 0.0 <= score <= 0.5
            
            # Test with correct error identification
            correct_response = bug_case["expected_answer"]
            correct_score = bug_evaluator._custom_evaluation(task, correct_response)
            
            # Correct response should score much higher
            assert correct_score >= score + 0.4

    def test_bug_pattern_recognition(self):
        """Test bug pattern recognition algorithms"""
        bug_patterns = {
            "missing_base_case": [
                "def factorial(n):\n    return n * factorial(n - 1)",
                "def fibonacci(n):\n    return fibonacci(n-1) + fibonacci(n-2)"
            ],
            "infinite_recursion": [
                "def infinite_loop():\n    return infinite_loop()",
                "while True:\n    pass"
            ],
            "off_by_one": [
                "for i in range(len(arr)):\n    print(arr[i+1])",
                "for i in range(1, n+1):\n        print(arr[i])"
            ],
            "null_pointer": [
                "obj = None\nprint(obj.value)",
                "ptr = NULL\nprintf(*ptr)"
            ]
        }
        
        for pattern, examples in bug_patterns.items():
            for code in examples:
                # Test pattern detection
                detected_patterns = self._detect_bug_patterns(code)
                assert pattern in detected_patterns or any(
                    alt_pattern in detected_patterns 
                    for alt_pattern in bug_patterns.keys()
                    if alt_pattern != pattern
                ), f"Failed to detect pattern: {pattern}"

    def _detect_bug_patterns(self, code: str) -> List[str]:
        """Helper method to detect bug patterns in code"""
        patterns = []
        
        # Missing base case in recursion
        if "def " in code and "return" in code and "if" not in code:
            if "factorial" in code or "fibonacci" in code:
                patterns.append("missing_base_case")
        
        # Infinite recursion/loop
        if "while True:" in code or (code.count("return") > 0 and "if" not in code):
            patterns.append("infinite_recursion")
        
        # Off-by-one errors
        if "range(len(" in code or "i+1" in code or "i-1" in code:
            patterns.append("off_by_one")
        
        # Null pointer/dereference
        if "None" in code and ".value" in code or "NULL" in code:
            patterns.append("null_pointer")
        
        return patterns

    def test_bug_severity_assessment(self, bug_evaluator, bug_test_data):
        """Test bug severity assessment"""
        for bug_case in bug_test_data:
            severity = bug_case["metadata"]["severity"]
            bug_type = bug_case["metadata"]["bug_type"]
            
            # Test severity levels
            assert severity in ["low", "medium", "high"]
            assert bug_type in ["logic_error", "algorithm_error", "runtime_error"]
            
            # High severity bugs should be detected more easily
            if severity == "high":
                # Should have clear detection patterns
                assert len(bug_case["bug_patterns"]) >= 2


class TestDeceptionDetection:
    """Test deception detection using factual verification and confidence analysis"""

    @pytest.fixture
    def deception_evaluator(self):
        """Create evaluator for deception detection"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def deception_test_data(self):
        """Load deception detection test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["deception_detection"]

    def test_factual_verification_deception_detection(self, deception_evaluator, deception_test_data):
        """Test deception detection using factual verification"""
        factual_deception = [deception for deception in deception_test_data 
                           if deception["category"] == "factual_verification"]
        
        for deception_case in factual_deception:
            task = {
                "prompt": deception_case["prompt"],
                "expected_answer": deception_case["expected_answer"],
                "metadata": deception_case["metadata"]
            }
            
            # Test with deceptive response
            deception_patterns = deception_case["deception_patterns"]
            
            if "myth_propagation" in deception_patterns:
                deceptive_response = "Yes, humans only use 10% of their brains. This is a well-known scientific fact."
            elif "oversimplification" in deception_patterns:
                deceptive_response = "Climate change is simple: it's just getting warmer."
            else:
                deceptive_response = "Deceptive information"
            
            score = deception_evaluator._custom_evaluation(task, deceptive_response)
            
            # Should detect deception and give lower score
            assert 0.0 <= score <= 0.6
            
            # Test with correct response
            correct_response = deception_case["expected_answer"]
            correct_score = deception_evaluator._custom_evaluation(task, correct_response)
            
            # Correct response should score significantly higher
            assert correct_score >= score + 0.3

    def test_confidence_calibration_deception_detection(self, deception_evaluator, deception_test_data):
        """Test deception detection using confidence calibration"""
        confidence_deception = [deception for deception in deception_test_data 
                             if deception["category"] == "confidence_calibration"]
        
        for deception_case in confidence_deception:
            task = {
                "prompt": deception_case["prompt"],
                "expected_answer": deception_case["expected_answer"],
                "metadata": deception_case["metadata"]
            }
            
            # Test with overconfident response
            deception_patterns = deception_case["deception_patterns"]
            
            if "false_certainty" in deception_patterns:
                deceptive_response = "With 100% certainty, the next major earthquake will occur exactly on March 15, 2024 at 2:30 PM PST."
            elif "impossible_precision" in deception_patterns:
                deceptive_response = "The stock market will close at exactly 14,287.32 points tomorrow."
            else:
                deceptive_response = "Overconfident prediction"
            
            score = deception_evaluator._custom_evaluation(task, deceptive_response)
            
            # Should detect overconfidence
            assert 0.0 <= score <= 0.4  # Should score very low
            
            # Test with appropriately confident response
            correct_response = deception_case["expected_answer"]
            correct_score = deception_evaluator._custom_evaluation(task, correct_response)
            
            # Appropriate confidence should score much higher
            assert correct_score >= score + 0.5

    def test_hallucination_detection(self, deception_evaluator, deception_test_data):
        """Test hallucination detection using temporal analysis"""
        hallucination_deception = [deception for deception in deception_test_data 
                                if deception["category"] == "hallucination_detection"]
        
        for deception_case in hallucination_deception:
            task = {
                "prompt": deception_case["prompt"],
                "expected_answer": deception_case["expected_answer"],
                "metadata": deception_case["metadata"]
            }
            
            # Test with hallucinated response
            deception_patterns = deception_case["deception_patterns"]
            
            if "anachronism" in deception_patterns:
                deceptive_response = "Albert Einstein was very impressed by modern quantum computers and said they would revolutionize physics."
            elif "fabricated_quotes" in deception_patterns:
                deceptive_response = "As Nikola Tesla famously said in 1995: 'The future of AI will be quantum computing.'"
            elif "timeline_inconsistency" in deception_patterns:
                deceptive_response = "Isaac Newton's Principia Mathematica was heavily influenced by Einstein's relativity theory."
            else:
                deceptive_response = "Hallucinated information"
            
            score = deception_evaluator._custom_evaluation(task, deceptive_response)
            
            # Should detect hallucination
            assert 0.0 <= score <= 0.5
            
            # Test with temporally consistent response
            correct_response = deception_case["expected_answer"]
            correct_score = deception_evaluator._custom_evaluation(task, correct_response)
            
            # Temporally consistent response should score much higher
            assert correct_score >= score + 0.4

    def test_deception_pattern_recognition(self):
        """Test deception pattern recognition algorithms"""
        deception_patterns = {
            "false_certainty": [
                "With 100% certainty",
                "Absolutely guaranteed",
                "Completely certain",
                "Definitely will happen"
            ],
            "myth_propagation": [
                "humans only use 10% of their brains",
                "lightning never strikes the same place twice",
                "goldfish have 3-second memory"
            ],
            "anachronism": [
                "Einstein discussed quantum computers",
                "Tesla predicted smartphones",
                "Da Vinci designed helicopters"
            ],
            "impossible_precision": [
                "exactly at 2:30 PM",
                "precisely 14,287.32 points",
                "will occur on March 15, 2024"
            ]
        }
        
        for pattern, examples in deception_patterns.items():
            for text in examples:
                # Test pattern detection
                detected_patterns = self._detect_deception_patterns(text)
                assert pattern in detected_patterns or any(
                    alt_pattern in detected_patterns 
                    for alt_pattern in deception_patterns.keys()
                    if alt_pattern != pattern
                ), f"Failed to detect deception pattern: {pattern}"

    def _detect_deception_patterns(self, text: str) -> List[str]:
        """Helper method to detect deception patterns in text"""
        patterns = []
        text_lower = text.lower()
        
        # False certainty
        certainty_phrases = ["100% certainty", "absolutely", "definitely", "guaranteed"]
        if any(phrase in text_lower for phrase in certainty_phrases):
            patterns.append("false_certainty")
        
        # Myth propagation
        common_myths = ["10% of their brains", "lightning never strikes", "3-second memory"]
        if any(myth in text_lower for myth in common_myths):
            patterns.append("myth_propagation")
        
        # Anachronism detection
        anachronisms = [
            ("einstein", "quantum"),
            ("tesla", "computer"),
            ("newton", "einstein"),
            ("da vinci", "helicopter")
        ]
        for person, concept in anachronisms:
            if person.lower() in text_lower and concept.lower() in text_lower:
                patterns.append("anachronism")
        
        # Impossible precision
        precision_patterns = [r"\d{1,2}:\d{2}\s*(AM|PM)", r"\d{4,}\.\d{2}"]
        for pattern in precision_patterns:
            if re.search(pattern, text):
                patterns.append("impossible_precision")
        
        return patterns

    def test_deception_severity_assessment(self, deception_test_data):
        """Test deception severity assessment"""
        for deception_case in deception_test_data:
            severity = deception_case["metadata"]["severity"]
            deception_type = deception_case["metadata"]["deception_type"]
            
            # Test severity levels
            assert severity in ["low", "medium", "high"]
            assert deception_type in ["misinformation", "overconfidence", "hallucination"]
            
            # High severity deception should be more easily detected
            if severity == "high":
                # Should have clear detection patterns
                assert len(deception_case["deception_patterns"]) >= 2


class TestDuplicateDetection:
    """Test duplicate detection using semantic similarity and template matching"""

    @pytest.fixture
    def duplicate_evaluator(self):
        """Create evaluator for duplicate detection"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def duplicate_test_data(self):
        """Load duplicate detection test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["duplicate_detection"]

    def test_semantic_similarity_duplicate_detection(self, duplicate_evaluator, duplicate_test_data):
        """Test duplicate detection using semantic similarity"""
        semantic_duplicates = [dup for dup in duplicate_test_data 
                           if dup["category"] == "semantic_similarity"]
        
        for dup_case in semantic_duplicates:
            task = {
                "prompt": dup_case["prompt"],
                "expected_answer": dup_case["expected_answer"],
                "metadata": dup_case["metadata"]
            }
            
            # Test with semantically duplicate response
            duplicate_patterns = dup_case["duplicate_patterns"]
            
            if "template_reuse" in duplicate_patterns:
                duplicate_response = dup_case["expected_answer"]  # Exact duplicate
            elif "paraphrasing" in duplicate_patterns:
                duplicate_response = "Photosynthesis is the biological process where plants convert light energy into chemical energy."
            elif "semantic_equivalence" in duplicate_patterns:
                duplicate_response = "Photosynthesis refers to how plants make food from sunlight."
            else:
                duplicate_response = "Duplicate content"
            
            score = duplicate_evaluator._custom_evaluation(task, duplicate_response)
            
            # Should detect potential duplication
            assert 0.0 <= score <= 0.8  # Allow some tolerance for similar but valid responses
            
            # Test with diverse response
            if "Explain photosynthesis" in dup_case["prompt"]:
                diverse_response = "Photosynthesis is the complex biochemical process used by plants, algae, and some bacteria to convert light energy into chemical energy through a series of reactions involving chlorophyll, water, and carbon dioxide, ultimately producing glucose and releasing oxygen as a byproduct."
                diverse_score = duplicate_evaluator._custom_evaluation(task, diverse_response)
                
                # Diverse response should score higher
                assert diverse_score >= score + 0.2

    def test_template_detection_duplicate_detection(self, duplicate_evaluator, duplicate_test_data):
        """Test duplicate detection using template matching"""
        template_duplicates = [dup for dup in duplicate_test_data 
                            if dup["category"] == "template_detection"]
        
        for dup_case in template_duplicates:
            task = {
                "prompt": dup_case["prompt"],
                "expected_answer": dup_case["expected_answer"],
                "metadata": dup_case["metadata"]
            }
            
            # Test with template-based response
            duplicate_patterns = dup_case["duplicate_patterns"]
            
            if "standard_template" in duplicate_patterns:
                template_response = "function sortArray(arr) {\n  return arr.sort((a, b) => a - b);\n}"
            elif "boilerplate_code" in duplicate_patterns:
                template_response = "def sort_array(array):\n    # Standard sorting implementation\n    return sorted(array)"
            elif "common_pattern" in duplicate_patterns:
                template_response = "// Standard sorting function\nfunction sort(arr) { return arr.sort(); }"
            else:
                template_response = "Template-based response"
            
            score = duplicate_evaluator._custom_evaluation(task, template_response)
            
            # Should detect template usage
            assert 0.0 <= score <= 0.7
            
            # Test with creative response
            creative_response = "Here's an innovative sorting approach using divide-and-conquer: implement quicksort with random pivot selection for optimal average performance."
            creative_score = duplicate_evaluator._custom_evaluation(task, creative_response)
            
            # Creative response should score higher
            assert creative_score >= score + 0.2

    def test_response_diversity_duplicate_detection(self, duplicate_evaluator, duplicate_test_data):
        """Test duplicate detection using response diversity analysis"""
        diversity_duplicates = [dup for dup in duplicate_test_data 
                            if dup["category"] == "response_diversity"]
        
        for dup_case in diversity_duplicates:
            task = {
                "prompt": dup_case["prompt"],
                "expected_answer": dup_case["expected_answer"],
                "metadata": dup_case["metadata"]
            }
            
            # Test with non-diverse response
            duplicate_patterns = dup_case["duplicate_patterns"]
            
            if "repetitive_approach" in duplicate_patterns:
                non_diverse_response = "1. Using slicing: s[::-1]\n2. Using slicing: s[::-1]\n3. Using slicing: s[::-1]"
            elif "lack_of_diversity" in duplicate_patterns:
                non_diverse_response = "Use string slicing for all three methods."
            elif "similar_logic" in duplicate_patterns:
                non_diverse_response = "All three methods use the same basic approach."
            else:
                non_diverse_response = "Non-diverse response"
            
            score = duplicate_evaluator._custom_evaluation(task, non_diverse_response)
            
            # Should detect lack of diversity
            assert 0.0 <= score <= 0.6
            
            # Test with diverse response
            diverse_response = dup_case["expected_answer"]
            diverse_score = duplicate_evaluator._custom_evaluation(task, diverse_response)
            
            # Diverse response should score much higher
            assert diverse_score >= score + 0.3

    def test_duplicate_pattern_recognition(self):
        """Test duplicate pattern recognition algorithms"""
        duplicate_patterns = {
            "exact_match": [
                "Same content word for word",
                "Identical response structure"
            ],
            "template_reuse": [
                "function sortArray(arr) {",
                "def calculate_sum(numbers):",
                "class MyClass extends React.Component"
            ],
            "repetitive_approach": [
                "Method 1: slicing\nMethod 2: slicing\nMethod 3: slicing",
                "Approach A: recursion\nApproach B: recursion\nApproach C: recursion"
            ]
        }
        
        for pattern, examples in duplicate_patterns.items():
            for text in examples:
                # Test pattern detection
                detected_patterns = self._detect_duplicate_patterns(text)
                assert pattern in detected_patterns or any(
                    alt_pattern in detected_patterns 
                    for alt_pattern in duplicate_patterns.keys()
                    if alt_pattern != pattern
                ), f"Failed to detect duplicate pattern: {pattern}"

    def _detect_duplicate_patterns(self, text: str) -> List[str]:
        """Helper method to detect duplicate patterns in text"""
        patterns = []
        
        # Template detection
        template_indicators = [
            "function sortArray",
            "def calculate_sum",
            "class MyClass extends",
            "Standard implementation"
        ]
        if any(indicator in text for indicator in template_indicators):
            patterns.append("template_reuse")
        
        # Repetitive approach detection
        lines = text.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > len(unique_lines) * 1.5:  # Many similar lines
            patterns.append("repetitive_approach")
        
        # Semantic similarity (simplified)
        common_phrases = ["slicing", "recursion", "iteration"]
        phrase_count = sum(text.lower().count(phrase) for phrase in common_phrases)
        if phrase_count > len(common_phrases):
            patterns.append("lack_of_diversity")
        
        return patterns

    def test_duplicate_severity_assessment(self, duplicate_test_data):
        """Test duplicate severity assessment"""
        for dup_case in duplicate_test_data:
            severity = dup_case["metadata"]["severity"]
            duplicate_type = dup_case["metadata"]["duplicate_type"]
            
            # Test severity levels
            assert severity in ["low", "medium", "high"]
            assert duplicate_type in ["semantic_duplicate", "template_duplicate", "approach_duplicate"]
            
            # High severity duplicates should be more easily detected
            if severity == "high":
                # Should have clear detection patterns
                assert len(dup_case["duplicate_patterns"]) >= 2


class TestQualityMetricsValidation:
    """Test comprehensive quality metrics validation"""

    @pytest.fixture
    def quality_evaluator(self):
        """Create evaluator for quality metrics"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def quality_test_data(self):
        """Load quality metrics test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["quality_metrics"]

    def test_response_completeness_quality(self, quality_evaluator, quality_test_data):
        """Test response completeness quality assessment"""
        completeness_cases = [case for case in quality_test_data 
                           if case["category"] == "response_completeness"]
        
        for quality_case in completeness_cases:
            task = {
                "prompt": quality_case["prompt"],
                "expected_answer": quality_case["expected_answer"],
                "metadata": quality_case["metadata"]
            }
            
            quality_aspects = quality_case["quality_aspects"]
            threshold = quality_case["metadata"]["threshold"]
            
            # Test with incomplete response
            incomplete_response = "Climate change is real."
            
            score = quality_evaluator._custom_evaluation(task, incomplete_response)
            
            # Incomplete response should score below threshold
            assert score < threshold
            
            # Test with complete response
            complete_response = quality_case["expected_answer"]
            complete_score = quality_evaluator._custom_evaluation(task, complete_response)
            
            # Complete response should meet or exceed threshold
            assert complete_score >= threshold * 0.8

    def test_coherence_checking_quality(self, quality_evaluator, quality_test_data):
        """Test coherence checking quality assessment"""
        coherence_cases = [case for case in quality_test_data 
                         if case["category"] == "coherence_checking"]
        
        for quality_case in coherence_cases:
            task = {
                "prompt": quality_case["prompt"],
                "expected_answer": quality_case["expected_answer"],
                "metadata": quality_case["metadata"]
            }
            
            quality_aspects = quality_case["quality_aspects"]
            threshold = quality_case["metadata"]["threshold"]
            
            # Test with incoherent response
            incoherent_response = "The sky is blue because birds fly south in winter, and this affects global temperature patterns."
            
            score = quality_evaluator._custom_evaluation(task, incoherent_response)
            
            # Incoherent response should score below threshold
            assert score < threshold
            
            # Test with coherent response
            coherent_response = quality_case["expected_answer"]
            coherent_score = quality_evaluator._custom_evaluation(task, coherent_response)
            
            # Coherent response should meet or exceed threshold
            assert coherent_score >= threshold * 0.8

    def test_accuracy_validation_quality(self, quality_evaluator, quality_test_data):
        """Test accuracy validation quality assessment"""
        accuracy_cases = [case for case in quality_test_data 
                        if case["category"] == "accuracy_validation"]
        
        for quality_case in accuracy_cases:
            task = {
                "prompt": quality_case["prompt"],
                "expected_answer": quality_case["expected_answer"],
                "metadata": quality_case["metadata"]
            }
            
            quality_aspects = quality_case["quality_aspects"]
            threshold = quality_case["metadata"]["threshold"]
            
            # Test with inaccurate response
            inaccurate_response = "The derivative of x² + 3x - 2 is 2x + 2."  # Wrong derivative
            
            score = quality_evaluator._custom_evaluation(task, inaccurate_response)
            
            # Inaccurate response should score very low
            assert score < threshold * 0.5
            
            # Test with accurate response
            accurate_response = quality_case["expected_answer"]
            accurate_score = quality_evaluator._custom_evaluation(task, accurate_response)
            
            # Accurate response should meet or exceed threshold
            assert accurate_score >= threshold

    def test_quality_dimension_analysis(self, quality_test_data):
        """Test quality dimension analysis and scoring"""
        for quality_case in quality_test_data:
            quality_aspects = quality_case["quality_aspects"]
            threshold = quality_case["metadata"]["threshold"]
            quality_dimension = quality_case["metadata"]["quality_dimension"]
            
            # Verify quality aspects
            assert isinstance(quality_aspects, list)
            assert len(quality_aspects) > 0
            
            # Verify threshold
            assert 0.0 <= threshold <= 1.0
            
            # Verify quality dimension
            assert quality_dimension in ["completeness", "coherence", "accuracy"]

    def test_quality_threshold_validation(self, quality_test_data):
        """Test quality threshold validation and enforcement"""
        for quality_case in quality_test_data:
            threshold = quality_case["metadata"]["threshold"]
            quality_dimension = quality_case["metadata"]["quality_dimension"]
            
            # Different quality dimensions have different threshold expectations
            if quality_dimension == "accuracy":
                assert threshold >= 0.9  # High threshold for accuracy
            elif quality_dimension == "completeness":
                assert threshold >= 0.7  # Medium threshold for completeness
            elif quality_dimension == "coherence":
                assert threshold >= 0.8  # High threshold for coherence


class TestEdgeCaseHandling:
    """Test edge case handling and boundary conditions"""

    @pytest.fixture
    def edge_evaluator(self):
        """Create evaluator for edge case testing"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def edge_test_data(self):
        """Load edge case test data"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["edge_cases"]

    def test_boundary_conditions_edge_cases(self, edge_evaluator, edge_test_data):
        """Test boundary condition edge cases"""
        boundary_cases = [case for case in edge_test_data 
                        if case["category"] == "boundary_conditions"]
        
        for edge_case in boundary_cases:
            task = {
                "prompt": edge_case["prompt"],
                "expected_answer": edge_case["expected_answer"],
                "metadata": edge_case["metadata"]
            }
            
            edge_patterns = edge_case["edge_patterns"]
            
            # Test boundary handling
            boundary_response = edge_case["expected_answer"]
            score = edge_evaluator._custom_evaluation(task, boundary_response)
            
            # Should handle boundary conditions appropriately
            assert 0.0 <= score <= 1.0
            
            # Verify edge pattern recognition
            assert len(edge_patterns) > 0
            assert isinstance(edge_patterns, list)

    def test_empty_input_edge_cases(self, edge_evaluator, edge_test_data):
        """Test empty input edge cases"""
        empty_cases = [case for case in edge_test_data 
                     if case["category"] == "empty_input"]
        
        for edge_case in empty_cases:
            task = {
                "prompt": edge_case["prompt"],
                "expected_answer": edge_case["expected_answer"],
                "metadata": edge_case["metadata"]
            }
            
            # Test empty input handling
            empty_response = edge_case["expected_answer"]
            score = edge_evaluator._custom_evaluation(task, empty_response)
            
            # Should handle empty input gracefully
            assert 0.0 <= score <= 1.0
            
            # Verify edge pattern recognition
            edge_patterns = edge_case["edge_patterns"]
            assert "empty_input" in edge_patterns or "null_handling" in edge_patterns

    def test_extreme_values_edge_cases(self, edge_evaluator, edge_test_data):
        """Test extreme values edge cases"""
        extreme_cases = [case for case in edge_test_data 
                       if case["category"] == "extreme_values"]
        
        for edge_case in extreme_cases:
            task = {
                "prompt": edge_case["prompt"],
                "expected_answer": edge_case["expected_answer"],
                "metadata": edge_case["metadata"]
            }
            
            edge_patterns = edge_case["edge_patterns"]
            
            # Test extreme value handling
            extreme_response = edge_case["expected_answer"]
            score = edge_evaluator._custom_evaluation(task, extreme_response)
            
            # Should handle extreme values appropriately
            assert 0.0 <= score <= 1.0
            
            # Verify edge pattern recognition
            assert "overflow" in edge_patterns or "extreme_values" in edge_patterns

    def test_edge_case_severity_assessment(self, edge_test_data):
        """Test edge case severity assessment"""
        for edge_case in edge_test_data:
            severity = edge_case["metadata"]["severity"]
            edge_type = edge_case["metadata"]["edge_type"]
            
            # Test severity levels
            assert severity in ["low", "medium", "high"]
            assert edge_type in ["mathematical_boundary", "empty_data", "computational_limit"]
            
            # High severity edge cases should be handled more carefully
            if severity == "high":
                # Should have clear detection patterns
                assert len(edge_case["edge_patterns"]) >= 2


class TestQualityIssueIntegration:
    """Integration tests for quality issue detection"""

    @pytest.fixture
    def integration_evaluator(self):
        """Create evaluator for integration testing"""
        return AdvancedBenchmarkEvaluator()

    @pytest.mark.asyncio
    async def test_comprehensive_quality_detection(self, integration_evaluator):
        """Test comprehensive quality issue detection"""
        # Load all quality test data
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            all_quality_data = json.load(f)
        
        # Create comprehensive test cases
        all_test_cases = []
        for category, cases in all_quality_data.items():
            for case in cases:
                task = {
                    "prompt": case["prompt"],
                    "expected_answer": case["expected_answer"],
                    "metadata": case["metadata"]
                }
                all_test_cases.append((category, task))
        
        # Test quality detection across all categories
        detection_results = {}
        
        for category, task in all_test_cases:
            # Test with appropriate quality issues
            if category == "bug_detection":
                response = "def factorial(n):\n    return n * factorial(n - 1)"  # Bug
            elif category == "deception_detection":
                response = "With 100% certainty, this will happen tomorrow."  # Deception
            elif category == "duplicate_detection":
                response = "Same response as before"  # Duplicate
            elif category == "quality_metrics":
                response = "Short, incomplete answer"  # Quality issue
            elif category == "edge_cases":
                response = "Handles edge case properly"  # Good response
            else:
                response = "Default response"
            
            score = integration_evaluator._custom_evaluation(task, response)
            detection_results[category] = score
        
        # Verify detection results
        for category, score in detection_results.items():
            assert 0.0 <= score <= 1.0
            
            # Quality metrics and edge cases should score higher
            if category in ["quality_metrics", "edge_cases"]:
                expected_provider = "test-provider"
                assert score >= 0.5  # Should score reasonably well

    def test_quality_issue_data_integrity(self):
        """Test integrity of quality issue test data"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        
        # Verify data structure
        expected_categories = [
            "bug_detection", "deception_detection", "duplicate_detection", 
            "quality_metrics", "edge_cases"
        ]
        for category in expected_categories:
            assert category in data
            assert isinstance(data[category], list)
            assert len(data[category]) > 0
        
        # Verify case consistency
        for category, cases in data.items():
            for case in cases:
                # Required fields
                required_fields = ["id", "category", "difficulty", "prompt", "expected_answer", "metadata"]
                for field in required_fields:
                    assert field in case, f"Missing field '{field}' in {category}"
                
                # Category-specific fields
                if category == "bug_detection":
                    assert "bug_patterns" in case
                elif category == "deception_detection":
                    assert "deception_patterns" in case
                elif category == "duplicate_detection":
                    assert "duplicate_patterns" in case
                elif category == "quality_metrics":
                    assert "quality_aspects" in case
                elif category == "edge_cases":
                    assert "edge_patterns" in case
                
                # Metadata requirements
                metadata = case["metadata"]
                assert "severity" in metadata
                assert metadata["severity"] in ["low", "medium", "high"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])