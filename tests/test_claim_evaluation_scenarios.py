"""
Domain-Specific Claim Evaluation Tests
Tests scientific claims, logical reasoning, evidence validation, and domain-specific scenarios
using DeepEval metrics for comprehensive evaluation
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EvaluationFramework, create_conjecture_wrapper
from src.benchmarking.deepeval_integration import AdvancedBenchmarkEvaluator


class TestScientificClaimsEvaluation:
    """Test evaluation of scientific claims across multiple domains"""

    @pytest.fixture
    def scientific_framework(self):
        """Create evaluation framework for scientific claims"""
        return EvaluationFramework()

    @pytest.fixture
    def scientific_test_data(self):
        """Load scientific claim test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["scientific_claims"]

    @pytest.mark.asyncio
    async def test_climate_science_claims(self, scientific_framework, scientific_test_data):
        """Test climate science claim evaluation"""
        climate_claims = [claim for claim in scientific_test_data 
                         if claim["category"] == "climate_science"]
        
        for claim in climate_claims:
            test_case = scientific_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "climate_science",
                    "claim_type": "factual"
                }
            )
            
            # Mock evaluation for testing
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await scientific_framework.evaluate_provider(
                    "ibm/granite-4-h-tiny", [test_case], use_conjecture=False
                )
                
                # Verify scientific claim evaluation
                assert result["success"] is True
                assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.8
                assert "faithfulness" in result["metrics_results"]
                assert "answer_relevancy" in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_physics_claims(self, scientific_framework, scientific_test_data):
        """Test physics claim evaluation"""
        physics_claims = [claim for claim in scientific_test_data 
                        if claim["category"] == "physics"]
        
        for claim in physics_claims:
            test_case = scientific_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "physics",
                    "claim_type": "factual"
                }
            )
            
            # Test with different providers
            providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
            
            for provider in providers:
                with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                    mock_wrapper = AsyncMock()
                    mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                    mock_create.return_value = mock_wrapper
                    
                    result = await scientific_framework.evaluate_provider(
                        provider, [test_case], use_conjecture=False
                    )
                    
                    # Physics claims should have high exact match scores
                    assert result["success"] is True
                    assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.9

    @pytest.mark.asyncio
    async def test_biology_claims(self, scientific_framework, scientific_test_data):
        """Test biology claim evaluation"""
        biology_claims = [claim for claim in scientific_test_data 
                         if claim["category"] == "biology"]
        
        for claim in biology_claims:
            test_case = scientific_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "biology",
                    "claim_type": "explanatory"
                }
            )
            
            # Test with Conjecture enhancement
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                # Test both direct and conjecture-enhanced
                direct_result = await scientific_framework.evaluate_provider(
                    "zai/GLM-4.6", [test_case], use_conjecture=False
                )
                
                conjecture_result = await scientific_framework.evaluate_provider(
                    "zai/GLM-4.6", [test_case], use_conjecture=True
                )
                
                # Conjecture should improve explanatory claims
                assert direct_result["success"] is True
                assert conjecture_result["success"] is True
                assert conjecture_result["overall_score"] >= direct_result["overall_score"]

    @pytest.mark.asyncio
    async def test_medicine_claims(self, scientific_framework, scientific_test_data):
        """Test medicine claim evaluation with nuance"""
        medicine_claims = [claim for claim in scientific_test_data 
                         if claim["category"] == "medicine"]
        
        for claim in medicine_claims:
            test_case = scientific_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "medicine",
                    "claim_type": "conditional"
                }
            )
            
            # Test with advanced evaluator for medical nuance
            evaluator = AdvancedBenchmarkEvaluator()
            
            # Test nuanced response
            nuanced_response = claim["expected_answer"]
            score = evaluator._custom_evaluation({
                "prompt": claim["prompt"],
                "expected_answer": claim["expected_answer"],
                "metadata": claim["metadata"]
            }, nuanced_response)
            
            # Medical claims require high accuracy
            assert score >= claim["metadata"]["confidence_threshold"] * 0.8

    def test_scientific_claim_validation(self, scientific_test_data):
        """Test scientific claim structure and validation"""
        for claim in scientific_test_data:
            # Verify claim structure
            assert "id" in claim
            assert "category" in claim
            assert "difficulty" in claim
            assert "prompt" in claim
            assert "expected_answer" in claim
            assert "metadata" in claim
            
            # Verify metadata structure
            metadata = claim["metadata"]
            assert "domain" in metadata
            assert "claim_type" in metadata
            assert "confidence_threshold" in metadata
            assert 0.0 <= metadata["confidence_threshold"] <= 1.0
            
            # Verify content quality
            assert len(claim["prompt"]) > 10
            assert len(claim["expected_answer"]) > 20
            assert claim["difficulty"] in ["easy", "medium", "hard"]


class TestLogicalReasoningEvaluation:
    """Test evaluation of logical reasoning and mathematical proofs"""

    @pytest.fixture
    def logical_framework(self):
        """Create evaluation framework for logical reasoning"""
        return EvaluationFramework()

    @pytest.fixture
    def logical_test_data(self):
        """Load logical reasoning test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["logical_reasoning"]

    @pytest.mark.asyncio
    async def test_mathematical_proofs(self, logical_framework, logical_test_data):
        """Test mathematical proof evaluation"""
        proof_claims = [claim for claim in logical_test_data 
                        if claim["category"] == "mathematics"]
        
        for claim in proof_claims:
            test_case = logical_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "mathematics",
                    "claim_type": "proof"
                }
            )
            
            # Test with Granite Tiny (good for mathematical reasoning)
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await logical_framework.evaluate_provider(
                    "ibm/granite-4-h-tiny", [test_case], use_conjecture=False
                )
                
                # Mathematical proofs should have high exact match
                assert result["success"] is True
                assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.85

    @pytest.mark.asyncio
    async def test_logic_puzzles(self, logical_framework, logical_test_data):
        """Test logic puzzle evaluation"""
        puzzle_claims = [claim for claim in logical_test_data 
                        if claim["category"] == "logic_puzzles"]
        
        for claim in puzzle_claims:
            test_case = logical_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "logic",
                    "claim_type": "deductive"
                }
            )
            
            # Test with GPT-OSS-20B (good for logical reasoning)
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await logical_framework.evaluate_provider(
                    "openrouter/gpt-oss-20b", [test_case], use_conjecture=False
                )
                
                # Logic puzzles should have high faithfulness
                assert result["success"] is True
                assert "faithfulness" in result["metrics_results"]
                assert result["metrics_results"]["faithfulness"]["score"] >= 0.8

    @pytest.mark.asyncio
    async def test_problem_solving(self, logical_framework, logical_test_data):
        """Test problem-solving evaluation"""
        problem_claims = [claim for claim in logical_test_data 
                         if claim["category"] == "mathematics" and 
                         claim["metadata"]["claim_type"] == "problem_solving"]
        
        for claim in problem_claims:
            test_case = logical_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "mathematics",
                    "claim_type": "problem_solving"
                }
            )
            
            # Test with GLM-4.6 (good for problem solving)
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await logical_framework.evaluate_provider(
                    "zai/GLM-4.6", [test_case], use_conjecture=False
                )
                
                # Problem solving should have good overall scores
                assert result["success"] is True
                assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.8

    def test_logical_reasoning_validation(self, logical_test_data):
        """Test logical reasoning claim structure"""
        for claim in logical_test_data:
            # Verify claim structure
            assert "id" in claim
            assert "category" in claim
            assert "difficulty" in claim
            assert "prompt" in claim
            assert "expected_answer" in claim
            assert "metadata" in claim
            
            # Verify logical claim properties
            assert len(claim["expected_answer"]) > 50  # Should be detailed
            assert claim["difficulty"] in ["easy", "medium", "hard"]
            
            # Verify metadata
            metadata = claim["metadata"]
            assert "claim_type" in metadata
            assert metadata["claim_type"] in ["proof", "deductive", "problem_solving"]


class TestEvidenceValidationEvaluation:
    """Test evaluation of evidence validation and fact-checking"""

    @pytest.fixture
    def evidence_framework(self):
        """Create evaluation framework for evidence validation"""
        return EvaluationFramework()

    @pytest.fixture
    def evidence_test_data(self):
        """Load evidence validation test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["evidence_validation"]

    @pytest.mark.asyncio
    async def test_fact_checking(self, evidence_framework, evidence_test_data):
        """Test fact-checking evaluation"""
        fact_claims = [claim for claim in evidence_test_data 
                      if claim["category"] == "fact_checking"]
        
        for claim in fact_claims:
            test_case = evidence_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "history",
                    "claim_type": "factual",
                    "sources": claim["metadata"].get("sources", [])
                }
            )
            
            # Test with multiple providers for fact-checking
            providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
            
            for provider in providers:
                with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                    mock_wrapper = AsyncMock()
                    mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                    mock_create.return_value = mock_wrapper
                    
                    result = await evidence_framework.evaluate_provider(
                        provider, [test_case], use_conjecture=False
                    )
                    
                    # Fact-checking should have high faithfulness
                    assert result["success"] is True
                    assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.9
                    assert "faithfulness" in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_source_credibility(self, evidence_framework, evidence_test_data):
        """Test source credibility evaluation"""
        credibility_claims = [claim for claim in evidence_test_data 
                           if claim["category"] == "source_credibility"]
        
        for claim in credibility_claims:
            test_case = evidence_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "medicine",
                    "claim_type": "evidence_evaluation",
                    "sources": claim["metadata"].get("sources", [])
                }
            )
            
            # Test with advanced evaluator for credibility assessment
            evaluator = AdvancedBenchmarkEvaluator()
            
            # Test credibility response
            credibility_response = claim["expected_answer"]
            score = evaluator._custom_evaluation({
                "prompt": claim["prompt"],
                "expected_answer": claim["expected_answer"],
                "metadata": claim["metadata"]
            }, credibility_response)
            
            # Source credibility should have high scores
            assert score >= claim["metadata"]["confidence_threshold"] * 0.85

    @pytest.mark.asyncio
    async def test_citation_verification(self, evidence_framework):
        """Test citation verification evaluation"""
        # Create citation verification test case
        citation_test = {
            "prompt": "According to WHO, what is the global vaccination coverage rate?",
            "expected_answer": "According to WHO data, global vaccination coverage is approximately 85% for basic vaccines.",
            "metadata": {
                "domain": "public_health",
                "claim_type": "citation_verification",
                "sources": ["WHO", "UNICEF", "CDC"],
                "confidence_threshold": 0.95
            }
        }
        
        test_case = evidence_framework.create_test_case(
            input_text=citation_test["prompt"],
            expected_output=citation_test["expected_answer"],
            additional_metadata=citation_test["metadata"]
        )
        
        # Test with Conjecture enhancement for better citation handling
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value=citation_test["expected_answer"])
            mock_create.return_value = mock_wrapper
            
            direct_result = await evidence_framework.evaluate_provider(
                "openrouter/gpt-oss-20b", [test_case], use_conjecture=False
            )
            
            conjecture_result = await evidence_framework.evaluate_provider(
                "openrouter/gpt-oss-20b", [test_case], use_conjecture=True
            )
            
            # Conjecture should improve citation verification
            assert direct_result["success"] is True
            assert conjecture_result["success"] is True
            assert conjecture_result["overall_score"] >= direct_result["overall_score"]

    def test_evidence_validation_structure(self, evidence_test_data):
        """Test evidence validation claim structure"""
        for claim in evidence_test_data:
            # Verify claim structure
            assert "id" in claim
            assert "category" in claim
            assert "difficulty" in claim
            assert "prompt" in claim
            assert "expected_answer" in claim
            assert "metadata" in claim
            
            # Verify evidence-specific properties
            metadata = claim["metadata"]
            assert "evidence_required" in metadata
            assert metadata["evidence_required"] is True
            assert "sources" in metadata or "confidence_threshold" in metadata
            
            # Verify content quality
            assert len(claim["expected_answer"]) > 30  # Should be detailed


class TestDomainSpecificEvaluation:
    """Test evaluation across different domains (technology, business, humanities)"""

    @pytest.fixture
    def domain_framework(self):
        """Create evaluation framework for domain-specific claims"""
        return EvaluationFramework()

    @pytest.fixture
    def domain_test_data(self):
        """Load domain-specific test cases"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        return data["domain_specific"]

    @pytest.mark.asyncio
    async def test_technology_domain(self, domain_framework, domain_test_data):
        """Test technology domain evaluation"""
        tech_claims = [claim for claim in domain_test_data 
                       if claim["category"] == "technology"]
        
        for claim in tech_claims:
            test_case = domain_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "technology",
                    "claim_type": "explanatory"
                }
            )
            
            # Test with all providers for technology questions
            providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
            
            for provider in providers:
                with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                    mock_wrapper = AsyncMock()
                    mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                    mock_create.return_value = mock_wrapper
                    
                    result = await domain_framework.evaluate_provider(
                        provider, [test_case], use_conjecture=False
                    )
                    
                    # Technology explanations should be clear and relevant
                    assert result["success"] is True
                    assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.8
                    assert "answer_relevancy" in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_business_domain(self, domain_framework, domain_test_data):
        """Test business domain evaluation"""
        business_claims = [claim for claim in domain_test_data 
                         if claim["category"] == "business"]
        
        for claim in business_claims:
            test_case = domain_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "business",
                    "claim_type": "definitional"
                }
            )
            
            # Test with GLM-4.6 for business concepts
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await domain_framework.evaluate_provider(
                    "zai/GLM-4.6", [test_case], use_conjecture=False
                )
                
                # Business definitions should be precise
                assert result["success"] is True
                assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.9
                assert "exact_match" in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_humanities_domain(self, domain_framework, domain_test_data):
        """Test humanities domain evaluation"""
        humanities_claims = [claim for claim in domain_test_data 
                           if claim["category"] == "humanities"]
        
        for claim in humanities_claims:
            test_case = domain_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata={
                    **claim["metadata"],
                    "domain": "philosophy",
                    "claim_type": "analytical"
                }
            )
            
            # Test with GPT-OSS-20B for humanities analysis
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value=claim["expected_answer"])
                mock_create.return_value = mock_wrapper
                
                result = await domain_framework.evaluate_provider(
                    "openrouter/gpt-oss-20b", [test_case], use_conjecture=False
                )
                
                # Humanities analysis should be nuanced
                assert result["success"] is True
                assert result["overall_score"] >= claim["metadata"]["confidence_threshold"] * 0.7
                assert "summarization" in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_cross_domain_comparison(self, domain_framework, domain_test_data):
        """Test cross-domain provider performance comparison"""
        # Select one claim from each domain
        cross_domain_claims = {}
        for claim in domain_test_data:
            domain = claim["category"]
            if domain not in cross_domain_claims:
                cross_domain_claims[domain] = claim
        
        test_cases = []
        for claim in cross_domain_claims.values():
            test_case = domain_framework.create_test_case(
                input_text=claim["prompt"],
                expected_output=claim["expected_answer"],
                additional_metadata=claim["metadata"]
            )
            test_cases.append(test_case)
        
        # Test all providers across all domains
        providers = ["ibm/granite-4-h-tiny", "zai/GLM-4.6", "openrouter/gpt-oss-20b"]
        
        provider_scores = {}
        
        for provider in providers:
            with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value="Cross-domain response")
                mock_create.return_value = mock_wrapper
                
                result = await domain_framework.evaluate_provider(
                    provider, test_cases, use_conjecture=False
                )
                
                provider_scores[provider] = result["overall_score"]
        
        # Verify all providers performed reasonably
        for provider, score in provider_scores.items():
            assert score >= 0.5  # Minimum acceptable performance

    def test_domain_specific_validation(self, domain_test_data):
        """Test domain-specific claim structure"""
        for claim in domain_test_data:
            # Verify claim structure
            assert "id" in claim
            assert "category" in claim
            assert "difficulty" in claim
            assert "prompt" in claim
            assert "expected_answer" in claim
            assert "metadata" in claim
            
            # Verify domain-specific properties
            metadata = claim["metadata"]
            assert "domain" in metadata
            assert "claim_type" in metadata
            assert "confidence_threshold" in metadata
            
            # Verify content quality
            assert len(claim["prompt"]) > 10
            assert len(claim["expected_answer"]) > 20


class TestClaimEvaluationIntegration:
    """Integration tests for comprehensive claim evaluation"""

    @pytest.fixture
    def integration_framework(self):
        """Create framework for integration testing"""
        return EvaluationFramework()

    @pytest.mark.asyncio
    async def test_comprehensive_claim_evaluation(self, integration_framework):
        """Test comprehensive evaluation across all claim types"""
        # Load all test data
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            all_claims = json.load(f)
        
        # Create test cases from all categories
        all_test_cases = []
        for category, claims in all_claims.items():
            for claim in claims:
                test_case = integration_framework.create_test_case(
                    input_text=claim["prompt"],
                    expected_output=claim["expected_answer"],
                    additional_metadata=claim["metadata"]
                )
                all_test_cases.append(test_case)
        
        # Test with all providers
        providers = [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6", 
            "openrouter/gpt-oss-20b"
        ]
        
        # Mock evaluation for integration testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Integration test response")
            mock_create.return_value = mock_wrapper
            
            # Run comprehensive evaluation
            results = await integration_framework.evaluate_multiple_providers(
                providers, all_test_cases[:10], compare_conjecture=True  # Limit for testing
            )
            
            # Verify comprehensive evaluation
            assert "providers" in results
            assert "comparison" in results
            assert len(results["providers"]) == len(providers) * 2  # direct + conjecture
            
            # Verify comparison data
            comparison = results["comparison"]
            assert "best_overall" in comparison
            assert "improvements" in comparison

    @pytest.mark.asyncio
    async def test_claim_evaluation_error_handling(self, integration_framework):
        """Test error handling in claim evaluation"""
        # Create problematic test cases
        error_test_cases = [
            integration_framework.create_test_case(
                input_text="",  # Empty prompt
                expected_output="",
                additional_metadata={"error_test": "empty_input"}
            ),
            integration_framework.create_test_case(
                input_text="Normal claim",
                expected_answer="Normal answer",
                additional_metadata={"error_test": "normal"}
            )
        ]
        
        # Mock wrapper that handles errors
        async def mock_generate_with_error_handling(prompt):
            if not prompt.strip():
                raise ValueError("Empty prompt detected")
            return "Normal response"
        
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(side_effect=mock_generate_with_error_handling)
            mock_create.return_value = mock_wrapper
            
            # Test error handling
            result = await integration_framework.evaluate_provider(
                "error-test-provider", error_test_cases, use_conjecture=False
            )
            
            # Should handle errors gracefully
            assert "error" in result or result["success"] is False

    def test_claim_data_integrity(self):
        """Test integrity of claim test data"""
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            data = json.load(f)
        
        # Verify data structure
        expected_categories = ["scientific_claims", "logical_reasoning", "evidence_validation", "domain_specific"]
        for category in expected_categories:
            assert category in data
            assert isinstance(data[category], list)
            assert len(data[category]) > 0
        
        # Verify claim consistency
        for category, claims in data.items():
            for claim in claims:
                # Required fields
                required_fields = ["id", "category", "difficulty", "prompt", "expected_answer", "metadata"]
                for field in required_fields:
                    assert field in claim, f"Missing field '{field}' in {category}"
                
                # Metadata requirements
                metadata = claim["metadata"]
                required_metadata = ["domain", "claim_type", "confidence_threshold"]
                for meta_field in required_metadata:
                    assert meta_field in metadata, f"Missing metadata '{meta_field}' in {claim['id']}"
                
                # Value constraints
                assert 0.0 <= metadata["confidence_threshold"] <= 1.0
                assert claim["difficulty"] in ["easy", "medium", "hard"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])