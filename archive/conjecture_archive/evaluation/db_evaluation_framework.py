Vector Database Evaluation Framework for Conjecture
Implements 40-point rubric for systematic comparison
"""

import time
import random
import string
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from ..core.basic_models import BasicClaim, ClaimState, ClaimType


@dataclass
class EvaluationResult:
    """Single evaluation metric result"""
    metric_name: str
    score: float
    max_score: float
    details: str
    passed_threshold: bool


@dataclass
class DatabaseEvaluation:
    """Complete evaluation for a database implementation"""
    database_name: str
    total_score: float
    max_score: float
    percentage: float
    results: List[EvaluationResult]
    performance_summary: Dict[str, float]
    recommendation: str


class VectorDatabaseInterface(ABC):
    """Abstract interface for all vector databases to test"""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize database with configuration"""
        pass

    @abstractmethod
    def add_claim(self, claim: BasicClaim) -> bool:
        """Add a single claim"""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[BasicClaim]:
        """Retrieve claim by ID"""
        pass

    @abstractmethod
    def search_similar(self, query: str, limit: int = 5) -> List[Tuple[BasicClaim, float]]:
        """Search for similar claims with similarity scores"""
        pass

    @abstractmethod
    def filter_claims(self, **filters) -> List[BasicClaim]:
        """Filter claims by metadata"""
        pass

    @abstractmethod
    def batch_add_claims(self, claims: List[BasicClaim]) -> bool:
        """Add multiple claims efficiently"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all data"""
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources"""
        pass


class DatabaseEvaluator:
    """Systematic evaluator for vector databases"""

    def __init__(self):
        self.test_claims = self._generate_test_dataset()
        self.large_dataset = self._generate_large_dataset(10000)

    def _generate_test_dataset(self) -> List[BasicClaim]:
        """Generate diverse test dataset covering all claim types and states"""
        claims = []

        # Test claims covering different types and content
        test_data = [
            ("quantum physics explains quantum entanglement", [ClaimType.CONCEPT], 0.95),
            ("Shakespeare wrote Hamlet around 1600", [ClaimType.REFERENCE], 0.98),
            ("Machine learning enables pattern recognition", [ClaimType.SKILL], 0.87),
            ("The Earth orbits the Sun annually", [ClaimType.CONCEPT], 0.99),
            ("Python is widely used for data science", [ClaimType.EXAMPLE], 0.92),
            ("Neural networks learn through backpropagation", [ClaimType.CONCEPT], 0.90),
            ("Einstein developed general relativity", [ClaimType.THESIS], 0.96),
            ("React uses virtual DOM for performance", [ClaimType.EXAMPLE], 0.85),
            ("Photosynthesis converts light to energy", [ClaimType.CONCEPT], 0.97),
            ("Git tracks changes in version control", [ClaimType.SKILL], 0.93)
        ]

        states = list(ClaimState)

        for i, (content, claim_types, confidence) in enumerate(test_data):
            claim = BasicClaim(
                id=f"test_claim_{i+1}",
                content=content,
                confidence=confidence,
                type=claim_types,
                tags=self._generate_relevant_tags(content),
                state=random.choice(states),
                created=datetime.now()
            )
            claims.append(claim)

        return claims

    def _generate_large_dataset(self, size: int) -> List[BasicClaim]:
        """Generate large dataset for performance testing"""
        claims = []
        claim_types = list(ClaimType)
        states = list(ClaimState)

        # Templates for realistic claim content
        templates = [
            "{} is a fundamental concept in {}",
            "{} demonstrates the principles of {}",
            "{} represents an important advancement in {}",
            "{} is commonly used in {} applications",
            "{} provides evidence for {} theory"
        ]

        domains = [
            "computer science", "physics", "mathematics", "biology",
            "chemistry", "engineering", "philosophy", "economics",
            "psychology", "literature", "history", "linguistics"
        ]

        for i in range(size):
            template = random.choice(templates)
            domain = random.choice(domains)

            # Generate semi-realistic content
            subject = ''.join(random.choices(string.ascii_uppercase, k=3))
            content = template.format(f"{subject}-system", domain)

            claim = BasicClaim(
                id=f"perf_claim_{i}",
                content=content,
                confidence=random.uniform(0.3, 0.95),
                type=[random.choice(claim_types)],
                tags=[domain, "performance-test"],
                state=random.choice(states),
                created=datetime.now()
            )
            claims.append(claim)

        return claims

    def _generate_relevant_tags(self, content: str) -> List[str]:
        """Generate relevant tags based on content"""
        tag_mapping = {
            "quantum": ["Quantum-Physics", "Science"],
            "shakespeare": ["Shakespeare", "Literature"],
            "machine learning": ["AI-Research", "Computer-Science"],
            "earth": ["Astronomy", "Science"],
            "python": ["Programming", "Software-Engineering"],
            "neural": ["AI-Research", "Deep-Learning"],
            "einstein": ["Physics", "Relativity"],
            "react": ["Web-Development", "JavaScript"],
            "photosynthesis": ["Biology", "Botany"],
            "git": ["Software-Engineering", "Version-Control"]
        }

        tags = []
        content_lower = content.lower()

        for keyword, related_tags in tag_mapping.items():
            if keyword in content_lower:
                tags.extend(related_tags)

        return list(set(tags))  # Remove duplicates

    def evaluate_database(self, db_impl: VectorDatabaseInterface, db_name: str) -> DatabaseEvaluation:
        """Run complete evaluation on a database implementation"""
        print(f"\n=== Evaluating {db_name} ===")
        results = []
        performance_summary = {}

        try:
            # Rubric Criterion 1: Connection & Configuration (10 points)
            result1 = self._test_connection_configuration(db_impl)
            results.append(result1)

            # Rubric Criterion 2: CRUD Operations (10 points)
            result2 = self._test_crud_operations(db_impl)
            results.append(result2)

            # Rubric Criterion 3: Performance Requirements (10 points)
            result3, perf_data = self._test_performance_requirements(db_impl)
            results.append(result3)
            performance_summary.update(perf_data)

            # Rubric Criterion 4: Schema Validation (5 points)
            result4 = self._test_schema_validation(db_impl)
            results.append(result4)

            # Rubric Criterion 5: Integration Workflow (5 points)
            result5 = self._test_integration_workflow(db_impl)
            results.append(result5)

        except Exception as e:
            print(f"Evaluation failed for {db_name}: {e}")
            # Add failure result
            results.append(EvaluationResult(
                "Overall Evaluation",
                0,
                40,
                f"Critical error during evaluation: {e}",
                False
            ))

        # Calculate total score
        total_score = sum(r.score for r in results)
        max_score = sum(r.max_score for r in results)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0

        # Generate recommendation
        recommendation = self._generate_recommendation(percentage, results, performance_summary)

        return DatabaseEvaluation(
            database_name=db_name,
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            results=results,
            performance_summary=performance_summary,
            recommendation=recommendation
        )

    def _test_connection_configuration(self, db_impl: VectorDatabaseInterface) -> EvaluationResult:
        """Test database connection and configuration setup"""
        score = 10
        issues = []

        try:
            # Test basic initialization
            stats = db_impl.get_stats()
            if not stats:
                score -= 3
                issues.append("Database stats not available")

            # Test persistence
            test_claim = self.test_claims[0]
            added = db_impl.add_claim(test_claim)
            if not added:
                score -= 4
                issues.append("Cannot add claims")

            # Test retrieval
            retrieved = db_impl.get_claim(test_claim.id)
            if not retrieved or retrieved.id != test_claim.id:
                score -= 3
                issues.append("Cannot retrieve claims")

        except Exception as e:
            score = 0
            issues.append(f"Connection error: {e}")

        return EvaluationResult(
            "Connection & Configuration",
            score,
            10,
            "Passed" if score >= 8 else f"Issues: {', '.join(issues)}",
            score >= 8
        )

    def _test_crud_operations(self, db_impl: VectorDatabaseInterface) -> EvaluationResult:
        """Test Create, Read, Update, Delete operations"""
        score = 10
        issues = []

        try:
            # Test Create
            test_claim = self.test_claims[1]
            create_success = db_impl.add_claim(test_claim)
            if not create_success:
                score -= 2
                issues.append("Create operation failed")

            # Test Read
            retrieved = db_impl.get_claim(test_claim.id)
            if not retrieved or retrieved.content != test_claim.content:
                score -= 2
                issues.append("Read operation failed")

            # Test Update (simulate by modifying and re-adding)
            updated_claim = BasicClaim(
                id=test_claim.id,
                content=test_claim.content + " [UPDATED]",
                confidence=test_claim.confidence,
                type=test_claim.type,
                tags=test_claim.tags,
                state=ClaimState.VALIDATED,
                created=test_claim.created
            )
            update_success = db_impl.add_claim(updated_claim)  # Mock update
            if not update_success:
                score -= 2
                issues.append("Update operation failed")

            # Test Delete
            delete_success = db_impl.clear_all()  # Mock delete
            if not delete_success:
                score -= 2
                issues.append("Delete operation failed")

            # Test Batch Operations
            batch_success = db_impl.batch_add_claims(self.test_claims[:5])
            if not batch_success:
                score -= 2
                issues.append("Batch operations failed")

        except Exception as e:
            score = 0
            issues.append(f"CRUD error: {e}")

        return EvaluationResult(
            "CRUD Operations",
            score,
            10,
            "Passed" if score >= 8 else f"Issues: {', '.join(issues)}",
            score >= 8
        )

    def _test_performance_requirements(self, db_impl: VectorDatabaseInterface) -> Tuple[EvaluationResult, Dict[str, float]]:
        """Test performance with large dataset"""
        score = 10
        issues = []
        perf_data = {}

        try:
            # Clear and load large dataset
            db_impl.clear_all()

            # Test batch insertion performance
            start_time = time.time()
            batch_success = db_impl.batch_add_claims(self.large_dataset[:1000])  # Start with 1k
            insertion_time = time.time() - start_time

            if not batch_success:
                score -= 4
                issues.append("Batch insertion failed")

            perf_data['insertion_time_1000'] = insertion_time

            # Test query performance
            if batch_success:
                start_time = time.time()
                results = db_impl.search_similar("computer science", limit=10)
                query_time = time.time() - start_time

                perf_data['query_time_10_results'] = query_time

                # Success criterion: <100ms queries
                if query_time > 0.1:  # 100ms
                    score -= 3
                    issues.append(f"Query too slow: {query_time*1000:.1f}ms")

                # Test with larger dataset if available
                if len(self.large_dataset) >= 10000:
                    start_time = time.time()
                    db_impl.batch_add_claims(self.large_dataset[1000:10000])
                    large_load_time = time.time() - start_time

                    start_time = time.time()
                    results = db_impl.search_similar("data", limit=5)
                    large_query_time = time.time() - start_time

                    perf_data['query_time_10k_claims'] = large_query_time
                    perf_data['load_time_9000_claims'] = large_load_time

                    if large_query_time > 0.2:  # 200ms for 10k
                        score -= 3
                        issues.append(f"Large query too slow: {large_query_time*1000:.1f}ms")

        except Exception as e:
            score = 0
            issues.append(f"Performance error: {e}")

        return EvaluationResult(
            "Performance Requirements",
            score,
            10,
            "Passed" if score >= 7 else f"Issues: {', '.join(issues)}",
            score >= 7
        ), perf_data

    def _test_schema_validation(self, db_impl: VectorDatabaseInterface) -> EvaluationResult:
        """Test proper schema handling and validation"""
        score = 5
        issues = []

        try:
            # Test with valid claim
            valid_claim = self.test_claims[0]
            success = db_impl.add_claim(valid_claim)
            if not success:
                score -= 2
                issues.append("Cannot add valid claim")

            # Test filtering capabilities
            filtered = db_impl.filter_claims(confidence_min=0.9)
            if not filtered:
                score -= 2
                issues.append("Filtering not working")

            # Test search returns proper claim objects
            search_results = db_impl.search_similar("test", limit=3)
            if not search_results:
                score -= 1
                issues.append("Search not returning results")

        except Exception as e:
            score = 0
            issues.append(f"Schema validation error: {e}")

        return EvaluationResult(
            "Schema Validation",
            score,
            5,
            "Passed" if score >= 4 else f"Issues: {', '.join(issues)}",
            score >= 4
        )

    def _test_integration_workflow(self, db_impl: VectorDatabaseInterface) -> EvaluationResult:
        """Test end-to-end workflow integration"""
        score = 5
        issues = []

        try:
            # Simulate typical workflow: add -> search -> filter -> retrieve
            claim = self.test_claims[2]

            # Add claim
            add_success = db_impl.add_claim(claim)
            if not add_success:
                score -= 2
                issues.append("Workflow: add failed")

            # Search for similar
            similar = db_impl.search_similar(claim.content[:10], limit=5)
            if not similar:
                score -= 1
                issues.append("Workflow: search failed")

            # Filter by type
            filtered = db_impl.filter_claims(claim_type=claim.type[0].value)
            if not filtered:
                score -= 1
                issues.append("Workflow: filter failed")

            # Get stats
            stats = db_impl.get_stats()
            if not stats:
                score -= 1
                issues.append("Workflow: stats failed")

        except Exception as e:
            score = 0
            issues.append(f"Integration error: {e}")

        return EvaluationResult(
            "Integration Workflow",
            score,
            5,
            "Passed" if score >= 4 else f"Issues: {', '.join(issues)}",
            score >= 4
        )

    def _generate_recommendation(self, percentage: float, results: List[EvaluationResult], perf_data: Dict[str, float]) -> str:
        """Generate recommendation based on evaluation results"""
        if percentage >= 90:
            return "HIGHLY RECOMMENDED - Excellent performance across all criteria"
        elif percentage >= 80:
            return "RECOMMENDED - Good performance with minor limitations"
        elif percentage >= 70:
            return "CONDITIONAL - Suitable for development, may need optimization"
        elif percentage >= 60:
            return "NOT RECOMMENDED FOR PRODUCTION - Significant limitations"
        else:
            return "NOT SUITABLE - Fails critical requirements"

    def print_evaluation_report(self, evaluation: DatabaseEvaluation):
        """Print detailed evaluation report"""
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {evaluation.database_name}")
        print(f"{'='*60}")
        print(f"Total Score: {evaluation.total_score}/{evaluation.max_score} ({evaluation.percentage:.1f}%)")
        print(f"Recommendation: {evaluation.recommendation}")

        print(f"\nDETAILED RESULTS:")
        print("-" * 60)
        for result in evaluation.results:
            status = "✅ PASS" if result.passed_threshold else "❌ FAIL"
            print(f"{result.metric_name:25} | {result.score:.1f}/{result.max_score} | {status}")
            if result.score < result.max_score:
                print(f"{'':27} | {result.details}")

        if evaluation.performance_summary:
            print(f"\nPERFORMANCE SUMMARY:")
            print("-" * 60)
            for metric, value in evaluation.performance_summary.items():
                if 'time' in metric:
                    unit = 's' if value > 1 else 'ms'
                    display_value = value if value > 1 else value * 1000
                    print(f"{metric:25} | {display_value:.3f} {unit}")
                else:
                    print(f"{metric:25} | {value}")

        print(f"\n{'='*60}\n")


def compare_evaluations(evaluations: List[DatabaseEvaluation]) -> Dict[str, DatabaseEvaluation]:
    """Compare multiple evaluations and rank databases"""
    rankings = {}

    # Sort by percentage score
    sorted_evaluations = sorted(evaluations, key=lambda x: x.percentage, reverse=True)

    print(f"\n{'='*80}")
    print(f"DATABASE COMPARISON RANKING")
    print(f"{'='*80}")

    for i, evaluation in enumerate(sorted_evaluations, 1):
        print(f"{i}. {evaluation.database_name:20} | {evaluation.percentage:5.1f}% | {evaluation.total_score}/{evaluation.max_score} | {evaluation.recommendation}")

    return {eval.database_name: eval for eval in sorted_evaluations}
