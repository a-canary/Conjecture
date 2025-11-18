"""
Additional test utilities, data providers, and helper functions
for comprehensive test coverage of the Conjecture data layer.
"""
import pytest
import asyncio
import random
import json
import numpy as np
from typing import List, Dict, Any, Generator, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from faker import Faker

from src.data.models import Claim, Relationship, ClaimFilter, DataConfig


@dataclass
class TestScenario:
    """Container for predefined test scenarios."""
    name: str
    description: str
    claims_data: List[Dict[str, Any]]
    expected_outcomes: Dict[str, Any]
    performance_requirements: Dict[str, Any]


class TestDataGenerators:
    """Comprehensive data generators for testing various scenarios."""

    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)  # Deterministic for consistent tests
        
        # Content templates organized by domain
        self.content_templates = {
            "science": [
                "Scientific study shows that {factor} affects {outcome} by {percentage}%",
                "Research published in {journal} demonstrates {phenomenon}",
                "Laboratory experiments confirm {hypothesis} with {confidence_level} certainty",
                "Peer-reviewed analysis reveals {discovery} about {subject}",
                "Empirical evidence suggests {relationship} between {cause} and {effect}"
            ],
            "technology": [
                "New {technology} improves {metric} by {improvement}%",
                "{company} released {product} with advanced {feature}",
                "Benchmark tests show {performance} gains for {system}",
                "Open-source {tool} achieves {result} through {method}",
                "Innovation in {field} enables new {capability}"
            ],
            "health": [
                "Clinical trial indicates {treatment} reduces {condition} by {effect}",
                "Medical research links {risk_factor} to {disease}",
                "Patient study shows {benefit} from {therapy} over {duration}",
                "Healthcare data reveals {trend} in {population}",
                "Preventive measures decrease {incidence} by {reduction}%"
            ]
        }
        
        # Tag prefixes for different categories
        self.tag_prefixes = [
            "research_", "clinical_", "experimental_", "peer-reviewed_",
            "preliminary_", "published_", "validated_", "theoretical_",
            "practical_", "observational_", "statistical_", "empirical_"
        ]

    def generate_realistic_claim(self, domain: str = None, complexity: str = "medium") -> Dict[str, Any]:
        """Generate a realistic claim with proper structure."""
        domain = domain or random.choice(list(self.content_templates.keys()))
        
        if domain == "science":
            return self._generate_scientific_claim(complexity)
        elif domain == "technology":
            return self._generate_technology_claim(complexity)
        elif domain == "health":
            return self._generate_health_claim(complexity)
        else:
            return self._generate_general_claim(complexity)

    def _generate_scientific_claim(self, complexity: str) -> Dict[str, Any]:
        """Generate a scientific claim."""
        template = random.choice(self.content_templates["science"])
        
        factors = ["temperature", "pressure", "pH level", "concentration", "frequency"]
        outcomes = ["reaction rate", "yield", "efficiency", "stability", "accuracy"]
        journals = ["Nature", "Science", "Cell", "PNAS", "Nature Medicine"]
        phenomena = ["quantum entanglement", "protein folding", "gene expression", "neural plasticity"]
        
        claim_content = template.format(
            factor=random.choice(factors),
            outcome=random.choice(outcomes),
            percentage=random.randint(5, 95),
            journal=random.choice(journals),
            phenomenon=random.choice(phenomena),
            confidence_level=random.choice(["moderate", "high", "very high"]),
            hypothesis=f"the {random.choice(['first', 'second', 'third'])} hypothesis",
            discovery=random.choice(["correlation", "causation", "interaction"]),
            subject=random.choice(["molecules", "cells", "organisms", "populations"]),
            relationship=random.choice(["positive", "negative", "inverse", "direct"]),
            cause=random.choice(["various factors", "environmental conditions", "genetic predisposition"]),
            effect=random.choice(["observed outcomes", "measured parameters", "biological responses"])
        )
        
        return {
            "content": claim_content,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "tags": self._generate_tags(["science", "research", random.choice(["physics", "chemistry", "biology"])]),
            "created_by": f"researcher_{random.randint(1, 20)}",
            "dirty": random.choice([True, False])
        }

    def _generate_technology_claim(self, complexity: str) -> Dict[str, Any]:
        """Generate a technology claim."""
        template = random.choice(self.content_templates["technology"])
        
        companies = ["Google", "Microsoft", "Apple", "Amazon", "Tesla"]
        technologies = ["AI", "machine learning", "blockchain", "quantum computing", "edge computing"]
        features = ["security", "performance", "scalability", "usability", "efficiency"]
        
        claim_content = template.format(
            technology=random.choice(technologies),
            metric=random.choice(["speed", "accuracy", "throughput", "latency", "reliability"]),
            improvement=random.randint(10, 200),
            company=random.choice(companies),
            product=f"{random.choice(['new', 'improved', 'next-generation'])} solution",
            feature=random.choice(features),
            system=random.choice(["existing", "legacy", "modern"]) + " systems",
            performance=random.choice(["significant", "moderate", "dramatic"]) + " improvements",
            tool=random.choice(["framework", "library", "platform", "toolkit"]),
            result=random.choice(["success", "breakthrough", "innovation"]),
            method=random.choice(["optimization", "redesign", "rearchitecture"]),
            field=random.choice(["software", "hardware", "networking", "security"]),
            capability=random.choice(["integration", "automation", "analytics"])
        )
        
        return {
            "content": claim_content,
            "confidence": round(random.uniform(0.7, 0.9), 2),
            "tags": self._generate_tags(["technology", random.choice(["software", "hardware", "AI", "cloud"])]),
            "created_by": f"tech_user_{random.randint(1, 15)}",
            "dirty": random.choice([True, False])
        }

    def _generate_health_claim(self, complexity: str) -> Dict[str, Any]:
        """Generate a health claim."""
        template = random.choice(self.content_templates["health"])
        
        treatments = ["medication", "therapy", "exercise", "diet", "intervention"]
        conditions = ["blood pressure", "cholesterol", "glucose levels", "pain", "inflammation"]
        risk_factors = ["smoking", "obesity", "sedentary lifestyle", "high stress", "poor diet"]
        diseases = ["diabetes", "heart disease", "cancer", "arthritis", "depression"]
        
        claim_content = template.format(
            treatment=random.choice(treatments),
            condition=random.choice(conditions),
            effect=random.randint(10, 50),
            risk_factor=random.choice(risk_factors),
            disease=random.choice(diseases),
            therapy=random.choice(["cognitive behavioral", "physical", "pharmacological"]),
            benefit=random.choice(["significant", "moderate", "noticeable"]) + " improvement",
            duration=random.choice(["2 weeks", "1 month", "3 months", "6 months"]),
            trend=random.choice(["increasing", "decreasing", "stable"]),
            population=random.choice(["elderly", "adults", "adolescents", "patients with comorbidities"]),
            incidence=random.randint(5, 40),
            reduction=random.randint(15, 60)
        )
        
        return {
            "content": claim_content,
            "confidence": round(random.uniform(0.5, 0.85), 2),
            "tags": self._generate_tags(["health", "medical", random.choice(["clinical", "preventive", "therapeutic"])]),
            "created_by": f"medical_professional_{random.randint(1, 25)}",
            "dirty": random.choice([True, False])
        }

    def _generate_general_claim(self, complexity: str) -> Dict[str, Any]:
        """Generate a general-purpose claim."""
        topics = ["education", "environment", "economics", "social policy", "urban planning"]
        topic = random.choice(topics)
        
        claim_templates = {
            "education": [
                "Students using {method} show {improvement}% better {outcome}",
                "{approach} increases {metric} among {population}",
                "Research demonstrates {finding} in {educational_level} education"
            ],
            "environment": [
                "{pollutant} levels decreased by {reduction}% in {location}",
                "Conservation efforts increased {species} population by {growth}%",
                "Climate data shows {trend} in {region} over {timeframe}"
            ],
            "economics": [
                "GDP grew by {growth}% following {policy} implementation",
                "Unemployment decreased by {reduction}% due to {factor}",
                "Inflation rate stabilized around {percentage}% after {intervention}"
            ],
            "social_policy": [
                "Access to {service} improved for {population} by {improvement}%",
                "{program} reduced {issue} by {reduction}% in target areas",
                "Social indicators show {trend} after {reform} implementation"
            ],
            "urban_planning": [
                "Public transit usage increased by {increase}% after {improvement}",
                "Green spaces reduced {metric} by {reduction}% in {area_type}",
                "Urban density affects {outcome} by {correlation}"
            ]
        }
        
        template = random.choice(claim_templates[topic])
        
        # Generate context-appropriate fillers
        fillers = {
            "method": ["online learning", "traditional instruction", "blended approaches", "project-based learning"],
            "approach": ["personalized learning", "early intervention", "technology integration"],
            "pollutant": ["PM2.5", "NO2", "CO2", "SO2"],
            "species": ["endangered birds", "coral reef", "forest coverage"],
            "growth": ["10", "25", "50"],
            "policy": ["fiscal stimulus", "trade agreement", "regulatory reform"],
            "service": ["healthcare", "education", "housing", "transportation"]
        }
        
        # Safe template filling with partial matches
        claim_content = template
        
        for key, options in fillers.items():
            if f"{{{key}}}" in claim_content:
                claim_content = claim_content.replace(f"{{{key}}}", random.choice(options))
        
        # Fill remaining placeholders with generic values
        placeholders = {}
        for i in range(1, 10):  # Up to 10 placeholders
            placeholder = f"placeholder_{i}"
            if placeholder in claim_content:
                if "percentage" in placeholder or "reduction" in placeholder:
                    placeholders[placeholder] = str(random.randint(5, 50))
                else:
                    placeholders[placeholder] = self.fake.catch_phrase()
        
        for key, value in placeholders.items():
            claim_content = claim_content.replace(f"{{{key}}}", value)
        
        return {
            "content": claim_content,
            "confidence": round(random.uniform(0.4, 0.8), 2),
            "tags": self._generate_tags([topic, "research", "data"]),
            "created_by": f"{topic}_analyst_{random.randint(1, 10)}",
            "dirty": random.choice([True, False])
        }

    def _generate_tags(self, base_tags: List[str]) -> List[str]:
        """Generate comprehensive tags for a claim."""
        tags = base_tags.copy()
        
        # Add contextual tags
        tags.append(random.choice(["validated", "preliminary", "confirmed", "disputed"]))
        tags.append(random.choice(["recent", "ongoing", "historical", "projected"]))
        
        # Add random specific tags
        for _ in range(random.randint(1, 3)):
            prefix = random.choice(self.tag_prefixes)
            specific = self.fake.word().lower()
            tags.append(f"{prefix}{specific}")
        
        # Remove duplicates and sort
        return sorted(list(set(tags)))

    def generate_domain_dataset(self, domain: str, size: int) -> List[Dict[str, Any]]:
        """Generate a dataset focused on a specific domain."""
        return [self.generate_realistic_claim(domain) for _ in range(size)]

    def generate_mixed_dataset(self, size: int) -> List[Dict[str, Any]]:
        """Generate a dataset with mixed domains."""
        domains = list(self.content_templates.keys()) + ["general"]
        return [
            self.generate_realistic_claim(random.choice(domains))
            for _ in range(size)
        ]

    def generate_relationship_network(self, claims: List[Claim], density: float = 0.2) -> List[Relationship]:
        """Generate a relationship network for given claims."""
        relationships = []
        
        # Create a percentage of possible relationships
        max_relationships = int(len(claims) * (len(claims) - 1) * density)
        
        for _ in range(max_relationships):
            supporter = random.choice(claims)
            supported = random.choice([c for c in claims if c.id != supporter.id])
            
            # Avoid duplicate relationships
            if not any(
                r.supporter_id == supporter.id and r.supported_id == supported.id
                for r in relationships
            ):
                rel = Relationship(
                    supporter_id=supporter.id,
                    supported_id=supported.id,
                    relationship_type=random.choice(["supports", "contradicts", "extends"]),
                    created_by="network_generator"
                )
                relationships.append(rel)
        
        return relationships


class TestScenarios:
    """Predefined test scenarios for comprehensive coverage."""

    def get_basic_functionality_scenario(self) -> TestScenario:
        """Basic functionality test scenario."""
        return TestScenario(
            name="basic_functionality",
            description="Test basic CRUD operations and simple workflows",
            claims_data=[
                {
                    "content": "The Earth orbits around the Sun in an elliptical path",
                    "created_by": "astronomer_1",
                    "confidence": 0.95,
                    "tags": ["astronomy", "orbital_mechanics"],
                    "dirty": False
                },
                {
                    "content": "Water freezes at 0 degrees Celsius at standard pressure",
                    "created_by": "physicist_1",
                    "confidence": 0.98,
                    "tags": ["physics", "thermodynamics"],
                    "dirty": False
                },
                {
                    "content": "DNA contains the genetic instructions for life",
                    "created_by": "biologist_1",
                    "confidence": 0.92,
                    "tags": ["biology", "genetics"],
                    "dirty": True
                }
            ],
            expected_outcomes={
                "total_claims": 3,
                "dirty_claims": 1,
                "successful_retrieval": True,
                "embedding_generation": True
            },
            performance_requirements={
                "create_claim_max_time": 0.1,
                "retrieve_claim_max_time": 0.01,
                "embeddings_created": 3
            }
        )

    def get_relationship_scenario(self) -> TestScenario:
        """Relationship-focused test scenario."""
        return TestScenario(
            name="relationship_network",
            description="Test complex relationship networks and traversals",
            claims_data=[
                {
                    "content": "Climate change is caused by greenhouse gas emissions",
                    "created_by": "climate_scientist_1",
                    "confidence": 0.9,
                    "tags": ["climate", "environment", "emissions"],
                    "dirty": False
                },
                {
                    "content": "CO2 levels have increased by 50% since pre-industrial times",
                    "created_by": "climate_scientist_2",
                    "confidence": 0.95,
                    "tags": ["climate", "co2", "data"],
                    "dirty": False
                },
                {
                    "content": "Global temperatures have risen by 1.1 degrees Celsius",
                    "created_by": "climate_scientist_3",
                    "confidence": 0.85,
                    "tags": ["climate", "temperature", "warming"],
                    "dirty": True
                },
                {
                    "content": "Renewable energy can reduce carbon emissions",
                    "created_by": "energy_expert_1",
                    "confidence": 0.8,
                    "tags": ["energy", "renewable", "solutions"],
                    "dirty": True
                }
            ],
            expected_outcomes={
                "supports_relationships": 2,
                "contradicts_relationships": 0,
                "network_depth": 2,
                "bidirectional_traversal": True
            },
            performance_requirements={
                "relationship_query_max_time": 0.02,
                "network_traversal_max_time": 0.05
            }
        )

    def get_search_scenario(self) -> TestScenario:
        """Search and similarity test scenario."""
        generator = TestDataGenerators()
        
        return TestScenario(
            name="semantic_search",
            description="Test semantic search and filtering capabilities",
            claims_data=generator.generate_mixed_dataset(50),
            expected_outcomes={
                "search_results_relevant": True,
                "filter_accuracy": 0.8,
                "similarity_ranking_sensible": True
            },
            performance_requirements={
                "search_max_time": 0.05,
                "filter_max_time": 0.02,
                "similarity_calculation_max_time": 0.01
            }
        )

    def get_performance_stress_scenario(self) -> TestScenario:
        """Performance stress test scenario."""
        generator = TestDataGenerators()
        
        return TestScenario(
            name="performance_stress",
            description="Stress test with large dataset and complex operations",
            claims_data=generator.generate_mixed_dataset(1000),
            expected_outcomes={
                "all_claims_created": True,
                "search_performance_acceptable": True,
                "memory_usage_reasonable": True
            },
            performance_requirements={
                "batch_create_time_per_claim": 0.005,
                "search_time": 0.1,
                "memory_usage_per_claim": 0.01  # MB
            }
        )

    def get_concurrency_scenario(self) -> TestScenario:
        """Concurrency test scenario."""
        return TestScenario(
            name="concurrent_operations",
            description="Test concurrent read/write operations",
            claims_data=[
                {
                    "content": f"Concurrent test claim {i}",
                    "created_by": f"concurrent_user_{i % 10}",
                    "confidence": 0.7,
                    "tags": ["concurrency", "test"],
                    "dirty": False
                }
                for i in range(100)
            ],
            expected_outcomes={
                "no_data_corruption": True,
                "race_conditions_handled": True,
                "throughput_acceptable": True
            },
            performance_requirements={
                "concurrent_ops_per_second": 20,
                "data_integrity_maintained": True
            }
        )


class TestAssertions:
    """Custom assertion helpers for comprehensive testing."""

    @staticmethod
    def assert_claim_validity(claim: Claim, expected_data: Dict[str, Any] = None):
        """Assert claim meets all validity requirements."""
        # Basic structure
        assert isinstance(claim, Claim)
        assert claim.id is not None
        assert claim.content is not None
        assert claim.created_by is not None
        
        # ID format
        assert claim.id.startswith('c')
        assert len(claim.id) == 8
        assert claim.id[1:].isdigit()
        
        # Content constraints
        assert len(claim.content) >= 10
        assert isinstance(claim.content, str)
        
        # Confidence range
        assert 0.0 <= claim.confidence <= 1.0
        assert isinstance(claim.confidence, (int, float))
        
        # Tags validation
        assert isinstance(claim.tags, list)
        assert all(isinstance(tag, str) and tag.strip() for tag in claim.tags)
        assert len(claim.tags) == len(set(claim.tags))  # No duplicates
        
        # Timestamps
        assert isinstance(claim.created_at, datetime)
        assert claim.updated_at is None or isinstance(claim.updated_at, datetime)
        
        # Expected data matches
        if expected_data:
            if "content" in expected_data:
                assert claim.content == expected_data["content"]
            if "created_by" in expected_data:
                assert claim.created_by == expected_data["created_by"]
            if "confidence" in expected_data:
                assert abs(claim.confidence - expected_data["confidence"]) < 1e-10
            if "tags" in expected_data:
                assert sorted(claim.tags) == sorted(expected_data["tags"])
            if "dirty" in expected_data:
                assert claim.dirty == expected_data["dirty"]

    @staticmethod
    def assert_relationship_validity(relationship: Relationship):
        """Assert relationship meets all validity requirements."""
        assert isinstance(relationship, Relationship)
        
        # ID formats
        assert relationship.supporter_id.startswith('c')
        assert len(relationship.supporter_id) == 8
        assert relationship.supporter_id[1:].isdigit()
        
        assert relationship.supported_id.startswith('c')
        assert len(relationship.supported_id) == 8
        assert relationship.supported_id[1:].isdigit()
        
        # No self-relationships
        assert relationship.supporter_id != relationship.supported_id
        
        # Relationship type
        assert relationship.relationship_type in ["supports", "contradicts", "extends", "clarifies"]
        
        # Timestamp
        assert isinstance(relationship.created_at, datetime)

    @staticmethod
    def assert_search_result_quality(results: List[Claim], query: str, domain: str = None):
        """Assess quality of search results."""
        assert isinstance(results, list)
        
        # Basic relevance check (mock-based might be limited)
        if results:
            for result in results:
                assert isinstance(result, Claim)
                # In real testing, would check semantic relevance
                # For now, just ensure structure is valid
                TestAssertions.assert_claim_validity(result)

    @staticmethod
    def assert_performance_metrics(metrics: Dict[str, Any], requirements: Dict[str, Any]):
        """Assert performance meets requirements."""
        for metric, requirement in requirements.items():
            if metric.endswith("_time"):
                actual = metrics.get(metric.replace("_time", "_avg_time"))
                if actual is not None:
                    assert actual <= requirement, f"{metric} average {actual:.4f}s exceeds requirement {requirement:.4f}s"
            
            elif metric.endswith("_rate") or metric.endswith("_throughput"):
                actual = metrics.get(metric)
                if actual is not None:
                    assert actual >= requirement, f"{metric} {actual:.2f} below requirement {requirement:.2f}"

    @staticmethod
    def assert_data_consistency(sqlite_data: List[Claim], chroma_data: List[Dict], expected_count: int):
        """Assert data consistency between SQLite and ChromaDB."""
        assert len(sqlite_data) == expected_count, f"SQLite has {len(sqlite_data)} claims, expected {expected_count}"
        assert len(chroma_data) == expected_count, f"ChromaDB has {len(chroma_data)} embeddings, expected {expected_count}"
        
        # Check all SQLite claims have corresponding embeddings
        sqlite_ids = {claim.id for claim in sqlite_data}
        chroma_ids = {data['id'] for data in chroma_data}
        
        assert sqlite_ids == chroma_ids, f"ID mismatch: SQLite={sqlite_ids}, ChromaDB={chroma_ids}"

    @staticmethod
    def assert_no_duplicates(items: List[Any]):
        """Assert list contains no duplicates."""
        if len(items) != len(set(items)):
            # Find duplicates for debugging
            from collections import Counter
            duplicates = [item for item, count in Counter(items).items() if count > 1]
            assert False, f"Duplicates found: {duplicates}"


class TestDataProvider:
    """Pytest data provider for parameterized tests."""

    @staticmethod
    def claim_content_samples():
        """Provide representative claim content samples."""
        return [
            pytest.param(
                "The Earth revolves around the Sun in an elliptical orbit",
                id="astronomy_basic"
            ),
            pytest.param(
                "Water boils at 100 degrees Celsius at standard atmospheric pressure",
                id="physics_thermodynamics"
            ),
            pytest.param(
                "DNA carries genetic information through sequences of nucleotides",
                id="biology_genetics"
            ),
            pytest.param(
                "Climate change is primarily driven by human greenhouse gas emissions",
                id="environmental_climate"
            ),
            pytest.param(
                "Machine learning algorithms improve with more training data",
                id="technology_ml"
            ),
            pytest.param(
                "Regular exercise reduces the risk of cardiovascular disease",
                id="health_fitness"
            ),
            pytest.param(
                "Economic policies affect inflation rates through monetary mechanisms",
                id="economics_policy"
            ),
            pytest.param(
                "Social media influences public opinion through content algorithms",
                id="social_media"
            ),
        ]

    @staticmethod
    def invalid_claims_data():
        """Provide systematically invalid claim data."""
        return [
            pytest.param(
                {"content": "Too short", "created_by": "user"},
                "content_too_short",
                id="content_min_length"
            ),
            pytest.param(
                {"content": "", "created_by": "user"},
                "content_empty",
                id="content_empty"
            ),
            pytest.param(
                {"content": "Valid content length", "created_by": ""},
                "creator_empty",
                id="creator_empty"
            ),
            pytest.param(
                {"content": "Valid content length", "created_by": "user", "confidence": -0.1},
                "confidence_negative",
                id="confidence_negative"
            ),
            pytest.param(
                {"content": "Valid content length", "created_by": "user", "confidence": 1.1},
                "confidence_too_high",
                id="confidence_too_high"
            ),
            pytest.param(
                {"content": "Valid content length", "created_by": "user", "tags": [""]},
                "empty_tag",
                id="empty_tag"
            ),
            pytest.param(
                {"content": "Valid content length", "created_by": "user", "tags": [None]},
                "null_tag",
                id="null_tag"
            ),
        ]

    @staticmethod
    def relationship_type_samples():
        """Provide all valid relationship types."""
        return [
            pytest.param("supports", id="supports"),
            pytest.param("contradicts", id="contradicts"),
            pytest.param("extends", id="extends"),
            pytest.param("clarifies", id="clarifies"),
        ]

    @staticmethod
    def filter_combinations():
        """Provide comprehensive filter combinations."""
        return [
            pytest.param(
                {"tags": ["science"]},
                "single_tag",
                id="single_tag_filter"
            ),
            pytest.param(
                {"confidence_min": 0.8},
                "confidence_min",
                id="confidence_min_filter"
            ),
            pytest.param(
                {"confidence_max": 0.7},
                "confidence_max",
                id="confidence_max_filter"
            ),
            pytest.param(
                {"confidence_min": 0.6, "confidence_max": 0.9},
                "confidence_range",
                id="confidence_range_filter"
            ),
            pytest.param(
                {"dirty_only": True},
                "dirty_only",
                id="dirty_filter"
            ),
            pytest.param(
                {"created_by": "test_user"},
                "creator",
                id="creator_filter"
            ),
            pytest.param(
                {"content_contains": "research"},
                "content_search",
                id="content_filter"
            ),
            pytest.param(
                {"tags": ["science", "research"], "confidence_min": 0.7, "limit": 5},
                "complex",
                id="complex_filter"
            ),
        ]

    @staticmethod
    def performance_test_parameters():
        """Provide parameters for performance testing."""
        return [
            pytest.param(10, "small_dataset", id="small_dataset"),
            pytest.param(50, "medium_dataset", id="medium_dataset"),
            pytest.param(200, "large_dataset", id="large_dataset"),
            pytest.param(1000, "stress_dataset", id="stress_dataset"),
        ]

    @staticmethod
    def edge_case_strings():
        """Provide string edge cases for testing."""
        return [
            pytest.param("Normal string", id="normal"),
            pytest.param("A" * 10, id="min_length"),
            pytest.param("A" * 10000, id="max_length"),
            pytest.param("Special chars: !@#$%^&*()", id="special_chars"),
            pytest.param("Unicode: αβγδε 中文 日本語 🚀", id="unicode"),
            pytest.param("Newlines:\nLine1\nLine2\n", id="newlines"),
            pytest.param("Tabs:\tTabbed\tContent\t", id="tabs"),
            pytest.param("Mixed \"quotes\" and 'apostrophes'", id="quotes"),
            pytest.param("  Leading and trailing spaces  ", id="whitespace"),
            pytest.param("Emoji test: 🧬🧪🔬📊📈", id="emojis"),
        ]


# Integration with pytest fixtures
@pytest.fixture
def test_data_generator():
    """Provide TestDataGenerator for tests."""
    return TestDataGenerators()


@pytest.fixture
def test_scenarios():
    """Provide TestScenarios for tests."""
    return TestScenarios()


@pytest.fixture
def test_assertions():
    """Provide TestAssertions for tests."""
    return TestAssertions()


@pytest.fixture
def test_data_provider():
    """Provide TestDataProvider for tests."""
    return TestDataProvider()