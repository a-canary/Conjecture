"""
Pytest fixtures and configuration for skill-based agency tests.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from src.core.skill_models import (
    SkillClaim, ExampleClaim, ExecutionResult, ToolCall,
    SkillParameter, SkillRegistry
)
from src.data.data_manager import DataManager
from src.processing.skill_manager import SkillManager
from src.processing.response_parser import ResponseParser
from src.processing.tool_executor import ToolExecutor, ExecutionLimits
from src.processing.example_generator import ExampleGenerator


@pytest.fixture
def data_manager_mock():
    """Mock DataManager for testing."""
    mock = AsyncMock(spec=DataManager)
    
    # Setup default return values
    mock.create_claim.return_value = MagicMock(id="c123", created_at=datetime.utcnow())
    mock.get_claim.return_value = MagicMock(id="c123", content="test", confidence=0.8)
    mock.update_claim.return_value = True
    mock.delete_claim.return_value = True
    mock.search_similar.return_value = []
    mock.filter_claims.return_value = []
    mock.get_relationships.return_value = []
    mock.add_relationship.return_value = "rel123"
    mock.get_stats.return_value = {
        'total_claims': 100,
        'total_relationships': 50
    }
    
    return mock


@pytest.fixture
def skill_manager(data_manager_mock):
    """SkillManager instance with mocked dependencies."""
    return SkillManager(data_manager_mock)


@pytest.fixture
def response_parser():
    """ResponseParser instance."""
    return ResponseParser()


@pytest.fixture
def execution_limits():
    """ExecutionLimits for testing."""
    return ExecutionLimits(
        max_execution_time=5.0,
        max_memory_mb=50,
        max_cpu_time=2.0,
        max_output_chars=1000,
        allow_network=False,
        allow_file_access=False
    )


@pytest.fixture
def tool_executor(execution_limits):
    """ToolExecutor instance with custom limits."""
    return ToolExecutor(execution_limits)


@pytest.fixture
def example_generator(data_manager_mock):
    """ExampleGenerator instance with mocked dependencies."""
    return ExampleGenerator(data_manager_mock)


@pytest.fixture
def sample_skill_parameter():
    """Sample SkillParameter for testing."""
    return SkillParameter(
        name="query",
        param_type="str",
        required=True,
        description="Search query string"
    )


@pytest.fixture
def sample_skill_claim(sample_skill_parameter):
    """Sample SkillClaim for testing."""
    return SkillClaim(
        id="c123",
        content="Search for claims matching the query",
        function_name="search_claims",
        parameters=[sample_skill_parameter],
        return_type="List[Dict]",
        skill_category="search",
        skill_version="1.0.0",
        confidence=0.9,
        tags=["type.skill", "search"],
        created_by="test_user"
    )


@pytest.fixture
def sample_execution_result():
    """Sample ExecutionResult for testing."""
    return ExecutionResult(
        success=True,
        result={"id": "c456", "content": "Found claim", "confidence": 0.8},
        execution_time_ms=150,
        memory_usage_mb=2.5,
        skill_id="c123",
        parameters_used={"query": "test", "limit": 10},
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_tool_call():
    """Sample ToolCall for testing."""
    return ToolCall(
        name="search_claims",
        parameters={"query": "test", "limit": 10},
        call_id="call123"
    )


@pytest.fixture
def sample_example_claim():
    """Sample ExampleClaim for testing."""
    return ExampleClaim(
        id="c789",
        content="Example: search_claims(query=\"test\", limit=10) -> [{'id': 'c456', 'content': 'Found claim'}]",
        skill_id="c123",
        input_parameters={"query": "test", "limit": 10},
        output_result={"id": "c456", "content": "Found claim", "confidence": 0.8},
        execution_time_ms=150,
        example_quality=0.85,
        confidence=0.8,
        tags=["type.example", "auto_generated"],
        created_by="example_generator"
    )


@pytest.fixture
def sample_skill_registry():
    """Sample SkillRegistry with skills."""
    registry = SkillRegistry()
    
    # Add sample skills
    skill1 = SkillClaim(
        id="c1",
        content="Search for claims",
        function_name="search_claims",
        parameters=[
            SkillParameter(name="query", param_type="str", required=True),
            SkillParameter(name="limit", param_type="int", required=False, default_value=10)
        ],
        skill_category="search",
        confidence=0.9
    )
    
    skill2 = SkillClaim(
        id="c2", 
        content="Create a new claim",
        function_name="create_claim",
        parameters=[
            SkillParameter(name="content", param_type="str", required=True),
            SkillParameter(name="confidence", param_type="float", required=False, default_value=0.5)
        ],
        skill_category="creation",
        confidence=0.8
    )
    
    registry.register_skill(skill1)
    registry.register_skill(skill2)
    
    return registry


# Test data fixtures
@pytest.fixture
def xml_response_samples():
    """Sample XML responses for testing."""
    return {
        "valid_single": """
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">test query</parameter>
                <parameter name="limit">10</parameter>
            </invoke>
        </tool_calls>
        """,
        "valid_multiple": """
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">test</parameter>
            </invoke>
            <invoke name="create_claim">
                <parameter name="content">New claim content</parameter>
                <parameter name="confidence">0.8</parameter>
            </invoke>
        </tool_calls>
        """,
        "with_text": """
        I'll help you search for claims.
        
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">machine learning</parameter>
            </invoke>
        </tool_calls>
        
        Let me know the results.
        """,
        "malformed": """
        <tool_calls>
            <invoke name="search_claims">
                <parameter name="query">test
            </invoke>
        </tool_calls>
        """,
        "empty": ""
    }


@pytest.fixture
def json_response_samples():
    """Sample JSON responses for testing."""
    return {
        "valid_single": """
        {
            "tool_calls": [
                {
                    "name": "search_claims",
                    "parameters": {
                        "query": "test",
                        "limit": 10
                    }
                }
            ]
        }
        """,
        "valid_multiple": """
        {
            "tool_calls": [
                {
                    "name": "search_claims",
                    "parameters": {"query": "test"}
                },
                {
                    "name": "create_claim", 
                    "parameters": {"content": "New claim"}
                }
            ]
        }
        """,
        "no_tool_calls": """
        {
            "message": "I'll help you with that."
        }
        """,
        "invalid_json": """
        {
            "tool_calls": [
                {
                    "name": "search_claims",
                    "parameters": {
                        "query": "test",
                        "limit": 
                    }
                }
            ]
        }
        """
    }


@pytest.fixture
def markdown_response_samples():
    """Sample markdown responses for testing."""
    return {
        "valid_single": """
        ```tool_call
        name: search_claims
        parameters: {"query": "test", "limit": 10}
        ```
        """,
        "valid_simple": """
        ```tool_call
        name: search_claims
        query: test
        limit: 10
        ```
        """,
        "with_text": """
        I'll search for the information you requested.
        
        ```tool_call
        name: search_claims
        query: machine learning
        limit: 5
        ```
        
        Let me find those results for you.
        """,
        "no_tool_calls": """
        I'll help you with that request. Let me search for the information.
        """
    }


@pytest.fixture
def code_execution_samples():
    """Sample code snippets for testing."""
    return {
        "safe_math": """
        result = sum([1, 2, 3, 4, 5])
        average = result / 5
        output = {"sum": result, "average": average}
        """,
        "safe_string": """
        text = "Hello, World!"
        words = text.split(", ")
        reversed_text = words[1] + " " + words[0]
        output = reversed_text
        """,
        "dangerous_import": """
        import os
        output = os.listdir("/")
        """,
        "dangerous_exec": """
        output = eval("open('secret.txt', 'r').read()")
        """,
        "complex_math": """
        import math
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(10)
        output = {"fibonacci_10": result}
        """
    }


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance testing fixtures
@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        "max_parsing_time_ms": 10,
        "max_execution_time_ms": 100,
        "max_memory_mb": 50,
        "min_code_coverage": 0.95
    }


# Helper fixtures
@pytest.fixture
def mock_async_functions():
    """Mock async functions for testing."""
    async def mock_search_claims(query: str, limit: int = 10):
        return [
            {"id": "c1", "content": f"Result for {query}", "confidence": 0.8}
            for _ in range(min(limit, 5))
        ]
    
    async def mock_create_claim(content: str, confidence: float = 0.5, tags: List[str] = None, created_by: str = "test"):
        return {"id": "c123", "content": content, "confidence": confidence, "tags": tags or []}
    
    return {
        "search_claims": mock_search_claims,
        "create_claim": mock_create_claim
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "missing_parameter": "Required parameter 'query' not provided",
        "invalid_parameter_type": "Parameter 'limit' must be of type int",
        "skill_not_found": "Skill 'unknown_skill' not found",
        "execution_timeout": "Execution timeout after 30.0 seconds",
        "security_violation": "Security validation failed: Dangerous function call: eval",
        "parsing_error": "No valid tool calls found in response",
        "database_error": "Failed to connect to database"
    }