"""
Dynamic Tool Creation System for the Conjecture skill-based agency system.
Handles LLM-driven tool discovery, creation, and validation.
"""

import asyncio
import re
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib

from .tool_manager import ToolManager
from .response_parser import ResponseParser
from .tool_executor import ToolExecutor, ExecutionLimits
from ..data.models import Claim
from ..data.data_manager import DataManager

logger = logging.getLogger(__name__)

class ToolDiscoveryEngine:
    """Discovers tool creation methods through web search and analysis."""

    def __init__(self):
        self.search_patterns = {
            "weather": [
                "how to get weather forecast by zipcode python",
                "python weather api zipcode",
                "weather data python library zipcode",
            ],
            "calculator": [
                "python calculator function basic operations",
                "how to create calculator in python",
                "python math operations function",
            ],
            "search": [
                "python search function text",
                "how to implement search in python",
                "python text search algorithm",
            ],
        }

    async def discover_tool_need(
        self, claim_content: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Analyze claim content to determine if a new tool is needed.

        Args:
            claim_content: Content of the claim being evaluated
            context: Additional context for analysis

        Returns:
            Tool need description or None if no tool needed
        """
        # Look for patterns that indicate tool needs
        tool_need_patterns = [
            r"need to.*(?:get|fetch|calculate|compute|search|find)",
            r"how to.*(?:weather|calculate|search|lookup)",
            r"want to.*(?:get|fetch|calculate|compute)",
            r"require.*(?:function|method|tool|utility)",
            r"looking for.*way to.*(?:get|fetch|calculate)",
        ]

        for pattern in tool_need_patterns:
            if re.search(pattern, claim_content, re.IGNORECASE):
                # Extract the specific need
                match = re.search(pattern, claim_content, re.IGNORECASE)
                if match:
                    return match.group(0)

        return None

    async def suggest_search_queries(self, tool_need: str) -> List[str]:
        """
        Suggest search queries for discovering tool creation methods.

        Args:
            tool_need: Description of the tool need

        Returns:
            List of search query suggestions
        """
        # Extract key terms from the need
        key_terms = []

        # Common tool categories
        categories = {
            "weather": ["weather", "forecast", "temperature", "zipcode", "climate"],
            "calculator": ["calculate", "compute", "math", "arithmetic", "operation"],
            "search": ["search", "find", "lookup", "query", "filter"],
            "data": ["data", "parse", "process", "transform", "convert"],
            "web": ["web", "http", "api", "request", "fetch"],
            "file": ["file", "read", "write", "save", "load"],
        }

        need_lower = tool_need.lower()

        for category, terms in categories.items():
            if any(term in need_lower for term in terms):
                key_terms.append(category)

        # Generate queries based on identified categories
        queries = []

        for category in key_terms:
            if category in self.search_patterns:
                queries.extend(self.search_patterns[category])

        # Generate generic queries if no specific category found
        if not queries:
            generic_queries = [
                f"python {tool_need} function",
                f"how to implement {tool_need} in python",
                f"python library for {tool_need}",
                f"{tool_need} python code example",
            ]
            queries.extend(generic_queries)

        return list(set(queries))  # Remove duplicates

    async def analyze_search_results(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze search results to find suitable tool creation methods.

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of analyzed methods with relevance scores
        """
        analyzed_methods = []

        for result in search_results:
            method = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "relevance_score": 0.0,
                "code_indicators": [],
                "complexity": "unknown",
            }

            # Look for code indicators
            code_patterns = [
                r"```python",
                r"def ",
                r"import ",
                r"from ",
                r"function",
                r"example",
                r"tutorial",
                r"code",
            ]

            for pattern in code_patterns:
                if re.search(pattern, method["snippet"], re.IGNORECASE):
                    method["code_indicators"].append(pattern)
                    method["relevance_score"] += 0.2

            # Look for simplicity indicators
            simple_patterns = [r"simple", r"basic", r"easy", r"quick", r"beginner"]

            for pattern in simple_patterns:
                if re.search(pattern, method["snippet"], re.IGNORECASE):
                    method["relevance_score"] += 0.1
                    method["complexity"] = "simple"

            # Look for complexity indicators
            complex_patterns = [r"advanced", r"complex", r"difficult", r"expert"]

            for pattern in complex_patterns:
                if re.search(pattern, method["snippet"], re.IGNORECASE):
                    method["relevance_score"] -= 0.1
                    method["complexity"] = "complex"

            # Cap relevance score
            method["relevance_score"] = min(1.0, max(0.0, method["relevance_score"]))

            if method["relevance_score"] > 0.3:  # Only include relevant results
                analyzed_methods.append(method)

        # Sort by relevance score
        analyzed_methods.sort(key=lambda x: x["relevance_score"], reverse=True)

        return analyzed_methods

class ToolCodeGenerator:
    """Generates Python code for tools based on discovered methods."""

    def __init__(self):
        self.code_templates = {
            "weather": '''
def get_weather_by_zipcode(zipcode: str) -> dict:
    """
    Get weather information for a given zipcode.
    
    Args:
        zipcode: 5-digit zipcode string
        
    Returns:
        Dictionary with weather information including temperature, conditions, etc.
    """
    import json
    import urllib.request
    import urllib.parse
    
    # Validate zipcode format
    if not zipcode.isdigit() or len(zipcode) != 5:
        return {{"error": "Invalid zipcode format. Use 5-digit zipcode."}}
    
    try:
        # Use a free weather API (you may need to sign up for API key)
        # For demo purposes, returning test data
        return {{
            "zipcode": zipcode,
            "temperature": "72°F",
            "conditions": "Sunny",
            "humidity": "45%",
            "source": "Demo Data"
        }}
    except Exception as e:
        return {{"error": f"Weather data unavailable: {{str(e)}}"}}
            '''
        }
        