"""
Web Search Tool for Conjecture
Provides simple DuckDuckGo search capabilities with basic validation
"""

from typing import List, Dict, Any
from ddgs import DDGS

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tools.registry import register_tool

@register_tool(name="WebSearch", is_core=True)
def webSearch(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo for information

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of search results with title, url, snippet, and metadata
    """
    # Basic input validation
    if not query or not query.strip():
        return [{
            'title': 'Error',
            'url': '',
            'snippet': 'Query cannot be empty',
            'type': 'error',
            'source': 'validation'
        }]

    if len(query) > 500:
        return [{
            'title': 'Error', 
            'url': '',
            'snippet': 'Query too long (max 500 characters)',
            'type': 'error',
            'source': 'validation'
        }]

    # Limit and normalize max_results
    if not isinstance(max_results, int):
        max_results = 10
    max_results = max(1, min(20, max_results))

    try:
        # Use ddgs library directly - no complex parsing needed
        ddgs = DDGS()
        raw_results = list(ddgs.text(query.strip(), max_results=max_results))

        # Simple format conversion - no HTML stripping (DuckDuckGo returns clean text)
        formatted_results = []
        for result in raw_results:
            formatted_results.append({
                'title': str(result.get('title', ''))[:200],
                'url': str(result.get('href', ''))[:500], 
                'snippet': str(result.get('body', ''))[:1000],
                'type': 'web_result',
                'source': 'duckduckgo'
            })

        return formatted_results

    except Exception:
        # Generic error message - no sensitive details exposed
        return [{
            'title': 'Search Failed',
            'url': '',
            'snippet': 'Search service temporarily unavailable',
            'type': 'error',
            'source': 'duckduckgo'
        }]

def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use this tool
    """
    return [
        "webSearch('Rust game development tutorial') returns list of tutorials and guides for creating games in Rust",
        "webSearch('minesweeper game rules') returns official rules, variations, and gameplay mechanics of minesweeper",
        "webSearch('Rust terminal UI libraries') returns available crates like ratatui, crossterm for TUI development",
        "webSearch('how to create Cargo project') returns step-by-step guides for initializing new Rust projects",
        "webSearch('minesweeper algorithm implementation') returns code examples and logic for mine placement and number calculation",
        "webSearch('Rust error handling best practices') returns patterns and examples for proper error handling in Rust",
        "webSearch('terminal game input handling Rust') returns methods for handling keyboard input in terminal applications"
    ]

if __name__ == "__main__":
    # Test the web search functionality
    print("Testing webSearch tool...")

    test_queries = [
        "Rust game development",
        "minesweeper rules",
        "Rust TUI libraries"
    ]

    for query in test_queries:
        print(f"\nSearching: {query}")
        results = webSearch(query, max_results=3)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()

    print("Examples for LLM context:")
    for example in examples():
        print(f"- {example}")