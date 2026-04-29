"""
Web Search Utilities for Conjecture

Implements F-0005: Web Research Integration
Automatically gather evidence from the web to support or challenge claims.
Web search results become claims with their own confidence scores and source attribution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from src.data.models import ClaimType

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """A single web search result ready to become a claim.

    Per F-0005: web search results become claims with confidence scores and source attribution.
    Per F-0007: search results use REFERENCE claim type.
    """

    query: str
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"
    confidence: float = 0.6  # Moderate confidence — not primary evidence
    claim_types: List[ClaimType] = field(
        default_factory=lambda: [ClaimType.REFERENCE]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_claim_content(self) -> str:
        """Format search result as claim-ready content with provenance.

        Returns content that includes title, URL (source attribution),
        and the relevant snippet for the claim.
        """
        return (
            f"{self.title}\n"
            f"Source: {self.url}\n"
            f"Query: {self.query}\n"
            f"Summary: {self.snippet}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage or API response."""
        return {
            "query": self.query,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "confidence": self.confidence,
            "claim_types": [ct.value for ct in self.claim_types],
            "metadata": self.metadata,
        }


class DuckDuckGoSearch:
    """DuckDuckGo web search client.

    Uses the DuckDuckGo HTML lite API (no API key required).
    Search results are returned as WebSearchResult objects ready for claim creation.
    """

    BASE_URL = "https://duckduckgo.com/html/"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def search(
        self, query: str, max_results: int = 10
    ) -> List[WebSearchResult]:
        """Search DuckDuckGo and return results as WebSearchResult objects.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20)

        Returns:
            List of WebSearchResult objects, each ready to become a claim.
        """
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 20))

        try:
            return await self._fetch_results(query, max_results)
        except Exception as exc:
            logger.warning(f"DuckDuckGo search failed for '{query}': {exc}")
            return []

    async def _fetch_results(
        self, query: str, max_results: int
    ) -> List[WebSearchResult]:
        """Fetch and parse results from DuckDuckGo HTML API."""
        params = {"q": query}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.BASE_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Accept": "application/json"},
            ) as response:
                response.raise_for_status()
                data = await response.json()

        return self._parse_results(data, query, max_results)

    def _parse_results(
        self, data: Dict[str, Any], query: str, max_results: int
    ) -> List[WebSearchResult]:
        """Parse JSON response into WebSearchResult objects."""
        raw_results = data.get("results", [])
        results = []

        for i, raw in enumerate(raw_results[:max_results]):
            result = WebSearchResult(
                query=query,
                title=raw.get("title", ""),
                url=raw.get("url", ""),
                snippet=raw.get("body", raw.get("description", "")),
                source="duckduckgo",
                confidence=self._compute_confidence(raw, i),
                claim_types=[ClaimType.REFERENCE],
                metadata={
                    "position": i,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            results.append(result)

        return results

    def _compute_confidence(self, raw: Dict[str, Any], position: int) -> float:
        """Assign confidence based on result position and content quality.

        Top results are more reliable. Confidence ranges 0.4-0.7.
        """
        base = 0.65 - (position * 0.03)  # Position decay
        # Penalize short snippets
        body = raw.get("body", "")
        if len(body) < 50:
            base -= 0.05
        return max(0.4, min(0.7, base))
