"""
Test suite for src/utils/web_search.py
Implements F-0005: Web Research Integration
Web search results become claims with confidence scores and source attribution.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import asdict


class TestWebSearchResult:
    """Tests for WebSearchResult dataclass"""

    def test_web_search_result_creation(self):
        """Test WebSearchResult can be created with all fields"""
        from src.utils.web_search import WebSearchResult

        result = WebSearchResult(
            query="quantum computing",
            title="Quantum Computing Overview",
            url="https://example.com/quantum",
            snippet="Quantum computing harnesses quantum mechanics...",
            source="duckduckgo",
        )

        assert result.query == "quantum computing"
        assert result.title == "Quantum Computing Overview"
        assert result.url == "https://example.com/quantum"
        assert "quantum" in result.snippet.lower()
        assert result.source == "duckduckgo"

    def test_web_search_result_to_claim_content(self):
        """Test WebSearchResult generates claim-ready content"""
        from src.utils.web_search import WebSearchResult

        result = WebSearchResult(
            query="python programming",
            title="Python Official Documentation",
            url="https://docs.python.org",
            snippet="Python is a high-level programming language...",
            source="duckduckgo",
        )

        content = result.to_claim_content()

        assert "Python Official Documentation" in content
        assert "https://docs.python.org" in content
        assert "python programming" in content.lower()

    def test_web_search_result_to_dict(self):
        """Test WebSearchResult serializes to dict"""
        from src.utils.web_search import WebSearchResult

        result = WebSearchResult(
            query="machine learning",
            title="ML Basics",
            url="https://example.com/ml",
            snippet="Machine learning is a subset of AI.",
            source="duckduckgo",
        )

        d = result.to_dict()

        assert d["query"] == "machine learning"
        assert d["title"] == "ML Basics"
        assert d["url"] == "https://example.com/ml"
        assert d["source"] == "duckduckgo"
        assert "metadata" in d


class TestDuckDuckGoSearch:
    """Tests for DuckDuckGoSearch client"""

    @pytest.fixture
    def search_client(self):
        """Create a DuckDuckGoSearch client for testing"""
        from src.utils.web_search import DuckDuckGoSearch

        return DuckDuckGoSearch(timeout=5.0)

    @pytest.mark.asyncio
    async def test_search_returns_results(self, search_client):
        """Test search returns a list of WebSearchResult objects"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Python Homepage",
                        "url": "https://python.org",
                        "body": "The Python Programming Language.",
                    }
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("python programming")

            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0].title == "Python Homepage"
            assert results[0].url == "https://python.org"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self, search_client):
        """Test empty query returns empty list"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            results = await search_client.search("")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_http_error(self, search_client):
        """Test search handles HTTP errors gracefully"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 429
            mock_response.raise_for_status.side_effect = Exception("Rate limited")
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("python")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_timeout(self, search_client):
        """Test search handles timeout errors gracefully"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.return_value.__aenter__.side_effect = Exception("Timeout")

            results = await search_client.search("python")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_multiple_results(self, search_client):
        """Test search returns multiple results when available"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "body": "Body of result 1",
                    },
                    {
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "body": "Body of result 2",
                    },
                    {
                        "title": "Result 3",
                        "url": "https://example.com/3",
                        "body": "Body of result 3",
                    },
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("example query")

            assert len(results) == 3
            assert results[0].title == "Result 1"
            assert results[1].title == "Result 2"
            assert results[2].title == "Result 3"

    def test_search_result_confidence_assignment(self, search_client):
        """Test web search results get appropriate confidence scores"""
        from src.utils.web_search import WebSearchResult

        result = WebSearchResult(
            query="test",
            title="Trusted Source",
            url="https://trusted-site.com",
            snippet="Reliable information here.",
            source="duckduckgo",
        )

        # Results from web search are not primary evidence
        # they should have moderate confidence
        assert 0.4 <= result.confidence <= 0.8

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self, search_client):
        """Test search respects max_results parameter"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {"title": f"R{i}", "url": f"https://e.com/{i}", "body": f"B{i}"}
                    for i in range(10)
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("test", max_results=3)

            assert len(results) == 3

    def test_client_default_timeout(self):
        """Test client uses sensible default timeout"""
        from src.utils.web_search import DuckDuckGoSearch

        client = DuckDuckGoSearch()
        assert client.timeout == 10.0

    def test_client_custom_timeout(self):
        """Test client accepts custom timeout"""
        from src.utils.web_search import DuckDuckGoSearch

        client = DuckDuckGoSearch(timeout=30.0)
        assert client.timeout == 30.0


class TestWebSearchEvidenceIntegration:
    """Integration tests for web search → claim pipeline (F-0005)"""

    @pytest.fixture
    def search_client(self):
        from src.utils.web_search import DuckDuckGoSearch

        return DuckDuckGoSearch(timeout=5.0)

    @pytest.mark.asyncio
    async def test_search_results_formatted_for_claim_creation(
        self, search_client
    ):
        """Test search results are ready to become claims (F-0005 requirement)"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Climate Change Evidence",
                        "url": "https://climate-study.org/evidence",
                        "body": "Global temperatures increased 1.1°C since 1880.",
                    },
                    {
                        "title": "NASA Climate Portal",
                        "url": "https://climate.nasa.gov",
                        "body": "NASA's official climate change resources.",
                    },
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("climate change evidence")

            # Each result is ready to become a claim
            for result in results:
                claim_content = result.to_claim_content()
                # Must have source attribution (URL) for provenance
                assert result.url in claim_content
                # Must have the topic
                assert "climate change" in claim_content.lower()
                # Must have result title
                assert result.title in claim_content

    def test_search_result_has_reference_claim_type(self):
        """Test search results have REFERENCE claim type (F-0007)"""
        from src.utils.web_search import WebSearchResult
        from src.data.models import ClaimType

        result = WebSearchResult(
            query="test",
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
            source="duckduckgo",
        )

        # Web search results are REFERENCE type claims per F-0007
        assert ClaimType.REFERENCE in result.claim_types

    @pytest.mark.asyncio
    async def test_search_includes_source_attribution(self, search_client):
        """Test search results include source attribution per F-0005"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "results": [
                    {
                        "title": "Source Article",
                        "url": "https://source.example.com/article",
                        "body": "Article content here.",
                    }
                ]
            }
            mock_get.return_value.__aenter__.return_value = mock_response

            results = await search_client.search("source example")

            assert len(results) == 1
            # Source attribution (URL) is present for provenance tracking
            assert "source.example.com" in results[0].url
            assert results[0].snippet == "Article content here."
