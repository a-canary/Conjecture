#!/usr/bin/env python3
"""
Hybrid Backend for Conjecture CLI
Combines both local and cloud services for optimal performance
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.config import get_primary_provider, validate_config

from ..base_cli import BackendNotAvailableError, BaseCLI
from .cloud import CloudBackend
from .local import LocalBackend


class HybridBackend(BaseCLI):
    """Backend implementation that combines local and cloud services."""

    def __init__(self):
        super().__init__("conjecture-hybrid", "Conjecture CLI with Hybrid Services")
        self.local_backend = LocalBackend()
        self.cloud_backend = CloudBackend()
        self.preferred_mode = "auto"  # "local", "cloud", or "auto"

    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which backends are available."""
        return {
            "local": self.local_backend.is_available(),
            "cloud": self.cloud_backend.is_available(),
        }

    def is_available(self) -> bool:
        """Check if at least one backend is available."""
        available = self._detect_available_backends()
        return available["local"] or available["cloud"]

    def _select_best_backend(self, operation: str = "create") -> BaseCLI:
        """Select the best backend for the given operation."""
        available = self._detect_available_backends()

        # Specific logic for different operations
        if self.preferred_mode == "local" and available["local"]:
            return self.local_backend
        elif self.preferred_mode == "cloud" and available["cloud"]:
            return self.cloud_backend
        elif self.preferred_mode == "auto":
            # Auto selection based on operation type
            if operation in ["create", "analyze"]:
                # Prefer cloud for analysis (more powerful models)
                if available["cloud"]:
                    return self.cloud_backend
                elif available["local"]:
                    return self.local_backend
            elif operation in ["search", "get"]:
                # Prefer local for search (faster, offline capable)
                if available["local"]:
                    return self.local_backend
                elif available["cloud"]:
                    return self.cloud_backend

        # Fallback to whatever is available
        if available["local"]:
            return self.local_backend
        elif available["cloud"]:
            return self.cloud_backend

        raise BackendNotAvailableError("No backends are available")

    def set_preferred_mode(self, mode: str):
        """Set preferred mode for backend selection."""
        if mode not in ["local", "cloud", "auto"]:
            raise ValueError("Mode must be 'local', 'cloud', or 'auto'")
        self.preferred_mode = mode

def create_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        analyze: bool = False,
        tags: list = None,
        **kwargs,
    ) -> str:
        """Create a new claim using the best available backend."""
        if not self.is_available():
            raise BackendNotAvailableError("Hybrid backend is not properly configured")

        backend = self._select_best_backend("create")
        backend_name = backend._get_backend_type()

        self.console.print(
            f"[blue]🔄 Using {backend_name} backend for claim creation...[/blue]"
        )

        try:
            claim_id = backend.create_claim(
                content, confidence, user_id, analyze, tags, **kwargs
            )

            # Store which backend was used
            if hasattr(backend, "_init_database"):  # Ensure database is initialized
                backend._init_database()

            return claim_id
        except Exception as e:
            # Try fallback backend if preferred one fails
            available = self._detect_available_backends()
            fallback = None

            if isinstance(backend, LocalBackend) and available["cloud"]:
                fallback = self.cloud_backend
            elif isinstance(backend, CloudBackend) and available["local"]:
                fallback = self.local_backend

            if fallback:
                self.console.print(
                    f"[yellow]⚠️ {backend_name} backend failed, trying {fallback._get_backend_type()}...[/yellow]"
                )
                return fallback.create_claim(
                    content, confidence, user, analyze, **kwargs
                )
            else:
                raise

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID (uses local database by default)."""
        if not self.is_available():
            raise BackendNotAvailableError("Hybrid backend is not properly configured")

        backend = self._select_best_backend("get")
        return backend.get_claim(claim_id)

    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims using the best available backend."""
        if not self.is_available():
            raise BackendNotAvailableError("Hybrid backend is not properly configured")

        backend = self._select_best_backend("search")
        backend_name = backend._get_backend_type()

        self.console.print(
            f"[blue]🔍 Using {backend_name} backend for search...[/blue]"
        )

        return backend.search_claims(query, limit, **kwargs)

    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using cloud services if available, otherwise local."""
        if not self.is_available():
            raise BackendNotAvailableError("Hybrid backend is not properly configured")

        available = self._detect_available_backends()

        # For analysis, prefer cloud for more powerful models
        if available["cloud"]:
            self.console.print(
                "[blue]🧠 Using cloud services for enhanced analysis...[/blue]"
            )
            analysis = self.cloud_backend.analyze_claim(claim_id, **kwargs)
            analysis["hybrid_mode"] = "cloud_preferred"
        elif available["local"]:
            self.console.print("[blue]💻 Using local services for analysis...[/blue]")
            analysis = self.local_backend.analyze_claim(claim_id, **kwargs)
            analysis["hybrid_mode"] = "local_fallback"
        else:
            raise BackendNotAvailableError("No backends available for analysis")

return analysis

    def process_prompt(
        self, 
        prompt_text: str, 
        confidence: float = 0.8,
        verbose: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation using hybrid backend."""
        if not self.is_available():
            raise BackendNotAvailableError("Hybrid backend is not properly configured")

        available = self._detect_available_backends()

        # For prompt processing, prefer cloud for better responses
        if available["cloud"]:
            if verbose >= 1:
                self.console.print(
                    "[blue]🧠 Using cloud services for prompt processing...[/blue]"
                )
            result = self.cloud_backend.process_prompt(prompt_text, confidence, verbose, **kwargs)
            result["hybrid_mode"] = "cloud_preferred"
        elif available["local"]:
            if verbose >= 1:
                self.console.print("[blue]💻 Using local services for prompt processing...[/blue]")
            result = self.local_backend.process_prompt(prompt_text, confidence, verbose, **kwargs)
            result["hybrid_mode"] = "local_fallback"
        else:
            raise BackendNotAvailableError("No backends available for prompt processing")

        return result

    def get_hybrid_status(self) -> Dict[str, Any]:
        """Get comprehensive status of hybrid backend."""
        local_status = self.local_backend.get_local_services_status()
        cloud_status = self.cloud_backend.get_cloud_services_status()
        available = self._detect_available_backends()

        status = {
            "backend_type": "hybrid",
            "available": self.is_available(),
            "preferred_mode": self.preferred_mode,
            "backends": {"local": local_status, "cloud": cloud_status},
            "active_backends": [
                name for name, is_available in available.items() if is_available
            ],
            "fallback_enabled": True,
        }

        return status

    def create_cross_backend_analysis(self, claim_id: str) -> Dict[str, Any]:
        """Create analysis using both backends for comparison."""
        if (
            not self._detect_available_backends()["local"]
            or not self._detect_available_backends()["cloud"]
        ):
            raise BackendNotAvailableError(
                "Both local and cloud backends required for cross-backend analysis"
            )

        self.console.print("[blue]🔄 Running cross-backend analysis...[/blue]")

        # Get claim
        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        # Analyze with both backends
        local_analysis = self.local_backend.analyze_claim(claim_id)
        cloud_analysis = self.cloud_backend.analyze_claim(claim_id)

        # Compare results
        comparison = {
            "claim_id": claim_id,
            "cross_backend_analysis": True,
            "local_analysis": local_analysis,
            "cloud_analysis": cloud_analysis,
            "comparison": {
                "local_sentiment": local_analysis.get("sentiment"),
                "cloud_sentiment": cloud_analysis.get("sentiment"),
                "sentiment_match": local_analysis.get("sentiment")
                == cloud_analysis.get("sentiment"),
                "local_topics": local_analysis.get("topics", []),
                "cloud_topics": cloud_analysis.get("topics", []),
                "topic_overlap": set(local_analysis.get("topics", [])).intersection(
                    set(cloud_analysis.get("topics", []))
                ),
            },
            "recommendation": "cloud"
            if cloud_analysis.get("verification_status") == "verified"
            else "local",
        }

        self.console.print("[green]✅ Cross-backend analysis complete[/green]")
        return comparison

    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize performance by selecting best backend for different operations."""
        available = self._detect_available_backends()
        optimization = {
            "optimization_applied": True,
            "current_mode": self.preferred_mode,
            "recommendations": {},
        }

        if available["local"] and available["cloud"]:
            optimization["recommendations"] = {
                "create": "local" if self.preferred_mode == "local" else "cloud",
                "search": "local",  # Local is faster for search
                "analyze": "cloud",  # Cloud is better for analysis
                "get": "local",  # Use local database
            }
            optimization["optimization_type"] = "auto"
        elif available["local"]:
            optimization["recommendations"] = {
                "create": "local",
                "search": "local",
                "analyze": "local",
                "get": "local",
            }
            optimization["optimization_type"] = "local_only"
        elif available["cloud"]:
            optimization["recommendations"] = {
                "create": "cloud",
                "search": "cloud",
                "analyze": "cloud",
                "get": "cloud",
            }
            optimization["optimization_type"] = "cloud_only"

        return optimization
