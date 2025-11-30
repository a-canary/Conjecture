"""
Hybrid Backend for Conjecture CLI
Provides intelligent switching between local and cloud backends
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from ..base_cli import BaseCLI, BackendNotAvailableError


class HybridBackend(BaseCLI):
    """Hybrid backend implementation that switches between local and cloud."""

    def __init__(self, preferred_mode: str = "local"):
        super().__init__(name="hybrid", help_text="Hybrid LLM backend")
        self.preferred_mode = preferred_mode  # "local", "cloud", or "auto"
        self.console = Console()
        self.error_console = Console(stderr=True)

        # Import backend classes
        from .local import LocalBackend
        from .cloud import CloudBackend

        self.local_backend = LocalBackend()
        self.cloud_backend = CloudBackend()

    def is_available(self) -> bool:
        """Check if at least one backend is available."""
        return self.local_backend.is_available() or self.cloud_backend.is_available()

    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which backends are available."""
        return {
            "local": self.local_backend.is_available(),
            "cloud": self.cloud_backend.is_available(),
        }

    def set_preferred_mode(self, mode: str) -> None:
        """Set the preferred mode for backend selection."""
        if mode in ["local", "cloud", "auto"]:
            self.preferred_mode = mode
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'local', 'cloud', or 'auto'"
            )

    def _select_best_backend(self, operation: str = "general") -> BaseCLI:
        """Select the best backend for the given operation."""
        available = self._detect_available_backends()

        if self.preferred_mode == "local" and available["local"]:
            return self.local_backend
        elif self.preferred_mode == "cloud" and available["cloud"]:
            return self.cloud_backend
        elif available["local"]:
            return self.local_backend
        elif available["cloud"]:
            return self.cloud_backend
        else:
            raise BackendNotAvailableError("No backends are available")

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
            f"[blue][HYBRID] Using {backend_name} backend for claim creation...[/blue]"
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
            fallback_backends = [
                b
                for b in [self.local_backend, self.cloud_backend]
                if b.is_available() and b != backend
            ]

            if fallback_backends:
                fallback = fallback_backends[0]
                fallback_name = fallback._get_backend_type()
                self.console.print(
                    f"[yellow][WARN] {backend_name} failed, trying {fallback_name} fallback...[/yellow]"
                )
                return fallback.create_claim(
                    content, confidence, user_id, analyze, tags, **kwargs
                )
            else:
                raise e

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        backend = self._select_best_backend("get")
        return backend.get_claim(claim_id)

    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims using the best available backend."""
        backend = self._select_best_backend("search")

        self.console.print(
            f"[blue][HYBRID] Using {backend._get_backend_type()} backend for search[/blue]"
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
                "[blue][HYBRID] Using cloud services for enhanced analysis...[/blue]"
            )
            analysis = self.cloud_backend.analyze_claim(claim_id, **kwargs)
            analysis["hybrid_mode"] = "cloud_preferred"
        elif available["local"]:
            self.console.print(
                "[blue][HYBRID] Using local services for analysis...[/blue]"
            )
            analysis = self.local_backend.analyze_claim(claim_id, **kwargs)
            analysis["hybrid_mode"] = "local_fallback"
        else:
            raise BackendNotAvailableError("No backends available for analysis")

        return analysis

    def process_prompt(
        self, prompt_text: str, confidence: float = 0.8, verbose: int = 0, **kwargs
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
            result = self.cloud_backend.process_prompt(
                prompt_text, confidence, verbose, **kwargs
            )
            result["hybrid_mode"] = "cloud_preferred"
        elif available["local"]:
            if verbose >= 1:
                self.console.print(
                    "[blue]💻 Using local services for prompt processing...[/blue]"
                )
            result = self.local_backend.process_prompt(
                prompt_text, confidence, verbose, **kwargs
            )
            result["hybrid_mode"] = "local_fallback"
        else:
            raise BackendNotAvailableError(
                "No backends available for prompt processing"
            )

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
            raise BackendNotAvailableError("Both backends required for cross-analysis")

        self.console.print("[blue]🔄 Running cross-backend analysis...[/blue]")

        local_analysis = self.local_backend.analyze_claim(claim_id)
        cloud_analysis = self.cloud_backend.analyze_claim(claim_id)

        return {
            "claim_id": claim_id,
            "local_analysis": local_analysis,
            "cloud_analysis": cloud_analysis,
            "comparison": {
                "confidence_diff": cloud_analysis.get("confidence_score", 0)
                - local_analysis.get("confidence_score", 0),
                "agreement": local_analysis.get("sentiment")
                == cloud_analysis.get("sentiment"),
            },
        }
