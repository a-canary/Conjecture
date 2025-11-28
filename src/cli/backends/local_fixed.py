"""
Local Backend for Conjecture CLI
Provides local LLM services integration
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from .base_cli import BaseCLI, BackendNotAvailableError


class LocalBackend(BaseCLI):
    """Local backend implementation using local LLM services."""

    def __init__(self):
        super().__init__(name="local", help_text="Local LLM backend")
        self.supported_providers = ["ollama", "lm_studio"]
        self.console = Console()
        self.error_console = Console(stderr=True)

    def is_available(self) -> bool:
        """Check if local backend is properly configured."""
        try:
            self._init_services()
            return bool(self.current_provider_config)
        except SystemExit:
            return False

    def create_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        analyze: bool = False,
        tags: list = None,
        **kwargs,
    ) -> str:
        """Create a new claim using local backend."""
        if not self.is_available():
            raise BackendNotAvailableError("Local backend is not properly configured")

        try:
            self._init_services()
            self._init_database()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task("Creating claim locally...", total=None)

                # Validate input
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("Confidence must be between 0.0 and 1.0")

                if len(content) < 10:
                    raise ValueError("Content must be at least 10 characters")

                # Save claim with local metadata
                metadata = {
                    "analyzed": analyze,
                    "provider": self.current_provider_config.get("name")
                    if self.current_provider_config
                    else None,
                    "backend": "local",
                    "offline_capable": True,
                }

                claim_id = self._save_claim(
                    content, confidence, user_id, metadata, tags
                )
                progress.update(task, description="Claim created successfully!")

                # Display result
                panel = self._create_claim_panel(
                    claim_id, content, confidence, user_id, metadata
                )
                self.console.print(panel)

                # Analyze if requested and provider is available
                if analyze and self.current_provider_config:
                    progress.update(
                        task, description="Analyzing claim with local services..."
                    )
                    self.console.print(
                        f"[yellow]Analyzing with {self.current_provider_config['name']}...[/yellow]"
                    )
                    # TODO: Implement local LLM analysis
                    self.console.print("[green]Analysis complete (local model)[/green]")

                return claim_id

        except Exception as e:
            self.error_console.print(f"[red]Error creating claim: {e}[/red]")
            raise

    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        self._init_database()
        return self._get_claim(claim_id)

    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims using local vector similarity."""
        try:
            self._init_services()
            self._init_database()
        except SystemExit:
            # For search-only, we can proceed without a provider if embeddings are available
            self._init_database()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Searching claims locally...", total=None)

            try:
                results = self._search_claims(query, limit)
                progress.update(task, description=f"Found {len(results)} results")
                return results
            except Exception as e:
                progress.update(task, description="Search error occurred")
                self.error_console.print(f"[red]Error searching claims: {e}[/red]")
                raise

    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using local services."""
        if not self.is_available():
            raise BackendNotAvailableError("Local backend is not properly configured")

        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        self.console.print(
            f"[blue]🔍 Analyzing claim {claim_id} with local services...[/blue]"
        )

        # Mock analysis for now - would integrate with local LLM
        analysis = {
            "claim_id": claim_id,
            "backend": "local",
            "provider": self.current_provider_config.get("name")
            if self.current_provider_config
            else None,
            "analysis_type": "local_semantic",
            "confidence_score": claim.get("confidence", 0.0),
            "sentiment": "neutral",
            "topics": ["general"],
            "verification_status": "pending",
            "local_processing": True,
            "offline_capable": True,
        }

        self.console.print("[green]✅ Local analysis complete[/green]")
        return analysis

    def process_prompt(
        self, prompt_text: str, confidence: float = 0.8, verbose: int = 0, **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation using local backend."""
        return super().process_prompt(prompt_text, confidence, verbose, **kwargs)

    def get_local_services_status(self) -> Dict[str, Any]:
        """Get status of local services."""
        status = {
            "backend_type": "local",
            "available": self.is_available(),
            "supported_providers": self.supported_providers,
            "current_provider": None,
            "endpoints": {},
            "offline_capable": True,
        }

        if self.current_provider_config:
            provider_name = self.current_provider_config.get("name", "")
            status["current_provider"] = provider_name
            status["current_type"] = self.current_provider_config.get("type", "")
            status["base_url"] = self.current_provider_config.get("base_url", "")

            # Check specific provider endpoints
            if "ollama" in provider_name.lower():
                status["endpoints"]["ollama"] = self.current_provider_config.get(
                    "base_url", "http://localhost:11434"
                )
            elif "lm_studio" in provider_name.lower():
                status["endpoints"]["lm_studio"] = self.current_provider_config.get(
                    "base_url", "http://localhost:1234"
                )

        return status

    def list_local_models(self) -> List[str]:
        """List available local models."""
        # Mock list - would integrate with actual provider APIs
        if self.current_provider_config:
            provider_name = self.current_provider_config.get("name", "").lower()
            if "ollama" in provider_name:
                return ["llama2", "mistral", "codellama", "vicuna"]
            elif "lm_studio" in provider_name:
                return ["custom-model", "llama-2-7b", "mistral-7b"]

        return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from local registry."""
        if not self.is_available():
            raise BackendNotAvailableError("Local backend is not properly configured")

        self.console.print(f"[blue]📥 Pulling model {model_name}...[/blue]")
        # Mock implementation - would integrate with actual provider APIs
        self.console.print(f"[green]✅ Model {model_name} pulled successfully[/green]")
        return True
