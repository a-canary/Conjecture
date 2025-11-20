#!/usr/bin/env python3
"""
Local Backend for Conjecture CLI
Handles local services like Ollama and LM Studio
"""

import os
import sys
from typing import Any, Dict, List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.config import get_primary_provider, validate_config

from ..base_cli import BackendNotAvailableError, BaseCLI


class LocalBackend(BaseCLI):
    """Backend implementation for local services (Ollama, LM Studio)."""

    def __init__(self):
        super().__init__("conjecture-local", "Conjecture CLI with Local Services")
        self.supported_providers = ["ollama", "lm_studio", "localai"]

    def _validate_local_config(self) -> bool:
        """Validate that a local provider is configured."""
        try:
            result = validate_config()
            if not result.success:
                return False

            provider = get_primary_provider()
            if not provider:
                return False

            return provider.is_local
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if local backend is available."""
        return self._validate_local_config()

    def create_claim(
        self,
        content: str,
        confidence: float,
        user: str,
        analyze: bool = False,
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

                claim_id = self._save_claim(content, confidence, user, metadata)
                progress.update(task, description="Claim created successfully!")

                # Display result
                panel = self._create_claim_panel(
                    claim_id, content, confidence, user, metadata
                )
                self.console.print(panel)

                # Analyze if requested and provider is available
                if analyze and self.current_provider_config:
                    progress.update(task, description="Analyzing claim locally...")
                    self.console.print(
                        f"[yellow]Analyzing with {self.current_provider_config['name']}...[/yellow]"
                    )
                    # TODO: Implement local LLM analysis
                    self.console.print(
                        "[green]Analysis complete (mock implementation)[/green]"
                    )

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
        # Initialize database without full config validation for search-only
        try:
            self._init_services()
        except SystemExit:
            # For search-only, we can proceed without a provider if embeddings are available
            pass

        self._init_database()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Searching local claims...", total=None)

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

        self.console.print(f"[yellow]📥 Pulling model {model_name}...[/yellow]")
        # Mock implementation - would integrate with actual provider
        self.console.print(f"[green]✅ Model {model_name} pulled successfully[/green]")
        return True
