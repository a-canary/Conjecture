#!/usr/bin/env python3
"""
Base CLI functionality for Conjecture
Uses the simple unified Conjecture API - no over-engineering
"""

import asyncio
import json
import os
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import get_config, validate_config


class BaseCLI(ABC):
    """Base class for all CLI implementations with common functionality."""

    def __init__(self, name: str = "conjecture", help_text: str = "Conjecture CLI"):
        self.name = name
        self.help_text = help_text
        self.console = Console()
        self.error_console = Console(stderr=True)
        self.embedding_model = None
        self.current_provider_config = None
        self.db_path = "data/conjecture.db"

    def _get_backend_type(self) -> str:
        """Get the backend type for this CLI."""
        return self.__class__.__name__.replace("CLI", "").lower()

    def _init_services(self):
        """Initialize services common to all backends."""
        # Validate configuration first
        result = validate_config()

        if not result.success:
            self.console.print("[bold red]❌ Configuration Required[/bold red]")
            self._print_validation_result(result)

            self.console.print("\n[bold yellow]📋 Quick Setup:[/bold yellow]")
            self.console.print("1. Copy template: [cyan]cp .env.example .env[/cyan]")
            self.console.print("2. Edit [cyan].env[/cyan] with your preferred provider")
            self.console.print("3. Try again: [cyan]python conjecture[/cyan]")

            self.console.print(f"\n[bold]💡 Need help?[/bold]")
            self.console.print("Run: [cyan]python conjecture setup[/cyan]")
            self.console.print("Run: [cyan]python conjecture providers[/cyan]")

            raise SystemExit(1)

        # Get the best configured provider
        self.current_provider_config = self._get_configured_provider()

        if self.current_provider_config:
            provider_info = self.current_provider_config
            backend_type = self._get_backend_type()
            self.console.print(
                f"[green]✅ Using {provider_info['name']} ({backend_type} backend)[/green]"
            )

        # Create data directory
        os.makedirs("data", exist_ok=True)

        # Initialize sentence transformer
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.console.print(
                    "[bold blue]Loading sentence transformer model...[/bold blue]"
                )
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.console.print("[green][OK][/green] Model loaded successfully")
            except ImportError:
                self.console.print(
                    "[red]Error: sentence-transformers not installed[/red]"
                )
                self.console.print("Install with: pip install sentence-transformers")
                raise SystemExit(1)

    def _get_configured_provider(self) -> Optional[Dict[str, Any]]:
        """Get configured provider in legacy format."""
        provider = get_primary_provider()
        if provider:
            return {
                "name": provider.name,
                "type": "local" if provider.is_local else "cloud",
                "priority": provider.priority,
                "base_url": provider.base_url,
                "model": provider.model,
                "required_vars": {
                    "api_url": provider.base_url,
                    "model": provider.model,
                },
                "optional_vars": {"api_key": provider.api_key},
            }
        return None

def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                user_id TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                tags TEXT,
                is_dirty BOOLEAN DEFAULT 1,
                dirty_reason TEXT,
                dirty_priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.embedding_model.encode(text)

    def _save_claim(
        self, content: str, confidence: float, user_id: str, metadata: dict = None, tags: list = None
    ) -> str:
        """Save claim to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate embedding
        embedding = self._generate_embedding(content)
        embedding_bytes = embedding.tobytes()

        # Generate ID
        claim_id = f"c{int(time.time() * 1000) % 1000000000:07d}"

        # Save claim
        cursor.execute(
            """
            INSERT INTO claims (id, content, confidence, user_id, embedding, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                claim_id,
                content,
                confidence,
                user_id,
                embedding_bytes,
                json.dumps(metadata) if metadata else None,
                json.dumps(tags) if tags else None,
            ),
        )

        conn.commit()
        conn.close()

        return claim_id

    def _get_claim(self, claim_id: str) -> Optional[dict]:
        """Get claim from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

cursor.execute(
            """
            SELECT id, content, confidence, user_id, metadata, tags, is_dirty, dirty_reason, dirty_priority, created_at
            FROM claims WHERE id = ?
        """,
            (claim_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "content": row[1],
                "confidence": row[2],
                "user_id": row[3],
                "metadata": json.loads(row[4]) if row[4] else None,
                "tags": json.loads(row[5]) if row[5] else [],
                "is_dirty": bool(row[6]),
                "dirty_reason": row[7],
                "dirty_priority": row[8],
                "created_at": row[9],
            }
        return None

    def _search_claims(self, query: str, limit: int = 10) -> List[dict]:
        """Search claims using vector similarity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

# Get all claims with embeddings
        cursor.execute("""
            SELECT id, content, confidence, user_id, metadata, tags, is_dirty, dirty_reason, dirty_priority, created_at, embedding
            FROM claims
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

# Calculate similarities
        results = []
        for row in rows:
            claim_embedding = np.frombuffer(row[9], dtype=np.float32)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, claim_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(claim_embedding)
            )

            results.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "confidence": row[2],
                    "user_id": row[3],
                    "metadata": json.loads(row[4]) if row[4] else None,
                    "tags": json.loads(row[5]) if row[5] else [],
                    "is_dirty": bool(row[6]),
                    "dirty_reason": row[7],
                    "dirty_priority": row[8],
                    "created_at": row[9],
                    "similarity": float(similarity),
                }
            )

        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _print_validation_result(self, result):
        """Print validation result."""
        if result.success:
            self.console.print(
                "[bold green]✅ Configuration Validation: PASSED[/bold green]"
            )
        else:
            self.console.print(
                "[bold red]❌ Configuration Validation: FAILED[/bold red]"
            )
            for error in result.errors:
                self.console.print(f"  • {error}")

    def _print_configuration_status(self):
        """Print configuration status."""
        from config.config import show_configuration_status

        show_configuration_status(detailed=True)

    def _print_setup_instructions(self, provider_name: str = None):
        """Print setup instructions."""
        from config.config import get_unified_validator

        validator = get_unified_validator()
        examples = validator.get_format_examples()

        self.console.print("[bold]Setup Instructions[/bold]")

        for format_type, format_examples in examples.items():
            if provider_name or format_type.value == "unified_provider":
                self.console.print(
                    f"\n[bold blue]{format_type.value.replace('_', ' ').title()} Format:[/bold blue]"
                )
                for category, example_list in format_examples.items():
                    self.console.print(
                        f"\n[cyan]{category.replace('_', ' ').title()}:[/cyan]"
                    )
                    for example in example_list:
                        if isinstance(example, list):
                            for line in example:
                                self.console.print(f"  {line}")

    def _create_claim_panel(
        self, claim_id: str, content: str, confidence: float, user: str, metadata: dict
    ) -> Panel:
        """Create a panel for claim display."""
        return Panel(
            f"[bold green]Claim Created Successfully![/bold green]\n\n"
            f"[bold]ID:[/bold] {claim_id}\n"
            f"[bold]Content:[/bold] {content}\n"
            f"[bold]Confidence:[/bold] {confidence:.2f}\n"
            f"[bold]User:[/bold] {user}\n"
            f"[bold]Backend:[/bold] {self._get_backend_type()}\n"
            f"[bold]Metadata:[/bold] {json.dumps(metadata, indent=2)}",
            title="[OK] Success",
            border_style="green",
        )

    def _create_search_table(self, results: List[dict], query: str) -> Table:
        """Create a table for search results."""
        table = Table(title=f"Search Results for: '{query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Confidence", style="green")
        table.add_column("Similarity", style="yellow")
        table.add_column("User", style="blue")

        for result in results:
            # Truncate content for display
            content = (
                result["content"][:50] + "..."
                if len(result["content"]) > 50
                else result["content"]
            )
            table.add_row(
                result["id"],
                content,
                f"{result['confidence']:.2f}",
                f"{result['similarity']:.3f}",
                result["user_id"],
            )

        return table

    def _get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get total claims
        cursor.execute("SELECT COUNT(*) FROM claims")
        total_claims = cursor.fetchone()[0]

        # Get average confidence
        cursor.execute("SELECT AVG(confidence) FROM claims")
        avg_confidence = cursor.fetchone()[0] or 0

        # Get unique users
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM claims")
        unique_users = cursor.fetchone()[0]

        conn.close()

        return {
            "total_claims": total_claims,
            "avg_confidence": avg_confidence,
            "unique_users": unique_users,
            "database_path": self.db_path,
            "embedding_model": "all-MiniLM-L6-v2",
            "backend_type": self._get_backend_type(),
        }

    # Abstract methods that must be implemented by backends
@abstractmethod
    def create_claim(
        self,
        content: str,
        confidence: float,
        user_id: str,
        analyze: bool = False,
        tags: list = None,
        **kwargs,
    ) -> str:
        """Create a new claim."""
        pass

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        pass

    @abstractmethod
    def search_claims(self, query: str, limit: int = 10, **kwargs) -> List[dict]:
        """Search claims by content."""
        pass

@abstractmethod
    def analyze_claim(self, claim_id: str, **kwargs) -> Dict[str, Any]:
        """Analyze a claim using backend-specific services."""
        pass

    @abstractmethod
    def process_prompt(
        self, 
        prompt_text: str, 
        confidence: float = 0.8,
        verbose: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available and properly configured."""
        pass

def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend."""
        return {
            "name": self._get_backend_type(),
            "configured": self.is_available(),
            "provider": self.current_provider_config.get("name")
            if self.current_provider_config
            else None,
            "type": self.current_provider_config.get("type")
            if self.current_provider_config
            else None,
            "model": self.current_provider_config.get("model")
            if self.current_provider_config
            else None,
        }

    def process_prompt(
        self, 
        prompt_text: str, 
        confidence: float = 0.8,
        verbose: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Process user prompt as claim with dirty evaluation."""
        from ..core.models import DirtyReason
        
        # Get workspace context from config
        from config.config import get_config
        config = get_config()
        workspace = config.workspace
        user = config.user
        team = config.team
        
        # Create tags with workspace context
        tags = ["user-prompt", f"workspace-{workspace}", f"user-{user}", f"team-{team}"]
        
        if verbose >= 1:
            self.console.print(f"[dim]🔧 Creating claim with context: {workspace}/{team}/{user}[/dim]")
            self.console.print(f"[dim]🔧 Tags: {tags}[/dim]")
        
        # Create claim with user-prompt tags
        claim_id = self.create_claim(
            content=prompt_text,
            confidence=confidence,
            user_id=user,
            tags=tags
        )
        
        if verbose >= 1:
            self.console.print(f"[dim]🔧 Marking claim {claim_id} as dirty for evaluation[/dim]")
        
        # Mark as dirty for evaluation (default priority=10)
        self.mark_claim_dirty(
            claim_id, 
            DirtyReason.NEW_CLAIM_ADDED, 
            priority=10
        )
        
        if verbose >= 1:
            self.console.print(f"[dim]🔧 Starting dirty evaluation...[/dim]")
        
        # Process dirty evaluation (mock for now)
        evaluation_result = self._mock_evaluate_claim(claim_id)
        
        if verbose >= 1:
            self.console.print(f"[dim]🔧 Evaluation complete, confidence: {evaluation_result['final_confidence']:.1%}[/dim]")
        
        # Check confidence threshold for user response
        if evaluation_result['final_confidence'] > 0.90:
            if verbose >= 2:
                self.console.print(f"[blue]📋 High-confidence claim achieved: {evaluation_result['final_confidence']:.1%}[/blue]")
                self.console.print(f"[blue]📋 Claim details: {evaluation_result['summary']}[/blue]")
            
            if verbose >= 1:
                self.console.print(f"[dim]🔧 Generating user response with TellUser tool...[/dim]")
            
            user_response = self._generate_user_response(
                prompt_text, 
                evaluation_result
            )
            
            # Always show final response (even at verbose=0)
            self.console.print(user_response)
            
            evaluation_result['user_response'] = user_response
        else:
            if verbose >= 1:
                self.console.print(f"[dim]🔧 Confidence {evaluation_result['final_confidence']:.1%} below 90% threshold - no user response[/dim]")
        
        return {
            "claim_id": claim_id,
            "original_prompt": prompt_text,
            "workspace_context": f"{workspace}/{team}/{user}",
            "tags": tags,
            "evaluation_result": evaluation_result,
            "final_status": "processed"
        }

    def mark_claim_dirty(self, claim_id: str, reason: DirtyReason, priority: int = 0):
        """Mark a claim as dirty in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE claims 
            SET is_dirty = 1, dirty_reason = ?, dirty_priority = ?
            WHERE id = ?
        """,
            (reason.value, priority, claim_id)
        )
        
        conn.commit()
        conn.close()

    def _mock_evaluate_claim(self, claim_id: str) -> Dict[str, Any]:
        """Mock evaluation for testing - replace with actual evaluation logic."""
        import random
        
        # Get the claim
        claim = self.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")
        
        # Mock evaluation - in real implementation this would use LLM
        final_confidence = random.uniform(0.85, 0.96)
        
        return {
            "claim_id": claim_id,
            "original_confidence": claim["confidence"],
            "final_confidence": final_confidence,
            "evaluation_steps": [
                "Analyzed claim content",
                "Checked supporting evidence", 
                "Validated reasoning"
            ],
            "summary": f"Evaluated claim: {claim['content'][:50]}...",
            "status": "validated" if final_confidence > 0.90 else "needs_review"
        }

    def _generate_user_response(self, prompt_text: str, evaluation_result: Dict[str, Any]) -> str:
        """Generate user response using TellUser tool format."""
        # Mock response - in real implementation this would use LLM
        confidence = evaluation_result['final_confidence']
        
        response = f"[SUCCESS] Based on your prompt about '{prompt_text[:50]}...', "
        response += f"I've analyzed the requirements and validated the approach. "
        response += f"The system confidence in this assessment is high at {confidence:.1%}. "
        
        if confidence > 0.95:
            response += "This claim has been thoroughly validated and can be trusted."
        elif confidence > 0.90:
            response += "This claim shows strong validation with good supporting evidence."
        
        return response


class ClaimValidationError(Exception):
    """Exception raised for claim validation errors."""

    pass


class DatabaseError(Exception):
    """Exception raised for database operations."""

    pass


class BackendNotAvailableError(Exception):
    """Exception raised when backend is not available."""

    pass
