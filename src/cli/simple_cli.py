#!/usr/bin/env python3
"""
Simple CLI implementation using the unified Conjecture API
Demonstrates the clean, direct interface pattern
"""

from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from contextflow import Conjecture


class SimpleCLI:
    """Simple CLI implementation using the unified Conjecture API."""

    def __init__(self):
        self.console = Console()
        self.error_console = Console(stderr=True)
        
        # Single unified API - no complexity needed
        self.cf = Conjecture()

    def explore(self, query: str, max_claims: int = 10) -> Dict[str, Any]:
        """Explore claims using the unified API."""
        try:
            result = self.cf.explore(query, max_claims=max_claims)
            return {
                "success": True,
                "result": result,
                "summary": result.summary()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def add_claim(self, content: str, confidence: float, claim_type: str, tags: List[str] = None) -> Dict[str, Any]:
        """Add a claim using the unified API."""
        try:
            claim = self.cf.add_claim(
                content=content,
                confidence=confidence,
                claim_type=claim_type,
                tags=tags or []
            )
            return {
                "success": True,
                "claim": claim,
                "claim_id": claim.id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics using the unified API."""
        try:
            stats = self.cf.get_statistics()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def display_results(self, results: Dict[str, Any]):
        """Display exploration results."""
        if not results["success"]:
            self.error_console.print(f"[red]❌ Error: {results['error']}[/red]")
            return

        result = results["result"]
        
        # Create results table
        table = Table(title=f"Exploration Results for: '{result.query}'")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content", style="white")
        table.add_column("Confidence", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Tags", style="yellow")

        for claim in result.claims:
            type_str = ", ".join([t.value for t in claim.type])
            tags_str = ", ".join(claim.tags) if claim.tags else "None"
            content = claim.content[:60] + "..." if len(claim.content) > 60 else claim.content
            
            table.add_row(
                claim.id,
                content,
                f"{claim.confidence:.2f}",
                type_str,
                tags_str
            )

        self.console.print(table)
        self.console.print(f"\n[green]Found {len(result.claims)} claims in {result.search_time:.2f}s[/green]")

    def display_claim(self, claim_result: Dict[str, Any]):
        """Display a created claim."""
        if not claim_result["success"]:
            self.error_console.print(f"[red]❌ Error: {claim_result['error']}[/red]")
            return

        claim = claim_result["claim"]
        
        panel = Panel(
            f"[bold green]Claim Created Successfully![/bold green]\n\n"
            f"[bold]ID:[/bold] {claim.id}\n"
            f"[bold]Content:[/bold] {claim.content}\n"
            f"[bold]Confidence:[/bold] {claim.confidence:.2f}\n"
            f"[bold]Type:[/bold] {', '.join([t.value for t in claim.type])}\n"
            f"[bold]Tags:[/bold] {', '.join(claim.tags) if claim.tags else 'None'}\n"
            f"[bold]State:[/bold] {claim.state.value}\n"
            f"[bold]Created:[/bold] {claim.created.strftime('%Y-%m-%d %H:%M:%S')}",
            title="[OK] Success",
            border_style="green"
        )
        
        self.console.print(panel)

    def display_stats(self, stats_result: Dict[str, Any]):
        """Display system statistics."""
        if not stats_result["success"]:
            self.error_console.print(f"[red]❌ Error: {stats_result['error']}[/red]")
            return

        stats = stats_result["stats"]
        
        table = Table(title="System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(table)


def main():
    """Example usage of the simple CLI."""
    cli = SimpleCLI()
    
    print("🚀 Conjecture Simple CLI Demo")
    print("=" * 40)
    
    # Example exploration
    print("\n🔍 Exploring 'machine learning'...")
    results = cli.explore("machine learning", max_claims=3)
    cli.display_results(results)
    
    # Example claim creation
    print("\n➕ Adding a new claim...")
    claim_result = cli.add_claim(
        content="Machine learning algorithms improve with more training data",
        confidence=0.85,
        claim_type="concept",
        tags=["machine learning", "algorithms", "training"]
    )
    cli.display_claim(claim_result)
    
    # Example statistics
    print("\n📊 System statistics...")
    stats = cli.get_stats()
    cli.display_stats(stats)


if __name__ == "__main__":
    main()