#!/usr/bin/env python3
"""
Simple CLI commands without Rich progress bars to avoid Unicode issues
"""

import typer
from rich.console import Console
import sys
import os

# Set environment variable for this process
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Simple console
console = Console(legacy_windows=True)

def simple_create_command(content: str, confidence: float, user: str, cli_backend, analyze: bool = False):
    """Simple create command without progress bars."""
    try:
        console.print("[bold blue]Creating claim...[/bold blue]")
        claim_id = cli_backend.create_claim(content, confidence, user, analyze)
        console.print(f"[bold green]+ Claim {claim_id} created successfully![/bold green]")
        return claim_id
    except Exception as e:
        console.print(f"[red]Error creating claim: {e}[/red]")
        raise typer.Exit(1)

def simple_get_command(claim_id: str, cli_backend):
    """Simple get command without progress bars."""
    try:
        console.print("[bold blue]Retrieving claim...[/bold blue]")
        claim = cli_backend.get_claim(claim_id)

        if claim:
            console.print("[bold green]+ Claim retrieved successfully![/bold green]")
            from rich.panel import Panel
            import json
            
            panel = Panel(
                f"[bold]ID:[/bold] {claim['id']}\n"
                f"[bold]Content:[/bold] {claim['content']}\n"
                f"[bold]Confidence:[/bold] {claim['confidence']:.2f}\n"
                f"[bold]User:[/bold] {claim['user_id']}\n"
                f"[bold]Created:[/bold] {claim['created_at']}\n"
                f"[bold]Metadata:[/bold] {json.dumps(claim['metadata'], indent=2)}",
                title=f"Claim Details: {claim_id}",
                border_style="blue"
            )
            console.print(panel)
            return claim
        else:
            console.print("[yellow]Claim not found[/yellow]")
            return None

    except Exception as e:
        console.print(f"[red]Error retrieving claim: {e}[/red]")
        raise typer.Exit(1)

def simple_search_command(query: str, limit: int, cli_backend):
    """Simple search command without progress bars."""
    try:
        console.print("[bold blue]Searching claims...[/bold blue]")
        results = cli_backend.search_claims(query, limit)
        
        console.print(f"[bold green]+ Found {len(results)} results[/bold green]")

        if results:
            from rich.table import Table
            
            # Use backend's table creation or fallback
            if hasattr(cli_backend, '_create_search_table'):
                table = cli_backend._create_search_table(results, query)
            else:
                # Fallback table creation
                table = Table(title=f"Search Results for: '{query}'")
                table.add_column("ID", style="cyan", no_wrap=True)
                table.add_column("Content", style="white")
                table.add_column("Confidence", style="green")
                table.add_column("Similarity", style="yellow")
                table.add_column("User", style="blue")

                for result in results:
                    content = result['content'][:50] + "..." if len(result['content']) > 50 else result['content']
                    table.add_row(
                        result['id'],
                        content,
                        f"{result['confidence']:.2f}",
                        f"{result['similarity']:.3f}" if 'similarity' in result else "N/A",
                        result['user_id']
                    )

            console.print(table)
        else:
            console.print("[yellow]No claims found matching your query[/yellow]")
            console.print("Try creating some claims first with: conjecture create")
            
        return results

    except Exception as e:
        console.print(f"[red]Error searching claims: {e}[/red]")
        raise typer.Exit(1)

def simple_analyze_command(claim_id: str, cli_backend):
    """Simple analyze command without progress bars."""
    try:
        console.print("[bold blue]Analyzing claim...[/bold blue]")
        analysis = cli_backend.analyze_claim(claim_id)
        console.print("[bold green]+ Analysis complete![/bold green]")

        # Display analysis results
        from rich.panel import Panel
        panel = Panel(
            f"[bold]Claim ID:[/bold] {analysis['claim_id']}\n"
            f"[bold]Backend:[/bold] {analysis.get('backend', 'unknown')}\n"
            f"[bold]Analysis Type:[/bold] {analysis.get('analysis_type', 'unknown')}\n"
            f"[bold]Confidence:[/bold] {analysis.get('confidence_score', 0):.2f}\n"
            f"[bold]Sentiment:[/bold] {analysis.get('sentiment', 'unknown')}\n"
            f"[bold]Topics:[/bold] {', '.join(analysis.get('topics', []))}\n"
            f"[bold]Status:[/bold] {analysis.get('verification_status', 'unknown')}",
            title=f"Analysis Results for {claim_id}",
            border_style="green"
        )
        console.print(panel)
        return analysis

    except Exception as e:
        console.print(f"[red]Error analyzing claim: {e}[/red]")
        raise typer.Exit(1)