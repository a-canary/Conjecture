#!/usr/bin/env python3
"""
Modular Conjecture CLI with Pluggable Backends - Fixed Version
Unified entry point for all Conjecture CLI functionality with auto-detection
(Fixed unicode characters for Windows compatibility)
"""

import asyncio
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import List, Optional
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Initialize Unicode and TensorFlow support
from .encoding_handler import setup_unicode_environment, get_safe_console
from .tf_suppression import suppress_tensorflow_warnings, print_ml_environment_info

# Setup environment
setup_unicode_environment()
suppress_tensorflow_warnings()

# Get console instances with UTF-8 support
from rich.console import Console
import sys
import os

# Set environment variable for this process
os.environ["PYTHONIOENCODING"] = "utf-8"

# Force UTF-8 encoding for Rich console with safe fallback
try:
    console = Console(
        file=sys.stdout,
        force_terminal=True,
        legacy_windows=True,  # Enable legacy Windows rendering for compatibility
        width=None,
        no_color=False,  # Allow colors but be safe about it
        ascii_only=False,  # Let Rich handle Unicode safely
    )
except Exception:
    # Fallback to safe console if Rich console creation fails
    console = get_safe_console()

error_console = get_safe_console()  # Use safe console for error printing

from .backends import (
    BACKEND_REGISTRY,
    LocalBackend,
    CloudBackend,
    HybridBackend,
    AutoBackend,
)
from .base_cli import BaseCLI, BackendNotAvailableError

# from .dirty_commands import dirty_app  # Temporarily disabled due to import issues
from config.config import validate_config

# Create the main Typer app
app = typer.Typer(
    name="conjecture",
    help="Conjecture CLI - Modular architecture with pluggable backends",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add dirty flag subcommand
# app.add_typer(dirty_app, name="dirty", help="Dirty flag system management")  # Temporarily disabled

# Backend selection options
backend_option = typer.Option(
    "auto",
    "--backend",
    "-b",
    help="Backend to use: auto, local, cloud, hybrid",
    rich_help_panel="Backend Options",
)

# Global backend instance
current_backend: BaseCLI = None


def get_backend(backend_type: str = "auto") -> BaseCLI:
    """Get or create the specified backend instance."""
    global current_backend

    if backend_type not in BACKEND_REGISTRY:
        error_console.print(f"[red]Unknown backend: {backend_type}[/red]")
        error_console.print(f"Available backends: {', '.join(BACKEND_REGISTRY.keys())}")
        raise typer.Exit(1)

    # Create backend instance
    backend_class = BACKEND_REGISTRY[backend_type]
    current_backend = backend_class()

    # Check availability
    if not current_backend.is_available():
        if backend_type == "auto":
            # For auto backend, provide detailed feedback
            detection = current_backend.get_detection_report()
            error_console.print(
                "[bold red]No backends are properly configured[/bold red]"
            )

            console.print("\n[bold yellow]Quick Setup:[/bold yellow]")
            console.print("1. Copy template: [cyan]cp .env.example .env[/cyan]")
            console.print("2. Edit [cyan].env[/cyan] with your preferred provider")
            console.print("3. Try again: [cyan]conjecture --backend auto[/cyan]")

            console.print(f"\n[bold]Need help?[/bold]")
            console.print("Run: [cyan]conjecture setup[/cyan]")
            console.print("Run: [cyan]conjecture providers[/cyan]")
        else:
            error_console.print(
                f"[red]{backend_type.title()} backend is not properly configured[/red]"
            )
            error_console.print("Run 'conjecture setup' to configure providers")

        raise typer.Exit(1)

    return current_backend


def print_backend_info(backend: BaseCLI):
    """Print information about the current backend."""
    info = backend.get_backend_info()

    panel_content = f"[bold]Backend:[/bold] {info['name']}\n"
    panel_content += (
        f"[bold]Configured:[/bold] {'OK' if info['configured'] else 'FAIL'}\n"
    )

    if info.get("provider"):
        panel_content += f"[bold]Provider:[/bold] {info['provider']}\n"
    if info.get("type"):
        panel_content += f"[bold]Type:[/bold] {info['type']}\n"
    if info.get("model"):
        panel_content += f"[bold]Model:[/bold] {info['model']}\n"

    panel = Panel(
        panel_content, title="[bold]Backend Information[/bold]", border_style="blue"
    )
    console.print(panel)


@app.callback()
def main(
    backend: str = backend_option,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Conjecture CLI - Unified interface with pluggable backends."""
    global current_backend

    # Initialize backend
    try:
        current_backend = get_backend(backend)

        if verbose:
            console.print(f"[green]Initialized {backend} backend[/green]")
            print_backend_info(current_backend)

    except typer.Exit:
        raise
    except Exception as e:
        error_console.print(f"[red]Error initializing backend: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create(
    content: str = typer.Argument(..., help="Content of the claim"),
    confidence: float = typer.Option(
        0.8, "--confidence", "-c", help="Confidence score (0.0-1.0)"
    ),
    user: str = typer.Option("user", "--user", "-u", help="User ID"),
    analyze: bool = typer.Option(
        False, "--analyze", "-a", help="Analyze with configured LLM"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Create a new claim."""
    try:
        # Override backend if specified
        if backend:
            cli_backend = get_backend(backend)
        else:
            cli_backend = current_backend

        if not cli_backend:
            error_console.print(
                "[red]No backend initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating claim...", total=None)

            try:
                claim_id = cli_backend.create_claim(content, confidence, user, analyze)
                progress.update(
                    task, description=f"Claim {claim_id} created successfully!"
                )

            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error creating claim: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def get(
    claim_id: str = typer.Argument(..., help="ID of the claim to retrieve"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Get a claim by ID."""
    try:
        # Override backend if specified
        if backend:
            cli_backend = get_backend(backend)
        else:
            cli_backend = current_backend

        if not cli_backend:
            error_console.print(
                "[red]No backend initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Retrieving claim...", total=None)

            try:
                claim = cli_backend.get_claim(claim_id)

                if claim:
                    progress.update(task, description="Claim retrieved successfully!")

                    panel = Panel(
                        f"[bold]ID:[/bold] {claim['id']}\n"
                        f"[bold]Content:[/bold] {claim['content']}\n"
                        f"[bold]Confidence:[/bold] {claim['confidence']:.2f}\n"
                        f"[bold]User:[/bold] {claim['user_id']}\n"
                        f"[bold]Created:[/bold] {claim['created_at']}\n"
                        f"[bold]Metadata:[/bold] {json.dumps(claim['metadata'], indent=2)}",
                        title=f"Claim Details: {claim_id}",
                        border_style="blue",
                    )
                    console.print(panel)
                else:
                    progress.update(task, description="Claim not found")
                    error_console.print(f"[red]Claim {claim_id} not found[/red]")
                    raise typer.Exit(1)

            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error retrieving claim: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Search claims by content."""
    try:
        # Override backend if specified
        if backend:
            cli_backend = get_backend(backend)
        else:
            cli_backend = current_backend

        if not cli_backend:
            error_console.print(
                "[red]No backend initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching claims...", total=None)

            try:
                results = cli_backend.search_claims(query, limit)

                progress.update(task, description=f"Found {len(results)} results")

                if results:
                    # Use backend's table creation or fallback
                    if hasattr(cli_backend, "_create_search_table"):
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
                            content = (
                                result["content"][:50] + "..."
                                if len(result["content"]) > 50
                                else result["content"]
                            )
                            table.add_row(
                                result["id"],
                                content,
                                f"{result['confidence']:.2f}",
                                f"{result['similarity']:.3f}"
                                if "similarity" in result
                                else "N/A",
                                result["user_id"],
                            )

                    console.print(table)
                else:
                    console.print(
                        "[yellow]No claims found matching your query[/yellow]"
                    )
                    console.print(
                        "Try creating some claims first with: conjecture create"
                    )

            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error searching claims: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def analyze(
    claim_id: str = typer.Argument(..., help="ID of the claim to analyze"),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Analyze a claim using LLM services."""
    try:
        # Override backend if specified
        if backend:
            cli_backend = get_backend(backend)
        else:
            cli_backend = current_backend

        if not cli_backend:
            error_console.print(
                "[red]No backend initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing claim...", total=None)

            try:
                analysis = cli_backend.analyze_claim(claim_id)
                progress.update(task, description="Analysis complete!")

                # Display analysis results
                panel = Panel(
                    f"[bold]Claim ID:[/bold] {analysis['claim_id']}\n"
                    f"[bold]Backend:[/bold] {analysis.get('backend', 'unknown')}\n"
                    f"[bold]Analysis Type:[/bold] {analysis.get('analysis_type', 'unknown')}\n"
                    f"[bold]Confidence:[/bold] {analysis.get('confidence_score', 0):.2f}\n"
                    f"[bold]Sentiment:[/bold] {analysis.get('sentiment', 'unknown')}\n"
                    f"[bold]Topics:[/bold] {', '.join(analysis.get('topics', []))}\n"
                    f"[bold]Status:[/bold] {analysis.get('verification_status', 'unknown')}",
                    title=f"Analysis Results for {claim_id}",
                    border_style="green",
                )
                console.print(panel)

            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error analyzing claim: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def config():
    """Show configuration status and validation."""
    console.print("[bold]Configuration Status[/bold]")
    console.print("=" * 50)

    result = validate_config()

    # Print validation result
    if result.success:
        console.print("[bold green]Configuration Validation: PASSED[/bold green]")
    else:
        console.print("[bold red]Configuration Validation: FAILED[/bold red]")
        for error in result.errors:
            console.print(f"  • {error}")

    console.print(f"\n[bold]Next Steps:[/bold]")
    if result.success:
        console.print("Configuration is valid!")
        console.print("You can now use: create, get, search, analyze commands")

        # Show available backends
        console.print(f"\n[bold]Available Backends:[/bold]")
        for name in BACKEND_REGISTRY.keys():
            backend_class = BACKEND_REGISTRY[name]
            backend_instance = backend_class()
            status = "OK" if backend_instance.is_available() else "FAIL"
            console.print(f"  {status} {name.lower()}")
    else:
        console.print("Configure at least one provider:")
        console.print("  • Run: [cyan]conjecture setup[/cyan]")
        console.print("  • Edit: [cyan].env[/cyan] file")
        console.print("  • Copy from: [cyan].env.example[/cyan]")


@app.command()
def providers():
    """Show available providers and setup instructions."""
    show_configuration_status()


@app.command()
def setup(
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive setup mode"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Setup specific provider"
    ),
):
    """Setup provider configuration."""
    console.print("[bold]Provider Setup[/bold]")
    console.print("=" * 50)

    if provider:
        # Setup specific provider
        console.print(f"\n[bold blue]Setup Guide for {provider.title()}:[/bold blue]")
        console.print("1. Copy template: [cyan]cp .env.example .env[/cyan]")
        console.print("2. Edit the file [cyan].env[/cyan]")
        console.print("3. Add the configuration below:")

        # Show provider-specific config
        from config.unified_validator import get_unified_validator

        validator = get_unified_validator()
        examples = validator.get_format_examples()

        for format_type, format_examples in examples.items():
            if provider.lower() in format_type.value.lower():
                for category, example_list in format_examples.items():
                    console.print(
                        f"\n[cyan]{category.replace('_', ' ').title()}:[/cyan]"
                    )
                    for example in example_list:
                        if isinstance(example, list):
                            for line in example:
                                console.print(f"  {line}")
    else:
        # Show all options
        console.print("\n[bold]Available Providers:[/bold]")
        console.print(
            "  • [cyan]ollama[/cyan] - Local models (recommended for privacy)"
        )
        console.print("  • [cyan]lm_studio[/cyan] - Local model server")
        console.print("  • [cyan]openai[/cyan] - GPT models (cloud)")
        console.print("  • [cyan]anthropic[/cyan] - Claude models (cloud)")
        console.print("  • [cyan]google[/cyan] - Gemini models (cloud)")
        console.print("  • [cyan]cohere[/cyan] - Cohere models (cloud)")

        if interactive:
            console.print("\n[bold]Interactive Setup[/bold]")
            provider_choice = Prompt.ask(
                "Choose provider to configure",
                choices=[
                    "ollama",
                    "lm_studio",
                    "openai",
                    "anthropic",
                    "google",
                    "cohere",
                    "skip",
                ],
                default="skip",
            )

            if provider_choice != "skip":
                console.print(
                    f"\n[bold blue]Setup Guide for {provider_choice.title()}:[/bold blue]"
                )
                console.print("1. Copy template: [cyan]cp .env.example .env[/cyan]")
                console.print("2. Edit the file [cyan].env[/cyan]")
                console.print("3. Run: [cyan]conjecture config[/cyan] to validate")


@app.command()
def stats(
    backend: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Show database and backend statistics."""
    try:
        # Override backend if specified
        if backend:
            cli_backend = get_backend(backend)
        else:
            cli_backend = current_backend

        if not cli_backend:
            error_console.print(
                "[red]No backend initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        # Get statistics
        if hasattr(cli_backend, "_get_database_stats"):
            stats = cli_backend._get_database_stats()
        else:
            # Fallback stats
            stats = {
                "total_claims": "Unknown",
                "avg_confidence": "Unknown",
                "unique_users": "Unknown",
                "backend_type": cli_backend._get_backend_type(),
            }

        # Create statistics table
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Backend", stats.get("backend_type", "Unknown"))
        table.add_row("Total Claims", str(stats.get("total_claims", "Unknown")))
        table.add_row(
            "Average Confidence",
            f"{stats.get('avg_confidence', 0):.3f}"
            if isinstance(stats.get("avg_confidence"), (int, float))
            else str(stats.get("avg_confidence", "Unknown")),
        )
        table.add_row("Unique Users", str(stats.get("unique_users", "Unknown")))

        if hasattr(cli_backend, "db_path"):
            table.add_row("Database Path", cli_backend.db_path)

        table.add_row("Embedding Model", "all-MiniLM-L6-v2")

        # Add backend-specific info
        backend_info = cli_backend.get_backend_info()
        if backend_info.get("provider"):
            table.add_row("Current Provider", backend_info["provider"])

        console.print(table)

    except typer.Exit:
        raise


@app.command()
def backends():
    """Show available backends and their status."""
    console.print("[bold]Available Backends[/bold]")
    console.print("=" * 50)

    table = Table(title="Backend Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description", style="white")

    descriptions = {
        "auto": "Intelligent auto-detection of optimal backend",
        "local": "Local services (Ollama, LM Studio) - offline capable",
        "cloud": "Cloud services (OpenAI, Anthropic) - advanced features",
        "hybrid": "Combines local and cloud for optimal performance",
    }

    for name, backend_class in BACKEND_REGISTRY.items():
        backend_instance = backend_class()
        status = "Available" if backend_instance.is_available() else "Not Configured"
        description = descriptions.get(name, "Unknown backend")

        table.add_row(name.title(), status, description)

    console.print(table)

    # Show recommendation
    console.print(f"\n[bold]Recommendation:[/bold]")
    console.print("Use [cyan]--backend auto[/cyan] for intelligent backend selection")
    console.print("Use [cyan]conjecture backends[/cyan] to check status anytime")


@app.command()
def health():
    """Check system health and backend availability."""
    console.print("[bold]System Health Check[/bold]")
    console.print("=" * 50)

    # Check each backend
    healthy_backends = []
    unhealthy_backends = []

    for name, backend_class in BACKEND_REGISTRY.items():
        try:
            backend_instance = backend_class()
            if backend_instance.is_available():
                healthy_backends.append(name)
                console.print(f"[green]OK {name.title()} Backend: Healthy[/green]")
            else:
                unhealthy_backends.append(name)
                console.print(f"[red]FAIL {name.title()} Backend: Not Available[/red]")
        except Exception as e:
            unhealthy_backends.append(name)
            console.print(f"[red]ERROR {name.title()} Backend: Error - {e}[/red]")

    # Overall status
    if healthy_backends:
        console.print(
            f"\n[bold green]System Status: {len(healthy_backends)} backend(s) healthy[/bold green]"
        )
        console.print(f"Available: {', '.join(healthy_backends)}")
    else:
        console.print(f"\n[bold red]System Status: No healthy backends[/bold red]")
        console.print("Please configure at least one provider")

    if unhealthy_backends:
        console.print(
            f"[yellow]Unhealthy backends: {', '.join(unhealthy_backends)}[/yellow]"
        )


@app.command()
def quickstart():
    """Quick start guide for new users."""
    console.print("[bold blue]Conjecture Quick Start Guide[/bold blue]")
    console.print("=" * 60)

    console.print("\n[bold]Step 1: Configure a Provider[/bold]")
    console.print("Choose ONE of these options:")

    console.print("\n[bold green]LOCAL (Recommended for Privacy)[/bold green]")
    console.print("  • [cyan]Ollama:[/cyan] Install from https://ollama.ai/")
    console.print("    - Run: [yellow]ollama serve[/yellow]")
    console.print(
        "    - Configure: [cyan]OLLAMA_ENDPOINT=http://localhost:11434[/cyan]"
    )

    console.print("\n[bold cyan]CLOUD (Internet Required)[/bold cyan]")
    console.print(
        "  • [cyan]OpenAI:[/cyan] Get key from https://platform.openai.com/api-keys"
    )
    console.print("    - Configure: [cyan]OPENAI_API_KEY=sk-your-key[/cyan]")

    console.print("\n[bold]Step 2: Create Configuration File[/bold]")
    console.print("  [yellow]cp .env.example .env[/yellow]")
    console.print("  [yellow]# Edit .env with your chosen provider[/yellow]")

    console.print("\n[bold]Step 3: Validate Configuration[/bold]")
    console.print("  [yellow]conjecture config[/yellow]")

    console.print("\n[bold]Step 4: Create Your First Claim[/bold]")
    console.print(
        '  [yellow]conjecture create "The sky is blue" --confidence 0.9[/yellow]'
    )

    console.print("\n[bold]Step 5: Search Claims[/bold]")
    console.print('  [yellow]conjecture search "sky"[/yellow]')

    console.print("\n[bold]Backend Selection:[/bold]")
    console.print("  • [cyan]conjecture --backend auto[/cyan] (recommended)")
    console.print("  • [cyan]conjecture --backend local[/cyan] (offline)")
    console.print("  • [cyan]conjecture --backend cloud[/cyan] (advanced)")
    console.print("  • [cyan]conjecture --backend hybrid[/cyan] (balanced)")

    console.print("\n[bold]Need More Help?[/bold]")
    console.print("  • Setup help: [cyan]conjecture setup[/cyan]")
    console.print("  • Provider options: [cyan]conjecture providers[/cyan]")
    console.print("  • Backend status: [cyan]conjecture backends[/cyan]")
    console.print("  • System health: [cyan]conjecture health[/cyan]")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
