#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Modular Conjecture CLI with Pluggable Backends - Fixed Version
Unified entry point for all Conjecture CLI functionality with auto-detection
(Fixed unicode characters for Windows compatibility)
"""

import asyncio
import typer
from rich.progress import Progress, TextColumn
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
# Use standard Rich Console for Progress compatibility
console = Console(
    file=sys.stdout,
    force_terminal=True,
    legacy_windows=True,  # Enable legacy Windows rendering for compatibility
    width=None,
    no_color=False,  # Allow colors but be safe about it
)

# Error console can use safe console
error_console = Console(stderr=True, legacy_windows=True)

from .base_cli import BaseCLI
from src.interfaces.processing_interface import ProcessingInterface

# from .dirty_commands import dirty_app  # Temporarily disabled due to import issues
from src.config.unified_config import validate_config

# Create the main Typer app
app = typer.Typer(
    name="conjecture",
    help="Conjecture CLI - Modular architecture with pluggable backends",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Add dirty flag subcommand
# app.add_typer(dirty_app, name="dirty", help="Dirty flag system management")  # Temporarily disabled

# Global processing interface instance
current_processing_interface: ProcessingInterface = None

def get_processing_interface(backend_type: str = "auto") -> ProcessingInterface:
    """Get or create the specified processing interface instance."""
    global current_processing_interface

    # If we already have a processing interface instance, reuse it
    if current_processing_interface is not None:
        return current_processing_interface

    # Import and create Conjecture instance (implements ProcessingInterface)
    from src.interfaces.conjecture_processing_interface import ConjectureProcessingInterface
    from src.config.unified_config import UnifiedConfig as Config

    try:
        # Create configuration
        config = Config()

        # Create Conjecture instance (implements ProcessingInterface)
        # Use database path from config if available, otherwise use default
        db_path = getattr(config, 'database_path', 'data/conjecture.db')
        interface = ConjectureProcessingInterface(db_path=db_path)
        current_processing_interface = interface

        # Initialize interface (synchronously if no event loop)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(current_processing_interface.initialize())
            else:
                loop.run_until_complete(current_processing_interface.initialize())
            console.print("[green]+ Processing interface initialized[/green]")
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # Create new loop and initialize
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(current_processing_interface.initialize())
                console.print("[green]+ Processing interface initialized[/green]")
            else:
                raise

        # Always return the interface
        return current_processing_interface
        
    except Exception as e:
        error_console.print(
            f"[bold red]Failed to initialize processing interface: {e}[/bold red]"
        )
        console.print("\n[bold yellow]Configuration Setup:[/bold yellow]")
        
        from src.config.pydantic_config import ConfigHierarchy
        hierarchy = ConfigHierarchy()
        
        console.print("1. [cyan]User config:[/cyan] ~/.conjecture/config.json")
        console.print(f"   Status: {'+ Exists' if hierarchy.user_config.exists() else '- Not found'}")
        
        console.print("2. [cyan]Workspace config:[/cyan] .conjecture/config.json")
        console.print(f"   Status: {'+ Exists' if hierarchy.workspace_config.exists() else '- Not found'}")
        
        console.print("\n[bold]Example config format:[/bold]")
        console.print("[cyan]{")
        console.print('[cyan]  "providers": [')
        console.print("[cyan]    {")
        console.print('[cyan]      "url": "http://localhost:11434",')
        console.print('[cyan]      "api": "",')
        console.print('[cyan]      "model": "llama2",')
        console.print('[cyan]      "name": "ollama"')
        console.print("[cyan]    }")
        console.print("[cyan]  ]")
        console.print("[cyan]}[/cyan]")
        
        console.print("\n[bold yellow]To create user config:[/bold yellow]")
        console.print("[cyan]cp .conjecture/config.example.json ~/.conjecture/config.json[/cyan]")
        console.print("[cyan]# Then either edit the api field, or export env vars (recommended):[/cyan]")
        console.print("[cyan]#   export OPENROUTER_API_KEY=sk-or-v1-...[/cyan]")
        console.print("[cyan]#   export CHUTES_API_KEY=cpk_...[/cyan]")
        console.print("[cyan]# config.json is gitignored; never commit API keys.[/cyan]")

        raise typer.Exit(1)

def get_backend(backend_type: str = "auto") -> ProcessingInterface:
    """Alias for get_processing_interface for backward compatibility."""
    return get_processing_interface(backend_type)

def print_backend_info(backend_type: str = "auto"):
    """Print backend information for backward compatibility."""
    processing_interface = get_processing_interface(backend_type)
    print_processing_interface_info(processing_interface)

def print_processing_interface_info(processing_interface: ProcessingInterface):
    """Print information about the current processing interface."""
    # Get health status to check if interface is properly configured
    import asyncio
    try:
        health = asyncio.run(processing_interface.get_health_status())
        
        panel_content = "Processing Interface: Conjecture\n"
        panel_content += f"Status: {'OK' if health.get('healthy', False) else 'FAIL'}\n"
        
        if health.get('services'):
            services = health['services']
            panel_content += f"Data Manager: {'OK' if services.get('data_manager', False) else 'FAIL'}\n"
            panel_content += f"LLM Bridge: {'OK' if services.get('llm_bridge', False) else 'FAIL'}\n"
            panel_content += f"Async Evaluation: {'OK' if services.get('async_evaluation', False) else 'FAIL'}\n"
        
        if health.get('active_sessions') is not None:
            panel_content += f"Active Sessions: {health['active_sessions']}\n"

    except Exception as e:
        panel_content = "Processing Interface: Conjecture\n"
        panel_content += f"Status: ERROR - {e}\n"

    panel = Panel(
        panel_content, title="Processing Interface Information", border_style="blue"
    )
    console.print(panel)

@app.callback()
def main(
    backend: str = "auto",
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Conjecture CLI - Unified interface with processing interface."""
    global current_processing_interface

    # Skip initialization for help/version flags to allow `command --help` to work
    import sys
    if any(arg in ("--help", "-h", "--version") for arg in sys.argv):
        return

    # Initialize processing interface
    try:
        current_processing_interface = get_processing_interface(backend)

        if verbose:
            console.print(f"[green]Initialized {backend} processing interface[/green]")

    except typer.Exit:
        raise
    except Exception as e:
        error_console.print(f"[red]Error initializing processing interface: {e}[/red]")
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
        # Override processing interface if specified
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            # Initialize processing interface if not already done
            processing_interface = get_processing_interface()

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)
            
        # Start services synchronously if not already started
        import asyncio
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one and initialize
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(processing_interface.initialize())
            # Don't close - we need it for later async operations

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Creating claim...", total=None)

            try:
                import asyncio
                claim_coroutine = processing_interface.create_claim(
                    content=content,
                    confidence=confidence,
                    tags=[user] if user else None
                )
                claim = asyncio.run(claim_coroutine)
                claim_id = claim.id
                
                # Analyze claim if requested
                if analyze:
                    progress.update(task, description="Analyzing claim...")
                    eval_coroutine = processing_interface.evaluate_claim(claim_id)
                    eval_result = asyncio.run(eval_coroutine)
                    if eval_result.success:
                        progress.update(task, description=f"Claim {claim_id} created and analyzed!")
                    else:
                        progress.update(task, description=f"Claim {claim_id} created but analysis failed")
                else:
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
        # Override processing interface if specified
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Retrieving claim...", total=None)

            try:
                import asyncio
                claim_coroutine = processing_interface.get_claim(claim_id)
                claim = asyncio.run(claim_coroutine)

                if claim:
                    progress.update(task, description="Claim retrieved successfully!")

                    user = claim.created_by if hasattr(claim, 'created_by') else "unknown"
                    created = claim.created if hasattr(claim, 'created') else claim.updated
                    metadata = claim.metadata if hasattr(claim, 'metadata') else {}
                    tags = claim.tags if hasattr(claim, 'tags') else []
                    state = claim.state.value if hasattr(claim, 'state') else "unknown"
                    types = [t.value for t in claim.type] if hasattr(claim, 'type') else ["unknown"]
                    
                    # Get supers/subs for provenance chain (UX-0005)
                    supers = claim.supers if hasattr(claim, 'supers') else []
                    subs = claim.subs if hasattr(claim, 'subs') else []
                    dirty = claim.is_dirty if hasattr(claim, 'is_dirty') else False

                    # Build provenance info
                    provenance_lines = []
                    if supers:
                        provenance_lines.append(f"[bold]Supports:[/bold] {', '.join(supers)}")
                    if subs:
                        provenance_lines.append(f"[bold]Supported by:[/bold] {', '.join(subs)}")
                    if dirty:
                        provenance_lines.append("[yellow]⚠ Marked for re-evaluation[/yellow]")

                    provenance_str = "\n".join(provenance_lines) if provenance_lines else "[dim]No provenance links[/dim]"

                    panel = Panel(
                        f"[bold]ID:[/bold] {claim.id}\n"
                        f"[bold]Content:[/bold] {claim.content}\n"
                        f"[bold]Type:[/bold] {', '.join(types)}\n"
                        f"[bold]Confidence:[/bold] {claim.confidence:.2f}\n"
                        f"[bold]State:[/bold] {state}\n"
                        f"[bold]User:[/bold] {user}\n"
                        f"[bold]Created:[/bold] {created}\n"
                        f"[bold]Tags:[/bold] {tags}\n"
                        f"\n[bold cyan]Provenance Chain[/bold cyan]\n{provenance_str}",
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
        # Override processing interface if specified
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Searching claims...", total=None)

            try:
                import asyncio
                # Note: limit parameter not supported by interface, apply post-filter
                results_coroutine = processing_interface.search_claims(query)
                results = asyncio.run(results_coroutine)
                # Apply limit locally
                if limit and len(results) > limit:
                    results = results[:limit]

                progress.update(task, description=f"Found {len(results)} results")

                if results:
                    # Create table for search results
                    table = Table(title=f"Search Results for: '{query}'")
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Content", style="white")
                    table.add_column("Confidence", style="green")
                    table.add_column("State", style="yellow")
                    table.add_column("User", style="blue")

                    for claim in results:
                        content = (
                            claim.content[:50] + "..."
                            if len(claim.content) > 50
                            else claim.content
                        )
                        user = claim.created_by if hasattr(claim, 'created_by') else "unknown"
                        state = claim.state.value if hasattr(claim, 'state') else "unknown"
                        
                        table.add_row(
                            claim.id,
                            content,
                            f"{claim.confidence:.2f}",
                            state,
                            user,
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
        # Override processing interface if specified
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing claim...", total=None)

            try:
                import asyncio
                eval_coroutine = processing_interface.evaluate_claim(claim_id)
                eval_result = asyncio.run(eval_coroutine)
                progress.update(task, description="Analysis complete!")

                # Display analysis results
                panel = Panel(
                    f"[bold]Claim ID:[/bold] {eval_result.claim_id}\n"
                    f"[bold]Success:[/bold] {eval_result.success}\n"
                    f"[bold]Original Confidence:[/bold] {eval_result.original_confidence:.2f}\n"
                    f"[bold]New Confidence:[/bold] {eval_result.new_confidence:.2f}\n"
                    f"[bold]State:[/bold] {eval_result.state.value}\n"
                    f"[bold]Summary:[/bold] {eval_result.evaluation_summary}\n"
                    f"[bold]Processing Time:[/bold] {eval_result.processing_time:.2f}s\n"
                    f"[bold]Supporting Evidence:[/bold] {len(eval_result.supporting_evidence)} items\n"
                    f"[bold]Counter Evidence:[/bold] {len(eval_result.counter_evidence)} items\n"
                    f"[bold]Recommendations:[/bold] {len(eval_result.recommendations)} items",
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
def prompt(
    prompt_text: str = typer.Argument(
        ..., help="The prompt text (will be created as a claim)"
    ),
    confidence: float = typer.Option(
        0.8, "--confidence", "-c", help="Initial confidence score"
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        help="Verbosity level: 0=final only, 1=tool calls, 2=claims>90%",
    ),
):
    """Process a prompt as a claim with workspace context."""
    try:
        # Use current processing interface (no backend override option for prompt)
        processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing prompt...", total=None)

            try:
                import asyncio
                
                # Create a claim from the prompt first
                claim_coroutine = processing_interface.create_claim(
                    content=prompt_text,
                    confidence=confidence,
                    tags=["prompt", "workspace-context"]
                )
                claim = asyncio.run(claim_coroutine)
                
                # Evaluate the claim to get LLM processing
                eval_coroutine = processing_interface.evaluate_claim(claim.id)
                eval_result = asyncio.run(eval_coroutine)
                
                progress.update(task, description="Prompt processed successfully!")
                
                # Display results
                if eval_result.success:
                    panel = Panel(
                        f"[bold]Prompt:[/bold] {prompt_text}\n"
                        f"[bold]Claim ID:[/bold] {claim.id}\n"
                        f"[bold]Confidence:[/bold] {eval_result.new_confidence:.2f}\n"
                        f"[bold]State:[/bold] {eval_result.state.value}\n"
                        f"[bold]Evaluation:[/bold] {eval_result.evaluation_summary}\n"
                        f"[bold]Processing Time:[/bold] {eval_result.processing_time:.2f}s",
                        title="Prompt Processing Results",
                        border_style="green",
                    )
                    console.print(panel)
                else:
                    panel = Panel(
                        f"[bold]Prompt:[/bold] {prompt_text}\n"
                        f"[bold]Claim ID:[/bold] {claim.id}\n"
                        f"[bold]Error:[/bold] {eval_result.evaluation_summary}\n"
                        f"[bold]Processing Time:[/bold] {eval_result.processing_time:.2f}s",
                        title="Prompt Processing Results",
                        border_style="red",
                    )
                    console.print(panel)

            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error processing prompt: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise

@app.command()
def config():
    """Show configuration status and validation."""
    console.print("[bold]Configuration Status[/bold]")
    console.print("=" * 50)

    try:
        is_valid = validate_config()
    except Exception as e:
        is_valid = False
        console.print(f"[yellow]Warning: Config validation error: {e}[/yellow]")

    # Print validation result
    if is_valid:
        console.print("[bold green]Configuration Validation: PASSED[/bold green]")
    else:
        console.print("[bold yellow]Configuration Validation: INCOMPLETE[/bold yellow]")
        console.print("  • Some configuration may be missing or invalid")

    console.print(f"\n[bold]Next Steps:[/bold]")
    if is_valid:
        console.print("Configuration is valid!")
        console.print("You can now use: create, get, search, analyze, prompt commands")

        # Show available providers
        console.print(f"\n[bold]Available Providers:[/bold]")
        try:
            # Just check the config directly without creating a ProcessingInterface
            from src.config.unified_config import UnifiedConfig as Config
            
            config = Config()
            if hasattr(config, 'providers') and config.providers:
                for i, provider in enumerate(config.providers):
                    console.print(
                        f"  • Provider {i + 1}: {provider.get('name', 'Unknown')} ({provider.get('url', 'No URL')})"
                    )
            else:
                console.print("  • No providers configured in config")
                
        except Exception as e:
            console.print(f"  • Error checking providers: {e}")

@app.command()
def providers():
    """Show available providers and setup instructions."""
    console.print("[bold]Available Providers[/bold]")
    console.print("=" * 50)

    try:
        # Create a temporary processing interface for this command
        from src.conjecture import Conjecture
        from src.config.unified_config import UnifiedConfig as Config
        
        config = Config()
        temp_interface = Conjecture(config=config)
        
        if hasattr(temp_interface, 'llm_bridge') and hasattr(temp_interface.llm_bridge, 'provider_manager'):
            providers = temp_interface.llm_bridge.provider_manager.get_providers()

            if not providers:
                console.print("[red]No providers configured[/red]")
                console.print("\n[yellow]Setup Instructions:[/yellow]")
                console.print("1. Run: conjecture setup")
                console.print("2. Edit: ~/.conjecture/config.json")
                return

            console.print(f"[bold green]Found {len(providers)} provider(s):[/bold green]")

            for i, provider in enumerate(providers):
                console.print(f"  • Provider {i + 1}: {provider.get('name', 'Unknown')}")
                console.print(f"    URL: {provider.get('url', 'No URL')}")
                console.print(f"    Status: {provider.get('status', 'Unknown')}")
                if provider.get("api"):
                    console.print(f"    API Key: {provider.get('api')}")
                console.print()
        else:
            console.print("[bold green]Using Conjecture processing interface[/bold green]")
            console.print("This interface doesn't use external providers")

    except Exception as e:
        console.print(f"[red]Error loading providers: {e}[/red]")

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
        from src.config.unified_provider_validator import get_unified_validator

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
        # Override processing interface if specified
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print(
                "[red]No processing interface initialized. Use --backend option.[/red]"
            )
            raise typer.Exit(1)

        # Get statistics using ProcessingInterface
        import asyncio
        try:
            stats = asyncio.run(processing_interface.get_statistics())
        except Exception:
            # Fallback stats if get_statistics is not implemented
            stats = {
                "total_claims": "Unknown",
                "avg_confidence": "Unknown",
                "unique_users": "Unknown",
                "backend_type": "Conjecture",
            }

        # Create statistics table
        table = Table(title="Database Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Backend", stats.get("backend_type", "Conjecture"))
        table.add_row("Total Claims", str(stats.get("total_claims", "Unknown")))
        table.add_row(
            "Average Confidence",
            f"{stats.get('avg_confidence', 0):.3f}"
            if isinstance(stats.get("avg_confidence"), (int, float))
            else str(stats.get("avg_confidence", "Unknown")),
        )
        table.add_row("Unique Users", str(stats.get("unique_users", "Unknown")))

        # Get database path from config if available
        try:
            from src.config.unified_config import UnifiedConfig as Config
            config = Config()
            if hasattr(config, 'database_path'):
                table.add_row("Database Path", config.database_path)
        except Exception:
            pass

        table.add_row("Embedding Model", "all-MiniLM-L6-v2")

        # Add processing interface info
        try:
            health = asyncio.run(processing_interface.get_health_status())
            if health.get('services', {}).get('llm_bridge', False):
                table.add_row("LLM Bridge", "OK")
            else:
                table.add_row("LLM Bridge", "FAIL")
        except Exception:
            table.add_row("LLM Bridge", "Unknown")

        console.print(table)

    except typer.Exit:
        raise

@app.command()
def backends():
    """Show available processing interface information."""
    console.print("[bold]Processing Interface Information[/bold]")
    console.print("=" * 50)

    table = Table(title="Processing Interface Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description", style="white")

    # Using ProcessingInterface with Conjecture implementation
    console.print("Using ProcessingInterface with Conjecture implementation")
    console.print("Provides clean architecture separation between CLI and processing layers")
    console.print("Configuration is loaded from ~/.conjecture/config.json")
    
    table.add_row("Processing Interface", "Available", "Conjecture - Async Evidence-Based AI Reasoning")
    table.add_row("Data Layer", "Integrated", "SQLite + FAISS for vector storage")
    table.add_row("LLM Bridge", "Configurable", "Supports multiple LLM providers")
    table.add_row("Async Evaluation", "Enabled", "Background claim evaluation service")
    table.add_row("Tool Management", "Dynamic", "Runtime tool creation and execution")
    table.add_row("Event Streaming", "Supported", "Real-time processing events")
    
    console.print(table)

@app.command()
def health():
    """Check system health and processing interface availability."""
    console.print("[bold]System Health Check[/bold]")
    console.print("=" * 50)

    # Check processing interface system
    try:
        processing_interface = current_processing_interface
        if processing_interface:
            import asyncio
            health = asyncio.run(processing_interface.get_health_status())
            
            if health.get("healthy", False):
                console.print("[bold green]System Status: Healthy[/bold green]")
                
                services = health.get("services", {})
                console.print("\n[bold]Service Status:[/bold]")
                for service_name, status in services.items():
                    status_text = "✓ OK" if status else "✗ Failed"
                    status_color = "green" if status else "red"
                    console.print(f"  • {service_name.title()}: [{status_color}]{status_text}[/{status_color}]")
                
                # Show additional health info
                if "active_sessions" in health:
                    console.print(f"\nActive Sessions: {health['active_sessions']}")
                if "event_subscribers" in health:
                    console.print(f"Event Subscribers: {health['event_subscribers']}")
                    
            else:
                console.print("[bold red]System Status: Unhealthy[/bold red]")
                if "error" in health:
                    console.print(f"Error: {health['error']}")
        else:
            console.print("[bold red]System Status: No processing interface[/bold red]")
            console.print("Please initialize the processing interface")

    except Exception as e:
        console.print(f"[bold red]System Status: Error - {e}[/bold red]")

    console.print("\n[bold]Health Check Complete[/bold]")

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


@app.command()
def tree(
    claim_id: str = typer.Argument(..., help="ID of the claim to visualize"),
    depth: int = typer.Option(3, "--depth", "-d", help="Maximum depth to traverse (default: 3, max: 10)"),
    min_confidence: float = typer.Option(0.0, "--confidence", "-c", help="Minimum confidence threshold (0.0-1.0)"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, ascii"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Override backend selection"),
):
    """Visualize claim support tree (UX-0007).

    Shows the claim and all its supporting claims (sub-claims) in a tree structure.
    This helps users understand the evidence chain supporting a conclusion.

    Example:
        conjecture tree c00000001 --depth 3
        conjecture tree c00000001 --confidence 0.5 --format json
    """
    try:
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print("[red]No processing interface initialized[/red]")
            raise typer.Exit(1)

        depth = min(depth, 10)  # Cap at 10

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Building claim tree...", total=None)

            try:
                import asyncio

                # Get the root claim
                claim = asyncio.run(processing_interface.get_claim(claim_id))
                if not claim:
                    progress.update(task, description="Claim not found")
                    error_console.print(f"[red]Claim not found: {claim_id}[/red]")
                    raise typer.Exit(1)

                # Build tree recursively
                async def get_claim_by_id(cid):
                    return await processing_interface.get_claim(cid)

                from src.utils.visualization import build_claim_tree, ClaimNode

                def get_claim_sync(cid):
                    return asyncio.run(get_claim_by_id(cid))

                tree_root = build_claim_tree(
                    claim, get_claim_sync, 
                    max_depth=depth, 
                    min_confidence=min_confidence
                )

                progress.update(task, description="Rendering tree...")

                if format == "json":
                    import json
                    result = tree_root.to_dict()
                    console.print(json.dumps(result, indent=2))
                elif format == "ascii":
                    from src.utils.visualization import render_tree_ascii
                    ascii_tree = render_tree_ascii(tree_root)
                    console.print(ascii_tree)
                else:  # rich
                    from rich.tree import Tree
                    from rich.console import Group
                    from src.utils.visualization import confidence_color

                    def build_rich_tree(node: ClaimNode, tree: Tree):
                        """Recursively build Rich Tree."""
                        conf = node.claim.confidence
                        color = confidence_color(conf)
                        conf_marker = "●" if conf >= 0.8 else "○"
                        content = node.claim.content[:60]
                        if len(node.claim.content) > 60:
                            content += "..."
                        
                        label = f"[{color}]{node.claim.id}[/{color}] {conf_marker} {content}"
                        
                        if node.children:
                            branch = tree.add(f"[bold]{label}[/bold]", guide_style=color)
                            for child in node.children:
                                build_rich_tree(child, branch)
                        else:
                            tree.add(f"{label}")

                    rich_tree = Tree(
                        f"[bold blue]Claim Tree: {claim_id}[/bold blue] (depth={depth})",
                        guide_style="dim"
                    )
                    for child in tree_root.children:
                        build_rich_tree(child, rich_tree)

                    console.print(rich_tree)

                progress.update(task, description="Done!")

            except Exception as e:
                progress.update(task, description="Error")
                error_console.print(f"[red]Error building tree: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def trace(
    claim_id: str = typer.Argument(..., help="ID of the claim to trace"),
    format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Override backend selection"),
):
    """Show chain from root to claim (UX-0007).

    Traces the path from the root claim (no supers) down to the specified claim.
    This helps users understand how a claim relates back to the original context.

    Example:
        conjecture trace c00000005
        conjecture trace c00000005 --format json
    """
    try:
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print("[red]No processing interface initialized[/red]")
            raise typer.Exit(1)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Building claim trace...", total=None)

            try:
                import asyncio

                # Get the target claim
                claim = asyncio.run(processing_interface.get_claim(claim_id))
                if not claim:
                    progress.update(task, description="Claim not found")
                    error_console.print(f"[red]Claim not found: {claim_id}[/red]")
                    raise typer.Exit(1)

                # Build trace recursively
                async def get_claim_by_id(cid):
                    return await processing_interface.get_claim(cid)

                from src.utils.visualization import build_claim_trace

                def get_claim_sync(cid):
                    return asyncio.run(get_claim_by_id(cid))

                trace_result = build_claim_trace(claim, get_claim_sync)

                progress.update(task, description="Rendering trace...")

                if format == "json":
                    import json
                    result = trace_result.to_dict()
                    console.print(json.dumps(result, indent=2))
                else:  # rich
                    from rich.table import Table
                    from src.utils.visualization import confidence_color

                    table = Table(title=f"[bold blue]Claim Trace: {claim_id}[/bold blue] ({len(trace_result.nodes)} hops)")
                    table.add_column("Hop", style="cyan", width=4)
                    table.add_column("Claim ID", style="bold")
                    table.add_column("Content", width=50)
                    table.add_column("Confidence", justify="right")

                    for i, node in enumerate(trace_result.nodes):
                        conf = node.claim.confidence
                        color = confidence_color(conf)
                        content = node.claim.content[:47]
                        if len(node.claim.content) > 47:
                            content += "..."
                        
                        is_target = node.claim.id == claim_id
                        marker = "[bold yellow]▶[/bold yellow] " if is_target else "   "
                        
                        table.add_row(
                            str(i),
                            f"[{color}]{node.claim.id}[/{color}]",
                            f"{marker}{content}",
                            f"[{color}]{conf:.2f}[/{color}]"
                        )

                    console.print(table)

                progress.update(task, description="Done!")

            except Exception as e:
                progress.update(task, description="Error")
                error_console.print(f"[red]Error building trace: {e}[/red]")
                raise typer.Exit(1)

    except typer.Exit:
        raise


@app.command()
def browse(
    root_id: str = typer.Argument(..., help="ID of the root claim to browse"),
    max_depth: int = typer.Option(5, "--max-depth", "-d", help="Maximum tree depth (default: 5, max: 10)"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Override backend selection"),
):
    """Launch interactive TUI claim tree browser (UX-0007 Phase 3).

    Keyboard controls:
        j / ↓ : Move cursor down
        k / ↑ : Move cursor up
        l / → / Space : Expand current node
        h / ←      : Collapse current node
        Enter     : Toggle expand/collapse of current node
        /         : Enter search mode
        Escape    : Exit search / cancel
        q         : Quit browser
        g         : Jump to top
        G         : Jump to bottom

    Example:
        conjecture browse c00000001
        conjecture browse c00000001 --max-depth 3
    """
    try:
        if backend:
            processing_interface = get_processing_interface(backend)
        else:
            processing_interface = current_processing_interface

        if not processing_interface:
            error_console.print("[red]No processing interface initialized[/red]")
            raise typer.Exit(1)

        max_depth = min(max_depth, 10)

        try:
            from src.cli.claim_browser import browse_claims
        except ImportError as e:
            error_console.print(f"[red]Failed to import claim browser: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Launching claim browser for [bold]{root_id}[/bold]...[/cyan]")

        try:
            browser = browse_claims(
                processing_interface=processing_interface,
                root_id=root_id,
                max_depth=max_depth,
            )
        except Exception as e:
            error_console.print(f"[red]Failed to build claim tree: {e}[/red]")
            raise typer.Exit(1)

        if browser.root_node is None:
            error_console.print(f"[red]Claim not found: {root_id}[/red]")
            raise typer.Exit(1)

        browser.run_interactive()

    except typer.Exit:
        raise


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Start Conjecture as an LLM endpoint server (M-0007).

    Runs an OpenAI-compatible HTTP server that enhances queries with claim context.
    Connect any OpenAI-compatible client to http://localhost:8000/v1/chat/completions

    Example:
        conjecture serve --port 8000

        curl http://localhost:8000/v1/chat/completions \\
          -H "Content-Type: application/json" \\
          -d '{"model":"conjecture","messages":[{"role":"user","content":"Hello"}]}'
    """
    console.print("[bold blue]Conjecture LLM Endpoint Server[/bold blue]")
    console.print(f"Starting on [cyan]{host}:{port}[/cyan]...")

    try:
        from src.endpoint.http_server import ConjectureServer, FASTAPI_AVAILABLE

        if not FASTAPI_AVAILABLE:
            console.print("[bold red]ERROR: FastAPI not installed[/bold red]")
            console.print("Run: [cyan]pip install fastapi uvicorn[/cyan]")
            raise typer.Exit(1)

        import asyncio
        server = ConjectureServer(host=host, port=port)
        asyncio.run(server.run())

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Run: [cyan]pip install fastapi uvicorn[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Server error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def mcp():
    """Start Conjecture as an MCP server (A-0013).

    Model Context Protocol server for Claude Desktop, Cursor, and other MCP clients.
    Provides tools: build_context, upsert_claim, explore_next, get_claim_support.

    Example:
        conjecture mcp
    """
    console.print("[bold blue]Conjecture MCP Server[/bold blue]")

    try:
        from src.endpoint.mcp_server import main as mcp_main
        mcp_main()
    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Run: [cyan]pip install mcp[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]MCP error: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
