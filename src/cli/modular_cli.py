#!/usr/bin/env python3
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
    from src.conjecture import Conjecture
    from src.config.unified_config import UnifiedConfig as Config
    
    try:
        # Create configuration
        config = Config()
        
        # Create Conjecture instance (implements ProcessingInterface)
        interface = Conjecture(config=config)
        current_processing_interface = interface
        
        # Start services only if there's an event loop
        try:
            import asyncio
            asyncio.create_task(current_processing_interface.start_services())
            console.print("[green]+ Processing interface initialized[/green]")
        except RuntimeError as e:
            if "no running event loop" in str(e):
                console.print("[yellow]+ Processing interface created (services not started - no event loop)[/yellow]")
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
        
        console.print("\n[bold yellow]To create user config with API keys:[/bold yellow]")
        console.print("[cyan]cp src/config/default_config.json ~/.conjecture/config.json[/cyan]")
        console.print("[cyan]# Then edit ~/.conjecture/config.json to add your API keys[/cyan]")

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
            # No running loop, create one and start services
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(processing_interface.start_services())
            loop.close()

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
                    
                    panel = Panel(
                        f"[bold]ID:[/bold] {claim.id}\n"
                        f"[bold]Content:[/bold] {claim.content}\n"
                        f"[bold]Confidence:[/bold] {claim.confidence:.2f}\n"
                        f"[bold]State:[/bold] {state}\n"
                        f"[bold]User:[/bold] {user}\n"
                        f"[bold]Created:[/bold] {created}\n"
                        f"[bold]Tags:[/bold] {tags}\n",
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
                results_coroutine = processing_interface.search_claims(query, limit=limit)
                results = asyncio.run(results_coroutine)

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
    table.add_row("Data Layer", "Integrated", "SQLite + ChromaDB for vector storage")
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

if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
