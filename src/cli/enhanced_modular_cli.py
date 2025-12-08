#!/usr/bin/env python3
"""
Enhanced Modular Conjecture CLI with Improved UX
Integrates all UI enhancements for better user experience
"""

import asyncio
import typer
from typing import List, Optional, Dict, Any
import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import enhanced UI components
from .ui_enhancements import (
    UIEnhancer, UIConfig, OutputMode, VerbosityLevel,
    create_ui_enhancer
)
from .encoding_handler import setup_unicode_environment, get_safe_console
from .tf_suppression import suppress_tensorflow_warnings

# Setup environment
setup_unicode_environment()
suppress_tensorflow_warnings()

# Import base CLI components
from .base_cli import BaseCLI
from .modular_cli import get_backend, print_backend_info
from src.config.unified_config import validate_config

# Create the enhanced Typer app
app = typer.Typer(
    name="conjecture",
    help="Conjecture CLI - Enhanced user experience with better progress indicators and accessibility",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Global UI enhancer instance
ui_enhancer: Optional[UIEnhancer] = None

# Global backend instance
current_backend: Optional[BaseCLI] = None


def get_ui_enhancer(
    output_mode: str = "normal",
    verbosity: str = "info",
    quiet: bool = False,
    verbose: bool = False,
    json_output: bool = False,
    no_progress: bool = False,
    no_color: bool = False,
    accessible: bool = False
) -> UIEnhancer:
    """Get or create a UI enhancer with the specified configuration"""
    global ui_enhancer

    # Determine output mode
    if json_output:
        output_mode = "json"
    elif quiet:
        output_mode = "quiet"
    elif verbose:
        output_mode = "verbose"
    elif accessible:
        output_mode = "accessible"

    # Determine verbosity level
    if quiet:
        verbosity = "error"
    elif verbose:
        verbosity = "debug"
    else:
        verbosity = verbosity

    # Create configuration
    config = UIConfig(
        output_mode=OutputMode(output_mode),
        verbosity=VerbosityLevel(verbosity),
        show_progress=not no_progress,
        color_enabled=not no_color,
        accessibility_mode=accessible
    )

    ui_enhancer = UIEnhancer(config)
    return ui_enhancer


@app.callback()
def main(
    output_mode: str = typer.Option(
        "normal", "--output", "-o",
        help="Output format: normal, quiet, verbose, json, markdown, accessible"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Minimal output"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Detailed output"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="JSON output format"
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable progress indicators"
    ),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable colored output"
    ),
    accessible: bool = typer.Option(
        False, "--accessible", help="Screen reader friendly output"
    ),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend selection"),
):
    """Conjecture CLI - Enhanced interface with improved user experience."""
    global current_backend, ui_enhancer

    # Initialize UI enhancer
    ui_enhancer = get_ui_enhancer(
        output_mode=output_mode,
        verbosity="info",
        quiet=quiet,
        verbose=verbose,
        json_output=json_output,
        no_progress=no_progress,
        no_color=no_color,
        accessible=accessible
    )

    # Initialize backend
    try:
        current_backend = get_backend(backend)

        if verbose:
            ui_enhancer.formatter.print_info(f"Initialized {backend} backend")
            print_backend_info(current_backend)

        # Show contextual help
        ui_enhancer.show_command_help("general")

    except typer.Exit:
        raise
    except Exception as e:
        ui_enhancer.formatter.print_error(
            "Failed to initialize backend",
            str(e),
            "Try running 'conjecture setup' to configure providers"
        )
        raise typer.Exit(1)


@app.command()
def create(
    content: str = typer.Argument(..., help="Content of the claim"),
    confidence: float = typer.Option(
        0.8, "--confidence", "-c", help="Confidence score (0.0-1.0)", min=0.0, max=1.0
    ),
    user: str = typer.Option("user", "--user", "-u", help="User ID"),
    analyze: bool = typer.Option(
        False, "--analyze", "-a", help="Analyze with configured LLM"
    ),
    tags: Optional[List[str]] = typer.Option(
        None, "--tag", "-t", help="Tags for the claim (can be used multiple times)"
    ),
    backend_override: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Create a new claim with enhanced progress tracking."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    # Show contextual help
    ui_enhancer.show_command_help("create")

    # Validate input
    if not content or len(content.strip()) < 5:
        ui_enhancer.formatter.print_error(
            "Claim content is required and must be at least 5 characters",
            suggestion="Provide meaningful content for your claim"
        )
        raise typer.Exit(1)

    # Override backend if specified
    cli_backend = current_backend
    if backend_override:
        try:
            cli_backend = get_backend(backend_override)
        except Exception as e:
            ui_enhancer.formatter.print_error(
                f"Failed to initialize backend {backend_override}",
                str(e)
            )
            raise typer.Exit(1)

    # Start operation with progress tracking
    task_id = ui_enhancer.start_operation("Creating claim")

    try:
        ui_enhancer.update_progress(task_id, "Validating claim content", 1)

        # Additional validation
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        ui_enhancer.update_progress(task_id, "Storing claim in database", 1)

        # Create the claim
        claim_id = cli_backend.create_claim(content, confidence, user, analyze)

        ui_enhancer.update_progress(task_id, "Finalizing claim creation", 1)

        # Show success message with timing
        ui_enhancer.complete_operation(
            task_id,
            f"Claim {claim_id} created successfully" +
            (" with analysis" if analyze else "")
        )

        # Show claim details in verbose mode
        if ui_enhancer.config.verbosity.value >= VerbosityLevel.VERBOSE.value:
            claim = cli_backend.get_claim(claim_id)
            if claim:
                details = ui_enhancer.formatter.format_claim_details(claim)
                ui_enhancer.formatter.print(details)

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Failed to create claim: {e}",
            "Check your claim content and try again"
        )
        raise typer.Exit(1)


@app.command()
def get(
    claim_id: str = typer.Argument(..., help="ID of the claim to retrieve"),
    backend_override: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Get a claim by ID with enhanced formatting."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    # Override backend if specified
    cli_backend = current_backend
    if backend_override:
        try:
            cli_backend = get_backend(backend_override)
        except Exception as e:
            ui_enhancer.formatter.print_error(
                f"Failed to initialize backend {backend_override}",
                str(e)
            )
            raise typer.Exit(1)

    task_id = ui_enhancer.start_operation(f"Retrieving claim {claim_id}")

    try:
        ui_enhancer.update_progress(task_id, "Searching for claim", 1)

        claim = cli_backend.get_claim(claim_id)

        ui_enhancer.update_progress(task_id, "Formatting claim details", 1)

        if claim:
            ui_enhancer.complete_operation(task_id, f"Claim {claim_id} retrieved successfully")

            # Format and display claim details
            details = ui_enhancer.formatter.format_claim_details(claim)
            ui_enhancer.formatter.print(details)

        else:
            ui_enhancer.fail_operation(
                task_id,
                f"Claim {claim_id} not found",
                "Check the claim ID or use 'conjecture search' to find claims"
            )
            raise typer.Exit(1)

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error retrieving claim {claim_id}: {e}",
            "Verify the claim ID and try again"
        )
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results", min=1, max=100),
    backend_override: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Search claims by content with enhanced results display."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    # Show contextual help
    ui_enhancer.show_command_help("search")

    # Override backend if specified
    cli_backend = current_backend
    if backend_override:
        try:
            cli_backend = get_backend(backend_override)
        except Exception as e:
            ui_enhancer.formatter.print_error(
                f"Failed to initialize backend {backend_override}",
                str(e)
            )
            raise typer.Exit(1)

    task_id = ui_enhancer.start_operation(f"Searching claims for: '{query}'")

    try:
        ui_enhancer.update_progress(task_id, "Building search query", 1)
        ui_enhancer.update_progress(task_id, "Searching database", 1)
        ui_enhancer.update_progress(task_id, "Collecting results", 1)

        results = cli_backend.search_claims(query, limit)

        ui_enhancer.update_progress(task_id, "Formatting results", 1)

        ui_enhancer.complete_operation(
            task_id,
            f"Found {len(results)} result{'s' if len(results) != 1 else ''}"
        )

        # Display results
        if results:
            table = ui_enhancer.formatter.format_claim_table(results, query)
            ui_enhancer.formatter.print(table)

            # Show search tips in verbose mode
            if ui_enhancer.config.verbosity.value >= VerbosityLevel.VERBOSE.value:
                ui_enhancer.formatter.print_info(
                    f"Search completed in {ui_enhancer.operation_start_time and (time.time() - ui_enhancer.operation_start_time):.2f}s"
                )
        else:
            ui_enhancer.formatter.print_warning(
                f"No claims found matching '{query}'",
                suggestion="Try different keywords or create some claims first"
            )

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error searching claims: {e}",
            "Try modifying your search query"
        )
        raise typer.Exit(1)


@app.command()
def analyze(
    claim_id: str = typer.Argument(..., help="ID of the claim to analyze"),
    backend_override: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Analyze a claim using LLM services with enhanced progress."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    # Show contextual help
    ui_enhancer.show_command_help("analyze")

    # Override backend if specified
    cli_backend = current_backend
    if backend_override:
        try:
            cli_backend = get_backend(backend_override)
        except Exception as e:
            ui_enhancer.formatter.print_error(
                f"Failed to initialize backend {backend_override}",
                str(e)
            )
            raise typer.Exit(1)

    task_id = ui_enhancer.start_operation(f"Analyzing claim {claim_id}")

    try:
        ui_enhancer.update_progress(task_id, "Retrieving claim", 1)

        # Verify claim exists
        claim = cli_backend.get_claim(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        ui_enhancer.update_progress(task_id, "Preparing analysis", 1)
        ui_enhancer.update_progress(task_id, "Sending to LLM for analysis", 2)
        ui_enhancer.update_progress(task_id, "Processing analysis results", 1)

        analysis = cli_backend.analyze_claim(claim_id)

        ui_enhancer.update_progress(task_id, "Formatting analysis", 1)

        ui_enhancer.complete_operation(task_id, f"Analysis completed for claim {claim_id}")

        # Display analysis results
        if ui_enhancer.config.output_mode == OutputMode.JSON:
            ui_enhancer.formatter.print(analysis)
        else:
            # Create a formatted analysis panel
            analysis_content = f"""[bold cyan]Claim ID:[/bold cyan] {analysis['claim_id']}
[bold]Backend:[/bold] {analysis.get('backend', 'unknown')}
[bold]Analysis Type:[/bold] {analysis.get('analysis_type', 'unknown')}
[bold green]Confidence:[/bold green] {analysis.get('confidence_score', 0):.2f}
[bold yellow]Sentiment:[/bold yellow] {analysis.get('sentiment', 'unknown')}
[bold blue]Topics:[/bold blue] {', '.join(analysis.get('topics', []))}
[bold]Status:[/bold] {analysis.get('verification_status', 'unknown')}"""

            if 'reasoning' in analysis:
                analysis_content += f"\n[bold]Reasoning:[/bold] {analysis['reasoning']}"

            from rich.panel import Panel
            if ui_enhancer.config.output_mode != OutputMode.QUIET:
                panel = Panel(
                    analysis_content,
                    title=f"Analysis Results for {claim_id}",
                    border_style="green"
                )
                ui_enhancer.formatter.print(panel)

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error analyzing claim {claim_id}: {e}",
            "Ensure the claim exists and LLM services are configured"
        )
        raise typer.Exit(1)


@app.command()
def prompt(
    prompt_text: str = typer.Argument(
        ..., help="The prompt text (will be created as a claim)"
    ),
    confidence: float = typer.Option(
        0.8, "--confidence", "-c", help="Initial confidence score", min=0.0, max=1.0
    ),
    verbosity: int = typer.Option(
        0,
        "--verbose",
        "-v",
        help="Verbosity level: 0=final only, 1=tool calls, 2=claims>90%",
    ),
):
    """Process a prompt as a claim with enhanced workspace context."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    task_id = ui_enhancer.start_operation("Processing prompt")

    try:
        ui_enhancer.update_progress(task_id, "Analyzing prompt content", 1)
        ui_enhancer.update_progress(task_id, "Collecting workspace context", 1)
        ui_enhancer.update_progress(task_id, "Processing with LLM", 2)
        ui_enhancer.update_progress(task_id, "Formatting results", 1)

        result = current_backend.process_prompt(prompt_text, confidence, verbosity)

        ui_enhancer.complete_operation(task_id, "Prompt processed successfully")

        # Display results based on output mode
        if ui_enhancer.config.output_mode == OutputMode.JSON:
            ui_enhancer.formatter.print(result)
        else:
            ui_enhancer.formatter.print_success(f"Prompt processed: {result}")

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error processing prompt: {e}",
            "Check your prompt content and LLM configuration"
        )
        raise typer.Exit(1)


@app.command()
def config():
    """Show configuration status with enhanced formatting."""
    if not ui_enhancer:
        print("CLI not properly initialized")
        raise typer.Exit(1)

    # Show contextual help
    ui_enhancer.show_command_help("config")

    task_id = ui_enhancer.start_operation("Checking configuration")

    try:
        ui_enhancer.update_progress(task_id, "Validating configuration files", 1)
        ui_enhancer.update_progress(task_id, "Checking provider status", 1)

        is_valid = validate_config()

        ui_enhancer.update_progress(task_id, "Formatting status report", 1)

        ui_enhancer.complete_operation(task_id, "Configuration check completed")

        # Display configuration status
        if ui_enhancer.config.output_mode == OutputMode.JSON:
            config_status = {
                "valid": is_valid,
                "providers": [],
                "configuration_files": {
                    "user_config": str(Path.home() / ".conjecture/config.json"),
                    "workspace_config": str(Path.cwd() / ".conjecture/config.json")
                }
            }
            ui_enhancer.formatter.print(config_status)
        else:
            if is_valid:
                ui_enhancer.formatter.print_success("Configuration is valid")
            else:
                ui_enhancer.formatter.print_warning(
                    "Configuration incomplete",
                    "Some configuration may be missing or invalid"
                )

            # Show provider information
            try:
                backend = get_backend()
                providers = backend.provider_manager.get_providers()

                if providers:
                    ui_enhancer.formatter.print_info(f"Found {len(providers)} configured provider(s)")
                    for i, provider in enumerate(providers):
                        ui_enhancer.formatter.print(f"  • {provider.get('name', f'Provider {i+1}')}")
                else:
                    ui_enhancer.formatter.print_warning(
                        "No providers configured",
                        "Run 'conjecture setup' to configure providers"
                    )
            except Exception:
                ui_enhancer.formatter.print_warning("Could not check provider status")

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error checking configuration: {e}",
            "Check your configuration files"
        )
        raise typer.Exit(1)


@app.command()
def setup(
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive setup mode"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Setup specific provider"
    ),
):
    """Setup provider configuration with enhanced guidance."""
    if not ui_enhancer:
        print("CLI not properly initialized")
        raise typer.Exit(1)

    task_id = ui_enhancer.start_operation("Provider setup")

    try:
        ui_enhancer.update_progress(task_id, "Loading setup configuration", 1)

        if provider:
            # Setup specific provider
            ui_enhancer.formatter.print_info(f"Setup guide for {provider.title()}")

            setup_steps = [
                f"1. Copy template: cp .env.example .env",
                f"2. Edit the file .env",
                f"3. Add the configuration for {provider}",
                f"4. Run: conjecture config to validate"
            ]

            for step in setup_steps:
                ui_enhancer.formatter.print_info(step)

        else:
            # Show all options
            if interactive:
                # Interactive provider selection
                providers = ["ollama", "lm_studio", "openai", "anthropic", "google", "cohere", "skip"]
                provider_choice = ui_enhancer.prompter.ask_choice(
                    "Choose provider to configure",
                    providers,
                    default="skip"
                )

                if provider_choice != "skip":
                    ui_enhancer.formatter.print_info(f"Setup guide for {provider_choice.title()}")
                    ui_enhancer.formatter.print_info("1. Copy template: cp .env.example .env")
                    ui_enhancer.formatter.print_info("2. Edit the file .env")
                    ui_enhancer.formatter.print_info("3. Run: conjecture config to validate")
            else:
                # Non-interactive - show all options
                ui_enhancer.formatter.print_info("Available providers:")
                provider_list = [
                    "• ollama - Local models (recommended for privacy)",
                    "• lm_studio - Local model server",
                    "• openai - GPT models (cloud)",
                    "• anthropic - Claude models (cloud)",
                    "• google - Gemini models (cloud)",
                    "• cohere - Cohere models (cloud)"
                ]

                for provider_info in provider_list:
                    ui_enhancer.formatter.print_info(provider_info)

        ui_enhancer.complete_operation(task_id, "Setup information displayed")

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error during setup: {e}",
            "Check the setup documentation for more details"
        )
        raise typer.Exit(1)


@app.command()
def stats(
    backend_override: Optional[str] = typer.Option(
        None, "--backend", "-b", help="Override backend selection"
    ),
):
    """Show database and backend statistics with enhanced formatting."""
    if not ui_enhancer or not current_backend:
        ui_enhancer.formatter.print_error("CLI not properly initialized")
        raise typer.Exit(1)

    # Override backend if specified
    cli_backend = current_backend
    if backend_override:
        try:
            cli_backend = get_backend(backend_override)
        except Exception as e:
            ui_enhancer.formatter.print_error(
                f"Failed to initialize backend {backend_override}",
                str(e)
            )
            raise typer.Exit(1)

    task_id = ui_enhancer.start_operation("Collecting statistics")

    try:
        ui_enhancer.update_progress(task_id, "Querying database", 1)
        ui_enhancer.update_progress(task_id, "Calculating metrics", 1)
        ui_enhancer.update_progress(task_id, "Formatting results", 1)

        # Get statistics
        if hasattr(cli_backend, "_get_database_stats"):
            stats = cli_backend._get_database_stats()
        else:
            stats = {
                "total_claims": "Unknown",
                "avg_confidence": "Unknown",
                "unique_users": "Unknown",
                "backend_type": cli_backend._get_backend_type(),
            }

        ui_enhancer.complete_operation(task_id, "Statistics collected")

        # Display statistics
        if ui_enhancer.config.output_mode == OutputMode.JSON:
            ui_enhancer.formatter.print(stats)
        else:
            # Create enhanced statistics table
            from rich.table import Table

            table = Table(title="Database Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", width=20)
            table.add_column("Value", style="green", width=30)

            table.add_row("Backend", str(stats.get("backend_type", "Unknown")))
            table.add_row("Total Claims", str(stats.get("total_claims", "Unknown")))

            avg_conf = stats.get("avg_confidence", 0)
            if isinstance(avg_conf, (int, float)):
                table.add_row("Average Confidence", f"{avg_conf:.3f}")
            else:
                table.add_row("Average Confidence", str(avg_conf))

            table.add_row("Unique Users", str(stats.get("unique_users", "Unknown")))

            if hasattr(cli_backend, "db_path"):
                table.add_row("Database Path", cli_backend.db_path)

            table.add_row("Embedding Model", "all-MiniLM-L6-v2")

            # Add backend-specific info
            backend_info = cli_backend.get_backend_info()
            if backend_info.get("provider"):
                table.add_row("Current Provider", backend_info["provider"])

            ui_enhancer.formatter.print(table)

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Error collecting statistics: {e}",
            "Check database connectivity"
        )
        raise typer.Exit(1)


@app.command()
def health():
    """Check system health with enhanced status reporting."""
    if not ui_enhancer:
        print("CLI not properly initialized")
        raise typer.Exit(1)

    task_id = ui_enhancer.start_operation("System health check")

    try:
        ui_enhancer.update_progress(task_id, "Checking provider system", 1)
        ui_enhancer.update_progress(task_id, "Testing connectivity", 1)
        ui_enhancer.update_progress(task_id, "Validating configuration", 1)

        # Check provider system
        backend = get_backend()
        providers = backend.provider_manager.get_providers()

        ui_enhancer.update_progress(task_id, "Compiling health report", 1)

        ui_enhancer.complete_operation(task_id, "Health check completed")

        # Display health status
        if ui_enhancer.config.output_mode == OutputMode.JSON:
            health_status = {
                "status": "healthy" if providers else "unhealthy",
                "providers_count": len(providers),
                "providers": providers,
                "timestamp": time.time()
            }
            ui_enhancer.formatter.print(health_status)
        else:
            if providers:
                ui_enhancer.formatter.print_success(
                    f"System Status: {len(providers)} provider(s) configured"
                )
                for i, provider in enumerate(providers):
                    ui_enhancer.formatter.print_info(
                        f"  • {provider.get('name', f'Provider {i + 1}')}: {provider.get('url', 'No URL')}"
                    )
            else:
                ui_enhancer.formatter.print_warning(
                    "System Status: No providers configured",
                    "Please configure providers in ~/.conjecture/config.json"
                )

            ui_enhancer.formatter.print_success("Health check complete")

    except Exception as e:
        ui_enhancer.fail_operation(
            task_id,
            f"Health check failed: {e}",
            "Check system configuration and connectivity"
        )
        raise typer.Exit(1)


@app.command()
def quickstart():
    """Quick start guide with enhanced interactive help."""
    if not ui_enhancer:
        print("CLI not properly initialized")
        raise typer.Exit(1)

    # Create enhanced quick start guide
    if ui_enhancer.config.output_mode == OutputMode.JSON:
        quickstart_data = {
            "title": "Conjecture Quick Start Guide",
            "steps": [
                {
                    "step": 1,
                    "title": "Configure a Provider",
                    "options": [
                        {
                            "name": "Ollama (Local)",
                            "description": "Install from https://ollama.ai/",
                            "commands": ["ollama serve", "OLLAMA_ENDPOINT=http://localhost:11434"]
                        },
                        {
                            "name": "OpenAI (Cloud)",
                            "description": "Get key from https://platform.openai.com/api-keys",
                            "commands": ["OPENAI_API_KEY=sk-your-key"]
                        }
                    ]
                },
                {
                    "step": 2,
                    "title": "Create Configuration File",
                    "commands": ["cp .env.example .env", "# Edit .env with your chosen provider"]
                },
                {
                    "step": 3,
                    "title": "Validate Configuration",
                    "commands": ["conjecture config"]
                },
                {
                    "step": 4,
                    "title": "Create Your First Claim",
                    "commands": ['conjecture create "The sky is blue" --confidence 0.9']
                },
                {
                    "step": 5,
                    "title": "Search Claims",
                    "commands": ['conjecture search "sky"']
                }
            ]
        }
        ui_enhancer.formatter.print(quickstart_data)
    else:
        from rich.panel import Panel
        from rich.text import Text

        quickstart_content = """[bold blue]Step 1: Configure a Provider[/bold blue]

[bold green]LOCAL (Recommended for Privacy)[/bold green]
  • [cyan]Ollama:[/cyan] Install from https://ollama.ai/
    - Run: [yellow]ollama serve[/yellow]
    - Configure: [cyan]OLLAMA_ENDPOINT=http://localhost:11434[/cyan]

[bold cyan]CLOUD (Internet Required)[/bold cyan]
  • [cyan]OpenAI:[/cyan] Get key from https://platform.openai.com/api-keys
    - Configure: [cyan]OPENAI_API_KEY=sk-your-key[/cyan]

[bold]Step 2: Create Configuration File[/bold]
  [yellow]cp .env.example .env[/yellow]
  [yellow]# Edit .env with your chosen provider[/yellow]

[bold]Step 3: Validate Configuration[/bold]
  [yellow]conjecture config[/yellow]

[bold]Step 4: Create Your First Claim[/bold]
  [yellow]conjecture create "The sky is blue" --confidence 0.9[/yellow]

[bold]Step 5: Search Claims[/bold]
  [yellow]conjecture search "sky"[/yellow]

[bold]Need More Help?[/bold]
  • Setup help: [cyan]conjecture setup[/cyan]
  • Provider options: [cyan]conjecture providers[/cyan]
  • Backend status: [cyan]conjecture backends[/cyan]
  • System health: [cyan]conjecture health[/cyan]"""

        panel = Panel(
            quickstart_content,
            title="[bold blue]Conjecture Quick Start Guide[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        ui_enhancer.formatter.print(panel)


# Cleanup function
def cleanup():
    """Clean up UI resources"""
    global ui_enhancer
    if ui_enhancer:
        ui_enhancer.cleanup()


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        if ui_enhancer:
            ui_enhancer.formatter.print_warning("Operation cancelled by user")
        else:
            print("Operation cancelled")
        sys.exit(1)
    except Exception as e:
        if ui_enhancer:
            ui_enhancer.formatter.print_error(f"Unexpected error: {e}")
        else:
            print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        cleanup()