#!/usr/bin/env python3
"""
Simple Conjecture CLI - Redirected to Modular CLI
This file has been consolidated into the new modular CLI system.
Please use 'conjecture' or 'python -m src.cli.modular_cli' instead.
"""

import sys
import os
from rich.console import Console

# Rich console for beautiful output
console = Console()

def main():
    """Main redirect function."""
    console.print("[bold yellow]⚠️ CLI Redirection Notice[/bold yellow]")
    console.print("=" * 50)
    
    console.print("\n[bold]This CLI has been consolidated into the new modular system.[/bold]")
    console.print("Please use one of these commands instead:")
    
    console.print("\n[bold green]📋 Recommended Commands:[/bold green]")
    console.print("  • [cyan]conjecture[/cyan] - Main command with auto-detection")
    console.print("  • [cyan]python conjecture[/cyan] - Alternative entry point")
    console.print("  • [cyan]python -m src.cli.modular_cli[/cyan] - Direct module access")
    
    console.print("\n[bold blue]🔄 Equivalent Commands:[/bold blue]")
    console.print("  • [cyan]conjecture create[/cyan] (was: python simple_conjecture_cli.py create)")
    console.print("  • [cyan]conjecture search[/cyan] (was: python simple_conjecture_cli.py search)")
    console.print("  • [cyan]conjecture config[/cyan] (was: python simple_conjecture_cli.py config)")
    console.print("  • [cyan]conjecture setup[/cyan] (was: python simple_conjecture_cli.py setup)")
    
    console.print("\n[bold purple]🔧 Backend Options:[/bold purple]")
    console.print("  • [cyan]conjecture --backend auto[/cyan] - Intelligent selection")
    console.print("  • [cyan]conjecture --backend local[/cyan] - Local services only")
    console.print("  • [cyan]conjecture --backend cloud[/cyan] - Cloud services only")
    console.print("  • [cyan]conjecture --backend hybrid[/cyan] - Combined services")
    
    console.print("\n[bold]📚 Migration Help:[/bold]")
    console.print("  • Run: [cyan]conjecture quickstart[/cyan] for getting started")
    console.print("  • Run: [cyan]conjecture backends[/cyan] to see available options")
    console.print("  • Run: [cyan]conjecture health[/cyan] to check system status")
    
    console.print("\n[bold yellow]🎯 Quick Start:[/bold yellow]")
    console.print("  1. [cyan]cp .env.example .env[/cyan] && edit with your provider")
    console.print("  2. [cyan]conjecture config[/cyan] to validate setup")
    console.print("  3. [cyan]conjecture create \"test claim\" --confidence 0.8[/cyan]")
    
    # Attempt to redirect automatically
    console.print(f"\n[blue]🚀 Auto-redirecting to new CLI...[/blue]")
    
    try:
        # Import and run the new CLI
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from cli.modular_cli import app
        app()
    except ImportError:
        console.print("\n[red]❌ Could not import new CLI. Please install dependencies.[/red]")
        console.print("Also try: [cyan]python -m src.cli.modular_cli[/cyan]")
    except SystemExit:
        # Allow normal exit from the new CLI
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        console.print("Please try running [cyan]conjecture[/cyan] directly.")

if __name__ == "__main__":
    main()