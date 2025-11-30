#!/usr/bin/env python3
"""
Simple Local CLI - Redirected to Modular CLI
This file has been consolidated into the new modular CLI system.
Please use 'conjecture --backend local' or 'python -m src.cli.modular_cli' instead.
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
    
    console.print("\n[bold green]📋 Local Backend Commands:[/bold green]")
    console.print("  • [cyan]conjecture --backend local[/cyan] - Use local services")
    console.print("  • [cyan]conjecture create --backend local[/cyan] - Create with local")
    console.print("  • [cyan]conjecture search --backend local[/cyan] - Search with local")
    
    console.print("\n[bold blue]🔄 Equivalent Commands:[/bold blue]")
    console.print("  • [cyan]conjecture --backend local create[/cyan] (was: python simple_local_cli.py create)")
    console.print("  • [cyan]conjecture --backend local search[/cyan] (was: python simple_local_cli.py search)")
    console.print("  • [cyan]conjecture --backend local stats[/cyan] (was: python simple_local_cli.py stats)")
    
    console.print("\n[bold purple]🔧 Local Backend Benefits:[/bold purple]")
    console.print("  • ✓ Offline capability")
    console.print("  • ✓ Complete data privacy")
    console.print("  • ✓ Fast local processing")
    console.print("  • ✓ No API costs")
    
    console.print("\n[bold]📚 Migration Help:[/bold]")
    console.print("  • Run: [cyan]conjecture quickstart[/cyan] for getting started")
    console.print("  • Run: [cyan]conjecture backends[/cyan] to see available options")
    console.print("  • Run: [cyan]conjecture config[/cyan] to check configuration")
    
    console.print("\n[bold yellow]🎯 Local Setup Quick Start:[/bold yellow]")
    console.print("  1. Install: [cyan]https://ollama.ai/[/cyan] or [cyan]https://lmstudio.ai/[/cyan]")
    console.print("  2. Configure: [cyan]OLLAMA_ENDPOINT=http://localhost:11434[/cyan] in .env")
    console.print("  3. Run: [cyan]conjecture --backend local create \"test\" --confidence 0.8[/cyan]")
    
    # Attempt to redirect automatically with local backend
    console.print(f"\n[blue]🚀 Auto-redirecting to new CLI with local backend...[/blue]")
    
    try:
        # Import and run the new CLI with local backend
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from cli.modular_cli import app
        
        # Prepend --backend local to arguments
        sys.argv = [sys.argv[0]] + ['--backend', 'local'] + sys.argv[1:]
        app()
    except ImportError:
        console.print("\n[red]❌ Could not import new CLI. Please install dependencies.[/red]")
        console.print("Also try: [cyan]python -m src.cli.modular_cli --backend local[/cyan]")
    except SystemExit:
        # Allow normal exit from the new CLI
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        console.print("Please try running [cyan]conjecture --backend local[/cyan] directly.")

if __name__ == "__main__":
    main()