#!/usr/bin/env python3
"""
Enhanced Conjecture CLI - Redirected to Modular CLI
This file has been consolidated into the new modular CLI system.
Please use 'conjecture --backend cloud' or 'python -m src.cli.modular_cli' instead.
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
    
    console.print("\n[bold]Enhanced CLI has been consolidated into the new modular system.[/bold]")
    console.print("The enhanced features are now available through the cloud or hybrid backends.")
    console.print("Please use one of these commands instead:")
    
    console.print("\n[bold green]📋 Enhanced Features Commands:[/bold green]")
    console.print("  • [cyan]conjecture --backend cloud[/cyan] - Cloud services with advanced analysis")
    console.print("  • [cyan]conjecture --backend hybrid[/cyan] - Best of both worlds")
    console.print("  • [cyan]conjecture analyze <claim_id> --backend cloud[/cyan] - Advanced analysis")
    
    console.print("\n[bold blue]🔄 Enhanced CLI Equivalents:[/bold blue]")
    console.print("  • [cyan]conjecture create --analyze --backend cloud[/cyan] (was enhanced create)")
    console.print("  • [cyan]conjecture analyze <id> --backend cloud[/cyan] (was enhanced analyze)")
    console.print("  • [cyan]conjecture search --backend cloud[/cyan] (was enhanced search)")
    console.print("  • [cyan]conjecture stats --backend cloud[/cyan] (was enhanced stats)")
    
    console.print("\n[bold purple]🔧 Cloud Backend Advantages:[/bold purple]")
    console.print("  • ✓ Advanced LLM models (GPT-4, Claude, etc.)")
    console.print("  • ✓ Fact-checking capabilities")
    console.print("  • ✓ Web search integration")
    console.print("  • ✓ Enhanced semantic analysis")
    console.print("  • ✓ Rubric evaluation")
    
    console.print("\n[bold]📚 Migration Help:[/bold]")
    console.print("  • Run: [cyan]conjecture quickstart[/cyan] for getting started")
    console.print("  • Run: [cyan]conjecture backends[/cyan] to see available options")
    console.print("  • Run: [cyan]conjecture health[/cyan] to check system status")
    console.print("  • Run: [cyan]conjecture config[/cyan] to check configuration")
    
    console.print("\n[bold yellow]🎯 Enhanced Examples:[/bold yellow]")
    console.print("  • [cyan]conjecture create \"claim\" --confidence 0.9 --analyze --backend cloud[/cyan]")
    console.print("  • [cyan]conjecture analyze c1234567 --backend cloud[/cyan]")
    console.print("  • [cyan]conjecture search \"query\" --backend cloud --limit 20[/cyan]")
    
    # Attempt to redirect automatically with cloud backend
    console.print(f"\n[blue]🚀 Auto-redirecting to new CLI with cloud backend...[/blue]")
    
    try:
        # Import and run the new CLI with cloud backend
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_dir)
        from cli.modular_cli import app
        
        # Prepend --backend cloud to arguments
        sys.argv = [sys.argv[0]] + ['--backend', 'cloud'] + sys.argv[1:]
        app()
    except ImportError:
        console.print("\n[red]❌ Could not import new CLI. Please install dependencies.[/red]")
        console.print("Also try: [cyan]python -m src.cli.modular_cli --backend cloud[/cyan]")
    except SystemExit:
        # Allow normal exit from the new CLI
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        console.print("Please try running [cyan]conjecture --backend cloud[/cyan] directly.")

if __name__ == "__main__":
    main()