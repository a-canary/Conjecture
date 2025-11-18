#!/usr/bin/env python3
"""
Full Conjecture CLI - Redirected to Modular CLI
This file has been consolidated into the new modular CLI system.
Please use 'conjecture --backend hybrid' or 'python -m src.cli.modular_cli' instead.
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
    
    console.print("\n[bold]Full CLI has been consolidated into the new modular system.[/bold]")
    console.print("All features are now available through the hybrid backend for optimal performance.")
    console.print("Please use one of these commands instead:")
    
    console.print("\n[bold green]📋 Full Feature Commands:[/bold green]")
    console.print("  • [cyan]conjecture --backend hybrid[/cyan] - All features with optimization")
    console.print("  • [cyan]conjecture --backend auto[/cyan] - Intelligent feature selection")
    console.print("  • [cyan]conjecture analyze <claim_id> --backend hybrid[/cyan] - Cross-platform analysis")
    
    console.print("\n[bold blue]🔄 Full CLI Equivalents:[/bold blue]")
    console.print("  • [cyan]conjecture --backend hybrid create[/cyan] (was full CLI create)")
    console.print("  • [cyan]conjecture --backend hybrid search[/cyan] (was full CLI search)")
    console.print("  • [cyan]conjecture --backend hybrid get[/cyan] (was full CLI get)")
    console.print("  • [cyan]conjecture --backend hybrid analyze[/cyan] (was full CLI analyze)")
    
    console.print("\n[bold purple]🔧 Hybrid Backend Advantages:[/bold purple]")
    console.print("  • ✓ Auto-fallback between local and cloud")
    console.print("  • ✓ Optimized performance for each operation")
    console.print("  • ✓ Offline capability when possible")
    console.print("  • ✓ Advanced cloud features when available")
    console.print("  • ✓ Cost optimization")
    console.print("  • ✓ Best of both worlds")
    
    console.print("\n[bold]📚 Migration Help:[/bold]")
    console.print("  • Run: [cyan]conjecture quickstart[/cyan] for getting started")
    console.print("  • Run: [cyan]conjecture backends[/cyan] to see available options")
    console.print("  • Run: [cyan]conjecture health[/cyan] to check system status")
    console.print("  • Run: [cyan]conjecture config[/cyan] to check configuration")
    
    console.print("\n[bold yellow]🎯 Full Feature Examples:[/bold yellow]")
    console.print("  • [cyan]conjecture --backend hybrid create \"claim\" --confidence 0.9 --analyze[/cyan]")
    console.print("  • [cyan]conjecture --backend hybrid search \"query\"[/cyan]")
    console.print("  • [cyan]conjecture --backend hybrid analyze c1234567[/cyan]")
    console.print("  • [cyan]conjecture --backend hybrid stats[/cyan]")
    
    # Attempt to redirect automatically with hybrid backend
    console.print(f"\n[blue]🚀 Auto-redirecting to new CLI with hybrid backend...[/blue]")
    
    try:
        # Import and run the new CLI with hybrid backend
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_dir)
        from cli.modular_cli import app
        
        # Prepend --backend hybrid to arguments
        sys.argv = [sys.argv[0]] + ['--backend', 'hybrid'] + sys.argv[1:]
        app()
    except ImportError:
        console.print("\n[red]❌ Could not import new CLI. Please install dependencies.[/red]")
        console.print("Also try: [cyan]python -m src.cli.modular_cli --backend hybrid[/cyan]")
    except SystemExit:
        # Allow normal exit from the new CLI
        pass
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        console.print("Please try running [cyan]conjecture --backend hybrid[/cyan] directly.")

if __name__ == "__main__":
    main()