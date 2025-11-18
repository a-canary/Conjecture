#!/usr/bin/env python3
"""
Conjecture Data Layer CLI - Redirected to Modular CLI
This file has been consolidated into the new modular CLI system.
Please use 'conjecture --backend auto' or 'python -m src.cli.modular_cli' instead.
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
    
    console.print("\n[bold]Data Layer CLI has been consolidated into the new modular system.[/bold]")
    console.print("Please use one of these commands instead:")
    
    console.print("\n[bold green]📋 Recommended Commands:[/bold green]")
    console.print("  • [cyan]conjecture[/cyan] - Main command with auto-detection")
    console.print("  • [cyan]python -m src.cli.modular_cli[/cyan] - Direct module access")
    
    console.print("\n[bold blue]🔄 Data Layer Equivalents:[/bold blue]")
    console.print("  • [cyan]conjecture create[/cyan] (was: python src/cli.py create)")
    console.print("  • [cyan]conjecture get <id>[/cyan] (was: python src/cli.py get <id>)")
    console.print("  • [cyan]conjecture search <query>[/cyan] (was: python src/cli.py search <query>)")
    console.print("  • [cyan]conjecture stats[/cyan] (was: python src/cli.py stats)")
    
    console.print("\n[bold purple]🔧 Enhanced Features in Modular CLI:[/bold purple]")
    console.print("  • ✓ Multiple backend support (auto, local, cloud, hybrid)")
    console.print("  • ✓ Rich console interface")
    console.print("  • ✓ Better error handling")
    console.print("  • ✓ Auto-detection of optimal backend")
    console.print("  • ✓ Comprehensive health checks")
    
    console.print("\n[bold]📚 Migration Help:[/bold]")
    console.print("  • Run: [cyan]conjecture quickstart[/cyan] for getting started")
    console.print("  • Run: [cyan]conjecture backends[/cyan] to see available options")
    console.print("  • Run: [cyan]conjecture health[/cyan] to check system status")
    console.print("  • Run: [cyan]conjecture config[/cyan] to check configuration")
    
    console.print("\n[bold yellow]🎯 Direct Examples:[/bold yellow]")
    console.print("  • [cyan]conjecture create \"claim content\" --confidence 0.9[/cyan]")
    console.print("  • [cyan]conjecture search \"query term\"[/cyan]")
    console.print("  • [cyan]conjecture get c1234567[/cyan]")
    console.print("  • [cyan]conjecture analyze c1234567[/cyan]")
    
    # Attempt to redirect automatically
    console.print(f"\n[blue]🚀 Auto-redirecting to new CLI...[/blue]")
    
    try:
        # Import and run the new CLI
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_dir)
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