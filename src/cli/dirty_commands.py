"""
Dirty Flag CLI Commands
Command-line interface for dirty flag system management
"""

import asyncio
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from typing import List, Optional
import json
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.models import Claim, DirtyReason
from core.dirty_flag import DirtyFlagSystem
from processing.dirty_evaluator import DirtyEvaluator, DirtyEvaluationConfig
from processing.llm.llm_manager import LLMManager
from config.dirty_flag_config import DirtyFlagConfig, DirtyFlagConfigManager


# Rich console for beautiful output
console = Console()
error_console = Console(stderr=True)

# Create Typer app for dirty flag commands
dirty_app = typer.Typer(
    name="dirty",
    help="Dirty flag system management commands",
    no_args_is_help=True
)

# Global instances
_dirty_flag_system: Optional[DirtyFlagSystem] = None
_dirty_evaluator: Optional[DirtyEvaluator] = None
_config_manager: Optional[DirtyFlagConfigManager] = None


def get_dirty_flag_system() -> DirtyFlagSystem:
    """Get or create dirty flag system instance."""
    global _dirty_flag_system
    if _dirty_flag_system is None:
        config = get_config_manager().get_config()
        _dirty_flag_system = DirtyFlagSystem(
            confidence_threshold=config.confidence_threshold,
            cascade_depth=config.cascade_depth
        )
    return _dirty_flag_system


def get_dirty_evaluator() -> DirtyEvaluator:
    """Get or create dirty evaluator instance."""
    global _dirty_evaluator
    if _dirty_evaluator is None:
        config = get_config_manager().get_config()
        
        # Create evaluation config
        eval_config = DirtyEvaluationConfig(
            batch_size=config.batch_size,
            max_parallel_batches=config.max_parallel_batches,
            confidence_threshold=config.confidence_threshold,
            confidence_boost_factor=config.confidence_boost_factor,
            enable_two_pass=config.two_pass_evaluation,
            relationship_threshold=config.relationship_threshold,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries
        )
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        
        # Create evaluator
        dirty_flag_system = get_dirty_flag_system()
        _dirty_evaluator = DirtyEvaluator(llm_manager, dirty_flag_system, eval_config)
    
    return _dirty_evaluator


def get_config_manager() -> DirtyFlagConfigManager:
    """Get configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DirtyFlagConfigManager()
    return _config_manager


def mock_get_claims() -> List[Claim]:
    """Mock function to get claims (should be replaced with actual data access)."""
    # This is a placeholder - in real implementation, you'd fetch from your data store
    claims = []
    
    # Create some mock claims for demonstration
    claims.append(Claim(
        id="claim_1",
        content="The sky is blue",
        confidence=0.85,
        type=["concept"],
        tags=["science", "observation"]
    ))
    
    claims.append(Claim(
        id="claim_2", 
        content="Water boils at 100°C at sea level",
        confidence=0.95,
        type=["concept"],
        tags=["physics", "temperature"]
    ))
    
    # Mark one as dirty for demonstration
    if claims:
        claims[0].mark_dirty(DirtyReason.MANUAL_MARK, priority=15)
    
    return claims


@dirty_app.command()
def status():
    """Show dirty flag system status and statistics."""
    console.print("[bold blue]Dirty Flag System Status[/bold blue]")
    console.print("=" * 50)
    
    try:
        dirty_system = get_dirty_flag_system()
        config = get_config_manager().get_config()
        
        # Get all claims
        claims = mock_get_claims()
        
        # Get statistics
        stats = dirty_system.get_dirty_statistics(claims)
        
        # Display configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Confidence Threshold", f"{config.confidence_threshold}")
        config_table.add_row("Cascade Depth", str(config.cascade_depth))
        config_table.add_row("Batch Size", str(config.batch_size))
        config_table.add_row("Auto Evaluation", "Enabled" if config.auto_evaluation_enabled else "Disabled")
        config_table.add_row("Two-Pass Evaluation", "Enabled" if config.two_pass_evaluation else "Disabled")
        
        console.print(config_table)
        console.print()
        
        # Display statistics
        stats_table = Table(title="Dirty Claim Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Dirty Claims", str(stats["total_dirty"]))
        stats_table.add_row("Priority Dirty Claims", str(stats["priority_dirty"]))
        stats_table.add_row("High Priority", str(stats["priority_ranges"].get("high", 0)))
        stats_table.add_row("Medium Priority", str(stats["priority_ranges"].get("medium", 0)))
        stats_table.add_row("Low Priority", str(stats["priority_ranges"].get("low", 0)))
        
        console.print(stats_table)
        
        # Display reasons breakdown
        if stats["reasons"]:
            console.print("\n[bold]Dirty Flag Reasons:[/bold]")
            for reason, count in stats["reasons"].items():
                console.print(f"  • {reason}: {count}")
        
        # Display list of dirty claims
        dirty_claims = dirty_system.get_dirty_claims(claims, prioritize=True)
        if dirty_claims:
            console.print(f"\n[bold]Dirty Claims ({len(dirty_claims)}):[/bold]")
            
            claims_table = Table()
            claims_table.add_column("ID", style="cyan")
            claims_table.add_column("Content", style="white", max_width=50)
            claims_table.add_column("Confidence", style="yellow")
            claims_table.add_column("Priority", style="green")
            claims_table.add_column("Reason", style="blue")
            claims_table.add_column("Dirty Time", style="magenta")
            
            for claim in dirty_claims:
                content = claim.content[:47] + "..." if len(claim.content) > 50 else claim.content
                dirty_time = claim.dirty_timestamp.strftime("%H:%M:%S") if claim.dirty_timestamp else "Unknown"
                
                claims_table.add_row(
                    claim.id,
                    content,
                    f"{claim.confidence:.2f}",
                    str(claim.dirty_priority),
                    claim.dirty_reason.value if claim.dirty_reason else "Unknown",
                    dirty_time
                )
            
            console.print(claims_table)
        else:
            console.print("\n[green]No dirty claims found[/green]")
    
    except Exception as e:
        error_console.print(f"[red]Error getting dirty flag status: {e}[/red]")


@dirty_app.command()
def evaluate(
    priority_only: bool = typer.Option(True, "--priority-only/--all", help="Only evaluate priority claims"),
    max_claims: Optional[int] = typer.Option(None, "--max-claims", "-n", help="Maximum claims to evaluate"),
    force: bool = typer.Option(False, "--force", "-f", help="Force evaluation of all dirty claims")
):
    """Evaluate dirty claims using LLM."""
    console.print("[bold blue]Evaluating Dirty Claims[/bold blue]")
    console.print("=" * 50)
    
    try:
        claims = mock_get_claims()
        dirty_system = get_dirty_flag_system()
        
        # Display what will be evaluated
        if priority_only:
            dirty_claims = dirty_system.get_priority_dirty_claims(claims, max_count=max_claims)
            console.print(f"[yellow]Will evaluate {len(dirty_claims)} priority dirty claims[/yellow]")
        else:
            dirty_claims = dirty_system.get_dirty_claims(claims, max_count=max_claims)
            console.print(f"[yellow]Will evaluate {len(dirty_claims)} dirty claims[/yellow]")
        
        if not dirty_claims:
            console.print("[green]No dirty claims to evaluate[/green]")
            return
        
        if not force:
            if not Confirm.ask(f"Evaluate {len(dirty_claims)} dirty claims?"):
                console.print("[yellow]Evaluation cancelled[/yellow]")
                return
        
        # Run evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating dirty claims...", total=None)
            
            try:
                # Get evaluator
                evaluator = get_dirty_evaluator()
                
                # Run evaluation
                result = asyncio.run(evaluator.evaluate_dirty_claims(
                    claims=dirty_claims,
                    priority_only=priority_only,
                    max_claims=max_claims
                ))
                
                progress.update(task, description="Evaluation complete!")
                
                # Display results
                if result.success:
                    console.print(f"[green]✓ Evaluation completed successfully[/green]")
                    console.print(f"  Processed: {result.processed_claims} claims")
                    console.print(f"  Updated: {result.updated_claims} claims")
                    if result.execution_time:
                        console.print(f"  Time: {result.execution_time:.2f} seconds")
                else:
                    console.print(f"[red]✗ Evaluation failed[/red]")
                    if result.errors:
                        for error in result.errors:
                            console.print(f"  Error: {error}")
                
            except Exception as e:
                progress.update(task, description="Error occurred")
                error_console.print(f"[red]Error during evaluation: {e}[/red]")
    
    except Exception as e:
        error_console.print(f"[red]Error setting up evaluation: {e}[/red]")


@dirty_app.command()
def mark(
    claim_id: str = typer.Argument(..., help="ID of claim to mark dirty"),
    reason: str = typer.Option("manual_mark", "--reason", "-r", help="Reason for marking dirty"),
    priority: int = typer.Option(10, "--priority", "-p", help="Priority for evaluation (higher = more urgent)"),
    cascade: bool = typer.Option(True, "--cascade/--no-cascade", help="Cascade to related claims")
):
    """Mark a claim as dirty manually."""
    console.print(f"[bold blue]Marking Claim {claim_id} as Dirty[/bold blue]")
    console.print("=" * 50)
    
    try:
        # Get claims
        claims = mock_get_claims()
        
        # Find the claim
        target_claim = None
        for claim in claims:
            if claim.id == claim_id:
                target_claim = claim
                break
        
        if not target_claim:
            error_console.print(f"[red]Claim {claim_id} not found[/red]")
            raise typer.Exit(1)
        
        # Validate reason
        try:
            dirty_reason = DirtyReason(reason)
        except ValueError:
            error_console.print(f"[red]Invalid reason: {reason}[/red]")
            error_console.print(f"Valid reasons: {[r.value for r in DirtyReason]}")
            raise typer.Exit(1)
        
        # Get dirty flag system
        dirty_system = get_dirty_flag_system()
        
        # Mark as dirty
        dirty_system.mark_claim_dirty(target_claim, dirty_reason, priority, cascade)
        
        console.print(f"[green]✓ Claim {claim_id} marked as dirty[/green]")
        console.print(f"  Reason: {dirty_reason.value}")
        console.print(f"  Priority: {priority}")
        console.print(f"  Cascade: {'Enabled' if cascade else 'Disabled'}")
        
        # Show what was cascaded
        if cascade:
            dirty_claims = dirty_system.get_dirty_claims(claims)
            cascaded_count = len(dirty_claims) - 1  # Subtract the original claim
            if cascaded_count > 0:
                console.print(f"  Cascaded to {cascaded_count} related claims")
    
    except typer.Exit:
        raise
    except Exception as e:
        error_console.print(f"[red]Error marking claim dirty: {e}[/red]")


@dirty_app.command()
def clean(
    claim_id: Optional[str] = typer.Argument(None, help="ID of claim to clean (cleans all if not provided)"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Only clean claims with specific reason")
):
    """Clear dirty flags from claims."""
    console.print("[bold blue]Cleaning Dirty Flags[/bold blue]")
    console.print("=" * 50)
    
    try:
        dirty_system = get_dirty_flag_system()
        claims = mock_get_claims()
        
        # Validate reason if provided
        dirty_reason_filter = None
        if reason:
            try:
                dirty_reason_filter = DirtyReason(reason)
            except ValueError:
                error_console.print(f"[red]Invalid reason: {reason}[/red]")
                error_console.print(f"Valid reasons: {[r.value for r in DirtyReason]}")
                raise typer.Exit(1)
        
        if claim_id:
            # Clean specific claim
            target_claims = []
            for claim in claims:
                if claim.id == claim_id and claim.is_dirty:
                    target_claims.append(claim)
            
            if not target_claims:
                console.print(f"[yellow]Claim {claim_id} is not dirty[/yellow]")
                return
            
            # Clean the claim
            for claim in target_claims:
                claim.mark_clean()
            
            console.print(f"[green]✓ Claim {claim_id} cleaned[/green]")
            
        else:
            # Clean multiple claims
            claims_to_clean = []
            for claim in claims:
                if claim.is_dirty:
                    if dirty_reason_filter is None or claim.dirty_reason == dirty_reason_filter:
                        claims_to_clean.append(claim)
            
            if not claims_to_clean:
                console.print("[yellow]No dirty claims to clean[/yellow]")
                return
            
            # Show what will be cleaned
            reason_text = f" with reason {dirty_reason_filter.value}" if dirty_reason_filter else ""
            console.print(f"[yellow]Will clean {len(claims_to_clean)} claims{reason_text}[/yellow]")
            
            if not Confirm.ask("Continue?"):
                console.print("[yellow]Cleaning cancelled[/yellow]")
                return
            
            # Clean claims
            cleaned_count = dirty_system.clear_dirty_flags(claims_to_clean, dirty_reason_filter)
            console.print(f"[green]✓ Cleaned {cleaned_count} claims[/green]")
    
    except typer.Exit:
        raise
    except Exception as e:
        error_console.print(f"[red]Error cleaning dirty flags: {e}[/red]")


@dirty_app.command()
def config_show():
    """Show dirty flag configuration."""
    console.print("[bold blue]Dirty Flag Configuration[/bold blue]")
    console.print("=" * 50)
    
    try:
        config = get_config_manager().get_config()
        manager = get_config_manager()
        
        # Show full configuration
        config_table = Table()
        config_table.add_column("Setting", style="cyan", min_width=25)
        config_table.add_column("Value", style="green", min_width=20)
        config_table.add_column("Description", style="white", min_width=30)
        
        config_rows = [
            ("confidence_threshold", f"{config.confidence_threshold}", "Confidence threshold for priority evaluation"),
            ("cascade_depth", str(config.cascade_depth), "Maximum depth for cascading dirty flags"),
            ("batch_size", str(config.batch_size), "Number of claims per evaluation batch"),
            ("max_parallel_batches", str(config.max_parallel_batches), "Maximum parallel evaluation batches"),
            ("confidence_boost_factor", f"{config.confidence_boost_factor}", "Confidence boost for re-evaluated claims"),
            ("two_pass_evaluation", "Enabled" if config.two_pass_evaluation else "Disabled", "Enable two-pass evaluation system"),
            ("relationship_threshold", f"{config.relationship_threshold}", "Threshold for relationship detection"),
            ("timeout_seconds", str(config.timeout_seconds), "Evaluation timeout in seconds"),
            ("max_retries", str(config.max_retries), "Maximum evaluation retries"),
            ("auto_evaluation_enabled", "Enabled" if config.auto_evaluation_enabled else "Disabled", "Enable automatic evaluation"),
            ("evaluation_interval_minutes", str(config.evaluation_interval_minutes), "Auto-evaluation interval"),
            ("max_claims_per_evaluation", str(config.max_claims_per_evaluation), "Maximum claims per evaluation"),
            ("min_dirty_claims_batch", str(config.min_dirty_claims_batch), "Minimum dirty claims for batch processing"),
            ("similarity_threshold", f"{config.similarity_threshold}", "Similarity threshold for claim matching"),
            ("cache_invalidation_minutes", str(config.cache_invalidation_minutes), "Cache invalidation time in minutes"),
        ]
        
        for setting, value, description in config_rows:
            config_table.add_row(setting, value, description)
        
        console.print(config_table)
        
        # Show priority weights
        console.print("\n[bold]Priority Weights:[/bold]")
        weights_table = Table()
        weights_table.add_column("Reason", style="cyan")
        weights_table.add_column("Weight", style="green")
        
        for reason, weight in config.priority_weights.items():
            weights_table.add_row(reason, f"{weight}")
        
        console.print(weights_table)
        
        # Show configuration summary
        console.print("\n[bold]Configuration Summary:[/bold]")
        summary = manager.get_config_summary()
        for key, value in summary.items():
            console.print(f"  {key}: {value}")
    
    except Exception as e:
        error_console.print(f"[red]Error showing configuration: {e}[/red]")


@dirty_app.command()
def config_set(
    setting: str = typer.Argument(..., help="Configuration setting to update"),
    value: str = typer.Argument(..., help="New value for setting")
):
    """Update a dirty flag configuration setting."""
    console.print(f"[bold blue]Updating Configuration: {setting}[/bold blue]")
    console.print("=" * 50)
    
    try:
        manager = get_config_manager()
        
        # Parse value based on setting type
        parsed_value = value
        
        # Try to convert to appropriate type
        if setting in ["confidence_threshold", "confidence_boost_factor", "relationship_threshold", "similarity_threshold"]:
            parsed_value = float(value)
            if not 0.0 <= parsed_value <= 1.0:
                raise ValueError("Value must be between 0.0 and 1.0")
        elif setting in ["cascade_depth", "batch_size", "max_parallel_batches", "timeout_seconds", "max_retries", "evaluation_interval_minutes", "max_claims_per_evaluation", "min_dirty_claims_batch", "cache_invalidation_minutes"]:
            parsed_value = int(value)
            if parsed_value < 0:
                raise ValueError("Value must be non-negative")
        elif setting in ["two_pass_evaluation", "auto_evaluation_enabled"]:
            parsed_value = value.lower() in ["true", "1", "yes", "on"]
        
        # Update configuration
        updated_config = manager.update_config(**{setting: parsed_value})
        
        console.print(f"[green]✓ Updated {setting} = {parsed_value}[/green]")
        
        # Show validation result
        console.print("[green]Configuration updated successfully[/green]")
    
    except ValueError as e:
        error_console.print(f"[red]Invalid value for {setting}: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Error updating configuration: {e}[/red]")
        raise typer.Exit(1)


@dirty_app.command()
def batch(
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Confidence threshold for marking dirty"),
    force: bool = typer.Option(False, "--force", "-f", help="Force batch marking without confirmation")
):
    """Mark multiple claims dirty based on confidence threshold."""
    console.print("[bold blue]Batch Marking Dirty Claims[/bold blue]")
    console.print("=" * 50)
    
    try:
        dirty_system = get_dirty_flag_system()
        config = get_config_manager().get_config()
        claims = mock_get_claims()
        
        # Use provided threshold or config default
        threshold = threshold or config.confidence_threshold
        
        console.print(f"[yellow]Will mark claims dirty with confidence < {threshold}[/yellow]")
        
        # Show what will be marked
        low_confidence_claims = [c for c in claims if c.confidence < threshold and not c.is_dirty]
        
        if not low_confidence_claims:
            console.print(f"[green]No claims with confidence < {threshold}[/green]")
            return
        
        console.print(f"[yellow]Found {len(low_confidence_claims)} claims to mark dirty:[/yellow]")
        for claim in low_confidence_claims[:5]:  # Show first 5
            console.print(f"  • {claim.id}: {claim.content[:50]}... (conf={claim.confidence:.2f})")
        
        if len(low_confidence_claims) > 5:
            console.print(f"  ... and {len(low_confidence_claims) - 5} more")
        
        if not force:
            if not Confirm.ask(f"Mark {len(low_confidence_claims)} claims dirty?"):
                console.print("[yellow]Batch marking cancelled[/yellow]")
                return
        
        # Mark claims dirty
        marked_count = dirty_system.mark_claims_dirty_by_confidence_threshold(claims, threshold)
        
        console.print(f"[green]✓ Marked {marked_count} claims dirty[/green]")
    
    except Exception as e:
        error_console.print(f"[red]Error in batch marking: {e}[/red]")


if __name__ == "__main__":
    dirty_app()