#!/usr/bin/env python3
"""
Enhanced UI Components for Conjecture CLI
Focus on improved user experience, progress indicators, and accessibility
"""

import time
import threading
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

try:
    from rich.console import Console, Group
    from rich.progress import (
        Progress, TaskID, BarColumn, TextColumn,
        TimeRemainingColumn, MofNCompleteColumn,
        SpinnerColumn, DownloadColumn, TransferSpeedColumn
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.rule import Rule
    from rich.align import Align
    from rich.columns import Columns
    from rich.status import Status
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.errors import MarkupError
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Some UI features will be disabled.")

import sys
import os

class OutputMode(Enum):
    """Output formatting modes for different use cases"""
    NORMAL = "normal"        # Balanced output with helpful formatting
    QUIET = "quiet"          # Minimal output, just results
    VERBOSE = "verbose"      # Detailed output with all information
    JSON = "json"            # Machine-readable JSON output
    MARKDOWN = "markdown"    # Markdown formatted output
    ACCESSIBLE = "accessible" # Screen reader friendly output

class VerbosityLevel(Enum):
    """Verbosity levels for controlling output detail"""
    SILENT = 0      # No output
    ERROR = 1       # Only errors
    WARN = 2        # Errors and warnings
    INFO = 3        # Errors, warnings, and info
    DEBUG = 4       # All output including debug
    TRACE = 5       # Everything including traces

    def __str__(self):
        return self.name.lower()

@dataclass
class UIConfig:
    """Configuration for UI components"""
    output_mode: OutputMode = OutputMode.NORMAL
    verbosity: VerbosityLevel = VerbosityLevel.INFO
    show_progress: bool = True
    show_timings: bool = True
    color_enabled: bool = True
    animated: bool = True
    accessibility_mode: bool = False
    compact_mode: bool = False
    show_tips: bool = True
    auto_save: bool = True

class EnhancedProgressTracker:
    """
    Enhanced progress tracking with multiple metrics and better UX
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self._tasks = {}
        self._subtasks = {}
        self._start_times = {}
        self._progress = None
        self._live = None
        self._active = False

        if RICH_AVAILABLE and config.show_progress:
            self._setup_progress_ui()

    def _setup_progress_ui(self):
        """Setup the Rich progress UI with custom columns"""
        if self.config.compact_mode:
            # Compact progress bar for limited space
            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=20),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=Console(file=sys.stderr),
                transient=True,
            )
        else:
            # Full-featured progress bar
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(elapsed_when_finished=True),
                console=Console(file=sys.stderr),
                transient=True,
            )

    def start_task(self, description: str, total: Optional[int] = None, parent_task: Optional[str] = None) -> str:
        """Start a new task and return its ID"""
        task_id = f"task_{len(self._tasks)}_{int(time.time())}"

        self._tasks[task_id] = {
            "description": description,
            "total": total or 100,
            "completed": 0,
            "parent": parent_task,
            "status": "running"
        }

        if self._progress and not self._active:
            self._live = Live(self._progress, console=Console(file=sys.stderr), refresh_per_second=10)
            self._live.start()
            self._active = True

        if self._progress:
            try:
                self._tasks[task_id]["progress_id"] = self._progress.add_task(
                    description, total=total or 100
                )
            except Exception as e:
                print(f"Progress error: {e}", file=sys.stderr)

        self._start_times[task_id] = time.time()
        return task_id

    def update_task(self, task_id: str, advance: int = 1, description: Optional[str] = None, status: Optional[str] = None):
        """Update task progress"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task["completed"] = min(task["completed"] + advance, task["total"])

        if description:
            task["description"] = description
        if status:
            task["status"] = status

        if self._progress and task_id in task:
            try:
                self._progress.update(
                    task["progress_id"],
                    advance=advance,
                    description=task["description"] or task["status"]
                )
            except Exception:
                pass  # Silently handle progress errors

    def complete_task(self, task_id: str, final_message: Optional[str] = None):
        """Mark a task as complete"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task["completed"] = task["total"]
        task["status"] = "completed"

        elapsed = time.time() - self._start_times.get(task_id, 0)

        if final_message and self.config.output_mode != OutputMode.QUIET:
            if self.config.show_timings:
                final_message = f"✓ {final_message} (took {elapsed:.2f}s)"
            else:
                final_message = f"✓ {final_message}"
            print(final_message, file=sys.stderr)

        if self._progress and task_id in task:
            try:
                self._progress.update(task["progress_id"], completed=task["total"])
            except Exception:
                pass

    def fail_task(self, task_id: str, error_message: Optional[str] = None):
        """Mark a task as failed"""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task["status"] = "failed"

        if error_message and self.config.output_mode != OutputMode.QUIET:
            print(f"✗ {error_message}", file=sys.stderr)

        if self._progress and task_id in task:
            try:
                self._progress.update(task["progress_id"], description=f"Failed: {task['description']}")
            except Exception:
                pass

    def finish_all(self):
        """Finish all progress tracking"""
        if self._live:
            self._live.stop()
            self._active = False

class EnhancedOutputFormatter:
    """
    Enhanced output formatting with multiple modes and accessibility features
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self.console = Console(
            file=sys.stdout,
            color_system="auto" if config.color_enabled else None,
            legacy_windows=True,
            width=None
        )
        self.error_console = Console(
            file=sys.stderr,
            color_system="auto" if config.color_enabled else None,
            legacy_windows=True
        )

    def format_success(self, message: str, details: Optional[str] = None) -> str:
        """Format success message"""
        if self.config.output_mode == OutputMode.JSON:
            return json.dumps({"status": "success", "message": message, "details": details})
        elif self.config.output_mode == OutputMode.MARKDOWN:
            output = f"✅ **{message}**"
            if details:
                output += f"\n\n{details}"
            return output
        elif self.config.output_mode == OutputMode.ACCESSIBLE:
            output = f"SUCCESS: {message}"
            if details:
                output += f". Details: {details}"
            return output
        else:
            if RICH_AVAILABLE:
                rich_msg = Text(f"✅ {message}", style="bold green")
                if details:
                    rich_msg.append(f"\n{details}", style="green")
                return rich_msg
            return f"✅ {message}"

    def format_error(self, message: str, details: Optional[str] = None, suggestion: Optional[str] = None) -> str:
        """Format error message with actionable suggestions"""
        if self.config.output_mode == OutputMode.JSON:
            return json.dumps({
                "status": "error",
                "message": message,
                "details": details,
                "suggestion": suggestion
            })
        elif self.config.output_mode == OutputMode.MARKDOWN:
            output = f"❌ **Error: {message}**"
            if details:
                output += f"\n\n{details}"
            if suggestion:
                output += f"\n\n💡 **Suggestion:** {suggestion}"
            return output
        elif self.config.output_mode == OutputMode.ACCESSIBLE:
            output = f"ERROR: {message}"
            if details:
                output += f". Details: {details}"
            if suggestion:
                output += f". Suggestion: {suggestion}"
            return output
        else:
            output_parts = [f"❌ {message}"]
            if details:
                output_parts.append(f"   {details}")
            if suggestion:
                output_parts.append(f"💡 Suggestion: {suggestion}")

            if RICH_AVAILABLE:
                return "\n".join(output_parts)
            return " ".join(output_parts)

    def format_warning(self, message: str, details: Optional[str] = None) -> str:
        """Format warning message"""
        if self.config.output_mode == OutputMode.JSON:
            return json.dumps({"status": "warning", "message": message, "details": details})
        elif self.config.output_mode == OutputMode.ACCESSIBLE:
            output = f"WARNING: {message}"
            if details:
                output += f". Details: {details}"
            return output
        else:
            if RICH_AVAILABLE:
                rich_msg = Text(f"⚠️ {message}", style="bold yellow")
                if details:
                    rich_msg.append(f"\n{details}", style="yellow")
                return rich_msg
            return f"⚠️ {message}"

    def format_info(self, message: str, details: Optional[str] = None) -> str:
        """Format info message"""
        if self.config.output_mode == OutputMode.QUIET:
            return ""
        elif self.config.output_mode == OutputMode.JSON:
            return json.dumps({"status": "info", "message": message, "details": details})
        elif self.config.output_mode == OutputMode.ACCESSIBLE:
            output = f"INFO: {message}"
            if details:
                output += f". Details: {details}"
            return output
        else:
            if RICH_AVAILABLE:
                rich_msg = Text(f"ℹ️ {message}", style="bold blue")
                if details:
                    rich_msg.append(f"\n{details}", style="blue")
                return rich_msg
            return f"ℹ️ {message}"

    def format_claim_table(self, claims: List[Dict[str, Any]], query: Optional[str] = None) -> Union[Table, str]:
        """Format claims as a table with enhanced readability"""
        if not claims:
            return self.format_info("No claims found" + (f" for query: {query}" if query else ""))

        if self.config.output_mode == OutputMode.JSON:
            return json.dumps({"claims": claims, "query": query, "count": len(claims)})
        elif self.config.output_mode == OutputMode.MARKDOWN:
            output = f"# Search Results{f' for: {query}' if query else ''}\n\n"
            output += "| ID | Content | Confidence | Tags | State |\n"
            output += "|---|---|---|---|---|\n"
            for claim in claims:
                content = claim.get("content", "")[:50] + "..." if len(claim.get("content", "")) > 50 else claim.get("content", "")
                tags = ", ".join(claim.get("tags", []))
                output += f"| {claim.get('id', 'N/A')} | {content} | {claim.get('confidence', 0):.2f} | {tags} | {claim.get('state', 'N/A')} |\n"
            return output
        else:
            if not RICH_AVAILABLE:
                # Fallback to plain text table
                output = f"\n{'='*80}\n"
                if query:
                    output += f"Search Results for: {query}\n"
                output += f"{'='*80}\n"
                for claim in claims:
                    output += f"\nID: {claim.get('id', 'N/A')}"
                    output += f"\nContent: {claim.get('content', 'N/A')}"
                    output += f"\nConfidence: {claim.get('confidence', 0):.2f}"
                    output += f"\nTags: {', '.join(claim.get('tags', []))}"
                    output += f"\nState: {claim.get('state', 'N/A')}"
                    output += f"\n{'-'*40}"
                return output

            table = Table(
                title=f"Search Results{f' for: {query}' if query else ''}",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("ID", style="cyan", no_wrap=True, width=10)
            table.add_column("Content", style="white", width=50)
            table.add_column("Confidence", style="green", width=12)
            table.add_column("Tags", style="blue", width=15)
            table.add_column("State", style="yellow", width=10)

            for claim in claims:
                content = claim.get("content", "")
                if len(content) > 47:
                    content = content[:47] + "..."

                tags = ", ".join(claim.get("tags", []))
                if len(tags) > 13:
                    tags = tags[:13] + "..."

                table.add_row(
                    claim.get("id", "N/A"),
                    content,
                    f"{claim.get('confidence', 0):.2f}",
                    tags,
                    claim.get("state", "N/A")
                )

            return table

    def format_claim_details(self, claim: Dict[str, Any]) -> Union[Panel, str]:
        """Format detailed claim information"""
        if self.config.output_mode == OutputMode.JSON:
            return json.dumps(claim, indent=2)
        elif self.config.output_mode == OutputMode.MARKDOWN:
            output = f"# Claim Details: {claim.get('id', 'N/A')}\n\n"
            output += f"**Content:** {claim.get('content', 'N/A')}\n\n"
            output += f"**Confidence:** {claim.get('confidence', 0):.2f}\n\n"
            output += f"**State:** {claim.get('state', 'N/A')}\n\n"
            output += f"**Tags:** {', '.join(claim.get('tags', []))}\n\n"
            if 'created_at' in claim:
                output += f"**Created:** {claim['created_at']}\n\n"
            if 'created_by' in claim:
                output += f"**User:** {claim['created_by']}\n\n"
            return output
        else:
            if not RICH_AVAILABLE:
                output = f"\n{'='*60}\n"
                output += f"Claim Details: {claim.get('id', 'N/A')}\n"
                output += f"{'='*60}\n"
                output += f"Content: {claim.get('content', 'N/A')}\n"
                output += f"Confidence: {claim.get('confidence', 0):.2f}\n"
                output += f"State: {claim.get('state', 'N/A')}\n"
                output += f"Tags: {', '.join(claim.get('tags', []))}\n"
                if 'created_at' in claim:
                    output += f"Created: {claim['created_at']}\n"
                if 'created_by' in claim:
                    output += f"User: {claim['created_by']}\n"
                return output

            content = f"""[bold cyan]ID:[/bold cyan] {claim.get('id', 'N/A')}
[bold]Content:[/bold] {claim.get('content', 'N/A')}
[bold green]Confidence:[/bold green] {claim.get('confidence', 0):.2f}
[bold yellow]State:[/bold yellow] {claim.get('state', 'N/A')}
[bold blue]Tags:[/bold blue] {', '.join(claim.get('tags', []))}"""

            if 'created_at' in claim:
                content += f"\n[bold]Created:[/bold] {claim['created_at']}"
            if 'created_by' in claim:
                content += f"\n[bold]User:[/bold] {claim['created_by']}"

            return Panel(content, title="Claim Details", border_style="blue")

    def print(self, content: Any, style: Optional[str] = None):
        """Print content using the appropriate formatter"""
        if self.config.output_mode == OutputMode.QUIET:
            return

        if RICH_AVAILABLE and isinstance(content, (Text, Table, Panel)):
            self.console.print(content)
        elif RICH_AVAILABLE and style:
            self.console.print(content, style=style)
        else:
            print(str(content))

    def print_error(self, content: Any, details: Optional[str] = None, suggestion: Optional[str] = None):
        """Print error message"""
        if details or suggestion:
            content = self.format_error(str(content), details, suggestion)

        if RICH_AVAILABLE and isinstance(content, (Text, Table, Panel)):
            self.error_console.print(content)
        elif RICH_AVAILABLE:
            self.error_console.print(content, style="red")
        else:
            print(str(content), file=sys.stderr)

    def print_success(self, content: Any):
        """Print success message"""
        self.print(content, style="green")

    def print_warning(self, content: Any):
        """Print warning message"""
        self.print(content, style="yellow")

    def print_info(self, content: Any):
        """Print info message"""
        self.print(content, style="blue")

class InteractivePrompter:
    """
    Enhanced interactive prompts with better UX and accessibility
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self.console = Console(
            color_system="auto" if config.color_enabled else None,
            legacy_windows=True
        )

    def ask_yes_no(self, question: str, default: bool = False) -> bool:
        """Ask a yes/no question with better defaults"""
        if self.config.output_mode == OutputMode.JSON:
            return default  # Can't prompt in JSON mode

        if not RICH_AVAILABLE:
            default_str = "Y/n" if default else "y/N"
            response = input(f"{question} ({default_str}): ").strip().lower()
            if not response:
                return default
            return response in ['y', 'yes']

        return Confirm.ask(question, default=default)

    def ask_text(self, question: str, default: str = "", password: bool = False) -> str:
        """Ask for text input"""
        if self.config.output_mode == OutputMode.JSON:
            return default

        if not RICH_AVAILABLE:
            if default:
                response = input(f"{question} [{default}]: ").strip()
                return response if response else default
            else:
                return input(f"{question}: ").strip()

        return Prompt.ask(question, default=default, password=password)

    def ask_choice(self, question: str, choices: List[str], default: Optional[str] = None) -> str:
        """Ask user to choose from options"""
        if self.config.output_mode == OutputMode.JSON:
            return default or choices[0]

        if not RICH_AVAILABLE:
            print(f"\n{question}")
            for i, choice in enumerate(choices):
                marker = "→" if choice == default else " "
                print(f"  {marker} {i+1}. {choice}")

            while True:
                try:
                    response = input(f"Choice (1-{len(choices)}) [{choices.index(default)+1 if default else ''}]: ").strip()
                    if not response and default:
                        return default
                    index = int(response) - 1
                    if 0 <= index < len(choices):
                        return choices[index]
                except (ValueError, IndexError):
                    pass

        return Prompt.ask(question, choices=choices, default=default)

    def ask_int(self, question: str, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
        """Ask for integer input with validation"""
        if self.config.output_mode == OutputMode.JSON:
            return default

        if not RICH_AVAILABLE:
            while True:
                try:
                    response = input(f"{question} [{default}]: ").strip()
                    value = int(response) if response else default
                    if (minimum is None or value >= minimum) and (maximum is None or value <= maximum):
                        return value
                    else:
                        print(f"Please enter an integer between {minimum} and {maximum}")
                except ValueError:
                    print("Please enter a valid integer")

        return IntPrompt.ask(question, default=default)

    def ask_float(self, question: str, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        """Ask for float input with validation"""
        if self.config.output_mode == OutputMode.JSON:
            return default

        if not RICH_AVAILABLE:
            while True:
                try:
                    response = input(f"{question} [{default}]: ").strip()
                    value = float(response) if response else default
                    if (minimum is None or value >= minimum) and (maximum is None or value <= maximum):
                        return value
                    else:
                        print(f"Please enter a number between {minimum} and {maximum}")
                except ValueError:
                    print("Please enter a valid number")

        return FloatPrompt.ask(question, default=default)

class ContextualHelp:
    """
    Contextual help system with tips and suggestions
    """

    def __init__(self, config: UIConfig):
        self.config = config
        self.tips = {
            "create": [
                "Tip: Use --analyze to automatically evaluate your claim with LLM",
                "Tip: Confidence should be between 0.0 (low) and 1.0 (high)",
                "Tip: Add relevant tags to improve search and organization"
            ],
            "search": [
                "Tip: Use quotes for exact phrase matching: \"quantum computing\"",
                "Tip: Try different keywords if you don't find what you're looking for",
                "Tip: Use --limit to control the number of results"
            ],
            "analyze": [
                "Tip: Analysis may take a few moments depending on claim complexity",
                "Tip: Results include sentiment analysis and topic extraction",
                "Tip: Analysis helps validate claims and identify related topics"
            ],
            "config": [
                "Tip: Run 'conjecture setup' for interactive configuration",
                "Tip: Configuration files are stored in ~/.conjecture/config.json",
                "Tip: Multiple providers can be configured for failover"
            ]
        }

    def get_random_tip(self, context: str) -> Optional[str]:
        """Get a random tip for the given context"""
        if not self.config.show_tips or self.config.output_mode in [OutputMode.QUIET, OutputMode.JSON]:
            return None

        tips = self.tips.get(context, [])
        if tips:
            import random
            return random.choice(tips)
        return None

    def show_help_for_command(self, command: str, formatter: EnhancedOutputFormatter):
        """Show contextual help for a specific command"""
        if command in self.tips:
            tip = self.get_random_tip(command)
            if tip:
                formatter.print_info(tip)

class UIEnhancer:
    """
    Main UI enhancement coordinator that brings all components together
    """

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        self.progress = EnhancedProgressTracker(self.config)
        self.formatter = EnhancedOutputFormatter(self.config)
        self.prompter = InteractivePrompter(self.config)
        self.help = ContextualHelp(self.config)

        # Track operation timing
        self.operation_start_time = None

    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation"""
        self.operation_start_time = time.time()
        if self.config.verbosity.value >= VerbosityLevel.INFO.value:
            task_id = self.progress.start_task(operation_name)
            return task_id
        return ""

    def complete_operation(self, task_id: str, result_message: Optional[str] = None):
        """Complete an operation and show timing if enabled"""
        if self.config.show_timings and self.operation_start_time:
            elapsed = time.time() - self.operation_start_time
            if result_message:
                result_message = f"{result_message} (completed in {elapsed:.2f}s)"
            else:
                result_message = f"Operation completed in {elapsed:.2f}s"

        if task_id:
            self.progress.complete_task(task_id, result_message)
        elif result_message and self.config.verbosity.value >= VerbosityLevel.INFO.value:
            self.formatter.print_success(result_message)

        self.progress.finish_all()

    def fail_operation(self, task_id: str, error_message: str, suggestion: Optional[str] = None):
        """Fail an operation with helpful error message"""
        if task_id:
            self.progress.fail_task(task_id, error_message)

        self.formatter.print_error(error_message, suggestion=suggestion)
        self.progress.finish_all()

    def update_progress(self, task_id: str, message: str, advance: int = 1):
        """Update progress for an operation"""
        if task_id:
            self.progress.update_task(task_id, advance, message)

    def show_command_help(self, command: str):
        """Show contextual help for a command"""
        self.help.show_help_for_command(command, self.formatter)

    def cleanup(self):
        """Clean up resources"""
        self.progress.finish_all()

# Convenience functions for quick access
def create_ui_enhancer(
    output_mode: str = "normal",
    verbosity: str = "info",
    show_progress: bool = True,
    **kwargs
) -> UIEnhancer:
    """Create a UI enhancer with common configuration"""
    # Map string values to enum values
    output_mode_map = {
        "normal": OutputMode.NORMAL,
        "quiet": OutputMode.QUIET,
        "verbose": OutputMode.VERBOSE,
        "json": OutputMode.JSON,
        "markdown": OutputMode.MARKDOWN,
        "accessible": OutputMode.ACCESSIBLE
    }

    verbosity_map = {
        "silent": VerbosityLevel.SILENT,
        "error": VerbosityLevel.ERROR,
        "warn": VerbosityLevel.WARN,
        "info": VerbosityLevel.INFO,
        "debug": VerbosityLevel.DEBUG,
        "trace": VerbosityLevel.TRACE
    }

    config = UIConfig(
        output_mode=output_mode_map.get(output_mode.lower(), OutputMode.NORMAL),
        verbosity=verbosity_map.get(verbosity.lower(), VerbosityLevel.INFO),
        show_progress=show_progress,
        **kwargs
    )
    return UIEnhancer(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced UI components
    import asyncio

    async def test_ui_enhancements():
        """Test the enhanced UI components"""
        print("Testing Enhanced UI Components")
        print("=" * 40)

        # Create different UI configurations
        configs = [
            UIConfig(output_mode=OutputMode.NORMAL, show_progress=True),
            UIConfig(output_mode=OutputMode.QUIET, show_progress=False),
            UIConfig(output_mode=OutputMode.JSON, show_progress=False),
            UIConfig(output_mode=OutputMode.ACCESSIBLE, show_progress=False)
        ]

        test_claims = [
            {
                "id": "c001",
                "content": "Quantum computing can solve certain problems exponentially faster than classical computers",
                "confidence": 0.85,
                "tags": ["quantum", "computing", "complexity"],
                "state": "validated"
            },
            {
                "id": "c002",
                "content": "Machine learning models can overfit training data",
                "confidence": 0.95,
                "tags": ["ml", "overfitting", "training"],
                "state": "evaluated"
            }
        ]

        for config in configs:
            print(f"\nTesting {config.output_mode.value} mode:")
            print("-" * 30)

            ui = UIEnhancer(config)

            # Test messages
            ui.formatter.print_success("Operation completed successfully")
            ui.formatter.print_warning("This is a warning message")
            ui.formatter.print_info("Here's some information")
            ui.formatter.print_error("Something went wrong", "Details about the error", "Try checking your configuration")

            # Test claim table
            table = ui.formatter.format_claim_table(test_claims, "quantum")
            ui.formatter.print(table)

            # Test operation tracking
            task_id = ui.start_operation("Processing claims")
            ui.update_progress(task_id, "Analyzing claims", 1)
            ui.update_progress(task_id, "Validating results", 1)
            ui.complete_operation(task_id, "All claims processed")

            ui.cleanup()

    asyncio.run(test_ui_enhancements())