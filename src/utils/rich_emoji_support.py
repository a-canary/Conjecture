"""
Enhanced emoji support with Rich library option
Provides beautiful cross-platform emoji support
"""

def is_rich_available():
    """Check if Rich library is available"""
    try:
        import rich
        return True
    except ImportError:
        return False

def create_rich_console():
    """Create a Rich console with emoji support"""
    if not is_rich_available():
        return None
    
    try:
        from rich.console import Console
        from rich.theme import Theme
        
        # Create a custom theme for better emoji display
        theme = Theme({
            "emoji": "bold blue",
            "message": "default",
            "timestamp": "dim cyan"
        })
        
        console = Console(theme=theme, force_terminal=True)
        return console
    except:
        return None

def rich_print(message: str, emoji: str = "", style: str = "default"):
    """Print using Rich library if available"""
    console = create_rich_console()
    if console is None:
        # Fallback to regular print
        try:
            print(f"{emoji} {message}")
        except UnicodeEncodeError:
            print(f"[INFO] {message}")
        return
    
    try:
        from rich.text import Text
        text = Text()
        if emoji:
            text.append(emoji + " ", style="emoji")
        text.append(message, style=style)
        console.print(text)
    except:
        # Fallback if Rich fails
        console.print(f"[INFO] {message}")

# Rich-specific emoji mappings with better styling
RICH_EMOJI_STYLES = {
    "🎯": "bold green",
    "✅": "bold green", 
    "⏳": "bold yellow",
    "🔧": "bold blue",
    "🚩": "bold red",
    "📊": "bold cyan",
    "⏱️": "bold magenta",
    "🔍": "bold blue",
    "💬": "bold cyan",
    "📝": "bold white",
    "🔗": "bold blue",
    "❌": "bold red",
    "⚡": "bold yellow",
    "✨": "bold green",
    "🛠️": "bold blue",
    "🧪": "bold purple",
    "📋": "bold cyan",
    "⚙️": "bold gray"
}