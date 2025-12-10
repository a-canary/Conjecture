"""
Unified emoji and symbol support for Conjecture
Combines emoji shortcodes, Rich styling, and cross-platform symbols
"""

import sys
import os
import platform
from typing import Dict, Optional

# Optional dependencies
try:
    import emoji

    HAS_EMOJI_PKG = True
except ImportError:
    HAS_EMOJI_PKG = False

try:
    from rich.console import Console
    from rich.theme import Theme
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class TerminalSupport:
    """Simple Unicode detection based on platform and terminal."""

    @staticmethod
    def supports_unicode() -> bool:
        """Detect if terminal supports Unicode characters."""
        if platform.system() == "Windows":
            # Modern Windows terminals that support Unicode
            return bool(
                os.environ.get("WT_SESSION")  # Windows Terminal
                or os.environ.get("TERM_PROGRAM") == "vscode"  # VS Code
                or os.environ.get("TERM_PROGRAM") == "Terminus-Sublime"
                or os.environ.get("TERM")
                in ["xterm-256color", "alacritty", "rxvt-unicode"]
            )
        return True  # Unix-like systems generally support Unicode

class UnifiedEmojiPrinter:
    """
    Unified emoji and symbol support combining the best features from:
    - emoji_support.py: Windows console handling + manual fallbacks
    - rich_emoji_support.py: Rich library integration with styling
    - terminal_emoji.py: Emoji package wrapper with shortcode support
    - symbols.py: Platform detection + symbol mapping
    """

    def __init__(self, enable_rich: bool = True, enable_emoji: bool = True):
        self.enable_rich = enable_rich and HAS_RICH
        self.enable_emoji = enable_emoji and HAS_EMOJI_PKG
        self.supports_unicode = TerminalSupport.supports_unicode()

        # Setup console encoding
        self._setup_console()

        # Initialize Rich console if available
        self.rich_console = None
        if self.enable_rich:
            self._setup_rich_console()

        # Symbol mappings (from symbols.py)
        self.UNICODE_SYMBOLS = {
            "success": "✔",
            "error": "✖",
            "warning": "⚠",
            "info": "ℹ",
            "target": "🎯",
            "loading": "⏳",
            "tool": "🔧",
            "chat": "💬",
            "resolved": "✅",
            "stats": "📊",
            "time": "⏱️",
            "search": "🔍",
            "flag": "🚩",
            "note": "📝",
            "link": "🔗",
            "process": "⚡",
            "complete": "✨",
            "setup": "🛠️",
            "test": "🧪",
            "result": "📋",
            "config": "⚙️",
        }

        # ASCII fallbacks (from symbols.py)
        self.ASCII_FALLBACKS = {
            "success": "✓",
            "error": "✗",
            "warning": "!",
            "info": "i",
            "target": "[TARGET]",
            "loading": "...",
            "tool": "[TOOL]",
            "chat": "[CHAT]",
            "resolved": "[OK]",
            "stats": "[STATS]",
            "time": "[TIME]",
            "search": "[SEARCH]",
            "flag": "[FLAG]",
            "note": "[NOTE]",
            "link": "[LINK]",
            "process": "[PROCESS]",
            "complete": "[DONE]",
            "setup": "[SETUP]",
            "test": "[TEST]",
            "result": "[RESULT]",
            "config": "[CONFIG]",
        }

        # Emoji fallback mappings (from emoji_support.py)
        self.EMOJI_FALLBACKS = {
            "🎯": "[TARGET]",
            "✅": "[OK]",
            "⏳": "[WAIT]",
            "🔧": "[TOOL]",
            "🚩": "[FLAG]",
            "📊": "[STATS]",
            "⏱️": "[TIME]",
            "🔍": "[SEARCH]",
            "💬": "[CHAT]",
            "📝": "[NOTE]",
            "🔗": "[LINK]",
            "❌": "[ERROR]",
            "⚡": "[PROCESS]",
            "✨": "[COMPLETE]",
            "🛠️": "[SETUP]",
            "🧪": "[TEST]",
            "📋": "[RESULT]",
            "⚙️": "[CONFIG]",
        }

        # Rich emoji styles (from rich_emoji_support.py)
        self.RICH_EMOJI_STYLES = {
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
            "⚙️": "bold gray",
        }

        # Emoji shortcode replacements (from terminal_emoji.py)
        self.SHORTCODE_REPLACEMENTS = {
            ":thumbs_up:": "[OK]",
            ":thumbs_down:": "[BAD]",
            ":warning:": "[WARN]",
            ":x:": "[ERROR]",
            ":check_mark:": "[OK]",
            ":heavy_check_mark:": "[OK]",
            ":information_source:": "[INFO]",
            ":gear:": "[CONFIG]",
            ":hammer_and_wrench:": "[TOOL]",
            ":speech_balloon:": "[CHAT]",
            ":target:": "[TARGET]",
            ":hourglass_flowing_sand:": "[LOADING]",
            ":bar_chart:": "[STATS]",
            ":stopwatch:": "[TIME]",
            ":magnifying_glass:": "[SEARCH]",
            ":triangular_flag_on_post:": "[FLAG]",
            ":memo:": "[NOTE]",
            ":link:": "[LINK]",
            ":zap:": "[PROCESS]",
            ":sparkles:": "[COMPLETE]",
            ":wrench:": "[SETUP]",
            ":microscope:": "[TEST]",
            ":clipboard:": "[RESULT]",
        }

    def _setup_console(self):
        """Configure console for emoji support on Windows."""
        if platform.system() == "Windows":
            try:
                # Set console code page to UTF-8 (Windows 10+)
                os.system("chcp 65001 > nul 2>&1")
                # Set Python encoding
                os.environ["PYTHONIOENCODING"] = "utf-8"
                # Configure stdout for UTF-8
                if hasattr(sys.stdout, "reconfigure"):
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except:
                pass  # Silently fail if commands don't work

    def _setup_rich_console(self):
        """Create a Rich console with emoji support."""
        try:
            theme = Theme(
                {"emoji": "bold blue", "message": "default", "timestamp": "dim cyan"}
            )
            self.rich_console = Console(theme=theme, force_terminal=True)
        except:
            self.enable_rich = False

    def get_symbol(self, key: str) -> str:
        """Get a symbol by key with automatic fallback."""
        if self.supports_unicode:
            return self.UNICODE_SYMBOLS.get(key, "?")
        else:
            return self.ASCII_FALLBACKS.get(key, "?")

    def emojize(self, text: str) -> str:
        """Convert emoji shortcodes to actual emojis."""
        if not self.enable_emoji:
            return self._remove_emoji_shortcodes(text)

        try:
            return emoji.emojize(text)
        except (UnicodeEncodeError, LookupError):
            return self._remove_emoji_shortcodes(text)

    def _remove_emoji_shortcodes(self, text: str) -> str:
        """Remove emoji shortcodes and replace with text alternatives."""
        result = text
        for shortcode, replacement in self.SHORTCODE_REPLACEMENTS.items():
            result = result.replace(shortcode, replacement)
        return result

    def print(self, text: str, use_rich: bool = False, style: str = "default"):
        """Unified printing with multiple backends."""
        if use_rich and self.rich_console:
            self._rich_print(text, style)
        else:
            self._simple_print(text)

    def _rich_print(self, text: str, style: str = "default"):
        """Print using Rich library with styling."""
        try:
            # Convert emoji shortcodes first
            emojized_text = self.emojize(text)

            # Create Rich Text with styling
            rich_text = Text()

            # Simple parsing - split on first emoji/symbol
            parts = emojized_text.split(" ", 1)
            if len(parts) == 2 and parts[0]:
                emoji_part = parts[0]
                message_part = parts[1]

                # Apply emoji style if available
                emoji_style = self.RICH_EMOJI_STYLES.get(emoji_part, "emoji")
                rich_text.append(emoji_part + " ", style=emoji_style)
                rich_text.append(message_part, style=style)
            else:
                rich_text.append(emojized_text, style=style)

            self.rich_console.print(rich_text)
        except:
            # Fallback to simple print
            self._simple_print(text)

    def _simple_print(self, text: str):
        """Simple print with fallback handling."""
        try:
            # Convert emoji shortcodes
            emojized_text = self.emojize(text)
            print(emojized_text)
        except UnicodeEncodeError:
            # Last resort: remove all emojis and use text fallbacks
            fallback_text = self._remove_emoji_shortcodes(text)
            for emoji_char, fallback in self.EMOJI_FALLBACKS.items():
                fallback_text = fallback_text.replace(emoji_char, fallback)
            print(fallback_text)

    def safe_print(self, message: str, emoji: str = "") -> bool:
        """Print with emoji support, returns True if emoji worked, False if fallback used."""
        try:
            print(f"{emoji} {message}")
            return True
        except UnicodeEncodeError:
            fallback = self.EMOJI_FALLBACKS.get(emoji, "[INFO]")
            print(f"{fallback} {message}")
            return False

# Global instance for easy use
emoji_printer = UnifiedEmojiPrinter()

# Backward compatibility functions
def safe_print(message: str, emoji: str = "") -> bool:
    """Backward compatibility: safe print with emoji fallback."""
    return emoji_printer.safe_print(message, emoji)

def get_emoji_or_fallback(preferred: str) -> str:
    """Backward compatibility: get emoji or fallback."""
    # Test if we can use emojis
    try:
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            preferred.encode(sys.stdout.encoding)
        else:
            preferred.encode("utf-8")
        return preferred
    except (UnicodeEncodeError, LookupError, AttributeError):
        return emoji_printer.EMOJI_FALLBACKS.get(preferred, preferred)

def rich_print(message: str, emoji: str = "", style: str = "default"):
    """Backward compatibility: Rich print."""
    if emoji:
        text = f"{emoji} {message}"
    else:
        text = message
    emoji_printer.print(text, use_rich=True, style=style)

# Convenience functions (from terminal_emoji.py)
def success(message: str = ""):
    """Print success message."""
    emoji_printer.print(f":thumbs_up: {message}" if message else ":thumbs_up:")

def error(message: str = ""):
    """Print error message."""
    emoji_printer.print(f":x: {message}" if message else ":x:")

def warning(message: str = ""):
    """Print warning message."""
    emoji_printer.print(f":warning: {message}" if message else ":warning:")

def info(message: str = ""):
    """Print info message."""
    emoji_printer.print(
        f":information_source: {message}" if message else ":information_source:"
    )

def target(message: str = ""):
    """Print target message."""
    emoji_printer.print(f":target: {message}" if message else ":target:")

def loading(message: str = ""):
    """Print loading message."""
    emoji_printer.print(
        f":hourglass_flowing_sand: {message}" if message else ":hourglass_flowing_sand:"
    )

def tool(message: str = ""):
    """Print tool message."""
    emoji_printer.print(
        f":hammer_and_wrench: {message}" if message else ":hammer_and_wrench:"
    )

def chat(message: str = ""):
    """Print chat message."""
    emoji_printer.print(
        f":speech_balloon: {message}" if message else ":speech_balloon:"
    )

def resolved(message: str = ""):
    """Print resolved message."""
    emoji_printer.print(f":check_mark: {message}" if message else ":check_mark:")

def stats(message: str = ""):
    """Print stats message."""
    emoji_printer.print(f":bar_chart: {message}" if message else ":bar_chart:")

# Test function
def test_emoji_support():
    """Test emoji support and show what's being used."""
    print("🧪 Conjecture Unified Emoji Support Test")
    print("=" * 50)
    print(f"Platform: {platform.system()}")
    print(f"Unicode support: {emoji_printer.supports_unicode}")
    print(f"Emoji package available: {HAS_EMOJI_PKG}")
    print(f"Rich library available: {HAS_RICH}")
    print(
        f"Using: {'Unicode symbols' if emoji_printer.supports_unicode else 'ASCII fallbacks'}"
    )
    print()

    # Test different features
    test_messages = [
        ("Symbol support", emoji_printer.get_symbol("success")),
        ("Emoji shortcode", ":thumbs_up: Success!"),
        ("Rich styling", "🎯 Target reached"),
        ("Fallback handling", "❌ Error occurred"),
    ]

    for desc, msg in test_messages:
        print(f"{desc}: {msg}")

    print()
    if emoji_printer.supports_unicode and HAS_EMOJI_PKG:
        print("✅ Full emoji support enabled!")
    elif emoji_printer.supports_unicode:
        print("⚠️  Basic emoji support (install 'emoji' package for shortcodes)")
    else:
        print("⚠️  Using ASCII fallbacks (upgrade your terminal for emojis)")

if __name__ == "__main__":
    test_emoji_support()
