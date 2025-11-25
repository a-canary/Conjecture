"""
Simple emoji wrapper for Conjecture using the popular 'emoji' package
Handles Windows console encoding gracefully
"""

import sys
import platform
import os
from typing import Optional

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("Warning: 'emoji' package not installed. Run: pip install emoji")

class TerminalEmoji:
    """Cross-platform emoji support with automatic fallbacks."""
    
    def __init__(self, enable_emoji: bool = True):
        self.enable_emoji = enable_emoji and HAS_EMOJI
        self._setup_console()
    
    def _setup_console(self):
        """Configure console for emoji support on Windows."""
        if platform.system() == 'Windows' and self.enable_emoji:
            try:
                # Configure stdout for UTF-8
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                # Set environment variable
                os.environ['PYTHONIOENCODING'] = 'utf-8'
            except:
                # Silently fail if configuration doesn't work
                pass
    
    def emojize(self, text: str) -> str:
        """Convert emoji shortcodes to actual emojis."""
        if not self.enable_emoji or not HAS_EMOJI:
            return self._remove_emoji_shortcodes(text)
        
        try:
            return emoji.emojize(text)
        except (UnicodeEncodeError, LookupError):
            # Fallback to text-only version
            return self._remove_emoji_shortcodes(text)
    
    def _remove_emoji_shortcodes(self, text: str) -> str:
        """Remove emoji shortcodes and replace with text alternatives."""
        if not HAS_EMOJI:
            return text
        
        # Common emoji shortcodes and their text alternatives
        replacements = {
            ':thumbs_up:': '[OK]',
            ':thumbs_down:': '[BAD]',
            ':warning:': '[WARN]',
            ':x:': '[ERROR]',
            ':check_mark:': '[OK]',
            ':heavy_check_mark:': '[OK]',
            ':information_source:': '[INFO]',
            ':gear:': '[CONFIG]',
            ':hammer_and_wrench:': '[TOOL]',
            ':speech_balloon:': '[CHAT]',
            ':target:': '[TARGET]',
            ':hourglass_flowing_sand:': '[LOADING]',
            ':bar_chart:': '[STATS]',
            ':stopwatch:': '[TIME]',
            ':magnifying_glass:': '[SEARCH]',
            ':triangular_flag_on_post:': '[FLAG]',
            ':memo:': '[NOTE]',
            ':link:': '[LINK]',
            ':zap:': '[PROCESS]',
            ':sparkles:': '[COMPLETE]',
            ':wrench:': '[SETUP]',
            ':microscope:': '[TEST]',
            ':clipboard:': '[RESULT]',
            ':gear:': '[CONFIG]',
        }
        
        result = text
        for shortcode, replacement in replacements.items():
            result = result.replace(shortcode, replacement)
        
        return result
    
    def print(self, text: str):
        """Print text with emoji support."""
        try:
            print(self.emojize(text))
        except UnicodeEncodeError:
            # Last resort: print without any emoji processing
            print(self._remove_emoji_shortcodes(text))

# Global instance for easy use
emoji_printer = TerminalEmoji()

# Convenience functions
def success(message: str = ""):
    """Print success message with thumbs up emoji."""
    emoji_printer.print(f":thumbs_up: {message}" if message else ":thumbs_up:")

def error(message: str = ""):
    """Print error message with X emoji."""
    emoji_printer.print(f":x: {message}" if message else ":x:")

def warning(message: str = ""):
    """Print warning message with warning emoji."""
    emoji_printer.print(f":warning: {message}" if message else ":warning:")

def info(message: str = ""):
    """Print info message with info emoji."""
    emoji_printer.print(f":information_source: {message}" if message else ":information_source:")

def target(message: str = ""):
    """Print target message with target emoji."""
    emoji_printer.print(f":target: {message}" if message else ":target:")

def loading(message: str = ""):
    """Print loading message with hourglass emoji."""
    emoji_printer.print(f":hourglass_flowing_sand: {message}" if message else ":hourglass_flowing_sand:")

def tool(message: str = ""):
    """Print tool message with wrench emoji."""
    emoji_printer.print(f":hammer_and_wrench: {message}" if message else ":hammer_and_wrench:")

def chat(message: str = ""):
    """Print chat message with speech bubble emoji."""
    emoji_printer.print(f":speech_balloon: {message}" if message else ":speech_balloon:")

def resolved(message: str = ""):
    """Print resolved message with checkmark emoji."""
    emoji_printer.print(f":check_mark: {message}" if message else ":check_mark:")

def stats(message: str = ""):
    """Print stats message with bar chart emoji."""
    emoji_printer.print(f":bar_chart: {message}" if message else ":bar_chart:")

def test_emoji_support():
    """Test emoji support and show what's being used."""
    print("🧪 Conjecture Emoji Package Test")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    print(f"Emoji package available: {HAS_EMOJI}")
    print(f"Emoji enabled: {emoji_printer.enable_emoji}")
    print()
    
    # Test common emojis
    test_messages = [
        ":thumbs_up: Success!",
        ":x: Error occurred",
        ":warning: Warning message",
        ":information_source: Information",
        ":target: Target reached",
        ":hourglass_flowing_sand: Loading...",
        ":hammer_and_wrench: Tool executed",
        ":check_mark: Task completed"
    ]
    
    for msg in test_messages:
        emoji_printer.print(msg)
    
    print()
    if emoji_printer.enable_emoji:
        print("✅ Full emoji support enabled!")
    else:
        print("⚠️  Using text fallbacks (install 'emoji' package for full support)")

if __name__ == "__main__":
    test_emoji_support()