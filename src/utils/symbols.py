"""
Simple, battle-tested emoji support for Conjecture
Based on Sindre Sorhus ecosystem approach used by 700+ CLI tools
"""

import platform
import os
import sys
from typing import Dict

class TerminalSupport:
    """Simple Unicode detection based on platform and terminal."""
    
    @staticmethod
    def supports_unicode() -> bool:
        """Detect if terminal supports Unicode characters."""
        if platform.system() == 'Windows':
            # Modern Windows terminals that support Unicode
            return bool(
                os.environ.get('WT_SESSION') or  # Windows Terminal
                os.environ.get('TERM_PROGRAM') == 'vscode' or  # VS Code
                os.environ.get('TERM_PROGRAM') == 'Terminus-Sublime' or
                os.environ.get('TERM') in ['xterm-256color', 'alacritty', 'rxvt-unicode']
            )
        return True  # Unix-like systems generally support Unicode

class Symbols:
    """Cross-platform terminal symbols with ASCII fallbacks."""
    
    # Unicode symbols (modern terminals)
    UNICODE: Dict[str, str] = {
        'success': '✔',
        'error': '✖', 
        'warning': '⚠',
        'info': 'ℹ',
        'target': '🎯',
        'loading': '⏳',
        'tool': '🔧',
        'chat': '💬',
        'resolved': '✅',
        'stats': '📊',
        'time': '⏱️',
        'search': '🔍',
        'flag': '🚩',
        'note': '📝',
        'link': '🔗',
        'process': '⚡',
        'complete': '✨',
        'setup': '🛠️',
        'test': '🧪',
        'result': '📋',
        'config': '⚙️',
    }
    
    # ASCII fallbacks (legacy terminals)
    FALLBACK: Dict[str, str] = {
        'success': '✓',
        'error': '✗',
        'warning': '!',
        'info': 'i',
        'target': '[TARGET]',
        'loading': '...',
        'tool': '[TOOL]',
        'chat': '[CHAT]',
        'resolved': '[OK]',
        'stats': '[STATS]',
        'time': '[TIME]',
        'search': '[SEARCH]',
        'flag': '[FLAG]',
        'note': '[NOTE]',
        'link': '[LINK]',
        'process': '[PROCESS]',
        'complete': '[DONE]',
        'setup': '[SETUP]',
        'test': '[TEST]',
        'result': '[RESULT]',
        'config': '[CONFIG]',
    }
    
    def __init__(self):
        self.supports_unicode = TerminalSupport.supports_unicode()
        self._symbols = self.UNICODE if self.supports_unicode else self.FALLBACK
    
    def get(self, key: str) -> str:
        """Get a symbol by key with automatic fallback."""
        return self._symbols.get(key, '?')
    
    def print(self, key: str, message: str = ""):
        """Print a symbol with optional message."""
        symbol = self.get(key)
        if message:
            print(f"{symbol} {message}")
        else:
            print(symbol)

# Global instance for easy access
symbols = Symbols()

# Convenience functions
def success(message: str = ""):
    """Print success symbol with message."""
    symbols.print('success', message)

def error(message: str = ""):
    """Print error symbol with message."""
    symbols.print('error', message)

def warning(message: str = ""):
    """Print warning symbol with message."""
    symbols.print('warning', message)

def info(message: str = ""):
    """Print info symbol with message."""
    symbols.print('info', message)

def target(message: str = ""):
    """Print target symbol with message."""
    symbols.print('target', message)

def loading(message: str = ""):
    """Print loading symbol with message."""
    symbols.print('loading', message)

# Test function
def test_emoji_support():
    """Test emoji support and show what's being used."""
    print("🧪 Conjecture Emoji Support Test")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    print(f"Unicode support: {symbols.supports_unicode}")
    print(f"Using: {'Unicode symbols' if symbols.supports_unicode else 'ASCII fallbacks'}")
    print()
    
    # Show sample symbols
    test_keys = ['success', 'error', 'warning', 'info', 'target', 'loading']
    for key in test_keys:
        symbol = symbols.get(key)
        print(f"{symbol} {key}")
    
    print()
    if symbols.supports_unicode:
        print("✅ Full emoji support detected!")
    else:
        print("⚠️  Using ASCII fallbacks (upgrade your terminal for emojis)")

if __name__ == "__main__":
    test_emoji_support()