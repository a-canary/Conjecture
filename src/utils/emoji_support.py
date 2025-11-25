"""
Cross-platform emoji support for Conjecture logging
Handles Windows console encoding issues gracefully
"""

import sys
import os
from typing import Dict

# Emoji fallback mappings for Windows console
EMOJI_FALLBACKS: Dict[str, str] = {
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
    "[CONFIDENT]": "🎯",
    "[EVALUATE]": "⏳", 
    "[RESOLVED]": "✅",
    "[RESPONSE]": "🎯",
    "[TOOL]": "🔧",
    "[CHAT]": "💬",
    "[TEST]": "🧪"
}

def setup_console_encoding():
    """Setup console for emoji support"""
    if sys.platform == "win32":
        try:
            # Set console code page to UTF-8 (Windows 10+)
            os.system("chcp 65001 > nul 2>&1")
            # Set Python encoding
            os.environ["PYTHONIOENCODING"] = "utf-8"
        except:
            pass  # Silently fail if commands don't work

def safe_print(message: str, emoji: str = "") -> bool:
    """
    Print with emoji support, returns True if emoji worked, False if fallback used
    """
    try:
        print(f"{emoji} {message}")
        return True
    except UnicodeEncodeError:
        # Use fallback
        fallback = EMOJI_FALLBACKS.get(emoji, "[INFO]")
        print(f"{fallback} {message}")
        return False

def get_emoji_or_fallback(preferred: str) -> str:
    """Get emoji if supported, otherwise return fallback"""
    # First check if it's already a fallback
    if preferred.startswith('[') and preferred.endswith(']'):
        return preferred

    # Test if we can use emojis
    try:
        # Try to encode a test emoji
        test_emoji = "🎯"
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            test_emoji.encode(sys.stdout.encoding)
        else:
            test_emoji.encode('utf-8')

        # If encoding works, check if we're in a Windows terminal that supports it
        if sys.platform == "win32":
            try:
                # Try to print to see if it actually works
                import io
                import contextlib

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    print(test_emoji, end='')
                output = f.getvalue()
                if output and len(output) > 0:
                    return preferred
            except:
                pass

        return preferred
    except (UnicodeEncodeError, LookupError, AttributeError):
        # Use fallback if encoding fails
        return EMOJI_FALLBACKS.get(preferred, preferred)

# Initialize console encoding when module is imported
setup_console_encoding()