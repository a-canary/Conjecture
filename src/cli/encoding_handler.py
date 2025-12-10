"""
Unicode compatibility utilities for Windows console support.
Ensures proper UTF-8 encoding across different platforms and terminals.
"""
import sys
import os
import locale
import ctypes
from pathlib import Path
from typing import Optional

def ensure_utf8_encoding() -> bool:
    """
    Ensure UTF-8 encoding for console output on all platforms.
    
    Returns:
        bool: True if UTF-8 encoding was successfully set, False otherwise
    """
    try:
        # Check current encoding
        current_encoding = sys.stdout.encoding or 'ascii'
        
        if 'utf' not in current_encoding.lower():
            # Set environment variable for subprocess calls
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # For Windows cmd.exe compatibility
            if sys.platform == 'win32':
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    # Enable UTF-8 mode in Windows console
                    kernel32.SetConsoleOutputCP(65001)  # CP_UTF8
                    kernel32.SetConsoleCP(65001)  # CP_UTF8
                except (AttributeError, OSError):
                    # Fallback if ctypes calls fail
                    pass
        
        return True
    except Exception:
        return False

def get_safe_console() -> 'SafeConsole':
    """
    Get a console instance with Unicode error handling.
    
    Returns:
        SafeConsole: Console instance that handles Unicode gracefully
    """
    from rich.console import Console
    return SafeConsole()

class SafeConsole:
    """
    Console wrapper with fallback for Unicode/markup errors.
    Provides graceful degradation when Rich console features fail.
    """
    
    def __init__(self):
        try:
            from rich.console import Console
            from rich.markup import escape_markup
            from rich.errors import MarkupError
            
            self.console = Console()
            self.escape_markup = escape_markup
            self.MarkupError = MarkupError
            self._rich_available = True
            
            # Delegate all Rich console methods to maintain compatibility
            for attr in dir(self.console):
                if not attr.startswith('_') and callable(getattr(self.console, attr)):
                    setattr(self, attr, getattr(self.console, attr))
                    
        except ImportError:
            self._rich_available = False
            self.console = None
    
    def safe_print(self, *args, **kwargs) -> None:
        """
        Print with markup and Unicode error handling.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for console.print
        """
        if not self._rich_available:
            # Fallback to basic print
            print(*args)
            return
        
        try:
            self.console.print(*args, **kwargs)
        except (self.MarkupError, UnicodeError, UnicodeEncodeError):
            # Fallback to plain text
            if args:
                # Escape markup and handle Unicode
                safe_args = []
                for arg in args:
                    try:
                        safe_arg = self.escape_markup(str(arg))
                        safe_args.append(safe_arg)
                    except (UnicodeError, AttributeError):
                        safe_args.append(str(arg))
                
                self.console.print(' '.join(safe_args), **kwargs)
    
    def print(self, *args, **kwargs) -> None:
        """Delegate to safe_print for compatibility."""
        self.safe_print(*args, **kwargs)
    
    def rule(self, title: str = "", **kwargs) -> None:
        """Print a horizontal rule with optional title."""
        if not self._rich_available:
            print(f"--- {title} ---" if title else "---")
            return
        
        try:
            self.console.rule(title, **kwargs)
        except (self.MarkupError, UnicodeError, UnicodeEncodeError):
            print(f"--- {title} ---" if title else "---")
    
    def panel(self, content: str, title: str = "", **kwargs) -> None:
        """Print content in a panel."""
        if not self._rich_available:
            border = f"=== {title} ===" if title else "======="
            print(border)
            print(content)
            print(border)
            return
        
        try:
            self.console.panel(content, title, **kwargs)
        except (self.MarkupError, UnicodeError, UnicodeEncodeError):
            border = f"=== {title} ===" if title else "======="
            print(border)
            print(content)
            print(border)

def setup_unicode_environment() -> bool:
    """
    Setup the complete Unicode environment for the application.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    success = True
    
    # Ensure UTF-8 encoding
    if not ensure_utf8_encoding():
        success = False
    
    # Set locale if possible
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        # Fallback to default locale
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # Keep system default
    
    return success

def get_unicode_error_context(error: Exception) -> str:
    """
    Provide context-specific error messages for Unicode issues.
    
    Args:
        error: The exception that occurred
        
    Returns:
        str: Helpful error message with solutions
    """
    error_str = str(error).lower()
    
    if 'charmap' in error_str and 'codec' in error_str:
        return """
🔤 Unicode Encoding Error Detected!

PROBLEM: Windows console cannot display Unicode characters
SOLUTION: Run this command first:
    set PYTHONIOENCODING=utf-8

PERMANENT FIX: Add to Windows Environment Variables:
    PYTHONIOENCODING=utf-8

ALTERNATIVE: Use Windows Terminal instead of cmd.exe
"""
    
    if 'utf-8' in error_str and 'can\'t decode' in error_str:
        return """
🔤 UTF-8 Decoding Error!

PROBLEM: File contains Unicode characters that can't be decoded
SOLUTION: Ensure files are saved with UTF-8 encoding
         Use UTF-8 BOM for Windows compatibility if needed
"""
    
    return f"""
🔤 Unicode/Encoding Error: {error}

Try setting PYTHONIOENCODING=utf-8 environment variable.
"""

# Initialize Unicode environment when module is imported
setup_unicode_environment()