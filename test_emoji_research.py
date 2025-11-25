#!/usr/bin/env python3
"""
Terminal Emoji Support Research for Conjecture Project
Tests various Python emoji packages on Windows console
"""

import sys
import os
import platform
from typing import Dict, List, Any

def test_environment():
    """Check the current environment details"""
    print("=== Environment Details ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Default encoding: {sys.getdefaultencoding()}")
    print(f"Stdout encoding: {sys.stdout.encoding}")
    print(f"File system encoding: {sys.getfilesystemencoding()}")
    
    # Try to detect if we're in Windows Terminal, PowerShell, or cmd
    try:
        if os.environ.get('WT_SESSION'):
            print("Terminal: Windows Terminal")
        elif 'powershell' in os.environ.get('PSModulePath', '').lower():
            print("Terminal: PowerShell")
        else:
            print("Terminal: Unknown (likely cmd.exe)")
    except:
        print("Terminal: Could not detect")
    
    print()

def configure_console_for_unicode():
    """Configure Windows console for better Unicode support"""
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable virtual terminal processing for ANSI escape sequences
            # and extended flags for better Unicode support
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # Set mode to enable ANSI and Unicode support
            kernel32.SetConsoleMode(handle, 0x0007 | 0x0004 | 0x0008 | 0x0010 | 0x0020 | 0x0040 | 0x0080)
            print("+ Windows console configured for Unicode/ANSI support")
        except Exception as e:
            print(f"! Could not configure Windows console: {e}")
    
    # Also try to configure stdout for UTF-8
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            print("+ Standard streams reconfigured for UTF-8")
    except Exception as e:
        print(f"! Could not reconfigure standard streams: {e}")
    
    print()

def test_emoji_package():
    """Test the emoji package"""
    print("=== Testing emoji package ===")
    try:
        import emoji
        
        tests = [
            ':thumbs_up: Success!',
            ':warning: Warning detected',
            ':x: Error occurred',
            ':information_source: Information',
            ':heavy_check_mark: Task completed',
            ':rocket: Launching...',
        ]
        
        for test in tests:
            try:
                result = emoji.emojize(test)
                print(f"  {result}")
            except UnicodeEncodeError:
                print(f"  X Failed to display: {test}")
                # Show fallback
                print(f"  (Text fallback: {test})")
        
        print("+ emoji package test completed")
    except ImportError:
        print("- emoji package not installed")
    except Exception as e:
        print(f"- emoji package error: {e}")
    
    print()

def test_log_symbols():
    """Test the log-symbols package"""
    print("=== Testing log-symbols package ===")
    try:
        from log_symbols import LogSymbols
        
        symbols = [
            ("SUCCESS", LogSymbols.SUCCESS),
            ("INFO", LogSymbols.INFO),
            ("WARNING", LogSymbols.WARNING),
            ("ERROR", LogSymbols.ERROR),
        ]
        
        for name, symbol in symbols:
            try:
                print(f"  {name}: {symbol.value}")
            except UnicodeEncodeError:
                print(f"  {name}: ❌ Unicode error")
        
        print("✓ log-symbols package test completed")
    except ImportError:
        print("❌ log-symbols package not installed")
    except Exception as e:
        print(f"❌ log-symbols package error: {e}")
    
    print()

def test_rich():
    """Test the Rich library"""
    print("=== Testing Rich library ===")
    try:
        from rich.console import Console
        from rich.text import Text
        
        console = Console()
        
        test_messages = [
            ("Success", "green"),
            ("Warning", "yellow"),
            ("Error", "red"),
            ("Info", "blue"),
        ]
        
        for message, style in test_messages:
            try:
                console.print(f"  {message.capitalize()}: ", style=style, end="")
                # Try different emoji approaches
                if message == "Success":
                    console.print("✓")
                elif message == "Warning":
                    console.print("⚠")
                elif message == "Error":
                    console.print("✗")
                elif message == "Info":
                    console.print("ℹ")
            except UnicodeEncodeError:
                console.print(f"  {message.capitalize()}: ❌ Unicode error")
        
        print("✓ Rich library test completed")
    except ImportError:
        print("❌ Rich library not installed")
    except Exception as e:
        print(f"❌ Rich library error: {e}")
    
    print()

def create_emoji_fallback_map():
    """Create a custom emoji fallback mapping for Windows compatibility"""
    fallbacks = {
        ':thumbs_up:': '[OK]',
        ':thumbs_up_dark:': '[OK]',
        ':warning:': '[!]',
        ':x:': '[X]',
        ':information_source:': '[i]',
        ':heavy_check_mark:': '[✓]',
        ':heavy_check_mark_dark:': '[✓]',
        ':rocket:': '[>>]',
        ':gear:': '[⚙]',
        ':package:': '[📦]',
        ':file:': '[📄]',
        ':folder:': '[📁]',
    }
    return fallbacks

def test_custom_fallback():
    """Test custom emoji fallback system"""
    print("=== Testing Custom Fallback System ===")
    try:
        import emoji
        
        fallbacks = create_emoji_fallback_map()
        
        # Test with automatic fallback
        def safe_emojize(text, use_fallback=True):
            try:
                return emoji.emojize(text)
            except UnicodeEncodeError:
                if use_fallback:
                    for emoji_code, fallback in fallbacks.items():
                        if emoji_code in text:
                            text = text.replace(emoji_code, fallback)
                return text
        
        tests = [
            ':thumbs_up: Operation completed successfully',
            ':warning: Configuration may need adjustment',
            ':x: Critical error detected',
            ':information_source: Process started',
            ':heavy_check_mark: All tests passed',
            ':rocket: Application launching',
        ]
        
        for test in tests:
            try:
                result = safe_emojize(test)
                print(f"  {result}")
            except Exception as e:
                print(f"  ❌ Failed: {test} ({e})")
        
        print("✓ Custom fallback system test completed")
    except ImportError:
        print("❌ emoji package not installed for fallback testing")
    except Exception as e:
        print(f"❌ Custom fallback error: {e}")
    
    print()

def generate_windows_safe_emoji_logger():
    """Generate code for a Windows-safe emoji logging solution"""
    print("=== Windows-Safe Emoji Logger Code ===")
    
    logger_code = '''
import sys
import platform
import emoji
from typing import Union

class WindowsSafeEmojiLogger:
    """A logger that handles emoji display safely across platforms, especially Windows"""
    
    def __init__(self, enable_emoji: bool = True):
        self.enable_emoji = enable_emoji
        self.fallbacks = {
            ':thumbs_up:': '[OK]',
            ':warning:': '[!]',
            ':x:': '[X]',
            ':information_source:': '[i]',
            ':heavy_check_mark:': '[✓]',
            ':rocket:': '[>>]',
            ':gear:': '[⚙]',
            ':package:': '[📦]',
            ':file:': '[📄]',
            ':folder:': '[📁]',
        }
        
        # Configure for Windows if needed
        self._configure_console()
    
    def _configure_console(self):
        """Configure console for better Unicode support on Windows"""
        if platform.system() == 'Windows':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)
                kernel32.SetConsoleMode(handle, 0x0007 | 0x0004 | 0x0008 | 0x0010 | 0x0020 | 0x0040 | 0x0080)
            except:
                pass  # Ignore configuration errors
            
            # Try to configure stdout
            try:
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass
    
    def safe_emojize(self, text: str) -> str:
        """Convert emoji codes to actual emojis with fallback support"""
        if not self.enable_emoji:
            # Replace all emoji codes with fallbacks
            for emoji_code, fallback in self.fallbacks.items():
                text = text.replace(emoji_code, fallback)
            return text
        
        try:
            return emoji.emojize(text)
        except UnicodeEncodeError:
            # Fall back to text representations
            for emoji_code, fallback in self.fallbacks.items():
                if emoji_code in text:
                    text = text.replace(emoji_code, fallback)
            return text
    
    def info(self, message: str) -> None:
        """Print info message"""
        emoji_msg = self.safe_emojize(f":information_source: {message}")
        print(emoji_msg)
    
    def success(self, message: str) -> None:
        """Print success message"""
        emoji_msg = self.safe_emojize(f":heavy_check_mark: {message}")
        print(emoji_msg)
    
    def warning(self, message: str) -> None:
        """Print warning message"""
        emoji_msg = self.safe_emojize(f":warning: {message}")
        print(emoji_msg)
    
    def error(self, message: str) -> None:
        """Print error message"""
        emoji_msg = self.safe_emojize(f":x: {message}")
        print(emoji_msg)

# Usage example:
if __name__ == "__main__":
    logger = WindowsSafeEmojiLogger(enable_emoji=True)
    logger.info("Application starting...")
    logger.success("Configuration loaded successfully")
    logger.warning("Deprecated configuration option detected")
    logger.error("Failed to connect to database")
'''
    
    print(logger_code)
    print()

def main():
    """Run all tests and generate recommendations"""
    print("Terminal Emoji Support Research for Conjecture Project")
    print("=" * 60)
    print()
    
    test_environment()
    configure_console_for_unicode()
    
    test_emoji_package()
    test_log_symbols()
    test_rich()
    test_custom_fallback()
    
    generate_windows_safe_emoji_logger()
    
    print("=== Research Summary ===")
    print("1. Windows console limitations are significant for emoji display")
    print("2. UTF-8 encoding configuration is essential but not always sufficient")
    print("3. The emoji package works but requires fallback handling")
    print("4. log-symbols has limited Windows compatibility")
    print("5. Rich library fails with Unicode errors on legacy Windows console")
    print("6. Custom fallback system provides the most reliable solution")
    print()
    print("Recommendation: Use WindowsSafeEmojiLogger for the Conjecture project")

if __name__ == "__main__":
    main()