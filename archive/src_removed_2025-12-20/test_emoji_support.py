"""
Comprehensive test suite for src/utils/emoji_support.py

Testing strategy:
- TerminalSupport Unicode detection
- UnifiedEmojiPrinter initialization and configuration
- Symbol retrieval with fallbacks
- Emoji shortcode conversion
- Print functionality with various backends
- Backward compatibility functions
- Convenience functions for different message types
- Platform-specific behavior

Target: 70%+ coverage of 147 statements
"""

import pytest
import sys
import os
import platform
from io import StringIO
from unittest.mock import patch, MagicMock

from src.utils.emoji_support import (
    TerminalSupport,
    UnifiedEmojiPrinter,
    emoji_printer,
    safe_print,
    get_emoji_or_fallback,
    rich_print,
    success,
    error,
    warning,
    info,
    target,
    loading,
    tool,
    chat,
    resolved,
    stats,
    test_emoji_support,
    HAS_EMOJI_PKG,
    HAS_RICH,
)


class TestTerminalSupport:
    """Test TerminalSupport class for Unicode detection"""

    def test_supports_unicode_windows_terminal(self):
        """Test Unicode support detection in Windows Terminal"""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"WT_SESSION": "123"}):
                assert TerminalSupport.supports_unicode() is True

    def test_supports_unicode_vscode(self):
        """Test Unicode support detection in VS Code"""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}):
                assert TerminalSupport.supports_unicode() is True

    def test_supports_unicode_terminus(self):
        """Test Unicode support detection in Terminus Sublime"""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"TERM_PROGRAM": "Terminus-Sublime"}):
                assert TerminalSupport.supports_unicode() is True

    def test_supports_unicode_xterm(self):
        """Test Unicode support detection in xterm-256color"""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"TERM": "xterm-256color"}):
                assert TerminalSupport.supports_unicode() is True

    def test_supports_unicode_windows_no_support(self):
        """Test Unicode detection when Windows has no supporting terminal"""
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {}, clear=True):
                assert TerminalSupport.supports_unicode() is False

    def test_supports_unicode_unix(self):
        """Test Unicode support on Unix-like systems"""
        with patch("platform.system", return_value="Linux"):
            assert TerminalSupport.supports_unicode() is True


class TestUnifiedEmojiPrinterInitialization:
    """Test UnifiedEmojiPrinter initialization and configuration"""

    def test_init_with_defaults(self):
        """Test initialization with default parameters"""
        printer = UnifiedEmojiPrinter()
        assert printer.enable_rich == HAS_RICH
        assert printer.enable_emoji == HAS_EMOJI_PKG
        assert isinstance(printer.supports_unicode, bool)

    def test_init_disable_rich(self):
        """Test initialization with Rich disabled"""
        printer = UnifiedEmojiPrinter(enable_rich=False)
        assert printer.enable_rich is False

    def test_init_disable_emoji(self):
        """Test initialization with emoji package disabled"""
        printer = UnifiedEmojiPrinter(enable_emoji=False)
        assert printer.enable_emoji is False

    def test_symbol_mappings_exist(self):
        """Test that symbol mappings are properly initialized"""
        printer = UnifiedEmojiPrinter()
        assert "success" in printer.UNICODE_SYMBOLS
        assert "error" in printer.UNICODE_SYMBOLS
        assert "warning" in printer.UNICODE_SYMBOLS
        assert "info" in printer.UNICODE_SYMBOLS
        assert len(printer.UNICODE_SYMBOLS) >= 20

    def test_ascii_fallbacks_exist(self):
        """Test that ASCII fallbacks are properly initialized"""
        printer = UnifiedEmojiPrinter()
        assert "success" in printer.ASCII_FALLBACKS
        assert "error" in printer.ASCII_FALLBACKS
        assert printer.ASCII_FALLBACKS["success"] == "✓"
        assert printer.ASCII_FALLBACKS["error"] == "✗"

    def test_emoji_fallbacks_exist(self):
        """Test that emoji fallbacks are properly initialized"""
        printer = UnifiedEmojiPrinter()
        assert "🎯" in printer.EMOJI_FALLBACKS
        assert "✅" in printer.EMOJI_FALLBACKS
        assert printer.EMOJI_FALLBACKS["🎯"] == "[TARGET]"

    def test_shortcode_replacements_exist(self):
        """Test that shortcode replacements are properly initialized"""
        printer = UnifiedEmojiPrinter()
        assert ":thumbs_up:" in printer.SHORTCODE_REPLACEMENTS
        assert ":thumbs_down:" in printer.SHORTCODE_REPLACEMENTS
        assert printer.SHORTCODE_REPLACEMENTS[":thumbs_up:"] == "[OK]"


class TestSymbolRetrieval:
    """Test symbol retrieval with automatic fallbacks"""

    def test_get_symbol_unicode_support(self):
        """Test getting symbol with Unicode support enabled"""
        printer = UnifiedEmojiPrinter()
        with patch.object(printer, "supports_unicode", True):
            symbol = printer.get_symbol("success")
            assert symbol == "✔"

    def test_get_symbol_ascii_fallback(self):
        """Test getting symbol with ASCII fallback"""
        printer = UnifiedEmojiPrinter()
        with patch.object(printer, "supports_unicode", False):
            symbol = printer.get_symbol("success")
            assert symbol == "✓"

    def test_get_symbol_unknown_key(self):
        """Test getting symbol with unknown key"""
        printer = UnifiedEmojiPrinter()
        symbol = printer.get_symbol("nonexistent_key")
        assert symbol == "?"

    def test_get_symbol_all_keys(self):
        """Test getting all defined symbols"""
        printer = UnifiedEmojiPrinter()
        for key in printer.UNICODE_SYMBOLS:
            symbol = printer.get_symbol(key)
            assert symbol is not None
            assert len(symbol) > 0


class TestEmojiConversion:
    """Test emoji shortcode conversion"""

    def test_emojize_with_emoji_package(self):
        """Test emoji conversion when emoji package available"""
        printer = UnifiedEmojiPrinter(enable_emoji=True)
        if HAS_EMOJI_PKG:
            # Test should work with emoji package
            result = printer.emojize(":thumbs_up:")
            assert result is not None
        else:
            # Should fall back to shortcode replacement
            result = printer.emojize(":thumbs_up: Great!")
            assert "[OK]" in result or ":thumbs_up:" in result

    def test_emojize_without_emoji_package(self):
        """Test emoji conversion without emoji package"""
        printer = UnifiedEmojiPrinter(enable_emoji=False)
        result = printer.emojize(":thumbs_up: Great!")
        assert "[OK]" in result
        assert ":thumbs_up:" not in result

    def test_remove_emoji_shortcodes(self):
        """Test shortcode removal and replacement"""
        printer = UnifiedEmojiPrinter()
        text = ":thumbs_up: Success :warning: Check this"
        result = printer._remove_emoji_shortcodes(text)
        assert "[OK]" in result
        assert "[WARN]" in result
        assert ":thumbs_up:" not in result
        assert ":warning:" not in result

    def test_remove_multiple_shortcodes(self):
        """Test removing multiple shortcodes from text"""
        printer = UnifiedEmojiPrinter()
        text = ":check_mark: Done :x: Failed :gear: Config"
        result = printer._remove_emoji_shortcodes(text)
        assert "[OK]" in result
        assert "[ERROR]" in result
        assert "[CONFIG]" in result


class TestPrintFunctionality:
    """Test various print methods"""

    def test_print_simple(self, capsys):
        """Test simple print without Rich"""
        printer = UnifiedEmojiPrinter()
        printer.print("Test message", use_rich=False)
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_print_with_rich_disabled(self, capsys):
        """Test print when Rich is disabled"""
        printer = UnifiedEmojiPrinter(enable_rich=False)
        printer.print("Test message", use_rich=True)
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_safe_print_success(self, capsys):
        """Test safe_print with successful emoji rendering"""
        printer = UnifiedEmojiPrinter()
        result = printer.safe_print("Success", "✅")
        captured = capsys.readouterr()
        # Should contain either emoji or fallback
        assert "Success" in captured.out
        assert isinstance(result, bool)

    def test_safe_print_unicode_error(self, capsys):
        """Test safe_print handling UnicodeEncodeError"""
        printer = UnifiedEmojiPrinter()
        # Create a proper UnicodeEncodeError
        try:
            "🎯".encode("ascii")
        except UnicodeEncodeError as e:
            with patch("builtins.print", side_effect=[e, None]):
                result = printer.safe_print("Message", "🎯")
                # Should return False when fallback is used
                assert result is False


class TestBackwardCompatibility:
    """Test backward compatibility functions"""

    def test_safe_print_global(self, capsys):
        """Test global safe_print function"""
        result = safe_print("Test message", "✅")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert isinstance(result, bool)

    def test_get_emoji_or_fallback_success(self):
        """Test getting emoji when encoding supports it"""
        result = get_emoji_or_fallback("✅")
        # Should return either emoji or fallback
        assert result is not None
        assert len(result) > 0

    def test_get_emoji_or_fallback_with_bad_encoding(self):
        """Test emoji fallback when encoding fails"""
        # Test with an emoji that has a fallback - test the actual fallback logic
        # by checking if encoding to ASCII would fail
        result = get_emoji_or_fallback("🎯")
        # Result should be either the emoji or a fallback
        assert result is not None
        assert len(result) > 0
        # Either it's the original emoji or a fallback like '[TARGET]'
        assert result == "🎯" or "[" in result


class TestConvenienceFunctions:
    """Test convenience functions for common message types"""

    def test_success_function(self, capsys):
        """Test success() function"""
        success("Operation completed")
        captured = capsys.readouterr()
        assert "Operation completed" in captured.out

    def test_error_function(self, capsys):
        """Test error() function"""
        error("Something went wrong")
        captured = capsys.readouterr()
        assert "Something went wrong" in captured.out

    def test_warning_function(self, capsys):
        """Test warning() function"""
        warning("Be careful")
        captured = capsys.readouterr()
        assert "Be careful" in captured.out

    def test_info_function(self, capsys):
        """Test info() function"""
        info("Information here")
        captured = capsys.readouterr()
        assert "Information here" in captured.out

    def test_target_function(self, capsys):
        """Test target() function"""
        target("Goal reached")
        captured = capsys.readouterr()
        assert "Goal reached" in captured.out

    def test_loading_function(self, capsys):
        """Test loading() function"""
        loading("Please wait")
        captured = capsys.readouterr()
        assert "Please wait" in captured.out

    def test_tool_function(self, capsys):
        """Test tool() function"""
        tool("Using hammer")
        captured = capsys.readouterr()
        assert "Using hammer" in captured.out

    def test_chat_function(self, capsys):
        """Test chat() function"""
        chat("Hello there")
        captured = capsys.readouterr()
        assert "Hello there" in captured.out

    def test_resolved_function(self, capsys):
        """Test resolved() function"""
        resolved("Issue fixed")
        captured = capsys.readouterr()
        assert "Issue fixed" in captured.out

    def test_stats_function(self, capsys):
        """Test stats() function"""
        stats("Metrics loaded")
        captured = capsys.readouterr()
        assert "Metrics loaded" in captured.out

    def test_convenience_functions_without_message(self, capsys):
        """Test convenience functions called without messages"""
        success()
        error()
        warning()
        info()
        captured = capsys.readouterr()
        # Should produce some output even without message
        assert len(captured.out) > 0


class TestRichIntegration:
    """Test Rich library integration (if available)"""

    def test_rich_emoji_styles_defined(self):
        """Test that Rich emoji styles are defined"""
        printer = UnifiedEmojiPrinter()
        assert "🎯" in printer.RICH_EMOJI_STYLES
        assert "✅" in printer.RICH_EMOJI_STYLES
        assert printer.RICH_EMOJI_STYLES["🎯"] == "bold green"

    @pytest.mark.skipif(not HAS_RICH, reason="Rich library not available")
    def test_rich_print_with_style(self, capsys):
        """Test rich_print with styling (requires Rich)"""
        rich_print("Styled message", "🎯", "bold green")
        captured = capsys.readouterr()
        assert "Styled message" in captured.out or len(captured.out) > 0


class TestPlatformSpecificBehavior:
    """Test platform-specific behavior"""

    def test_windows_console_setup(self):
        """Test Windows console setup doesn't crash"""
        with patch("platform.system", return_value="Windows"):
            printer = UnifiedEmojiPrinter()
            # Should not raise exception
            assert printer is not None

    def test_unix_console_setup(self):
        """Test Unix console setup"""
        with patch("platform.system", return_value="Linux"):
            printer = UnifiedEmojiPrinter()
            assert printer is not None
            assert printer.supports_unicode is True


class TestTestFunction:
    """Test the test_emoji_support() function"""

    def test_emoji_support_function_runs(self, capsys):
        """Test that test_emoji_support() runs without errors"""
        test_emoji_support()
        captured = capsys.readouterr()
        assert "Conjecture Unified Emoji Support Test" in captured.out
        assert "Platform:" in captured.out


class TestGlobalInstance:
    """Test the global emoji_printer instance"""

    def test_global_instance_exists(self):
        """Test that global emoji_printer instance exists"""
        assert emoji_printer is not None
        assert isinstance(emoji_printer, UnifiedEmojiPrinter)

    def test_global_instance_usable(self):
        """Test that global instance can be used"""
        symbol = emoji_printer.get_symbol("success")
        assert symbol is not None
        assert len(symbol) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_text_print(self, capsys):
        """Test printing empty text"""
        printer = UnifiedEmojiPrinter()
        printer.print("")
        captured = capsys.readouterr()
        # Should not crash
        assert True

    def test_none_symbol_key(self):
        """Test getting symbol with None key"""
        printer = UnifiedEmojiPrinter()
        # Should handle gracefully
        try:
            symbol = printer.get_symbol(None)
            assert symbol == "?"
        except (TypeError, AttributeError):
            # Also acceptable to raise exception
            pass

    def test_unicode_text_with_ascii_fallback(self, capsys):
        """Test Unicode text when only ASCII fallback available"""
        printer = UnifiedEmojiPrinter()
        with patch.object(printer, "supports_unicode", False):
            printer.print("Test 🎯 message")
            captured = capsys.readouterr()
            # Should produce some output
            assert len(captured.out) > 0

    def test_shortcode_without_replacement(self):
        """Test shortcode that doesn't have a replacement"""
        printer = UnifiedEmojiPrinter()
        text = ":nonexistent_shortcode: Test"
        result = printer._remove_emoji_shortcodes(text)
        # Original shortcode should remain if no replacement exists
        assert ":nonexistent_shortcode:" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
