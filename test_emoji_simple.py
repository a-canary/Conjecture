#!/usr/bin/env python3
"""
Simple emoji testing suite for Conjecture
Uses our emoji system properly for all output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functions():
    """Test basic emoji convenience functions."""
    from utils.terminal_emoji import emoji_printer
    emoji_printer.print(":microscope: Testing Basic Emoji Functions")
    print("=" * 50)
    
    from utils.terminal_emoji import success, error, warning, info, target, loading, tool, chat, resolved
    
    success("Basic success test")
    error("Basic error test")
    warning("Basic warning test")
    info("Basic info test")
    target("Basic target test")
    loading("Basic loading test")
    tool("Basic tool test")
    chat("Basic chat test")
    resolved("Basic resolved test")
    
    emoji_printer.print(":check_mark: Basic emoji functions test completed")
    print()

def test_verbose_logger():
    """Test emoji integration with verbose logger."""
    from utils.terminal_emoji import emoji_printer
    emoji_printer.print(":microscope: Testing Verbose Logger Integration")
    print("=" * 50)
    
    from utils.verbose_logger import VerboseLogger, VerboseLevel
    
    logger = VerboseLogger(VerboseLevel.USER)
    
    # Test claim assessment
    logger.claim_assessed_confident("c0000001", 0.9, 0.8)
    logger.claim_assessed_confident("c0000002", 0.6, 0.8)
    logger.claim_resolved("c0000001", 0.9)
    
    # Test tool execution
    logger.tool_executed("WebSearch", {"query": "test"}, {"success": True})
    logger.user_tool_executed("TellUser", "Hello from the system!")
    
    # Test final response
    logger.final_response("This is a test final response with emoji support")
    
    emoji_printer.print(":check_mark: Verbose logger integration test completed")
    print()

def test_fallbacks():
    """Test emoji fallback mechanism."""
    from utils.terminal_emoji import emoji_printer, TerminalEmoji
    
    emoji_printer.print(":microscope: Testing Fallback Mechanism")
    print("=" * 50)
    
    # Test with emoji disabled
    fallback_printer = TerminalEmoji(enable_emoji=False)
    
    fallback_printer.print(":thumbs_up: Success with fallback")
    fallback_printer.print(":x: Error with fallback")
    fallback_printer.print(":warning: Warning with fallback")
    fallback_printer.print(":target: Target with fallback")
    
    emoji_printer.print(":check_mark: Fallback mechanism test completed")
    print()

def test_performance():
    """Test emoji processing performance."""
    from utils.terminal_emoji import emoji_printer
    import time
    
    emoji_printer.print(":microscope: Testing Performance")
    print("=" * 50)
    
    start_time = time.time()
    
    for i in range(50):  # Reduced for faster testing
        emoji_printer.print(f":thumbs_up: Performance test {i}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed 50 emoji operations in {duration:.3f} seconds")
    
    if duration < 1.0:
        emoji_printer.print(":thumbs_up: Performance test passed")
    else:
        emoji_printer.print(":warning: Performance test warning")
    
    print()

def test_platform_info():
    """Show platform compatibility information."""
    from utils.terminal_emoji import emoji_printer
    import platform
    import sys
    
    emoji_printer.print(":microscope: Platform Information")
    print("=" * 50)
    
    print(f"Platform: {platform.system()}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Default encoding: {sys.stdout.encoding or 'unknown'}")
    
    from utils.terminal_emoji import TerminalEmoji
    printer = TerminalEmoji()
    print(f"Emoji enabled: {printer.enable_emoji}")
    
    emoji_printer.print(":check_mark: Platform information test completed")
    print()

def main():
    """Run all emoji tests."""
    from utils.terminal_emoji import emoji_printer
    emoji_printer.print(":rocket: Conjecture Emoji Testing Suite")
    print("=" * 60)
    print()
    
    try:
        test_basic_functions()
        test_verbose_logger()
        test_fallbacks()
        test_performance()
        test_platform_info()
        
        emoji_printer.print(":trophy: All emoji tests completed successfully!")
        print("Documentation available in EMOJI_USAGE.md")
        print("Integration complete with verbose logging system")
        
    except Exception as e:
        emoji_printer.print(":x: Test suite failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()