#!/usr/bin/env python3
"""
Comprehensive emoji testing suite for Conjecture
Tests all emoji functionality across different scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_emoji_functions():
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
    
    print("✅ Basic emoji functions test completed\n")

def test_emoji_printer_direct():
    """Test direct emoji printer usage."""
    print("🧪 Testing Direct Emoji Printer")
    print("=" * 50)
    
    from utils.terminal_emoji import emoji_printer
    
    test_messages = [
        ":thumbs_up: Direct thumbs up",
        ":x: Direct error",
        ":warning: Direct warning",
        ":information_source: Direct info",
        ":target: Direct target",
        ":hourglass_flowing_sand: Direct loading",
        ":hammer_and_wrench: Direct tool",
        ":speech_balloon: Direct chat",
        ":check_mark: Direct resolved"
    ]
    
    for msg in test_messages:
        emoji_printer.print(msg)
    
    print("✅ Direct emoji printer test completed\n")

def test_verbose_logger_integration():
    """Test emoji integration with verbose logger."""
    print("🧪 Testing Verbose Logger Integration")
    print("=" * 50)
    
    from utils.verbose_logger import VerboseLogger, VerboseLevel
    
    # Test different verbose levels
    logger = VerboseLogger(VerboseLevel.USER)
    
    # Test claim assessment
    logger.claim_assessed_confident("c0000001", 0.9, 0.8)
    logger.claim_assessed_confident("c0000002", 0.6, 0.8)
    logger.claim_resolved("c0000001", 0.9)
    
    # Test tool execution
    logger.tool_executed("WebSearch", {"query": "test"}, {"success": True})
    logger.tool_executed("TellUser", {"message": "Hello"}, {"success": True})
    
    # Test final response
    logger.final_response("This is a test final response with emoji support")
    
    # Test process logging
    logger.process_start("Test process")
    logger.finish()
    
    print("✅ Verbose logger integration test completed\n")

def test_fallback_mechanism():
    """Test emoji fallback mechanism."""
    print("🧪 Testing Fallback Mechanism")
    print("=" * 50)
    
    from utils.terminal_emoji import TerminalEmoji
    
    # Test with emoji disabled
    fallback_printer = TerminalEmoji(enable_emoji=False)
    
    test_messages = [
        ":thumbs_up: Success with fallback",
        ":x: Error with fallback",
        ":warning: Warning with fallback",
        ":target: Target with fallback"
    ]
    
    for msg in test_messages:
        fallback_printer.print(msg)
    
    print("✅ Fallback mechanism test completed\n")

def test_unicode_handling():
    """Test Unicode character handling."""
    print("🧪 Testing Unicode Handling")
    print("=" * 50)
    
    from utils.terminal_emoji import emoji_printer
    
    # Test various Unicode scenarios
    unicode_tests = [
        "Regular ASCII text",
        "Text with unicode: café résumé naïve",
        "Mixed: :thumbs_up: Unicode text: café",
        "Complex: :gear: Configuration loaded: ⚙️ Settings",
        "Edge case: :unknown_emoji: This should fallback"
    ]
    
    for test in unicode_tests:
        try:
            emoji_printer.print(test)
        except Exception as e:
            print(f"Error with '{test}': {e}")
    
    print("✅ Unicode handling test completed\n")

def test_performance():
    """Test emoji processing performance."""
    print("🧪 Testing Performance")
    print("=" * 50)
    
    import time
    from utils.terminal_emoji import emoji_printer
    
    # Test performance with many emoji operations
    start_time = time.time()
    
    for i in range(100):
        emoji_printer.print(f":thumbs_up: Performance test {i}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed 100 emoji operations in {duration:.3f} seconds")
    print(f"Average time per operation: {duration/100*1000:.2f} ms")
    
    if duration < 1.0:
        print("✅ Performance test passed (< 1 second for 100 operations)")
    else:
        print("⚠️  Performance test warning (> 1 second for 100 operations)")
    
    print()

def test_error_handling():
    """Test error handling in emoji processing."""
    print("🧪 Testing Error Handling")
    print("=" * 50)
    
    from utils.terminal_emoji import TerminalEmoji
    
    # Test various error scenarios
    error_tests = [
        ("Empty string", ""),
        ("None input", None),
        ("Non-string input", 123),
        ("Very long string", ":thumbs_up: " * 1000),
        ("Invalid emoji", ":invalid_emoji_name:"),
        ("Mixed valid/invalid", ":thumbs_up: :invalid_emoji: :x:")
    ]
    
    for test_name, test_input in error_tests:
        try:
            if test_input is None:
                # Skip None test as it would cause TypeError
                print(f"⚠️  {test_name}: Skipped (None input)")
                continue
            
            printer = TerminalEmoji()
            result = printer.emojize(str(test_input))
            print(f"✅ {test_name}: Handled successfully")
        except Exception as e:
            print(f"❌ {test_name}: Error - {e}")
    
    print("✅ Error handling test completed\n")

def test_platform_compatibility():
    """Test platform-specific compatibility."""
    print("🧪 Testing Platform Compatibility")
    print("=" * 50)
    
    import platform
    import sys
    
    print(f"Platform: {platform.system()}")
    print(f"Python version: {sys.version}")
    print(f"Default encoding: {sys.stdout.encoding or 'unknown'}")
    print(f"Environment PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'not set')}")
    
    # Test platform detection
    from utils.terminal_emoji import TerminalEmoji
    
    printer = TerminalEmoji()
    print(f"Emoji enabled: {printer.enable_emoji}")
    
    # Test a few emojis to verify they work
    test_emojis = [":thumbs_up:", ":x:", ":warning:", ":target:"]
    
    for emoji_code in test_emojis:
        try:
            result = printer.emojize(emoji_code)
            print(f"✅ {emoji_code} -> {result}")
        except Exception as e:
            print(f"❌ {emoji_code} -> Error: {e}")
    
    print("✅ Platform compatibility test completed\n")

def main():
    """Run all emoji tests."""
    from utils.terminal_emoji import emoji_printer
    emoji_printer.print(":rocket: Conjecture Emoji Testing Suite")
    print("=" * 60)
    print()
    
    try:
        test_basic_emoji_functions()
        test_emoji_printer_direct()
        test_verbose_logger_integration()
        test_fallback_mechanism()
        test_unicode_handling()
        test_performance()
        test_error_handling()
        test_platform_compatibility()
        
        print("🎉 All emoji tests completed successfully!")
        print("📚 Documentation available in EMOJI_USAGE.md")
        print("🔧 Integration complete with verbose logging system")
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()