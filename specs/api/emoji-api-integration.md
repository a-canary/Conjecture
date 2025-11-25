# Emoji API Integration Documentation

**Last Updated:** November 21, 2025
**Version:** 1.0
**Author:** Design Documentation Writer

## Overview

This document outlines how the emoji system integrates with existing Conjecture APIs and provides guidelines for developers who want to incorporate emoji support into their components.

## Integration Points

### 1. VerboseLogger Integration

The primary integration point is the `VerboseLogger` class, which uses emojis for different log levels and operations.

#### Emoji Mapping in VerboseLogger

| LogLevel | Operation | Emoji Shortcode | Fallback Text | Usage Context |
|----------|-----------|-----------------|---------------|---------------|
| USER | User communication | `:speech_balloon:` | `[CHAT]` | TellUser, AskUser |
| USER | Claim confident | `:target:` | `[TARGET]` | Confidence threshold met |
| USER | Claim needs evaluation | `:hourglass_flowing_sand:` | `[LOADING]` | Below confidence threshold |
| USER | Claim resolved | `:check_mark:` | `[OK]` | Final resolution |
| USER | Final response | `:target:` | `[TARGET]` | System response |
| TOOLS | Tool execution | `:hammer_and_wrench:` | `[TOOL]` | Any tool call |
| TOOLS | Claim validation success | `:check_mark:` | `[OK]` | Structural validation passed |
| TOOLS | Claim validation failure | `:x:` | `[ERROR]` | Structural validation failed |
| TOOLS | Claim created | `:memo:` | `[NOTE]` | New claim stored |
| TOOLS | Support relationship added | `:link:` | `[LINK]` | Support connections |
| DEBUG | Process start | `:magnifying_glass:` | `[SEARCH]` | Operation begins |
| DEBUG | Context built | `:gear:` | `[CONFIG]` | Context preparation |
| DEBUG | LLM processing | `:robot_face:` | `[PROCESS]` | AI model interaction |
| DEBUG | Evaluation start | `:zap:` | `[PROCESS]` | Claim evaluation |
| DEBUG | Evaluation complete | `:sparkles:` | `[COMPLETE]` | Evaluation finished |
| DEBUG | Tool registry | `:wrench:` | `[SETUP]` | Tool system ready |
| ERROR | Any error | `:x:` | `[ERROR]` | System errors |

#### Integration Implementation

```python
from utils.terminal_emoji import emoji_printer

class VerboseLogger:
    def _log(self, level: VerboseLevel, message: str, emoji_shortcode: str = ""):
        if self.level.value >= level.value:
            timestamp = datetime.now().strftime("%H:%M:%S") if level == VerboseLevel.DEBUG else ""
            prefix = f"[{timestamp}] " if timestamp else ""
            if emoji_shortcode:
                emoji_printer.print(f"{prefix}{emoji_shortcode} {message}")
            else:
                print(f"{prefix}{message}")

    def claim_assessed_confident(self, claim_id: str, confidence: float, threshold: float):
        """Log when claim meets confidence threshold"""
        if confidence >= threshold:
            self.stats['claims_assessed_confident'] += 1
            self._log(VerboseLevel.USER, 
                     f"Claim confident: {claim_id} (confidence: {confidence:.2f} >= {threshold:.2f})", 
                     ":target:")
        else:
            self.stats['claims_needing_evaluation'] += 1
            self._log(VerboseLevel.USER, 
                     f"Claim needs evaluation: {claim_id} (confidence: {confidence:.2f} < {threshold:.2f})", 
                     ":hourglass_flowing_sand:")
```

### 2. CLI Integration

The CLI interfaces automatically include emoji support through the verbose logger.

#### Output Formatting

```python
from utils.terminal_emoji import success, error, warning, info

# Direct integration in CLI commands
def stats_command(verbose: VerboseLevel = VerboseLevel.NONE):
    """Display system statistics with emoji indicators."""
    logger = VerboseLogger(verbose)
    
    success("Loading statistics...")
    stats = get_statistics()
    
    if stats['total_claims'] > 0:
        info(f"Found {stats['total_claims']} claims")
        target(f"Confident claims: {stats['confident_claims']}")
    else:
        warning("No claims found in database")
```

### 3. Conjecture Core Integration

The main Conjecture class integrates emojis through its verbose logging system.

#### Error Handling Integration

```python
class Conjecture:
    def process_request(self, request: str, verbose: VerboseLevel = VerboseLevel.NONE):
        """Process user request with emoji-enhanced logging."""
        logger = VerboseLogger(verbose)
        
        try:
            logger.process_start("Processing request")
            # ... processing logic ...
            logger.final_response(response)
            return response
        except Exception as e:
            logger.error("Request processing failed", e)
            raise
        finally:
            logger.finish()
```

## API Reference

### TerminalEmoji Class

#### Constructor

```python
TerminalEmoji(enable_emoji: bool = True)
```

**Parameters:**
- `enable_emoji` (bool): Enable emoji processing. If False, always uses fallback text.

**Returns:**
- `TerminalEmoji` instance

**Example:**
```python
# Create emoji-enabled printer
emoji_printer = TerminalEmoji(enable_emoji=True)

# Create text-only printer
text_printer = TerminalEmoji(enable_emoji=False)
```

#### Methods

##### emojize

```python
emojize(text: str) -> str
```

Convert emoji shortcodes to actual emojis or fallback text.

**Parameters:**
- `text` (str): Text containing emoji shortcodes

**Returns:**
- `str`: Processed text with emojis or fallback alternatives

**Example:**
```python
emoji_printer = TerminalEmoji()
result = emoji_printer.emojize(":thumbs_up: Success!")
# Returns: "👍 Success!" or "[OK] Success!"
```

##### print

```python
print(text: str) -> None
```

Print text with emoji support, handling encoding errors gracefully.

**Parameters:**
- `text` (str): Text to print with emoji shortcodes

**Example:**
```python
emoji_printer.print(":warning: Configuration file not found")
# Prints: "⚠️ Configuration file not found" or "[WARN] Configuration file not found"
```

##### _setup_console (Internal)

```python
_setup_console() -> None
```

Configure console for emoji support, particularly on Windows. Called automatically in constructor.

##### _remove_emoji_shortcodes (Internal)

```python
_remove_emoji_shortcodes(text: str) -> str
```

Convert emoji shortcodes to their text fallback alternatives.

**Parameters:**
- `text` (str): Text with emoji shortcodes

**Returns:**
- `str`: Text with fallback alternatives

### Convenience Functions

#### success

```python
success(message: str = "") -> None
```

Print success message with thumbs up emoji.

**Parameters:**
- `message` (str): Optional success message

**Example:**
```python
success("Operation completed successfully")
# Output: "👍 Operation completed successfully" or "[OK] Operation completed successfully"
```

#### error

```python
error(message: str = "") -> None
```

Print error message with X emoji.

**Parameters:**
- `message` (str): Optional error message

**Example:**
```python
error("Failed to connect to database")
# Output: "❌ Failed to connect to database" or "[ERROR] Failed to connect to database"
```

#### warning

```python
warning(message: str = "") -> None
```

Print warning message with warning emoji.

**Parameters:**
- `message` (str): Optional warning message

**Example:**
```python
warning("Low memory detected")
# Output: "⚠️ Low memory detected" or "[WARN] Low memory detected"
```

#### info

```python
info(message: str = "") -> None
```

Print info message with info emoji.

**Parameters:**
- `message` (str): Optional info message

**Example:**
```python
info("Starting initialization process...")
# Output: "ℹ️ Starting initialization process..." or "[INFO] Starting initialization process..."
```

#### target

```python
target(message: str = "") -> None
```

Print target message with target emoji (for goals and achievements).

**Parameters:**
- `message` (str): Optional target message

**Example:**
```python
target("Confidence threshold reached")
# Output: "🎯 Confidence threshold reached" or "[TARGET] Confidence threshold reached"
```

#### loading

```python
loading(message: str = "") -> None
```

Print loading message with hourglass emoji (for operations in progress).

**Parameters:**
- `message` (str): Optional loading message

**Example:**
```python
loading("Evaluating claim...")
# Output: "⏳ Evaluating claim..." or "[LOADING] Evaluating claim..."
```

#### tool

```python
tool(message: str = "") -> None
```

Print tool message with wrench emoji (for tool execution).

**Parameters:**
- `message` (str): Optional tool message

**Example:**
```python
tool("WebSearch executed")
# Output: "🛠️ WebSearch executed" or "[TOOL] WebSearch executed"
```

#### chat

```python
chat(message: str = "") -> None
```

Print chat message with speech bubble emoji (for user communication).

**Parameters:**
- `message` (str): Optional chat message

**Example:**
```python
chat("User message received")
# Output: "💬 User message received" or "[CHAT] User message received"
```

#### resolved

```python
resolved(message: str = "") -> None
```

Print resolved message with checkmark emoji (for completed tasks).

**Parameters:**
- `message` (str): Optional resolved message

**Example:**
```python
resolved("Claim fully resolved")
# Output: "✅ Claim fully resolved" or "[OK] Claim fully resolved"
```

#### stats

```python
stats(message: str = "") -> None
```

Print stats message with bar chart emoji (for statistical information).

**Parameters:**
- `message` (str): Optional stats message

**Example:**
```python
stats("15 claims analyzed")
# Output: "📊 15 claims analyzed" or "[STATS] 15 claims analyzed"
```

## Integration Guidelines

### 1. Adding Emoji Support to New Components

When adding new components to Conjecture, follow these integration patterns:

#### Step 1: Import Emoji System

```python
from utils.terminal_emoji import TerminalEmoji, success, error, warning, info
```

#### Step 2: Use Convenience Functions

```python
def process_data(data):
    try:
        info("Starting data processing...")
        
        # Processing logic here
        if data:
            success(f"Processed {len(data)} items")
        else:
            warning("No data to process")
            
    except Exception as e:
        error(f"Data processing failed: {e}")
```

#### Step 3: Custom Emoji Integration (Advanced)

```python
from utils.terminal_emoji import emoji_printer

def custom_operation():
    # For custom emoji usage
    emoji_printer.print(":gear: Custom configuration applied")
    emoji_printer.print(":zap: Optimization completed")
```

### 2. Logging Integration Best Practices

#### Use Appropriate Log Levels

```python
from utils.verbose_logger import VerboseLogger, VerboseLevel

class CustomComponent:
    def __init__(self, verbose: VerboseLevel = VerboseLevel.NONE):
        self.logger = VerboseLogger(verbose)
    
    def operation(self):
        # User-level information (will show with emojis)
        self.logger.claim_assessed_confident("claim123", 0.9, 0.8)
        
        # Tool-level information (shows with emojis at TOOLS level)
        self.logger.tool_executed("CustomOperation", {"param": "value"})
        
        # Debug-level information (shows with timestamps and emojis)
        self.logger.process_start("Custom operation processing")
```

#### Maintain Emoji Semantics

| Operation | Recommended Emoji | Fallback | Context |
|-----------|-------------------|----------|---------|
| Success | `:thumbs_up:`, `:check_mark:` | `[OK]` | Completed operations |
| Error | `:x:` | `[ERROR]` | Failures and exceptions |
| Warning | `:warning:` | `[WARN]` | Non-critical issues |
| Information | `:information_source:` | `[INFO]` | General info |
| Target/Goal | `:target:` | `[TARGET]` | Objectives met |
| Processing | `:hourglass_flowing_sand:`, `:zap:` | `[LOADING]`, `[PROCESS]` | Operations in progress |
| Tool | `:hammer_and_wrench:` | `[TOOL]` | Tool execution |
| Results | `:bar_chart:` | `[STATS]` | Statistical data |

### 3. Error Handling Integration

#### Wrap Emoji Operations

```python
def safe_emoji_print(message: str):
    """Safely print with emoji support."""
    try:
        from utils.terminal_emoji import emoji_printer
        emoji_printer.print(message)
    except Exception:
        # Fallback to plain print if emoji system fails
        print(message.replace(":", "").replace("_", " "))
```

#### Handle Missing Dependencies Gracefully

```python
def enhanced_logging(message: str, level: str = "info"):
    """Enhanced logging with emoji support."""
    try:
        from utils.terminal_emoji import info, warning, error
        
        if level == "info":
            info(message)
        elif level == "warning":
            warning(message)
        elif level == "error":
            error(message)
    except ImportError:
        # Fallback to standard logging
        print(f"[{level.upper()}] {message}")
```

## Performance Considerations

### 1. Emoji Processing Overhead

The emoji system is designed for minimal performance impact:

- **Package Detection**: One-time import verification
- **Fast Fallbacks**: Text replacements are O(n) string operations
- **Minimal Memory**: Stateless design with single global instance

### 2. Optimization Guidelines

#### Use Global Instance

```python
# Good: Use the provided global instance
from utils.terminal_emoji import emoji_printer
emoji_printer.print(":thumbs_up: Operation complete")

# Avoid: Creating multiple instances
from utils.terminal_emoji import TerminalEmoji
# Multiple instances unnecessary
```

#### Batch Processing

```python
# For multiple emoji operations
def process_multiple_messages(messages):
    for msg in messages:
        if msg['status'] == 'success':
            success(msg['text'])
        elif msg['status'] == 'error':
            error(msg['text'])
        elif msg['status'] == 'warning':
            warning(msg['text'])
```

## Configuration and Customization

### 1. Disabling Emoji Support

Users can disable emoji support globally:

```python
from utils.terminal_emoji import TerminalEmoji

# Create text-only instance
text_printer = TerminalEmoji(enable_emoji=False)
text_printer.print(":thumbs_up: Text only")
# Output: "[OK] Text only"
```

### 2. Custom Fallback Mapping

For advanced customization, modify the fallback system:

```python
from utils.terminal_emoji import TerminalEmoji

class CustomEmoji(TerminalEmoji):
    def _remove_emoji_shortcodes(self, text: str) -> str:
        # Custom fallback logic
        custom_replacements = {
            ':thumbs_up:': '[SUCCESS]',
            ':x:': '[FAILURE]',
            # ... add custom mappings
        }
        
        result = text
        for shortcode, replacement in custom_replacements.items():
            result = result.replace(shortcode, replacement)
        
        return result
```

### 3. Environment-Based Configuration

Control emoji behavior through environment variables:

```python
import os

def create_emoji_printer():
    # Check for emoji preference
    enable_emoji = os.environ.get('CONJECTURE_DISABLE_EMOJI', '').lower() != 'true'
    return TerminalEmoji(enable_emoji=enable_emoji)
```

## Migration Guide for Existing Code

### 1. Update Print Statements

Convert existing print statements to use emojis:

```python
# Before
print("Success: Operation completed")
print("Error: Connection failed")
print("Warning: Configuration outdated")

# After
success("Operation completed")
error("Connection failed")
warning("Configuration outdated")
```

### 2. Update Logging

Enhance existing logging with emojis:

```python
# Before
print(f"Processing: {operation_name}")
print(f"Result: {result_status}")

# After
info(f"Processing: {operation_name}")
if result_status == 'success':
    resolved(f"Completed: {operation_name}")
else:
    error(f"Failed: {operation_name}")
```

### 3. Maintain Backward Compatibility

Ensure existing code continues to work:

```python
def enhanced_print(message: str, use_emoji: bool = True):
    """Print with optional emoji support."""
    if use_emoji:
        from utils.terminal_emoji import emoji_printer
        emoji_printer.print(message)
    else:
        print(message)
```

## Testing Integration

### 1. Unit Testing Emoji Components

```python
import pytest
from unittest.mock import patch
from utils.terminal_emoji import TerminalEmoji, success

class TestCustomComponent:
    def test_emoji_integration(self):
        # Test that custom component uses emojis correctly
        with patch('builtins.print') as mock_print:
            success("Test message")
            mock_print.assert_called_once()
            
            # Check that emoji or fallback was used
            call_arg = str(mock_print.call_args[0][0])
            assert "👍" in call_arg or "[OK]" in call_arg
```

### 2. Integration Testing

```python
def test_full_workflow_emoji(self):
    """Test that emojis appear throughout the workflow."""
    with patch('builtins.print') as mock_print:
        # Run complete workflow
        run_conjecture_workflow()
        
        # Check for emoji usage
        output = str(mock_print.call_args_list)
        has_emoji_indicators = any(
            emoji in output for emoji in ["🎯", "✅", "🛠️", "❌", "⚠️"]
        )
        has_fallback_indicators = any(
            fallback in output for fallback in ["[TARGET]", "[OK]", "[TOOL]", "[ERROR]", "[WARN]"]
        )
        
        assert has_emoji_indicators or has_fallback_indicators
```

## Troubleshooting

### Common Integration Issues

#### Issue: Emojis Not Displaying
```python
# Solution: Check platform compatibility
from utils.terminal_emoji import emoji_printer
emoji_printer.print("🧪 Testing emoji support")
```

#### Issue: Encoding Errors
```python
# Solution: Use safe printing
try:
    success("Test message")
except UnicodeEncodeError:
    print("[OK] Test message")  # Fallback
```

#### Issue: Performance Impact
```python
# Solution: Use global instance instead of creating new ones
# Good
emoji_printer.print(":thumbs_up: Message")

# Avoid
new_instance = TerminalEmoji()  # Don't create multiple instances
new_instance.print(":thumbs_up: Message")
```

## Conclusion

The emoji integration provides a seamless way to enhance user experience across the Conjecture system. By following the guidelines and best practices outlined in this document, developers can effectively integrate emoji support into their components while maintaining compatibility and performance.

The system's design ensures that emojis enhance rather than hinder functionality, providing visual context while gracefully falling back to text alternatives when needed.

---

## Related Documents

- [Emoji Implementation Design](../implementation/emoji-implementation-design.md) - Technical details
- [Emoji Usage Guide](../user-guides/emoji-usage-guide.md) - User documentation
- [Emoji Testing Plan](../testing/emoji-testing-plan.md) - Testing strategy
- [Troubleshooting Guide](../support/emoji-troubleshooting.md) - Common issues