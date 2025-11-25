# Emoji Support in Conjecture

Conjecture uses the popular **`emoji`** package for cross-platform emoji support in terminal output.

## 🎯 Quick Start

```python
from utils.terminal_emoji import success, error, warning, info

# Simple emoji functions
success("Operation completed!")
error("Something went wrong")
warning("Configuration needs attention")
info("Processing started...")
```

## 📦 Installation

The emoji package is included in `requirements.txt`:

```bash
pip install emoji>=2.15.0
```

## 🚀 Usage Examples

### Basic Emoji Functions

```python
from utils.terminal_emoji import (
    success, error, warning, info,
    target, loading, tool, chat, resolved
)

# Common operations
success("Core tools setup complete!")
error("Failed to initialize configuration")
warning("Low confidence detected")
info("Starting analysis process...")

# Conjecture-specific
target("Claim meets confidence threshold")
loading("Evaluating claim...")
tool("WebSearch executed")
chat("User message received")
resolved("Claim fully resolved")
```

### Direct Emoji Shortcodes

```python
from utils.terminal_emoji import emoji_printer

# Use any emoji shortcode
emoji_printer.print(":thumbs_up: Success!")
emoji_printer.print(":warning: Warning message")
emoji_printer.print(":gear: Configuration loaded")
```

### Integration with Verbose Logger

```python
from utils.verbose_logger import VerboseLogger, VerboseLevel

logger = VerboseLogger(VerboseLevel.USER)

# Automatic emoji support
logger.claim_assessed_confident("c0000001", 0.9, 0.8)  # 🎯
logger.claim_resolved("c0000001", 0.9)                 # ✅
logger.tool_executed("WebSearch", {"query": "test"})   # 🛠️
```

## 🛡️ Cross-Platform Compatibility

### Windows Support
- **Automatic UTF-8 configuration** for emoji display
- **Graceful fallbacks** to text alternatives on older consoles
- **No manual setup required** - works out of the box

### Fallback Examples
```
🎯 Target reached    →  [TARGET] Target reached
⏳ Loading...        →  [LOADING] Loading...
🛠️ Tool executed     →  [TOOL] Tool executed
```

## 📋 Available Emoji Shortcodes

### Common Operations
- `:thumbs_up:` / `:check_mark:` - Success/Completion
- `:x:` - Error/Failure
- `:warning:` - Warning
- `:information_source:` - Information

### Conjecture-Specific
- `:target:` - Target/Goal achieved
- `:hourglass_flowing_sand:` - Loading/Processing
- `:hammer_and_wrench:` - Tool execution
- `:speech_balloon:` - User communication
- `:bar_chart:` - Statistics/Results

### Full List
Visit [emoji-cheat-sheet.com](https://www.webfx.com/tools/emoji-cheat-sheet/) for all available shortcodes.

## 🔧 Advanced Usage

### Custom Emoji Wrapper

```python
from utils.terminal_emoji import TerminalEmoji

# Create custom emoji printer
emoji = TerminalEmoji(enable_emoji=True)

# Process text with emojis
text = emoji.emojize(":thumbs_up: Custom message!")
print(text)
```

### Error Handling

```python
from utils.terminal_emoji import emoji_printer

try:
    emoji_printer.print(":thumbs_up: This will work!")
except UnicodeEncodeError:
    # Automatic fallback handling
    print("[OK] This will work!")
```

## 🎨 Best Practices

1. **Use descriptive shortcodes** - `:thumbs_up:` is clearer than `:+1:`
2. **Test on target platforms** - Verify emoji display works for your users
3. **Provide context** - Don't rely on emojis alone for important information
4. **Consider accessibility** - Some users may have difficulty distinguishing emojis

## 🐛 Troubleshooting

### Emojis Not Showing on Windows

```python
# Force UTF-8 encoding (usually automatic)
import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### Fallback Mode

If emojis aren't working, the system automatically falls back to text alternatives:

```
🎯 → [TARGET]
⏳ → [LOADING]  
🛠️ → [TOOL]
```

### Package Not Found

```bash
# Install the emoji package
pip install emoji>=2.15.0

# Or update requirements
pip install -r requirements.txt
```

## 📚 References

- [emoji package on PyPI](https://pypi.org/project/emoji/)
- [Emoji shortcode reference](https://www.webfx.com/tools/emoji-cheat-sheet/)
- [Unicode emoji documentation](https://unicode.org/emoji/)

---

**Note**: Conjecture's emoji system is designed to work everywhere. Modern terminals will display beautiful emojis, while legacy systems get clear text alternatives.