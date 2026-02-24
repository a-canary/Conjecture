# Emoji Troubleshooting Guide

**Last Updated:** November 21, 2025
**Version:** 1.0
**Author:** Design Documentation Writer**

## 🔧 Common Emoji Issues and Solutions

This guide helps you resolve common emoji-related problems in Conjecture. Most issues are easily fixable with the right configuration or approach.

---

## 🚨 Quick Diagnosis

First, let's check what's happening with emoji support:

```python
# Run this diagnostic script
from utils.terminal_emoji import test_emoji_support
test_emoji_support()
```

**Expected Output:**
```
🧪 Conjecture Emoji Package Test
========================================
Platform: Windows  # or macOS/Linux
Emoji package available: True
Emoji enabled: True
👍 Success!
❌ Error occurred
⚠️ Warning message
ℹ️ Information
🎯 Target reached
⏳ Loading...
🛠️ Tool executed
✅ Task completed

✅ Full emoji support enabled!
```

If you see any red flags or failures, continue reading for solutions.

---

## 🖥️ Platform-Specific Issues

### Windows Issues

#### ❌ Problem: Emojis Not Showing in Command Prompt

**Symptoms:**
- Seeing squares or question marks instead of emojis
- Getting `[OK]`, `[ERROR]` text alternatives

**Solutions:**

**Option 1: Use Windows Terminal (Recommended)**
1. Download Windows Terminal from Microsoft Store
2. Configure it to use PowerShell or Command Prompt
3. Emojis will work automatically

**Option 2: Configure Command Prompt**
```cmd
# Run this command before using Conjecture
chcp 65001
set PYTHONIOENCODING=utf-8
```

**Option 3: PowerShell UTF-8 Setup**
```powershell
# Set encoding for the session
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"
```

**Option 4: Permanent Fix (Advanced)**
Create a PowerShell profile:
```powershell
# Add to $PROFILE
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"
```

#### ❌ Problem: Old Windows Console

**Symptoms:**
- Very old Windows (pre-Windows 10)
- Legacy Enterprise environments

**Solutions:**
1. **Accept fallbacks:** The system will show clear text alternatives
2. **Use SSH/Remote terminal:** Connect to a Linux system with proper emoji support
3. **Upgrade:** Consider upgrading to Windows 10/11

### macOS Issues

#### ❌ Problem: Old Terminal Font

**Symptoms:**
- Blank spaces where emojis should be
- Inconsistent emoji rendering

**Solutions:**

**Option 1: Update Terminal Font**
1. Open Terminal.app
2. Go to Terminal → Preferences → Profiles → Text
3. Change font to "SF Pro" or another modern font
4. Restart Terminal

**Option 2: Use iTerm2 (Recommended)**
1. Install iTerm2 from https://iterm2.com/
2. Configure font preferences
3. Emoji support is much better

**Option 3: System Font Update**
```bash
# Install updated system fonts
brew install font-roboto font-noto-sans font-noto-emoji
```

#### ❌ Problem: Terminal Encoding Issues

**Symptoms:**
- Garbled characters
- Mixed-up rendering

**Solution:**
```bash
# Check current locale
locale

# Set proper UTF-8 locale if needed
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### Linux Issues

#### ❌ Problem: Missing Emoji Fonts

**Symptoms:**
- Empty rectangles or tofu characters (□)
- Inconsistent emoji rendering

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install fonts-noto-color-emoji
```

**Fedora/CentOS:**
```bash
sudo dnf install google-noto-emoji-fonts
sudo dnf install emoji-font
```

**Arch Linux:**
```bash
sudo pacman -S noto-fonts-emoji
```

**General Linux:**
```bash
# Test font support
fc-list | grep -i emoji

# Install Noto Color Emoji (if available)
sudo apt-get install fonts-noto-color-emoji
```

#### ❌ Problem: Terminal Doesn't Support Unicode

**Symptoms:**
- Very old terminal emulators
- Minimal Linux environments

**Solutions:**
1. **Use fallbacks:** System works without emojis
2. **Upgrade terminal:** Install GNOME Terminal, Konsole, or Alacritty
3. **Change locale:**
   ```bash
   export LANG=en_US.UTF-8
   export LC_ALL=en_US.UTF-8
   ```

---

## 📦 Package Issues

### ❌ Problem: Emoji Package Not Installed

**Symptoms:**
```
Warning: 'emoji' package not installed. Run: pip install emoji
```

**Solution:**
```bash
# Install the emoji package
pip install emoji>=2.15.0

# Or using requirements.txt
pip install -r requirements.txt
```

### ❌ Problem: Outdated Emoji Package

**Symptoms:**
- Missing new emojis
- Incorrect emoji rendering
- Version compatibility warnings

**Solution:**
```bash
# Update the emoji package
pip install --upgrade emoji

# Check current version
pip show emoji
```

### ❌ Problem: Package Conflict

**Symptoms:**
- Import errors
- Dependency conflicts

**Solutions:**

**Option 1: Clean Reinstall**
```bash
pip uninstall emoji
pip install emoji>=2.15.0
```

**Option 2: Virtual Environment**
```bash
python -m venv conjecture_env
source conjecture_env/bin/activate  # Linux/macOS
# or conjecture_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Option 3: Check Dependencies**
```bash
pip check
# Fix any reported conflicts
```

---

## 🔧 Configuration Issues

### ❌ Problem: Encoding Problems

**Symptoms:**
- `UnicodeEncodeError` exceptions
- Crashes when processing emojis

**Solutions:**

**Option 1: Set Environment Variables**
```bash
# Linux/macOS
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

# Windows (Command Prompt)
set PYTHONIOENCODING=utf-8

# Windows (PowerShell)
$env:PYTHONIOENCODING="utf-8"
```

**Option 2: Python Code Fix**
```python
import sys
import os

# Force UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Now use emoji functions
from utils.terminal_emoji import success
success("This should work now!")
```

**Option 3: Safe Wrapper**
```python
def safe_emoji_function(func, message):
    """Safely call emoji function with fallback."""
    try:
        from utils.terminal_emoji import emoji_printer
        func(message)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback to plain text
        print(f"[{func.__name__.upper()}] {message}")

# Usage
safe_emoji_function(success, "Operation complete")
```

### ❌ Problem: Performance Issues

**Symptoms:**
- Slow emoji processing
- High memory usage
- Delayed output

**Solutions:**

**Option 1: Use Global Functions**
```python
# Good: Use provided functions
from utils.terminal_emoji import success, error

# Avoid: Creating multiple instances
from utils.terminal_emoji import TerminalEmoji
printer = TerminalEmoji()  # Don't do this in loops
```

**Option 2: Batch Processing**
```python
# Process multiple messages at once
def process_results(results):
    for result in results:
        if result.success:
            success(result.message)
        else:
            error(result.message)
```

**Option 3: Disable for Performance**
```python
# Create text-only instance for high-volume operations
from utils.terminal_emoji import TerminalEmoji
fast_printer = TerminalEmoji(enable_emoji=False)

fast_printer.print(":thumbs_up: Message")
# Outputs: [OK] Message (fast)
```

---

## 🔄 Integration Issues

### ❌ Problem: Verbose Logger Not Showing Emojis

**Symptoms:**
- Verbose logging works but no emojis
- Text appears but no visual indicators

**Solutions:**

**Option 1: Check Verbose Level**
```python
from utils.verbose_logger import VerboseLogger, VerboseLevel

# Ensure level is set correctly
logger = VerboseLogger(VerboseLevel.USER)  # Shows emojis
# logger = VerboseLogger(VerboseLevel.NONE)  # Shows nothing
```

**Option 2: Check Implementation**
```python
# Make sure you're calling the right methods
logger.claim_assessed_confident("claim123", 0.9, 0.8)  # 🎯 Shows emoji
# logger.custom_method("message")  # Won't show emoji unless implemented
```

**Option 3: Test Directly**
```python
# Test emoji system directly
from utils.terminal_emoji import test_emoji_support
test_emoji_support()

# Test logger
from utils.verbose_logger import VerboseLogger, VerboseLevel
logger = VerboseLogger(VerboseLevel.USER)
logger.claim_assessed_confident("test", 0.9, 0.8)  # Should show 🎯 or [TARGET]
```

### ❌ Problem: Custom Code Not Using Emojis

**Symptoms:**
- Built-in Conjecture commands show emojis
- Your custom code doesn't

**Solutions:**

**Option 1: Import Emoji Functions**
```python
# Add to your custom code
from utils.terminal_emoji import success, error, warning, info

def your_function():
    try:
        # your logic
        success("Operation completed!")
    except Exception as e:
        error(f"Failed: {e}")
```

**Option 2: Wrap Existing Print Statements**
```python
# Instead of:
print("Success: something happened")

# Use:
from utils.terminal_emoji import emoji_printer
emoji_printer.print(":thumbs_up: Success: something happened")

# Or better:
success("Success: something happened")
```

**Option 3: Create Custom Emoji Functions**
```python
from utils.terminal_emoji import emoji_printer

def database_operation(msg=""):
    emoji_printer.print(f":database: {msg}" if msg else ":database:")

def security_check(msg=""):
    emoji_printer.print(f":shield: {msg}" if msg else ":shield:")
```

---

## 🐛 Advanced Troubleshooting

### Debug Mode Testing

```python
# Enable debug mode to see what's happening
import sys
import os
from utils.terminal_emoji import TerminalEmoji

# Show configuration
print(f"Platform: {sys.platform}")
print(f"Python version: {sys.version}")
print(f"Default encoding: {sys.stdout.encoding}")
print(f"PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING', 'not set')}")

# Test emoji package
try:
    import emoji
    print(f"Emoji package version: {emoji.__version__}")
    HAS_EMOJI = True
except ImportError:
    print("Emoji package not installed")
    HAS_EMOJI = False

# Test specific emoji
emoji_printer = TerminalEmoji()
test_emoji = ":thumbs_up: Test"
result = emoji_printer.emojize(test_emoji)

print(f"Input: {test_emoji}")
print(f"Output: {result}")
print(f"Processed correctly: {'✅' if '👍' in result else '⚠️' if '[OK]' in result else '❌'}")
```

### Force Specific Behavior

```python
# Test with different configurations
from utils.terminal_emoji import TerminalEmoji

# Force emoji enabled
emoji_enabled = TerminalEmoji(enable_emoji=True)
print("Emoji enabled:", emoji_enabled.emojize(":thumbs_up: test"))

# Force emoji disabled (text only)
text_only = TerminalEmoji(enable_emoji=False)
print("Emoji disabled:", text_only.emojize(":thumbs_up: test"))

# Test customshortcode
custom = Terminalemoji()
print("Custom:", custom.emojize(":gear: Configuration loaded"))
```

### Performance Profiling

```python
import time

from utils.terminal_emoji import TerminalEmoji

def benchmark_emoji_processing():
    """Test emoji processing performance."""
    printer = TerminalEmoji()
    
    test_messages = [
        ":thumbs_up: Success message",
        ":x: Error occurred",
        ":warning: Warning message",
        ":information_source: Info message",
        ":target: Target reached",
        ":hourglass_flowing_sand: Loading...",
        ":hammer_and_wrench: Tool executed",
        ":speech_balloon: Chat message",
        ":bar_chart: Statistics data",
        ":gear: Configuration updated"
    ]
    
    iterations = 1000
    
    start_time = time.time()
    for _ in range(iterations):
        for msg in test_messages:
            printer.emojize(msg)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / (iterations * len(test_messages))
    print(f"Average processing time: {avg_time * 1000:.3f}ms per message")
    
    # Performance should be under 1ms per message
    if avg_time < 0.001:
        print("✅ Performance is acceptable")
    else:
        print("⚠️  Performance may be slow")

# Run benchmark
benchmark_emoji_processing()
```

---

## ❓ Frequently Asked Questions

### Q: Why do I see `[OK]`, `[ERROR]`, etc. instead of emojis?

**A:** This is normal fallback behavior! It happens when:
- Your terminal doesn't support emoji display
- The emoji package isn't installed
- Your system has encoding limitations

The text alternatives are designed to be clear and readable, so your application works everywhere.

### Q: Can I force emojis to always show?

**A:** You can try, but it might cause encoding errors:

```python
from utils.terminal_emoji import TerminalEmoji

# Force emoji enabled (may cause errors)
printer = TerminalEmoji(enable_emoji=True)

# Better approach: Configure your terminal for UTF-8
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### Q: How do I disable emojis completely?

**A:** Easy! Create a text-only instance:

```python
from utils.terminal_emoji import TerminalEmoji

text_printer = TerminalEmoji(enable_emoji=False)
text_printer.print(":thumbs_up: Message")  # Always shows [OK] Message
```

### Q: Why are emojis slow in my application?

**A:** Common causes:
- Creating multiple TerminalEmoji instances (use the global functions instead)
- Processing very large amounts of emoji text
- Running on very old hardware

**Solutions:**
```python
# Good: Use provided functions
from utils.terminal_emoji import success, error

# Avoid: This creates new instances
from utils.terminal_emoji import TerminalEmoji
# Don't do this in loops!
```

### Q: Can I use my own emoji icons?

**A:** Yes! Create custom emoji functions:

```python
from utils.terminal_emoji import emoji_printer

def custom_success(message=""):
    emoji_printer.print(f":custom_icon: {message}" if message else ":custom_icon:")

# Use your custom function
custom_success("Custom operation complete")
```

### Q: Do emojis work in all terminals?

**A:** Emojis work in most modern terminals:
- ✅ Windows Terminal, PowerShell (with UTF-8)
- ✅ macOS Terminal.app, iTerm2
- ✅ Linux: GNOME Terminal, Konsole, Alacritty
- ⚠️ Legacy Command Prompt (works with fallbacks)
- ⚠️ Very old terminals (works with fallbacks)

### Q: How much memory do emojis use?

**A:** Very little! The emoji system is designed to be lightweight:
- Minimal memory overhead (< 1MB for typical usage)
- No caching to avoid memory leaks
- Fast string processing

### Q: Can emojis break my application?

**A:** No! The emoji system is designed to fail gracefully:
- All encoding errors are caught
- Automatic fallbacks prevent crashes
- Your application continues working even if emojis fail

---

## 🆘 Getting Help

### Self-Help Resources

1. **Run Diagnostic:**
   ```python
   from utils.terminal_emoji import test_emoji_support
   test_emoji_support()
   ```

2. **Check Configuration:**
   ```bash
   # Show current environment
   echo $PYTHONIOENCODING
   echo $LANG
   python -c "import sys; print(sys.stdout.encoding)"
   ```

3. **Test Package:**
   ```bash
   pip show emoji
   python -c "import emoji; print(emoji.emojize(':thumbs_up: test'))"
   ```

### Reporting Issues

If you're still having problems, please include:

1. **System Information:**
   - Operating system and version
   - Terminal application
   - Python version

2. **Symptoms:**
   - What you expected to happen
   - What actually happened
   - Any error messages

3. **Diagnostic Output:**
   - Results of `test_emoji_support()`
   - Configuration information

### Community Support

- Check the main Conjecture documentation
- Look for existing issues in the repository
- Ask questions in the appropriate forums

---

## 🎯 Quick Reference Solutions

| Problem | Try This First |
|---------|----------------|
| **No emojis on Windows** | Use Windows Terminal or run `chcp 65001` |
| **Package missing** | `pip install emoji>=2.15.0` |
| **Encoding errors** | Set `PYTHONIOENCODING=utf-8` |
| **Slow performance** | Use global functions (`success()`, `error()`) |
| **Custom code no emojis** | Import from `utils.terminal_emoji` |
| **Verbose logging no emojis** | Check `VerboseLevel` is set correctly |
| **Old terminal issues** | Accept fallbacks or upgrade terminal |
| **Font problems** | Install Noto Color Emoji fonts |
| **Linux no emojis** | `sudo apt install fonts-noto-color-emoji` |

---

## 📚 Additional Resources

- [Complete Emoji Usage Guide](../user-guides/emoji-usage-guide.md)
- [Technical Implementation Details](../implementation/emoji-implementation-design.md)
- [API Integration Documentation](../api/emoji-api-integration.md)
- [Testing Information](../testing/emoji-testing-plan.md)

---

**Remember:** The emoji system is designed to work everywhere. If you see text alternatives like `[OK]`, `[ERROR]`, that's intentional fallback behavior, not an error! The system ensures your messages are always clear and readable.