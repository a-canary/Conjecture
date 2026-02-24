# Emoji Usage Guide

**Last Updated:** November 21, 2025
**Version:** 1.0
**Author:** Design Documentation Writer**

## 🎯 Welcome to Emojis in Conjecture!

Conjecture uses emojis to make your terminal experience more visual and intuitive. Emojis help you quickly understand what's happening without reading through long text messages. Best of all, they work everywhere - even in environments that don't support emojis!

## 🚀 Quick Start

### Basic Usage

```python
from utils.terminal_emoji import success, error, warning, info

# Show different types of messages
success("Database connected!")
error("Failed to load configuration")
warning("Low memory detected")
info("Starting analysis...")
```

**Sample Output:**
```
👍 Database connected!
❌ Failed to load configuration
⚠️  Low memory detected
ℹ️  Starting analysis...
```

### Enabling Verbose Logging with Emojis

```python
from utils.verbose_logger import VerboseLogger, VerboseLevel

# Create logger with emoji support
logger = VerboseLogger(VerboseLevel.USER)

# Your commands will automatically include emojis
# (These are called internally by Conjecture)
logger.claim_assessed_confident("claim123", 0.9, 0.8)  # 🎯
logger.tool_executed("WebSearch", {"query": "Python"})  # 🛠️
logger.final_response("Analysis complete")  # 🎯
```

## 🎨 Available Emojis

### Success & Status
| Function | Emoji | Fallback | When to Use |
|----------|-------|----------|-------------|
| `success("msg")` | 👍 | [OK] | Operations completed successfully |
| `resolved("msg")` | ✅ | [OK] | Tasks finished or problems solved |
| `target("msg")` | 🎯 | [TARGET] | Goals achieved or targets reached |

### Errors & Warnings
| Function | Emoji | Fallback | When to Use |
|----------|-------|----------|-------------|
| `error("msg")` | ❌ | [ERROR] | Failures, exceptions, critical issues |
| `warning("msg")` | ⚠️ | [WARN] | Non-critical issues, cautions |

### Information & Processing
| Function | Emoji | Fallback | When to Use |
|----------|-------|----------|-------------|
| `info("msg")` | ℹ️ | [INFO] | General information, status updates |
| `loading("msg")` | ⏳ | [LOADING] | Operations in progress |
| `tool("msg")` | 🛠️ | [TOOL] | Tool execution actions |
| `stats("msg")` | 📊 | [STATS] | Statistical information, results |

### Communication
| Function | Emoji | Fallback | When to Use |
|----------|-------|----------|-------------|
| `chat("msg")` | 💬 | [CHAT] | User messages, conversations |

## 💡 Common Usage Patterns

### 1. Command-Line Operations

```python
def install_dependencies():
    info("Checking dependencies...")
    
    if check_dependencies():
        success("All dependencies satisfied!")
    else:
        warning("Some dependencies missing")
        loading("Installing missing packages...")
        
        if install_packages():
            success("Dependencies installed!")
        else:
            error("Failed to install dependencies")
```

### 2. Data Processing

```python
def process_dataset(data):
    info(f"Processing {len(data)} records...")
    
    try:
        # Validate data
        if not validate_data(data):
            warning("Data validation issues detected")
        
        # Process data
        loading("Analyzing data...")
        results = analyze_data(data)
        
        # Show results
        target(f"Analysis complete: {results.summary}")
        stats(f"Processed {len(results.processed)} records successfully")
        
    except Exception as e:
        error(f"Data processing failed: {e}")
```

### 3. Configuration Management

```python
def load_config(path):
    info(f"Loading configuration from {path}...")
    
    if not os.path.exists(path):
        warning(f"Configuration file not found: {path}")
        loading("Creating default configuration...")
        create_default_config(path)
    
    try:
        config = parse_config(path)
        success("Configuration loaded successfully!")
        return config
    except Exception as e:
        error(f"Failed to load configuration: {e}")
        return None
```

### 4. API and Network Operations

```python
def fetch_data(url):
    info(f"Fetching data from {url}...")
    
    try:
        loading("Connecting to server...")
        response = make_request(url)
        
        if response.success:
            target(f"Data retrieved: {len(response.data)} items")
            return response.data
        else:
            warning(f"Partial success: {response.status}")
            return response.data
            
    except ConnectionError:
        error("Network connection failed")
    except TimeoutError:
        warning("Request timed out, retrying...")
        return retry_request(url)
```

## 🔧 Advanced Usage

### Custom Emoji Messages

If you need more control, you can use the emoji printer directly:

```python
from utils.terminal_emoji import emoji_printer

# Use any emoji shortcode
emoji_printer.print(":gear: Configuration loaded")
emoji_printer.print(":rocket: Launching application...")
emoji_printer.print(":shield: Security scan complete")

# Combine messages
emoji_printer.print(":thumbs_up: Success! :check_mark: All tests passed")
```

### Creating Your Own Emoji Functions

```python
from utils.terminal_emoji import emoji_printer

def database(msg=""):
    """Custom database operation emoji."""
    emoji_printer.print(f":database: {msg}" if msg else ":database:")

def security(msg=""):
    """Custom security operation emoji."""
    emoji_printer.print(f":shield: {msg}" if msg else ":security:")

# Usage
database("Connecting to PostgreSQL...")
security("Authentication verified!")
```

### Verbose Logging Integration

When enabling verbose logging in Conjecture, emojis are used automatically:

```python
# Using the CLI with emojis
conjecture --verbose --level USER create "Python is a programming language"

# Output will include:
# 🎯 Claim confident: c0000001 (confidence: 0.95 >= 0.80)
# 🛠️ Tool: WebSearch(query="Python programming language")
# 💬 User message: Create claim "Python is a programming language"
# 📝 Claim created: c0000001 - "Python is a programming language"
```

## 🌍 Platform Compatibility

### Windows

**Automatic Setup:** Conjecture automatically configures Windows for emoji support.

**Best Experience:** Use **Windows Terminal** for full emoji support.

**Command Prompt/PowerShell:** Will show text alternatives like `[OK]`, `[ERROR]`, etc.

### macOS

**Full Support:** Most macOS terminals support emojis natively.

**Recommended:** Terminal.app or iTerm2 work perfectly.

### Linux

**Modern Support:** Most Linux distributions support emojis in modern terminals.

**Tested:** GNOME Terminal, Konsole - both work well.

**Note:** Some older terminals might show text alternatives.

## 🔄 Fallback Behavior

If emojis can't be displayed, you'll see clear text alternatives:

| Emoji | Fallback | Example |
|-------|----------|---------|
| 👍 | [OK] | `[OK] Operation complete` |
| ❌ | [ERROR] | `[ERROR] Connection failed` |
| ⚠️ | [WARN] | `[WARN] Low memory` |
| ℹ️ | [INFO] | `[INFO] Starting process` |
| 🎯 | [TARGET] | `[TARGET] Goal reached` |
| ⏳ | [LOADING] | `[LOADING] Processing...` |
| 🛠️ | [TOOL] | `[TOOL] Executed command` |
| 📊 | [STATS] | `[STATS] 15 items processed` |
| 💬 | [CHAT] | `[CHAT] User input received` |

## 🎯 Best Practices

### 1. Choose Clear Emojis

- **Good:** `success("File saved successfully")` 👍
- **Good:** `error("Invalid file format")` ❌
- **Avoid:** Using obscure emojis that might confuse users

### 2. Provide Context

```python
# Good: Clear context with emoji
success(f"Database connected to {config.database_name}")

# Avoid: Emoji without context
success()
```

### 3. Be Consistent

```python
# Consistent pattern in your application
if result.success:
    success(f"Operation completed: {result.action}")
else:
    error(f"Operation failed: {result.error}")
```

### 4. Think About Accessibility

```python
# Good: Emoji + text
success(f"Backup completed: {backup.size}GB saved")

# Better: Check user preference
if user_prefers_text:
    print(f"Backup completed: {backup.size}GB saved")
else:
    success(f"Backup completed: {backup.size}GB saved")
```

## 🎨 Styling Your Output

### Progress Indicators

```python
def show_progress(current, total):
    percentage = (current / total) * 100
    
    if current < total:
        loading(f"Progress: {percentage:.1f}% ({current}/{total})")
    else:
        target(f"Complete: {total} items processed")
```

### Status Reports

```python
def show_status_report():
    info("Generating status report...")
    
    # System status
    stats(f"Memory usage: {get_memory_usage()}%")
    stats(f"CPU usage: {get_cpu_usage()}%")
    
    # Application status
    if system_healthy():
        target("All systems operational")
    else:
        warning("Some systems need attention")
```

### Multi-step Processes

```python
def run_complex_operation():
    steps = [
        ("Initializing", lambda: init_system()),
        ("Loading data", lambda: load_data()),
        ("Processing", lambda: process_data()),
        ("Saving results", lambda: save_results())
    ]
    
    for step_name, step_func in steps:
        info(f"Step: {step_name}")
        
        try:
            loading(f"Executing {step_name.lower()}...")
            step_func()
            success(f"Completed: {step_name}")
        except Exception as e:
            error(f"Failed: {step_name} - {e}")
            break
```

## 🔍 Troubleshooting Quick Fixes

### Emojis Not Showing?

1. **Check your terminal:** Use a modern terminal (Windows Terminal, iTerm2, etc.)
2. **Windows users:** Run `chcp 65001` in Command Prompt
3. **Install emoji package:** `pip install emoji`

### Strange Characters?

1. **UTF-8 encoding:** Ensure your terminal supports UTF-8
2. **Fallback mode:** System automatically uses text alternatives
3. **Font support:** Some terminals need emoji-compatible fonts

### Performance Issues?

1. **Use global functions:** `success()`, `error()` are optimized
2. **Avoid creating instances:** Use provided emoji_printer
3. **Batch operations:** Group emoji messages together

## 📚 Full Emoji Reference

### Conjecture-Specific Emojis

| Shortcode | Emoji | Fallback | Context |
|-----------|-------|----------|---------|
| `:target:` | 🎯 | [TARGET] | Claims meeting confidence threshold |
| `:hammer_and_wrench:` | 🛠️ | [TOOL] | Tool execution |
| `:hourglass_flowing_sand:` | ⏳ | [LOADING] | Processing in progress |
| `:speech_balloon:` | 💬 | [CHAT] | User communication |
| `:bar_chart:` | 📊 | [STATS] | Statistical information |
| `:magnifying_glass:` | 🔍 | [SEARCH] | Investigation/analysis |
| `:zap:` | ⚡ | [PROCESS] | Quick operations |
| `:sparkles:` | ✨ | [COMPLETE] | Completion with quality |

### Universal Emojis

| Shortcode | Emoji | Fallback | Context |
|-----------|-------|----------|---------|
| `:thumbs_up:` | 👍 | [OK] | Success, approval |
| `:x:` | ❌ | [ERROR] | Error, failure |
| `:warning:` | ⚠️ | [WARN] | Warning, caution |
| `:check_mark:` | ✅ | [OK] | Verified, complete |
| `:information_source:` | ℹ️ | [INFO] | Information, help |
| `:gear:` | ⚙️ | [CONFIG] | Configuration, settings |
| `:link:` | 🔗 | [LINK] | Connections, relationships |
| `:robot_face:` | 🤖 | [PROCESS] | AI/LLM operations |
| `:shield:` | 🛡️ | [SECURE] | Security, protection |
| `:rocket:` | 🚀 | [LAUNCH] | Starting, launching |

## 🎉 Examples Gallery

### Example 1: File Processing

```python
def process_files(file_list):
    info(f"Found {len(file_list)} files to process")
    
    for i, file in enumerate(file_list, 1):
        loading(f"Processing file {i}/{len(file_list)}: {file.name}")
        
        try:
            process_single_file(file)
            success(f"Processed: {file.name}")
        except Exception as e:
            error(f"Failed: {file.name} - {e}")
    
    target(f"Batch processing complete: {len(file_list)} files")
```

### Example 2: System Health Check

```python
def health_check():
    info("Running system health check...")
    
    checks = [
        ("Database", check_database),
        ("API Connectivity", check_api),
        ("Disk Space", check_disk_space),
        ("Memory", check_memory)
    ]
    
    healthy_count = 0
    
    for name, check_func in checks:
        result = check_func()
        if result:
            success(f"{name}: OK")
            healthy_count += 1
        else:
            warning(f"{name}: Issue detected")
    
    if healthy_count == len(checks):
        target("All systems healthy!")
    else:
        warning(f"{len(checks) - healthy_count} systems need attention")
```

### Example 3: Data Validation

```python
def validate_dataset(data):
    info("Starting data validation...")
    
    # Check structure
    if not has_required_columns(data):
        error("Missing required columns")
        return False
    
    # Check content
    if has_duplicates(data):
        warning("Duplicate records found")
    
    if has_missing_values(data):
        warning("Missing values detected")
    
    # Check quality
    quality_score = data_quality_score(data)
    stats(f"Data quality score: {quality_score:.1f}/10")
    
    if quality_score >= 8.0:
        target(f"High quality dataset: {quality_score:.1f}/10")
        success("Validation passed!")
        return True
    elif quality_score >= 6.0:
        warning(f"Acceptable quality: {quality_score:.1f}/10")
        info("Consider cleaning the data")
        return True
    else:
        error(f"Low quality dataset: {quality_score:.1f}/10")
        return False
```

## 🎓 Resources

- [Complete Technical Documentation](../implementation/emoji-implementation-design.md)
- [API Reference](../api/emoji-api-integration.md)
- [Troubleshooting Guide](../support/emoji-troubleshooting.md)
- [Testing Information](../testing/emoji-testing-plan.md)

## 💬 Feedback

Found a way to improve emoji usage? Have suggestions for new emoji functions? Your feedback helps make Conjecture better for everyone!

---

**Enjoy using emojis with Conjecture!** 🎉

Remember: Emojis are meant to enhance, not complicate. The system always has a fallback to ensure your messages are clear and readable, no matter what environment you're using.