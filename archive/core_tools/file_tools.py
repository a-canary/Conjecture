"""
Additional File Tools for Conjecture
Provides file editing and searching capabilities beyond basic read/write operations
"""

import os
import re
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tools.registry import register_tool

@register_tool(name="EditFile", is_core=False)
def EditFile(file_path: str, old_text: str, new_text: str, 
             encoding: str = 'utf-8', backup: bool = True) -> Dict[str, Any]:
    """
    Edit a file by replacing a specific text pattern with new text.

    Args:
        file_path: Path to the file to edit
        old_text: The exact text to replace (must match exactly)
        new_text: The new text to insert in place of old_text
        encoding: File encoding (default: 'utf-8')
        backup: Whether to create a backup before editing (default: True)

    Returns:
        Dictionary with edit operation result and metadata
    """
    # Input validation
    if not file_path or not file_path.strip():
        return {
            'success': False,
            'error': 'File path cannot be empty'
        }

    if old_text is None or new_text is None:
        return {
            'success': False,
            'error': 'Old text and new text cannot be None'
        }

    # Security: Prevent path traversal
    if '..' in file_path or file_path.startswith('/'):
        return {
            'success': False,
            'error': 'Invalid file path'
        }

    # Security: Limit text sizes
    if len(old_text) > 10000 or len(new_text) > 10000:
        return {
            'success': False,
            'error': 'Text too long (max 10000 characters each)'
        }

    try:
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath('.')

        # Security: Only allow files in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'success': False,
                'error': 'Access denied'
            }

        if not os.path.exists(abs_path):
            return {
                'success': False,
                'error': 'File not found'
            }

        if not os.path.isfile(abs_path):
            return {
                'success': False,
                'error': 'Target is not a file'
            }

        # Security: Check file size
        stat = os.stat(abs_path)
        if stat.st_size > 1024 * 1024:  # 1MB limit
            return {
                'success': False,
                'error': 'File too large (max 1MB)'
            }

        # Read the file
        try:
            with open(abs_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to read file: {str(e)}'
            }

        # Check if old_text exists
        if old_text not in content:
            return {
                'success': False,
                'error': 'Text to replace not found in file',
                'old_text_length': len(old_text),
                'file_size': len(content)
            }

        # Count occurrences
        occurrences = content.count(old_text)

        # Create backup if requested
        backup_path = None
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{abs_path}.backup_{timestamp}"
            try:
                with open(backup_path, 'w', encoding=encoding) as f:
                    f.write(content)
            except Exception:
                return {
                    'success': False,
                    'error': 'Failed to create backup'
                }

        # Perform replacement
        new_content = content.replace(old_text, new_text, 1)  # Replace only first occurrence

        # Write the edited content
        try:
            with open(abs_path, 'w', encoding=encoding) as f:
                f.write(new_content)
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to write file: {str(e)}',
                'backup_path': backup_path
            }

        # Get updated stats
        new_stat = os.stat(abs_path)

        return {
            'success': True,
            'occurrences_found': occurrences,
            'replacements_made': 1,
            'original_size': stat.st_size,
            'new_size': new_stat.st_size,
            'size_change': new_stat.st_size - stat.st_size,
            'backup_path': backup_path,
            'encoding': encoding
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Edit operation failed: {str(e)}'
        }

@register_tool(name="GrepFiles", is_core=False)
def GrepFiles(pattern: str, path: str = ".", file_pattern: str = "*", 
              max_results: int = 100, case_sensitive: bool = False, 
              context_lines: int = 0) -> Dict[str, Any]:
    """
    Search for text patterns in files using grep-like functionality.

    Args:
        pattern: Regular expression pattern to search for
        path: Directory path to search (default: current directory)
        file_pattern: File pattern to match (default: "*")
        max_results: Maximum number of matches to return (default: 100)
        case_sensitive: Whether search should be case sensitive (default: False)
        context_lines: Number of context lines to include (default: 0)

    Returns:
        Dictionary with search results and metadata
    """
    # Input validation
    if not pattern or not pattern.strip():
        return {
            'success': False,
            'error': 'Search pattern cannot be empty'
        }

    if not path or not path.strip():
        path = "."

    # Security: Prevent path traversal
    if '..' in path:
        return {
            'success': False,
            'error': 'Invalid search path'
        }

    # Limit pattern length
    if len(pattern) > 1000:
        return {
            'success': False,
            'error': 'Search pattern too long (max 1000 characters)'
        }

    # Limit max_results
    if not isinstance(max_results, int) or max_results < 1:
        max_results = 100
    max_results = min(1000, max_results)  # Hard limit

    try:
        abs_path = os.path.abspath(path)
        current_dir = os.path.abspath('.')

        # Security: Only allow searching in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'success': False,
                'error': 'Access denied'
            }

        if not os.path.exists(abs_path):
            return {
                'success': False,
                'error': 'Search path not found'
            }

        # Compile regex pattern
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {
                'success': False,
                'error': f'Invalid regex pattern: {str(e)}'
            }

        # Find matching files
        search_pattern = os.path.join(abs_path, "**", file_pattern)
        file_paths = glob.glob(search_pattern, recursive=True)

        # Filter to safe paths only
        safe_files = []
        for file_path in file_paths:
            if (os.path.isfile(file_path) and 
                file_path.startswith(current_dir) and
                os.path.getsize(file_path) <= 1024 * 1024):  # 1MB limit
                safe_files.append(file_path)

        results = []
        total_matches = 0

        for file_path in safe_files[:200]:  # Limit files to process
            if total_matches >= max_results:
                break

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if total_matches >= max_results:
                        break

                    matches = list(regex.finditer(line))
                    if matches:
                        # Prepare context
                        start_context = max(0, line_num - context_lines - 1)
                        end_context = min(len(lines), line_num + context_lines)
                        
                        context_lines_list = []
                        for i in range(start_context, end_context):
                            prefix = ">>> " if i == line_num - 1 else "    "
                            context_lines_list.append(f"{prefix}{i+1}: {lines[i].rstrip()}")

                        result = {
                            'file_path': os.path.relpath(file_path, current_dir),
                            'line_number': line_num,
                            'line_content': line.strip(),
                            'matches': [{'start': m.start(), 'end': m.end(), 'text': m.group()} for m in matches],
                            'context_lines': context_lines_list if context_lines > 0 else [],
                            'match_count': len(matches)
                        }

                        results.append(result)
                        total_matches += sum(len(m) for m in [matches])

            except Exception:
                # Skip files that can't be read
                continue

        return {
            'success': True,
            'results': results,
            'total_matches': total_matches,
            'files_searched': len(safe_files),
            'pattern': pattern,
            'search_path': path,
            'file_pattern': file_pattern,
            'case_sensitive': case_sensitive
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Search failed: {str(e)}'
        }

@register_tool(name="CountLines", is_core=False)
def CountLines(file_path: str, count_empty: bool = True, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Count lines in a file with various options.

    Args:
        file_path: Path to the file to analyze
        count_empty: Whether to count empty lines (default: True)
        encoding: File encoding (default: 'utf-8')

    Returns:
        Dictionary with line count statistics
    """
    # Input validation
    if not file_path or not file_path.strip():
        return {
            'success': False,
            'error': 'File path cannot be empty'
        }

    # Security: Prevent path traversal
    if '..' in file_path or file_path.startswith('/'):
        return {
            'success': False,
            'error': 'Invalid file path'
        }

    try:
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath('.')

        # Security: Only allow files in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'success': False,
                'error': 'Access denied'
            }

        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return {
                'success': False,
                'error': 'File not found'
            }

        # Security: Check file size
        stat = os.stat(abs_path)
        if stat.st_size > 5 * 1024 * 1024:  # 5MB limit
            return {
                'success': False,
                'error': 'File too large (max 5MB)'
            }

        # Count lines
        total_lines = 0
        non_empty_lines = 0
        with open(abs_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                total_lines += 1
                if line.strip():
                    non_empty_lines += 1

        empty_lines = total_lines - non_empty_lines

        return {
            'success': True,
            'file_path': file_path,
            'total_lines': total_lines,
            'non_empty_lines': non_empty_lines,
            'empty_lines': empty_lines,
            'counted_empty': count_empty,
            'effective_line_count': total_lines if count_empty else non_empty_lines,
            'file_size_bytes': stat.st_size
        }

    except Exception as e:
        return {
            'success': False,
            'error': f'Line count failed: {str(e)}'
        }

def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use these tools
    """
    return [
        "EditFile('src/main.rs', 'println!(\"Hello\")', 'println!(\"Hello, World!\")') replaces a specific string in a Rust source file",
        "GrepFiles('fn main', './src/', '*.rs', case_sensitive=True) finds all main functions in Rust source files",
        "GrepFiles('TODO|FIXME', '.', '*.py', context_lines=2) searches for TODO comments with 2 lines of context in Python files",
        "CountLines('README.md') counts total lines in the README file, including empty lines",
        "EditFile('config.yml', old_text, new_text, backup=True) safely edits configuration file with backup creation",
        "GrepFiles('import.*requests', './', '*.py', max_results=50) finds all Python imports of requests package",
        "CountLines('src/lib.rs', count_empty=False) counts non-empty lines in a Rust library file",
        "EditFile('Cargo.toml', 'version = \"1.0\"', 'version = \"1.1\"') updates version number in Cargo.toml"
    ]

if __name__ == "__main__":
    # Test the file tools
    print("Testing file tools...")

    # Create test file
    test_file = "test_output/sample.txt"
    os.makedirs("test_output", exist_ok=True)
    
    test_content = """Line 1: Hello World
Line 2: This is a test
Line 3: TODO: Improve this
Line 4: Another line
Line 5: FIXME: Bug here
Line 6: Final line
"""
    
    with open(test_file, 'w') as f:
        f.write(test_content)

    # Test editing file
    print("\n1. Testing file edit:")
    edit_result = EditFile(test_file, "Hello World", "Hello, World!", backup=True)
    print(f"   Edit success: {edit_result['success']}")
    print(f"   Replacements: {edit_result.get('replacements_made', 0)}")

    # Test grep search
    print("\n2. Testing grep search:")
    grep_result = GrepFiles("TODO|FIXME", "test_output", "*.txt", context_lines=1)
    print(f"   Search success: {grep_result['success']}")
    print(f"   Total matches: {grep_result.get('total_matches', 0)}")
    for result in grep_result.get('results', []):
        print(f"   Found in {result['file_path']} line {result['line_number']}")

    # Test line counting
    print("\n3. Testing line count:")
    count_result = CountLines(test_file)
    print(f"   Count success: {count_result['success']}")
    print(f"   Total lines: {count_result.get('total_lines', 0)}")
    print(f"   Non-empty lines: {count_result.get('non_empty_lines', 0)}")

    # Clean up
    try:
        os.unlink(test_file)
        # Also remove backup files
        for backup in glob.glob("test_output/sample.txt.backup_*"):
            os.unlink(backup)
        os.rmdir("test_output")
        print(f"\n✓ Cleaned up test files")
    except:
        pass

    print("\nExamples for LLM context:")
    for example in examples():
        print(f"- {example}")