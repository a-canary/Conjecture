"""
Read Files Tool for Conjecture
Provides secure file reading capabilities with input validation
"""

import os
import glob
from typing import List, Dict, Any


def readFiles(path_pattern: str, max_files: int = 50, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Read files from disk using glob patterns

    Args:
        path_pattern: Glob pattern for file selection (e.g., "*.py", "src/**/*.rs", "docs/*.md")
        max_files: Maximum number of files to read (default: 50)
        encoding: File encoding (default: 'utf-8')

    Returns:
        List of file contents with metadata (path, size, lines, etc.)
    """
    # Input validation
    if not path_pattern or not path_pattern.strip():
        return [{
            'path': '',
            'filename': '',
            'content': '',
            'read_success': False,
            'error': 'Path pattern cannot be empty'
        }]

    # Security: Prevent path traversal
    if '..' in path_pattern or path_pattern.startswith('/'):
        return [{
            'path': path_pattern,
            'filename': '',
            'content': '',
            'read_success': False,
            'error': 'Invalid path pattern'
        }]

    # Limit max_files
    if not isinstance(max_files, int) or max_files < 1:
        max_files = 50
    max_files = min(100, max_files)  # Hard limit of 100 files

    try:
        # Expand glob pattern with security restrictions
        file_paths = glob.glob(path_pattern, recursive=True)
        
        # Security: Only allow files in current directory and subdirectories
        current_dir = os.path.abspath('.')
        safe_paths = []
        
        for file_path in file_paths:
            abs_path = os.path.abspath(file_path)
            if abs_path.startswith(current_dir) and os.path.isfile(abs_path):
                safe_paths.append(abs_path)
        
        # Limit number of files
        safe_paths = safe_paths[:max_files]
        
        results = []
        
        for file_path in safe_paths:
            try:
                # Security: Check file size (prevent reading huge files)
                stat = os.stat(file_path)
                if stat.st_size > 1024 * 1024:  # 1MB limit
                    results.append({
                        'path': file_path,
                        'filename': os.path.basename(file_path),
                        'content': '',
                        'read_success': False,
                        'error': 'File too large (max 1MB)'
                    })
                    continue
                
                # Read file content with error handling
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                
                # Security: Truncate content to prevent memory issues
                if len(content) > 50000:  # 50KB limit
                    content = content[:50000] + "\n... (content truncated)"
                
                # Simple metadata
                results.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'extension': os.path.splitext(file_path)[1],
                    'size_bytes': stat.st_size,
                    'line_count': content.count('\n') + 1,
                    'content': content,
                    'encoding': encoding,
                    'read_success': True
                })
                
            except Exception:
                # Generic error - no sensitive details
                results.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'content': '',
                    'read_success': False,
                    'error': 'Failed to read file'
                })
        
        return results
        
    except Exception:
        return [{
            'path': path_pattern,
            'filename': '',
            'content': '',
            'read_success': False,
            'error': 'Failed to process file pattern'
        }]


def readFile(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Read a single file

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: 'utf-8')

    Returns:
        Dictionary with file content and metadata
    """
    # Input validation
    if not file_path or not file_path.strip():
        return {
            'path': '',
            'filename': '',
            'content': '',
            'read_success': False,
            'error': 'File path cannot be empty'
        }
    
    # Security: Prevent path traversal
    if '..' in file_path or file_path.startswith('/'):
        return {
            'path': file_path,
            'filename': '',
            'content': '',
            'read_success': False,
            'error': 'Invalid file path'
        }
    
    try:
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath('.')
        
        # Security: Only allow files in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'path': file_path,
                'filename': '',
                'content': '',
                'read_success': False,
                'error': 'Access denied'
            }
        
        if not os.path.isfile(abs_path):
            return {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'content': '',
                'read_success': False,
                'error': 'File not found'
            }
        
        # Security: Check file size
        stat = os.stat(abs_path)
        if stat.st_size > 1024 * 1024:  # 1MB limit
            return {
                'path': file_path,
                'filename': os.path.basename(file_path),
                'content': '',
                'read_success': False,
                'error': 'File too large (max 1MB)'
            }
        
        # Read file content
        with open(abs_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        # Security: Truncate content
        if len(content) > 50000:  # 50KB limit
            content = content[:50000] + "\n... (content truncated)"
        
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'size_bytes': stat.st_size,
            'line_count': content.count('\n') + 1,
            'content': content,
            'encoding': encoding,
            'read_success': True
        }
        
    except Exception:
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'content': '',
            'read_success': False,
            'error': 'Failed to read file'
        }


def listFiles(directory: str, pattern: str = "*", recursive: bool = False) -> List[Dict[str, Any]]:
    """
    List files in a directory without reading content

    Args:
        directory: Directory path to list
        pattern: File pattern to match (default: "*")
        recursive: Whether to search recursively (default: False)

    Returns:
        List of file information without content
    """
    # Input validation
    if not directory or not directory.strip():
        return [{
            'path': '',
            'filename': '',
            'list_success': False,
            'error': 'Directory cannot be empty'
        }]
    
    # Security: Prevent path traversal
    if '..' in directory or directory.startswith('/'):
        return [{
            'path': directory,
            'filename': '',
            'list_success': False,
            'error': 'Invalid directory path'
        }]
    
    try:
        abs_dir = os.path.abspath(directory)
        current_dir = os.path.abspath('.')
        
        # Security: Only allow directories in current directory and subdirectories
        if not abs_dir.startswith(current_dir):
            return [{
                'path': directory,
                'filename': '',
                'list_success': False,
                'error': 'Access denied'
            }]
        
        if not os.path.isdir(abs_dir):
            return [{
                'path': directory,
                'filename': '',
                'list_success': False,
                'error': 'Directory not found'
            }]
        
        # Build search pattern
        if recursive:
            search_pattern = os.path.join(abs_dir, "**", pattern)
        else:
            search_pattern = os.path.join(abs_dir, pattern)
        
        # Find matching files
        file_paths = glob.glob(search_pattern, recursive=recursive)
        
        # Security: Filter to safe paths only
        safe_paths = []
        for file_path in file_paths:
            abs_path = os.path.abspath(file_path)
            if abs_path.startswith(current_dir) and os.path.isfile(abs_path):
                safe_paths.append(abs_path)
        
        results = []
        
        for file_path in safe_paths[:100]:  # Limit to 100 files
            try:
                stat = os.stat(file_path)
                results.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'extension': os.path.splitext(file_path)[1],
                    'size_bytes': stat.st_size,
                    'list_success': True
                })
            except Exception:
                results.append({
                    'path': file_path,
                    'filename': os.path.basename(file_path),
                    'list_success': False,
                    'error': 'Failed to get file info'
                })
        
        return results
        
    except Exception:
        return [{
            'path': directory,
            'filename': '',
            'list_success': False,
            'error': 'Failed to list directory'
        }]


def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use this tool
    """
    return [
        "readFiles('*.py') returns contents of all Python files in current directory",
        "readFiles('src/**/*.rs') returns contents of all Rust source files in src directory recursively",
        "readFiles('Cargo.toml') returns the Cargo.toml configuration file content",
        "readFiles('docs/*.md') returns all markdown documentation files",
        "readFile('src/main.rs') returns the main Rust source file content with metadata",
        "listFiles('src/', pattern='*.rs', recursive=True) returns list of all Rust files in src directory without reading content",
        "readFiles('*.rs', max_files=10) returns up to 10 Rust source files to avoid reading too many files",
        "readFiles('README.md') returns project documentation and setup instructions"
    ]


if __name__ == "__main__":
    # Test the read files functionality
    print("Testing readFiles tool...")

    # Test reading Python files
    print("\n1. Reading Python files:")
    results = readFiles("*.py", max_files=3)
    for result in results:
        if result['read_success']:
            print(f"   ✓ {result['filename']} ({result['line_count']} lines)")
        else:
            print(f"   ✗ {result.get('error', 'Unknown error')}")

    # Test reading a single file
    print("\n2. Reading single file:")
    single_result = readFile("tools/readFiles.py")
    if single_result['read_success']:
        print(f"   ✓ {single_result['filename']} ({single_result['line_count']} lines)")
        print(f"   Content preview: {single_result['content'][:100]}...")
    else:
        print(f"   ✗ {single_result.get('error', 'Unknown error')}")

    # Test listing files
    print("\n3. Listing files:")
    list_results = listFiles("tools/", pattern="*.py")
    for result in list_results:
        if result['list_success']:
            print(f"   ✓ {result['filename']} ({result['size_bytes']} bytes)")
        else:
            print(f"   ✗ {result.get('error', 'Unknown error')}")

    print("\nExamples for LLM context:")
    for example in examples():
        print(f"- {example}")