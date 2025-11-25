"""
Write Files Tool for Conjecture
Provides secure file writing capabilities with input validation
"""

import os
import shutil
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

# Import the registry system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.tools.registry import register_tool


@register_tool(name="WriteFile", is_core=True)
def writeFile(file_path: str, content: str, create_dirs: bool = True,
              backup: bool = False, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Write content to a file

    Args:
        file_path: Path where to write the file
        content: Content to write to the file
        create_dirs: Whether to create parent directories (default: True)
        backup: Whether to create backup of existing file (default: False)
        encoding: File encoding (default: 'utf-8')

    Returns:
        Dictionary with operation result and metadata
    """
    # Input validation
    if not file_path or not file_path.strip():
        return {
            'path': '',
            'filename': '',
            'write_success': False,
            'error': 'File path cannot be empty'
        }
    
    if content is None:
        content = ''
    
    # Security: Prevent path traversal
    if '..' in file_path or file_path.startswith('/'):
        return {
            'path': file_path,
            'filename': '',
            'write_success': False,
            'error': 'Invalid file path'
        }
    
    # Security: Limit content size
    if len(content) > 1024 * 1024:  # 1MB limit
        return {
            'path': file_path,
            'filename': '',
            'write_success': False,
            'error': 'Content too large (max 1MB)'
        }
    
    try:
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath('.')
        
        # Security: Only allow files in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'path': file_path,
                'filename': '',
                'write_success': False,
                'error': 'Access denied'
            }
        
        # Create parent directories if requested
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and create_dirs:
            os.makedirs(parent_dir, exist_ok=True)
        elif parent_dir and not os.path.exists(parent_dir):
            return {
                'path': file_path,
                'filename': '',
                'write_success': False,
                'error': 'Parent directory does not exist'
            }
        
        # Create backup if requested and file exists
        backup_path = None
        if backup and os.path.exists(abs_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{abs_path}.backup_{timestamp}"
            shutil.copy2(abs_path, backup_path)
        
        # Write the file
        with open(abs_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        # Get file stats
        stat = os.stat(abs_path)
        line_count = content.count('\n') + 1 if content else 0
        
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'size_bytes': stat.st_size,
            'line_count': line_count,
            'encoding': encoding,
            'write_success': True,
            'backup_path': backup_path
        }
        
    except Exception:
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'write_success': False,
            'error': 'Failed to write file'
        }


@register_tool(name="AppendFile", is_core=False)
def appendFile(file_path: str, content: str, create_dirs: bool = True,
               encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Append content to an existing file

    Args:
        file_path: Path to the file to append to
        content: Content to append
        create_dirs: Whether to create parent directories (default: True)
        encoding: File encoding (default: 'utf-8')

    Returns:
        Dictionary with operation result and metadata
    """
    # Input validation
    if not file_path or not file_path.strip():
        return {
            'path': '',
            'filename': '',
            'append_success': False,
            'error': 'File path cannot be empty'
        }
    
    if content is None:
        content = ''
    
    # Security: Prevent path traversal
    if '..' in file_path or file_path.startswith('/'):
        return {
            'path': file_path,
            'filename': '',
            'append_success': False,
            'error': 'Invalid file path'
        }
    
    # Security: Limit content size
    if len(content) > 1024 * 1024:  # 1MB limit
        return {
            'path': file_path,
            'filename': '',
            'append_success': False,
            'error': 'Content too large (max 1MB)'
        }
    
    try:
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath('.')
        
        # Security: Only allow files in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'path': file_path,
                'filename': '',
                'append_success': False,
                'error': 'Access denied'
            }
        
        # Create parent directories if requested
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and create_dirs:
            os.makedirs(parent_dir, exist_ok=True)
        elif parent_dir and not os.path.exists(parent_dir):
            return {
                'path': file_path,
                'filename': '',
                'append_success': False,
                'error': 'Parent directory does not exist'
            }
        
        # Check if file exists and get current size
        file_existed = os.path.exists(abs_path)
        if file_existed:
            stat = os.stat(abs_path)
            if stat.st_size > 10 * 1024 * 1024:  # 10MB limit for existing files
                return {
                    'path': file_path,
                    'filename': '',
                    'append_success': False,
                    'error': 'Target file too large'
                }
        
        # Append to the file
        with open(abs_path, 'a', encoding=encoding) as f:
            f.write(content)
        
        # Get updated file stats
        stat = os.stat(abs_path)
        
        # Read the file to get line count
        with open(abs_path, 'r', encoding=encoding, errors='ignore') as f:
            full_content = f.read()
        
        line_count = full_content.count('\n') + 1
        
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            'size_bytes': stat.st_size,
            'line_count': line_count,
            'encoding': encoding,
            'append_success': True,
            'file_existed': file_existed
        }
        
    except Exception:
        return {
            'path': file_path,
            'filename': os.path.basename(file_path),
            'append_success': False,
            'error': 'Failed to append to file'
        }


@register_tool(name="CreateDirectory", is_core=False)
def createDirectory(dir_path: str, parents: bool = True) -> Dict[str, Any]:
    """
    Create a directory

    Args:
        dir_path: Path of the directory to create
        parents: Whether to create parent directories (default: True)

    Returns:
        Dictionary with operation result
    """
    # Input validation
    if not dir_path or not dir_path.strip():
        return {
            'path': '',
            'directory_name': '',
            'create_success': False,
            'error': 'Directory path cannot be empty'
        }
    
    # Security: Prevent path traversal
    if '..' in dir_path or dir_path.startswith('/'):
        return {
            'path': dir_path,
            'directory_name': '',
            'create_success': False,
            'error': 'Invalid directory path'
        }
    
    try:
        abs_path = os.path.abspath(dir_path)
        current_dir = os.path.abspath('.')
        
        # Security: Only allow directories in current directory and subdirectories
        if not abs_path.startswith(current_dir):
            return {
                'path': dir_path,
                'directory_name': '',
                'create_success': False,
                'error': 'Access denied'
            }
        
        os.makedirs(abs_path, exist_ok=parents)
        
        return {
            'path': dir_path,
            'directory_name': os.path.basename(dir_path),
            'create_success': True,
            'existed_before': os.path.exists(abs_path)
        }
        
    except Exception:
        return {
            'path': dir_path,
            'directory_name': os.path.basename(dir_path),
            'create_success': False,
            'error': 'Failed to create directory'
        }


def examples() -> List[str]:
    """
    Return example usage claims for LLM context
    These examples help the LLM understand when and how to use this tool
    """
    return [
        "writeFile('src/main.rs', 'fn main() { println!(\"Hello, World!\"); }') creates a new Rust main file",
        "writeFile('Cargo.toml', cargo_toml_content, create_dirs=True) creates Cargo.toml in project root",
        "writeFile('docs/design.md', design_content, backup=True) creates design document with backup of existing file",
        "appendFile('src/lib.rs', 'pub mod utils;') adds module declaration to existing Rust library file",
        "createDirectory('src/modules') creates new directory for source modules",
        "writeFile('src/game/board.rs', board_impl) creates board implementation file with directory creation",
        "writeFile('README.md', readme_content) creates project documentation file",
        "writeFile('tests/integration_test.rs', test_content) creates test file in tests directory",
        "appendFile('Cargo.toml', '\\n[dependencies]\\nrand = \"0.8\"') adds dependencies to Cargo.toml"
    ]


if __name__ == "__main__":
    # Test the write files functionality
    print("Testing writeFiles tool...")
    
    # Test writing a new file
    print("\n1. Writing new file:")
    test_content = """fn main() {
    println!("Hello from Rust!");
}"""
    result = writeFile("test_output/main.rs", test_content, create_dirs=True)
    if result['write_success']:
        print(f"   ✓ Created {result['filename']} ({result['line_count']} lines)")
    else:
        print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Test appending to file
    print("\n2. Appending to file:")
    append_content = "\n// This is an appended comment\n"
    result = appendFile("test_output/main.rs", append_content)
    if result['append_success']:
        print(f"   ✓ Appended to {result['filename']} (now {result['line_count']} lines)")
    else:
        print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Test creating directory
    print("\n3. Creating directory:")
    result = createDirectory("test_output/modules")
    if result['create_success']:
        print(f"   ✓ Created directory {result['directory_name']}")
    else:
        print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Clean up test files
    try:
        shutil.rmtree("test_output")
        print("\n✓ Cleaned up test files")
    except:
        pass
    
    print("\nExamples for LLM context:")
    for example in examples():
        print(f"- {example}")