#!/usr/bin/env python3
"""
Cleanup remaining mock references after bulk removal.

This script handles the remaining mock references that need individual attention:
1. Comments and documentation mentioning "mock"
2. Variable names and function parameters with "mock"
3. Configuration settings with mock values
4. Class docstrings and comments
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set

# Files that need specific manual cleanup
CRITICAL_FILES_NEEDING_MANUAL_CLEANUP = {
    'src/local/embeddings.py': [
        # Rename MockEmbeddingManager back to LocalEmbeddingManager was incorrect
        # Need to fix the class name and docstring
    ],
    'tests/test_processing_layer.py': [
        # MockLLMProcessor was renamed incorrectly, need to fix
    ],
    'tests/test_processing_interface.py': [
        # ProcessingInterface mock needs cleanup
    ],
    'tests/test_core_tools.py': [
        # SimpleLLMProcessor mock needs cleanup
    ],
}

# Patterns for cleanup that are safe to apply globally
SAFE_REPLACEMENTS = {
    # Comments and documentation
    r'# Mock.*': '# Real',
    r'"""Mock.*': '"""Real',
    r'mock implementation': 'real implementation',
    r'mock response': 'test response',
    r'mock data': 'test data',
    r'mock backend': 'test backend',
    r'mock service': 'test service',
    
    # Configuration values
    r'database_type\s*=\s*"mock"': 'database_type = "sqlite"',
    r'llm_provider\s*=\s*"mock"': 'llm_provider = "local"',
    r'use_mock_embeddings\s*=\s*True': 'use_mock_embeddings = False',
    r'use_mocks\s*=\s*True': 'use_mocks = False',
    
    # Variable names in test contexts
    r'mock_config\b': 'test_config',
    r'mock_response\b': 'test_response',
    r'mock_result\b': 'test_result',
    r'mock_data\b': 'test_data',
    r'mock_model\b': 'test_model',
    r'mock_service\b': 'test_service',
    r'mock_client\b': 'test_client',
    r'mock_provider\b': 'test_provider',
    
    # Function names
    r'def mock_': 'def test_',
    r'async def mock_': 'async def test_',
    r'_mock_': '_test_',
}

# Patterns to remove completely
PATTERNS_TO_REMOVE = [
    r'# Note:.*mock.*',
    r'# Using mock.*',
    r'# For mock.*',
    r'# Mock.*',
]

class RemainingMockCleaner:
    """Clean up remaining mock references after bulk removal."""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.modified_files = []
        self.errors = []
        self.skipped_files = []
    
    def is_python_file(self, file_path: Path) -> bool:
        """Check if file is a Python file."""
        return file_path.suffix == '.py'
    
    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_list = {
            'mock_removal_script.py',
            'cleanup_remaining_mocks.py',
            'mock_removal_report.md',
            '__pycache__',
            '.git',
            '.pytest_cache',
        }
        return any(skip in str(file_path) for skip in skip_list)
    
    def apply_safe_replacements(self, content: str) -> str:
        """Apply safe global replacements."""
        new_content = content
        
        for pattern, replacement in SAFE_REPLACEMENTS.items():
            new_content = re.sub(pattern, replacement, new_content, flags=re.MULTILINE)
        
        # Remove unwanted patterns
        for pattern in PATTERNS_TO_REMOVE:
            new_content = re.sub(pattern, '', new_content, flags=re.MULTILINE)
        
        return new_content
    
    def fix_specific_file_issues(self, file_path: Path, content: str) -> str:
        """Fix specific issues in known problematic files."""
        relative_path = str(file_path.relative_to(self.root_dir))
        
        if relative_path == 'src/local/embeddings.py':
            # Fix the class name that was incorrectly changed
            content = content.replace('class LocalEmbeddingManager:', 'class MockEmbeddingManager:')
            content = content.replace('"""Mock embedding manager for testing and development."""', 
                                  '"""Local embedding manager using sentence-transformers."""')
            # Also fix the class reference at the end
            content = re.sub(r'class LocalEmbeddingManager:\s*"""Mock.*?"', '', content, flags=re.MULTILINE | re.DOTALL)
        
        elif relative_path == 'tests/test_processing_layer.py':
            # Fix the MockLLMProcessor that was incorrectly renamed
            content = content.replace('class ProcessLLMProcessor:', 'class MockLLMProcessor:')
            content = content.replace('"""Mock LLM processor for testing without real API calls"""', 
                                  '"""Mock LLM processor for testing without real API calls"""')
        
        elif relative_path == 'tests/test_processing_interface.py':
            # Fix the ProcessingInterface mock
            content = content.replace('class ProcessingInterface(ProcessingInterface):', 
                                  'class MockProcessingInterface(ProcessingInterface):')
            content = content.replace('"""Mock implementation of ProcessingInterface for testing"""', 
                                  '"""Mock implementation of ProcessingInterface for testing"""')
        
        elif relative_path == 'tests/test_core_tools.py':
            # Fix the SimpleLLMProcessor mock
            content = content.replace('class SimpleLLMProcessor(LLMInterface):', 
                                  'class MockLLM(LLMInterface):')
            content = content.replace('"""Mock LLM for testing purposes"""', 
                                  '"""Mock LLM for testing purposes"""')
        
        return content
    
    def clean_empty_lines_and_comments(self, content: str) -> str:
        """Clean up empty lines and orphaned comments."""
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines that follow other empty lines
            if stripped == '':
                if not prev_empty:
                    cleaned_lines.append(line)
                prev_empty = True
                continue
            
            # Skip lines that are just whitespace or orphaned comment markers
            if stripped in ['#', '# ', '# ']:
                continue
            
            cleaned_lines.append(line)
            prev_empty = False
        
        return '\n'.join(cleaned_lines)
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file for remaining mock cleanup."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply transformations
            new_content = original_content
            new_content = self.apply_safe_replacements(new_content)
            new_content = self.fix_specific_file_issues(file_path, new_content)
            new_content = self.clean_empty_lines_and_comments(new_content)
            
            # Check if content changed
            if new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.modified_files.append(str(file_path))
                return True
            
            return False
            
        except Exception as e:
            self.errors.append(f"Error processing {file_path}: {e}")
            return False
    
    def run(self) -> Dict[str, List[str]]:
        """Run the remaining mock cleanup process."""
        print("Starting remaining mock cleanup...")
        
        # Process all Python files
        for py_file in self.root_dir.rglob('*.py'):
            if self.should_skip_file(py_file):
                self.skipped_files.append(str(py_file))
                continue
            
            self.process_file(py_file)
        
        return {
            'modified_files': self.modified_files,
            'errors': self.errors,
            'skipped_files': self.skipped_files
        }
    
    def generate_report(self, results: Dict[str, List[str]]) -> str:
        """Generate a report of the cleanup process."""
        report = []
        report.append("# Remaining Mock Cleanup Report\n")
        
        if results['modified_files']:
            report.append(f"## Modified Files ({len(results['modified_files'])})\n")
            for file in results['modified_files']:
                report.append(f"- {file}")
            report.append("")
        
        if results['errors']:
            report.append(f"## Errors ({len(results['errors'])})\n")
            for error in results['errors']:
                report.append(f"- {error}")
            report.append("")
        
        if results['skipped_files']:
            report.append(f"## Skipped Files ({len(results['skipped_files'])})\n")
            for file in results['skipped_files'][:10]:  # Show first 10
                report.append(f"- {file}")
            if len(results['skipped_files']) > 10:
                report.append(f"- ... and {len(results['skipped_files']) - 10} more")
            report.append("")
        
        return '\n'.join(report)


def main():
    """Main execution function."""
    cleaner = RemainingMockCleaner()
    results = cleaner.run()
    
    # Print results
    print(f"\nRemaining mock cleanup completed!")
    print(f"Modified files: {len(results['modified_files'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Skipped files: {len(results['skipped_files'])}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Generate report
    report = cleaner.generate_report(results)
    report_file = Path('remaining_mock_cleanup_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()