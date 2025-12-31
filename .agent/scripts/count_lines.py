import os
import sys
from pathlib import Path

def count_lines_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def count_lines(directory):
    total_lines = 0
    file_count = 0
    extensions = {'.py', '.json', '.yaml', '.yml', '.md', '.txt'}

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

        for file in files:
            if Path(file).suffix in extensions:
                filepath = os.path.join(root, file)
                lines = count_lines_in_file(filepath)
                total_lines += lines
                file_count += 1

    return total_lines, file_count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_lines.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    total_lines, file_count = count_lines(directory)
    print(f"{total_lines}")
