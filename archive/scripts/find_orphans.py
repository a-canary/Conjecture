
import ast
import os
import sys
from pathlib import Path
from typing import Set, List, Tuple

def get_imports(file_path: Path) -> Set[Tuple[str, str, int]]:
    """Parse AST to find all imports in a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception as e:
        # print(f"Error parsing {file_path}: {e}")
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(("absolute", alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(("absolute", node.module, 0))
            elif node.level > 0:
                # relative import
                if node.module:
                    imports.add(("relative", node.module, node.level))
                else:
                    for alias in node.names:
                        imports.add(("relative_alias", alias.name, node.level))
    return imports

def resolve_import(import_name: str, current_file: Path, root: Path) -> Path:
    """Attempt to resolve an import string to a file path."""
    # 1. Absolute path check (relative to root)
    parts = import_name.split('.')
    potential_path = root.joinpath(*parts)
    
    # Check for dir/package (__init__.py)
    if potential_path.is_dir() and (potential_path / "__init__.py").exists():
        return potential_path / "__init__.py"
    
    # Check for .py file
    if potential_path.with_suffix(".py").exists():
        return potential_path.with_suffix(".py")
        
    # Check src/ prefix (common pattern)
    # If the import already starts with 'src', use it directly relative to root
    if parts[0] == "src":
         potential_path_direct = root / Path(*parts)
         if potential_path_direct.is_dir() and (potential_path_direct / "__init__.py").exists():
            return potential_path_direct / "__init__.py"
         if potential_path_direct.with_suffix(".py").exists():
            return potential_path_direct.with_suffix(".py")
    else:
        # If it doesn't start with src, try looking in src/
        potential_path_src = root / "src" / Path(*parts)
        if potential_path_src.is_dir() and (potential_path_src / "__init__.py").exists():
            return potential_path_src / "__init__.py"
        if potential_path_src.with_suffix(".py").exists():
            return potential_path_src.with_suffix(".py")

    return None

def scan_project(root: Path, entry_points: List[Path]):
    print(f"Scanning project root: {root}")
    print(f"Entry points: {[str(p.relative_to(root)) for p in entry_points]}")

    # 1. Index all Python files
    all_py_files = set(root.rglob("*.py"))
    # Filter out venv, .git, etc
    all_py_files = {p for p in all_py_files if ".venv" not in str(p) and "site-packages" not in str(p)}
    
    print(f"Total Python files found: {len(all_py_files)}")

    # 2. Build Reachability Graph
    reachable = set()
    queue = list(entry_points)
    processed = set()
    
    while queue:
        current = queue.pop(0).resolve()
        if current in processed:
            continue
        processed.add(current)
        reachable.add(current)
        
        # Get imports from this file
        raw_imports = get_imports(current)
        
        # Resolve imports to files
        for imp_type, imp_name, level in raw_imports:
            resolved = None
            if imp_type == "absolute":
                resolved = resolve_import(imp_name, current, root)
            elif imp_type == "relative":
                # from .foo import bar
                try:
                    base_dir = current.parents[level - 1]
                    parts = imp_name.split('.')
                    potential_path = base_dir.joinpath(*parts)
                    if potential_path.is_dir() and (potential_path / "__init__.py").exists():
                         resolved = potential_path / "__init__.py"
                    elif potential_path.with_suffix(".py").exists():
                         resolved = potential_path.with_suffix(".py")
                except IndexError:
                    pass
            elif imp_type == "relative_alias":
                 # from . import foo
                try:
                    base_dir = current.parents[level - 1]
                    parts = imp_name.split('.')
                    potential_path = base_dir.joinpath(*parts)
                    if potential_path.is_dir() and (potential_path / "__init__.py").exists():
                         resolved = potential_path / "__init__.py"
                    elif potential_path.with_suffix(".py").exists():
                         resolved = potential_path.with_suffix(".py")
                except IndexError:
                    pass

            if resolved:
                res_abs = resolved.resolve()
                if res_abs in {p.resolve() for p in all_py_files} and res_abs not in processed:
                    queue.append(resolved)

    # 3. Calculate Dangling
    # We compare absolute paths
    reachable_abs = result = {p.resolve() for p in reachable}
    all_abs = {p.resolve() for p in all_py_files}
    
    dangling = all_abs - reachable_abs
    
    print(f"\nReachable files: {len(reachable_abs)}")
    print(f"Dangling files: {len(dangling)}")
    print("\n--- POSSIBLE ORPHANS (Top 20) ---")
    for p in sorted(list(dangling))[:20]:
        print(f"  {p.relative_to(root.resolve())}")

if __name__ == "__main__":
    project_root = Path(os.getcwd())
    # Define Entry Points
    entries = [
        project_root / "src" / "conjecture.py",
        project_root / "src" / "cli" / "modular_cli.py",
    ]
    # Verify entries exist
    valid_entries = [e for e in entries if e.exists()]
    
    scan_project(project_root, valid_entries)
