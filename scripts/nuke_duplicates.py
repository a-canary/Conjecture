
import ast
import os
import sys
import hashlib
from pathlib import Path
from typing import Set, List, Tuple, Dict
from datetime import datetime

# --- Reachability Logic (from find_orphans.py) ---

def get_imports(file_path: Path) -> Set[Tuple[str, str, int]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
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
                if node.module:
                    imports.add(("relative", node.module, node.level))
                else:
                    for alias in node.names:
                        imports.add(("relative_alias", alias.name, node.level))
    return imports

def resolve_import(import_name: str, current_file: Path, root: Path) -> Path:
    parts = import_name.split('.')
    potential_path = root.joinpath(*parts)
    
    if potential_path.is_dir() and (potential_path / "__init__.py").exists():
        return potential_path / "__init__.py"
    if potential_path.with_suffix(".py").exists():
        return potential_path.with_suffix(".py")
        
    if parts[0] == "src":
         potential_path_direct = root / Path(*parts)
         if potential_path_direct.is_dir() and (potential_path_direct / "__init__.py").exists():
            return potential_path_direct / "__init__.py"
         if potential_path_direct.with_suffix(".py").exists():
            return potential_path_direct.with_suffix(".py")
    else:
        potential_path_src = root / "src" / Path(*parts)
        if potential_path_src.is_dir() and (potential_path_src / "__init__.py").exists():
            return potential_path_src / "__init__.py"
        if potential_path_src.with_suffix(".py").exists():
            return potential_path_src.with_suffix(".py")
    return None

def get_reachable_files(root: Path, entry_points: List[Path]) -> Set[Path]:
    all_py_files = set(root.rglob("*.py"))
    all_py_files = {p for p in all_py_files if ".venv" not in str(p) and "site-packages" not in str(p)}
    
    reachable = set()
    queue = list(entry_points)
    processed = set()
    
    while queue:
        current = queue.pop(0).resolve()
        if current in processed:
            continue
        processed.add(current)
        reachable.add(current)
        
        raw_imports = get_imports(current)
        for imp_type, imp_name, level in raw_imports:
            resolved = None
            if imp_type == "absolute":
                resolved = resolve_import(imp_name, current, root)
            elif imp_type == "relative":
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
    return {p.resolve() for p in reachable}

# --- Hashing Logic ---

def get_file_content(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def get_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

# --- Main Nuke Logic ---

def nuke_dead_duplicates(root: Path):
    print(f"Scanning project root: {root}")
    
    # 1. Identify Reachable Files
    entries = [
        root / "src" / "conjecture.py",
        root / "src" / "cli" / "modular_cli.py",
    ]
    valid_entries = [e for e in entries if e.exists()]
    print("Calculating reachability graph...")
    reachable_set = get_reachable_files(root, valid_entries)
    print(f"Reachable files: {len(reachable_set)}")

    # 2. Hash All Files
    all_files = list(root.rglob("*.py"))
    all_files = [p for p in all_files if ".venv" not in str(p) and "site-packages" not in str(p)]
    
    print(f"Hashing {len(all_files)} files...")
    hash_map: Dict[str, List[Path]] = {}
    
    for p in all_files:
        try:
            content = get_file_content(p)
            h = get_hash(content)
            if h not in hash_map:
                hash_map[h] = []
            hash_map[h].append(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")

    # 3. Find Candidates
    deleted_count = 0
    preserved_dirs = ["core_tools", "experiments", "archive"]
    
    print("\n--- DELETION LOG ---")
    
    for h, group in hash_map.items():
        if len(group) < 2:
            continue
            
        # We have duplication
        # Sort by modification time (oldest first)
        group.sort(key=lambda p: os.path.getmtime(p))
        
        # Identify "Keepers" (Reachable files)
        keepers = [p for p in group if p.resolve() in reachable_set]
        orphans = [p for p in group if p.resolve() not in reachable_set]
        
        if not orphans:
            continue
            
        # Strategy:
        # If we have Keepers, any Orphan that matches a Keeper is basically a dead copy -> Delete Orphan
        # If we have NO Keepers, but Multiple Orphans -> Delete all but the Newest Orphan? (Risky, user said "older")
        
        # Let's trust the "Delete Dead Duplicate" instruction strictly: 
        # Only delete an Orphan if it duplicates another file (Reachable OR Orphan) and is OLDER.
        
        # Actually, simpler:
        # The group is sorted by AGE (Oldest -> Newest).
        # We want to keep the "Best" file.
        # Preference: Reachable > Newest Orphan.
        
        best_file = None
        if keepers:
            best_file = keepers[-1] # Pick the newest reachable one as 'primary' just in case?
            # Actually any reachable file makes the orphans Redundant.
        else:
            best_file = group[-1] # The newest file
            
        for candidate in group:
            if candidate == best_file:
                continue
                
            # Check constraints
            is_protected = any(d in str(candidate) for d in preserved_dirs)
            if is_protected:
                print(f"SKIPPING Protected: {candidate.relative_to(root)}")
                continue
                
            if candidate.resolve() in reachable_set:
                # Iterate logic: we picked a best_file. If multiple reachable, we might delete reachable generic dupes? 
                # No, that's dangerous. "Dead Code" means Unreachable.
                # Do NOT delete valid reachable code even if duplicate, unless we are very sure.
                # User constraint: "Start with dead code"
                continue
                
            # It is Dead (Unreachable). It is a Duplicate of best_file. It is NOT protected.
            # And since group is sorted by time, if best_file is Newest, candidate (if time differs) is older.
            
            print(f"DELETING: {candidate.relative_to(root)}")
            print(f"  -> Duplicate of: {best_file.relative_to(root)}")
            print(f"  -> Status: Orphan vs {'Reachable' if best_file in keepers else 'Orphan'}")
            
            try:
                os.remove(candidate)
                deleted_count += 1
            except Exception as e:
                print(f"  FAILED to delete: {e}")

    print(f"\nTotal files deleted: {deleted_count}")

if __name__ == "__main__":
    nuke_dead_duplicates(Path(os.getcwd()))
