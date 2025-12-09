import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    print(f"\n{'='*20} Running {description} {'='*20}")
    try:
        # Run command and stream output
        result = subprocess.run(
            command, 
            check=False, # Don't raise exception on failure immediately
            shell=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False

def main():
    conjecture_root = Path(__file__).parent.parent
    
    tools = [
        ("ruff check .", "Ruff (Linter)"),
        ("mypy .", "Mypy (Type Checker)"),
        ("vulture .", "Vulture (Dead Code Finder)"),
        ("bandit -r src", "Bandit (Security Scanner)"),
        # Pylint can be noisy, maybe run only if explicitly asked or just run it.
        # "pylint src tests", "Pylint" 
    ]

    success = True
    for cmd, desc in tools:
        if not run_command(cmd, desc):
            success = False
            print(f"FAILED: {desc}")
        else:
            print(f"PASSED: {desc}")

    if not success:
        print("\nSome checks failed.")
        sys.exit(1)
    else:
        print("\nAll checks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
