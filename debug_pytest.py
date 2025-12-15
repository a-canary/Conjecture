import pytest
import sys
import traceback

with open('pytest_debug_out.txt', 'w', encoding='utf-8') as f:
    try:
        # Capture stdout/stderr
        sys.stdout = f
        sys.stderr = f
        print("Starting pytest run on tests/ directory...")
        # Removed --collect-only
        ret = pytest.main(["-v", "--tb=long", "-p", "no:warnings", "tests/"])
        print(f"Pytest finished with code {ret}")
    except Exception:
        traceback.print_exc(file=f)
