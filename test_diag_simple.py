import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from src.config.diagnostics import SystemDiagnostics
    
    print("Testing SystemDagnostics...")
    diagnostics = SystemDiagnostics()
    results = diagnostics.run_all_diagnostics()
    
    print(f"Overall status: {results['summary']['overall_status']}")
    print(f"Total checks: {results['summary']['total_checks']}")
    print("SUCCESS: All diagnostics completed")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()