#!/usr/bin/env python3
"""
Conjecture Configuration Wizard Entry Point

Run this script to start the streamlined configuration wizard with diagnostics.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from src.config.streamlined_wizard import run_wizard
    
    if __name__ == "__main__":
        print("🚀 Starting Conjecture Configuration Wizard...")
        
        # Run the wizard
        success = run_wizard()
        
        if success:
            print("\\n✅ Wizard completed successfully!")
            sys.exit(0)
        else:
            print("\\n❌ Wizard was cancelled or failed")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the Conjecture project directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)