#!/usr/bin/env python3
"""
Conjecture Provider Startup Script
Starts the Conjecture local LLM provider server on port 5678
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

def check_port_available(port):
    """Check if port is available"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except:
            return False

def start_provider():
    """Start the Conjecture provider server"""
    print("Starting Conjecture Local Provider...")

    # Check if port 5680 is available
    if not check_port_available(5680):
        print(
            "Port 5680 is already in use. Please stop the existing service or use a different port."
        )
        sys.exit(1)

    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root)

    # Start the provider
    provider_script = project_root / "src" / "providers" / "conjecture_provider.py"

    if not provider_script.exists():
        print(f"❌ Provider script not found: {provider_script}")
        sys.exit(1)

    try:
        print(f"Starting provider from: {provider_script}")
        print("   Available endpoints:")
        print("   POST /v1/chat/completions - Main chat endpoint")
        print("   POST /tools/tell_user - TellUser tool")
        print("   POST /tools/ask_user - AskUser tool")
        print("   GET /models - List models")
        print("   GET /health - Health check")
        print("\nStarting server on http://127.0.0.1:5678")
        print("Press Ctrl+C to stop the server")

        # Run the provider
        subprocess.run([sys.executable, str(provider_script)], cwd=project_root)

    except KeyboardInterrupt:
        print("\nStopping Conjecture provider...")
    except Exception as e:
        print(f"Error starting provider: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_provider()
