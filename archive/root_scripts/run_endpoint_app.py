#!/usr/bin/env python3
"""
Startup script for Conjecture EndPoint App
Convenient launcher with development and production modes
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import websockets
        print("All dependencies found")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install with: pip install -r requirements_endpoint.txt")
        return False

def setup_environment():
    """Setup environment for endpoint app"""
    # Ensure PYTHONPATH includes project root
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ["PYTHONPATH"] = str(project_root)
    
    print(f"Project root: {project_root}")
    print(f"Python path includes: {project_root}")

def run_endpoint_app(host="0.0.0.0", port=8000, reload=False, log_level="info"):
    """Run the endpoint app"""
    print(f"Starting Conjecture EndPoint App...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print()
    
    # Run the endpoint app
    cmd = [
        sys.executable, "src/endpoint_app.py",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level
    ]
    
    if reload:
        cmd.append("--reload")
        print("Auto-reload enabled for development")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start endpoint app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Conjecture EndPoint App Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_endpoint_app.py                    # Start with defaults (localhost:8000)
  python run_endpoint_app.py --port 8080       # Start on port 8080
  python run_endpoint_app.py --dev              # Development mode with auto-reload
  python run_endpoint_app.py --prod             # Production mode
        """
    )
    
    # Mode options
    parser.add_argument("--dev", action="store_true", help="Development mode with auto-reload")
    parser.add_argument("--prod", action="store_true", help="Production mode")
    
    # Server options
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level (default: info)")
    
    # Utility options
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--test", action="store_true", help="Run tests after starting")
    
    args = parser.parse_args()
    
    # Handle dependency check
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies satisfied")
        else:
            sys.exit(1)
        return
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine mode settings
    reload_mode = args.dev
    log_level = "debug" if args.dev else ("warning" if args.prod else args.log_level)
    
    if args.prod:
        print("🏭 Production mode enabled")
        reload_mode = False
        log_level = "warning"
    elif args.dev:
        print("🛠️ Development mode enabled")
        reload_mode = True
        log_level = "debug"
    
    # Run the endpoint app
    try:
        run_endpoint_app(
            host=args.host,
            port=args.port,
            reload=reload_mode,
            log_level=log_level
        )
        
        # Run tests if requested
        if args.test:
            print("\n🧪 Running endpoint tests...")
            import subprocess
            test_result = subprocess.run([
                sys.executable, "test_endpoint_app.py",
                "--url", f"http://{args.host}:{args.port}"
            ], capture_output=True, text=True)
            
            if test_result.returncode == 0:
                print("✅ All tests passed")
            else:
                print("❌ Some tests failed")
                print(test_result.stdout)
                print(test_result.stderr)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()