#!/usr/bin/env python3
"""
Endpoint Manager for Conjecture
Manages endpoint app lifecycle as subprocess with proper startup/shutdown handling
"""

import asyncio
import subprocess
import sys
import time
import signal
import requests
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EndpointManager:
    """Manages Conjecture EndPoint App as a subprocess"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8001, 
                 log_level: str = "info", project_root: Optional[Path] = None):
        self.host = host
        self.port = port
        self.log_level = log_level
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.base_url = f"http://{host}:{port}"
        self.health_url = f"{self.base_url}/health"
        
        self.process: Optional[subprocess.Popen] = None
        self.startup_timeout = 60  # seconds
        self.shutdown_timeout = 10  # seconds
        self.health_check_interval = 2  # seconds
        self.max_health_retries = 30
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def _setup_environment(self):
        """Setup environment for the endpoint app"""
        # Ensure PYTHONPATH includes project root
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.project_root)
        return env
    
    async def start(self) -> bool:
        """Start the endpoint app as subprocess"""
        if self.process and self.process.poll() is None:
            logger.warning("Endpoint app is already running")
            return await self.health_check()
        
        logger.info(f"Starting Conjecture EndPoint App on {self.base_url}")
        
        # Prepare command
        cmd = [
            sys.executable, 
            str(self.project_root / "src" / "endpoint_app.py"),
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", self.log_level
        ]
        
        # Setup environment
        env = self._setup_environment()
        
        try:
            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                env=env,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"Started endpoint app with PID: {self.process.pid}")
            
            # Wait for health check
            if await self._wait_for_health():
                logger.info("✅ Endpoint app started successfully")
                return True
            else:
                logger.error("❌ Endpoint app failed health check")
                await self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start endpoint app: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the endpoint app gracefully"""
        if not self.process or self.process.poll() is not None:
            logger.info("Endpoint app is not running")
            return True
        
        logger.info("Stopping Conjecture EndPoint App...")
        
        try:
            # Try graceful shutdown first
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=self.shutdown_timeout)
                logger.info("✅ Endpoint app stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                logger.warning("Graceful shutdown timeout, forcing termination")
                self.process.kill()
                self.process.wait()
                logger.info("✅ Endpoint app force-terminated")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping endpoint app: {e}")
            return False
        finally:
            self.process = None
    
    async def restart(self) -> bool:
        """Restart the endpoint app"""
        logger.info("Restarting Conjecture EndPoint App...")
        await self.stop()
        await asyncio.sleep(2)  # Brief pause between stop and start
        return await self.start()
    
    async def health_check(self) -> bool:
        """Check if the endpoint app is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200 and response.json().get("success", False)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def _wait_for_health(self) -> bool:
        """Wait for the endpoint app to become healthy"""
        logger.info("Waiting for endpoint app to become healthy...")
        
        for attempt in range(self.max_health_retries):
            if await self.health_check():
                return True
            
            logger.debug(f"Health check attempt {attempt + 1}/{self.max_health_retries} failed")
            await asyncio.sleep(self.health_check_interval)
        
        return False
    
    def is_running(self) -> bool:
        """Check if the process is running"""
        return self.process is not None and self.process.poll() is None
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get information about the running process"""
        if not self.process:
            return {"status": "not_running"}
        
        return {
            "status": "running" if self.process.poll() is None else "stopped",
            "pid": self.process.pid,
            "returncode": self.process.returncode,
            "base_url": self.base_url,
            "health_url": self.health_url
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# Context manager for temporary endpoint usage
class TemporaryEndpoint:
    """Context manager for temporary endpoint usage during tests"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8001, 
                 log_level: str = "warning", project_root: Optional[Path] = None):
        self.manager = EndpointManager(host, port, log_level, project_root)
    
    async def __aenter__(self):
        """Start endpoint for temporary use"""
        success = await self.manager.start()
        if not success:
            raise RuntimeError("Failed to start endpoint app")
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop endpoint after use"""
        await self.manager.stop()


# Utility function for common usage patterns
async def with_endpoint(func, host: str = "127.0.0.1", port: int = 8001, 
                      log_level: str = "warning", **kwargs):
    """
    Run a function with a temporary endpoint
    
    Args:
        func: Async function that takes endpoint_manager as first argument
        host: Endpoint host
        port: Endpoint port
        log_level: Log level for endpoint
        **kwargs: Additional arguments to pass to func
    
    Returns:
        Result of func
    """
    async with TemporaryEndpoint(host, port, log_level) as manager:
        return await func(manager, **kwargs)


# Signal handling for graceful shutdown
def setup_signal_handlers(manager: EndpointManager):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(manager.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Import os for environment setup
import os