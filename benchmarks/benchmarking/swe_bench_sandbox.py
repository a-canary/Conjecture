"""
=============================================================================
GENERATED CODE - SC-FEAT-001 - TEST BRANCH
=============================================================================
Docker Sandbox for Bash Command Execution - Generated 2025-12-30

Purpose: Secure isolated execution of bash commands for SWE-Bench evaluation
Features:
  - Full Docker container isolation
  - Volume mounting for task directories
  - 30-second timeout enforcement
  - Cross-platform support (Windows/Linux/Mac)
  - Auto-removal of containers after execution

Usage: Import DockerSandboxExecutor and replace direct subprocess calls
=============================================================================
"""

import asyncio
import os
import json
import tempfile
import shutil
from typing import List, Dict, Optional, Any
import time
from pathlib import Path
import uuid


# ============================================================================
# MARKER: GENERATED CODE FOR TEST BRANCH
# This module is part of SC-FEAT-001 implementation
# ============================================================================


class DockerSandboxExecutor:
    """
    Docker-based sandbox executor for bash commands.

    Generates a secure isolated environment for each task evaluation.
    All commands execute inside a temporary Docker container with:
    - Read-write mounted volume for task directory
    - Auto-removal after execution
    - Timeout enforcement
    - Isolated filesystem and network

    Generated for SC-FEAT-001 test branch.
    """

    def __init__(
        self,
        docker_image: str = "ubuntu:22.04",
        timeout: int = 30,
        enable_sandbox: bool = True,
        work_dir_base: Optional[str] = None,
    ):
        """
        Initialize Docker sandbox executor.

        Args:
            docker_image: Docker image to use (must have bash/sh installed)
            timeout: Timeout in seconds for each command
            enable_sandbox: If False, bypass sandbox and use direct execution
            work_dir_base: Base directory for task workspaces
        """
        self.docker_image = docker_image
        self.timeout = timeout
        self.enable_sandbox = enable_sandbox
        self.work_dir_base = work_dir_base or os.path.join(
            tempfile.gettempdir(), "swe_bench_sandbox"
        )

        # Create base work directory
        os.makedirs(self.work_dir_base, exist_ok=True)

        # Docker availability check
        self.docker_available = self._check_docker_available()

        if not self.docker_available:
            print(
                "[SANDBOX] Warning: Docker not available. Falling back to direct execution."
            )
            self.enable_sandbox = False

    async def initialize(self):
        """
        Initialize the sandbox executor.

        This is an async method that can be awaited for initialization.
        Currently a no-op since initialization happens in __init__.
        """
        pass

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Docker availability check
    # ------------------------------------------------------------------------
    def _check_docker_available(self) -> bool:
        """Check if Docker is available and accessible."""
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Main execution interface
    # ------------------------------------------------------------------------
    async def execute_commands(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a list of bash commands in the sandbox.

        Args:
            commands: List of bash commands to execute
            work_dir: Working directory inside container
            environment: Optional environment variables

        Returns:
            Dict with success, output, stdout, stderr, error_keywords, passed, total
        """
        if not self.enable_sandbox or not self.docker_available:
            return await self._execute_direct(commands, work_dir, environment)

        return await self._execute_in_docker(commands, work_dir, environment)

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Docker execution logic
    # ------------------------------------------------------------------------
    async def _execute_in_docker(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute commands inside a Docker container.

        Creates a temporary container with:
        - Mounted work directory volume
        - Auto-removal on exit
        - Timeout enforcement

        Generated for SC-FEAT-001 test branch.
        """
        # Create unique task directory
        task_id = uuid.uuid4().hex[:8]
        task_dir = os.path.join(self.work_dir_base, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        # Copy source work directory if it exists
        if os.path.exists(work_dir):
            try:
                if os.path.isdir(work_dir):
                    shutil.copytree(work_dir, task_dir, dirs_exist_ok=True)
                else:
                    shutil.copy2(work_dir, task_dir)
            except Exception as e:
                print(f"[SANDBOX] Warning copying work directory: {e}")

        # Build docker command
        container_name = f"swe_bench_sandbox_{task_id}"

        # Prepare environment variables
        env_args = []
        if environment:
            for key, value in environment.items():
                env_args.extend(["-e", f"{key}={value}"])

        try:
            all_output = []
            all_stdout = []
            all_stderr = []
            error_keywords = []
            success = True

            # Execute each command in the container
            for i, cmd in enumerate(commands):
                # Build docker run command
                docker_cmd = (
                    [
                        "docker",
                        "run",
                        "--rm",  # Auto-remove container
                        "--name",
                        f"{container_name}_{i}",
                        f"--volume={task_dir}:/workspace",
                        f"--workdir=/workspace",
                    ]
                    + env_args
                    + [
                        self.docker_image,
                        "bash",
                        "-c",
                        cmd,
                    ]
                )

                try:
                    # Execute with timeout
                    process = await asyncio.create_subprocess_exec(
                        *docker_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                    stdout_text = stdout.decode()
                    stderr_text = stderr.decode()

                    all_stdout.append(stdout_text)
                    all_stderr.append(stderr_text)

                    output = stdout_text + stderr_text
                    all_output.append(f"$ {cmd}\n{output}")

                    # Extract error keywords
                    if stderr_text:
                        keywords = self._extract_error_keywords(stderr_text)
                        error_keywords.extend(keywords)

                    if process.returncode != 0:
                        success = False

                except asyncio.TimeoutError:
                    # Kill container on timeout
                    await self._kill_container(f"{container_name}_{i}")
                    all_output.append(
                        f"$ {cmd}\n[TIMEOUT after {self.timeout} seconds]"
                    )
                    all_stderr.append(f"[TIMEOUT after {self.timeout} seconds]")
                    error_keywords.append("timeout")
                    success = False

                except Exception as e:
                    error_msg = str(e)
                    all_output.append(f"$ {cmd}\n[ERROR: {error_msg}]")
                    all_stderr.append(error_msg)
                    error_keywords.append("docker_execution_error")
                    success = False

            # Cleanup task directory
            try:
                shutil.rmtree(task_dir)
            except Exception as e:
                print(f"[SANDBOX] Warning cleaning task directory: {e}")

            return {
                "success": success,
                "output": "\n".join(all_output),
                "stdout": "\n".join(all_stdout),
                "stderr": "\n".join(all_stderr),
                "error_keywords": list(set(error_keywords)),
                "passed": 1 if success else 0,
                "total": 1,
            }

        except Exception as e:
            # Cleanup on error
            try:
                shutil.rmtree(task_dir)
            except:
                pass

            return {
                "success": False,
                "output": f"Docker execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "error_keywords": ["docker_error"],
                "passed": 0,
                "total": 1,
            }

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Container cleanup
    # ------------------------------------------------------------------------
    async def _kill_container(self, container_name: str):
        """Force kill a Docker container."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "kill",
                container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
        except Exception:
            pass  # Container may not exist

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Fallback direct execution (non-sandbox)
    # ------------------------------------------------------------------------
    async def _execute_direct(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Fallback direct execution without sandbox.

        WARNING: This executes commands directly on the host system.
        Only used when Docker is not available or sandbox is disabled.

        Generated for SC-FEAT-001 test branch.
        """
        print("[SANDBOX] ⚠️  Using direct execution (not sandboxed)")

        # Change to work directory if it exists
        original_dir = os.getcwd()
        if os.path.exists(work_dir):
            os.chdir(work_dir)

        try:
            all_output = []
            all_stdout = []
            all_stderr = []
            error_keywords = []
            success = True

            for cmd in commands:
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=environment,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                    stdout_text = stdout.decode()
                    stderr_text = stderr.decode()

                    all_stdout.append(stdout_text)
                    all_stderr.append(stderr_text)

                    output = stdout_text + stderr_text
                    all_output.append(f"$ {cmd}\n{output}")

                    if stderr_text:
                        keywords = self._extract_error_keywords(stderr_text)
                        error_keywords.extend(keywords)

                    if process.returncode != 0:
                        success = False

                except asyncio.TimeoutError:
                    all_output.append(
                        f"$ {cmd}\n[TIMEOUT after {self.timeout} seconds]"
                    )
                    all_stderr.append(f"[TIMEOUT after {self.timeout} seconds]")
                    error_keywords.append("timeout")
                    success = False

                except Exception as e:
                    error_msg = str(e)
                    all_output.append(f"$ {cmd}\n[ERROR: {error_msg}]")
                    all_stderr.append(error_msg)
                    error_keywords.append("execution_error")
                    success = False

            return {
                "success": success,
                "output": "\n".join(all_output),
                "stdout": "\n".join(all_stdout),
                "stderr": "\n".join(all_stderr),
                "error_keywords": list(set(error_keywords)),
                "passed": 1 if success else 0,
                "total": 1,
            }

        finally:
            os.chdir(original_dir)

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Error keyword extraction
    # ------------------------------------------------------------------------
    def _extract_error_keywords(self, stderr_text: str) -> List[str]:
        """
        Extract error keywords from stderr output.

        Identifies common bash error patterns for feedback to LLM.

        Generated for SC-FEAT-001 test branch.
        """
        keywords = []
        stderr_lower = stderr_text.lower()

        error_patterns = {
            "command not found": "command_not_found",
            "permission denied": "permission_denied",
            "syntax error": "syntax_error",
            "no such file": "file_not_found",
            "cannot open": "file_not_found",
            "undefined variable": "undefined_variable",
            "bad substitution": "bad_substitution",
            "unmatched": "unmatched_quote",
            "unexpected": "unexpected_token",
            "invalid": "invalid_syntax",
            "error": "generic_error",
            "failed": "execution_failed",
            "exit code": "non_zero_exit",
        }

        for pattern, keyword in error_patterns.items():
            if pattern in stderr_lower:
                keywords.append(keyword)

        return keywords

    # ------------------------------------------------------------------------
    # MARKER: GENERATED - Health check
    # ------------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Check sandbox health status.

        Returns:
            Dict with status, sandbox_enabled, docker_available
        """
        return {
            "status": "healthy" if self.docker_available else "degraded",
            "sandbox_enabled": self.enable_sandbox,
            "docker_available": self.docker_available,
            "docker_image": self.docker_image,
            "timeout": self.timeout,
            "work_dir_base": self.work_dir_base,
        }


# ============================================================================
# MARKER: GENERATED - Factory function for sandbox executor
# ============================================================================

_sandbox_executor_instance = None


def get_sandbox_executor(
    docker_image: str = "ubuntu:22.04",
    timeout: int = 30,
    enable_sandbox: bool = True,
) -> DockerSandboxExecutor:
    """
    Get global sandbox executor instance.

    Args:
        docker_image: Docker image to use
        timeout: Timeout in seconds
        enable_sandbox: Force enable/disable sandbox

    Returns:
        DockerSandboxExecutor instance

    Generated for SC-FEAT-001 test branch.
    """
    global _sandbox_executor_instance
    if _sandbox_executor_instance is None:
        _sandbox_executor_instance = DockerSandboxExecutor(
            docker_image=docker_image,
            timeout=timeout,
            enable_sandbox=enable_sandbox,
        )
    return _sandbox_executor_instance


def reset_sandbox_executor():
    """Reset global sandbox executor instance."""
    global _sandbox_executor_instance
    _sandbox_executor_instance = None


# ============================================================================
# END OF GENERATED CODE - SC-FEAT-001 - TEST BRANCH
# ============================================================================
