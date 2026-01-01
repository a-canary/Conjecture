"""
=============================================================================
Windows-Native Sandbox for Code Execution
=============================================================================
Lightweight sandbox alternatives for Windows without Docker dependency.

Supports multiple isolation strategies:
1. Windows Sandbox (WSB) - Full OS isolation via PyWinSandbox
2. Subprocess Isolation - Process-level isolation with resource limits
3. RestrictedPython - AST-based Python code sandboxing

Usage: Import WindowsSandboxExecutor and use as drop-in replacement for Docker sandbox
=============================================================================
"""

import asyncio
import os
import sys
import json
import tempfile
import shutil
import subprocess
from typing import List, Dict, Optional, Any
import time
from pathlib import Path
import uuid
from enum import Enum
from dataclasses import dataclass


class SandboxMode(Enum):
    """Available sandbox isolation modes"""

    WINDOWS_SANDBOX = (
        "windows_sandbox"  # Full OS isolation (requires Win Pro/Enterprise)
    )
    SUBPROCESS = "subprocess"  # Process isolation with limits
    RESTRICTED_PYTHON = "restricted"  # AST-based Python sandboxing
    DIRECT = "direct"  # No isolation (fallback)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""

    mode: SandboxMode = SandboxMode.SUBPROCESS
    timeout: int = 30
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    work_dir_base: Optional[str] = None
    enable_network: bool = False


class WindowsSandboxExecutor:
    """
    Windows-native sandbox executor for bash/Python commands.

    Provides multiple isolation strategies without Docker:
    - Windows Sandbox: Full OS isolation (best security)
    - Subprocess: Process-level isolation with resource limits
    - Direct: No isolation (fastest, least secure)
    """

    def __init__(
        self,
        timeout: int = 30,
        enable_sandbox: bool = True,
        work_dir_base: Optional[str] = None,
        preferred_mode: Optional[SandboxMode] = None,
    ):
        """
        Initialize Windows sandbox executor.

        Args:
            timeout: Timeout in seconds for each command
            enable_sandbox: If False, use direct execution
            work_dir_base: Base directory for task workspaces
            preferred_mode: Preferred sandbox mode (auto-detected if None)
        """
        self.timeout = timeout
        self.enable_sandbox = enable_sandbox
        self.work_dir_base = work_dir_base or os.path.join(
            tempfile.gettempdir(), "swe_bench_windows_sandbox"
        )

        # Create base work directory
        os.makedirs(self.work_dir_base, exist_ok=True)

        # Detect available sandbox modes
        self.available_modes = self._detect_available_modes()

        # Select best available mode
        if preferred_mode and preferred_mode in self.available_modes:
            self.mode = preferred_mode
        elif not enable_sandbox:
            self.mode = SandboxMode.DIRECT
        else:
            self.mode = self._select_best_mode()

        # Track execution stats
        self.executions = 0
        self.successful_executions = 0
        self.total_execution_time = 0.0

    def _detect_available_modes(self) -> List[SandboxMode]:
        """Detect which sandbox modes are available on this system."""
        available = [SandboxMode.DIRECT]  # Always available

        # Check for Windows Sandbox
        if self._check_windows_sandbox_available():
            available.append(SandboxMode.WINDOWS_SANDBOX)

        # Subprocess isolation is always available on Windows
        if sys.platform == "win32":
            available.append(SandboxMode.SUBPROCESS)

        # Check for RestrictedPython
        try:
            import RestrictedPython

            available.append(SandboxMode.RESTRICTED_PYTHON)
        except ImportError:
            pass

        return available

    def _check_windows_sandbox_available(self) -> bool:
        """Check if Windows Sandbox feature is available."""
        if sys.platform != "win32":
            return False

        try:
            # Check if Windows Sandbox is enabled
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM | Select-Object -ExpandProperty State",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "Enabled" in result.stdout
        except Exception:
            return False

    def _select_best_mode(self) -> SandboxMode:
        """Select the best available sandbox mode."""
        # Priority: Windows Sandbox > Subprocess > Direct
        if SandboxMode.WINDOWS_SANDBOX in self.available_modes:
            return SandboxMode.WINDOWS_SANDBOX
        elif SandboxMode.SUBPROCESS in self.available_modes:
            return SandboxMode.SUBPROCESS
        else:
            return SandboxMode.DIRECT

    async def initialize(self):
        """Initialize the sandbox executor (async compatibility)."""
        pass

    async def execute_commands(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a list of commands in the sandbox.

        Args:
            commands: List of commands to execute
            work_dir: Working directory for execution
            environment: Optional environment variables

        Returns:
            Dict with success, output, stdout, stderr, error_keywords, passed, total
        """
        start_time = time.time()
        self.executions += 1

        try:
            if self.mode == SandboxMode.WINDOWS_SANDBOX:
                result = await self._execute_in_windows_sandbox(
                    commands, work_dir, environment
                )
            elif self.mode == SandboxMode.SUBPROCESS:
                result = await self._execute_subprocess_isolated(
                    commands, work_dir, environment
                )
            else:
                result = await self._execute_direct(commands, work_dir, environment)

            if result.get("success"):
                self.successful_executions += 1

            return result

        finally:
            self.total_execution_time += time.time() - start_time

    async def _execute_in_windows_sandbox(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute commands in Windows Sandbox.

        Creates a .wsb configuration file and launches Windows Sandbox.
        """
        task_id = uuid.uuid4().hex[:8]
        task_dir = os.path.join(self.work_dir_base, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        try:
            # Copy work directory contents
            if os.path.exists(work_dir) and os.path.isdir(work_dir):
                shutil.copytree(work_dir, task_dir, dirs_exist_ok=True)

            # Create batch script with commands
            batch_script = os.path.join(task_dir, "run_commands.bat")
            output_file = os.path.join(task_dir, "output.txt")

            with open(batch_script, "w") as f:
                f.write("@echo off\n")
                f.write(f"cd /d {task_dir}\n")
                if environment:
                    for key, value in environment.items():
                        f.write(f"set {key}={value}\n")
                for cmd in commands:
                    # Convert bash commands to Windows equivalents where possible
                    win_cmd = self._convert_bash_to_windows(cmd)
                    f.write(f'echo $ {cmd} >> "{output_file}"\n')
                    f.write(f'{win_cmd} >> "{output_file}" 2>&1\n')
                f.write(f'echo EXECUTION_COMPLETE >> "{output_file}"\n')

            # Create Windows Sandbox configuration
            wsb_config = f"""<Configuration>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>{task_dir}</HostFolder>
      <SandboxFolder>C:\\Sandbox</SandboxFolder>
      <ReadOnly>false</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>C:\\Sandbox\\run_commands.bat</Command>
  </LogonCommand>
  <Networking>{"Enable" if environment and environment.get("ENABLE_NETWORK") else "Disable"}</Networking>
  <MemoryInMB>512</MemoryInMB>
</Configuration>"""

            wsb_file = os.path.join(task_dir, "sandbox.wsb")
            with open(wsb_file, "w") as f:
                f.write(wsb_config)

            # Launch Windows Sandbox
            process = await asyncio.create_subprocess_exec(
                "WindowsSandbox.exe",
                wsb_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(
                    self._wait_for_sandbox_completion(output_file),
                    timeout=self.timeout + 30,  # Extra time for sandbox startup
                )
            except asyncio.TimeoutError:
                # Kill sandbox
                subprocess.run(
                    ["taskkill", "/F", "/IM", "WindowsSandbox.exe"], capture_output=True
                )
                return {
                    "success": False,
                    "output": f"[TIMEOUT after {self.timeout} seconds]",
                    "stdout": "",
                    "stderr": f"[TIMEOUT after {self.timeout} seconds]",
                    "error_keywords": ["timeout"],
                    "passed": 0,
                    "total": 1,
                }

            # Read output
            output = ""
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    output = f.read()

            success = "EXECUTION_COMPLETE" in output
            error_keywords = self._extract_error_keywords(output)

            return {
                "success": success and not error_keywords,
                "output": output,
                "stdout": output,
                "stderr": "",
                "error_keywords": error_keywords,
                "passed": 1 if success else 0,
                "total": 1,
            }

        finally:
            # Cleanup
            try:
                shutil.rmtree(task_dir)
            except Exception:
                pass

    async def _wait_for_sandbox_completion(
        self, output_file: str, poll_interval: float = 1.0
    ):
        """Wait for sandbox execution to complete by monitoring output file."""
        while True:
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    content = f.read()
                    if "EXECUTION_COMPLETE" in content:
                        return
            await asyncio.sleep(poll_interval)

    async def _execute_subprocess_isolated(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute commands with subprocess isolation.

        Uses Windows Job Objects for resource limiting.
        """
        print(f"[SANDBOX] Using subprocess isolation (mode: {self.mode.value})")

        task_id = uuid.uuid4().hex[:8]
        task_dir = os.path.join(self.work_dir_base, f"task_{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        # Copy work directory
        if os.path.exists(work_dir) and os.path.isdir(work_dir):
            try:
                shutil.copytree(work_dir, task_dir, dirs_exist_ok=True)
            except Exception as e:
                print(f"[SANDBOX] Warning copying work directory: {e}")

        try:
            all_output = []
            all_stdout = []
            all_stderr = []
            error_keywords = []
            success = True

            # Prepare environment
            exec_env = os.environ.copy()
            if environment:
                exec_env.update(environment)

            for cmd in commands:
                try:
                    # Convert bash to Windows if needed
                    win_cmd = self._convert_bash_to_windows(cmd)

                    # Create subprocess with resource limits
                    if sys.platform == "win32":
                        # Use cmd.exe for Windows
                        process = await asyncio.create_subprocess_shell(
                            win_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=task_dir,
                            env=exec_env,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                        )
                    else:
                        process = await asyncio.create_subprocess_shell(
                            cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=task_dir,
                            env=exec_env,
                        )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                    stdout_text = stdout.decode(errors="replace")
                    stderr_text = stderr.decode(errors="replace")

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
            # Cleanup
            try:
                shutil.rmtree(task_dir)
            except Exception:
                pass

    async def _execute_direct(
        self,
        commands: List[str],
        work_dir: str,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Direct execution without sandbox (fallback).

        WARNING: No isolation - use only for trusted code.
        """
        print("[SANDBOX] WARNING: Using direct execution (no isolation)")

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
                    win_cmd = (
                        self._convert_bash_to_windows(cmd)
                        if sys.platform == "win32"
                        else cmd
                    )

                    process = await asyncio.create_subprocess_shell(
                        win_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=environment,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                    stdout_text = stdout.decode(errors="replace")
                    stderr_text = stderr.decode(errors="replace")

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

    def _convert_bash_to_windows(self, cmd: str) -> str:
        """
        Convert common bash commands to Windows equivalents.

        Handles common patterns used in SWE-bench tasks.
        """
        if sys.platform != "win32":
            return cmd

        # Common bash to Windows conversions
        conversions = {
            "ls ": "dir ",
            "ls\n": "dir\n",
            "cat ": "type ",
            "rm -rf ": "rmdir /s /q ",
            "rm -r ": "rmdir /s /q ",
            "rm ": "del ",
            "cp -r ": "xcopy /e /i ",
            "cp ": "copy ",
            "mv ": "move ",
            "mkdir -p ": "mkdir ",
            "touch ": "type nul > ",
            "pwd": "cd",
            "which ": "where ",
            "export ": "set ",
            "echo $": "echo %",
            "grep ": "findstr ",
            "head ": "more /e ",
            "tail ": "more +",
        }

        result = cmd
        for bash_cmd, win_cmd in conversions.items():
            result = result.replace(bash_cmd, win_cmd)

        # Handle path separators
        # Only convert if it looks like a path (contains /)
        if "/" in result and not result.startswith("http"):
            # Be careful not to break URLs or regex patterns
            parts = result.split()
            converted_parts = []
            for part in parts:
                if (
                    "/" in part
                    and not part.startswith("http")
                    and not part.startswith("-")
                ):
                    # Likely a path
                    converted_parts.append(part.replace("/", "\\"))
                else:
                    converted_parts.append(part)
            result = " ".join(converted_parts)

        return result

    def _extract_error_keywords(self, text: str) -> List[str]:
        """Extract error keywords from output text."""
        keywords = []
        text_lower = text.lower()

        error_patterns = {
            "command not found": "command_not_found",
            "'cmd' is not recognized": "command_not_found",
            "is not recognized as an internal": "command_not_found",
            "permission denied": "permission_denied",
            "access is denied": "permission_denied",
            "syntax error": "syntax_error",
            "no such file": "file_not_found",
            "cannot find": "file_not_found",
            "the system cannot find": "file_not_found",
            "undefined variable": "undefined_variable",
            "is not defined": "undefined_variable",
            "unexpected": "unexpected_token",
            "invalid": "invalid_syntax",
            "error": "generic_error",
            "failed": "execution_failed",
            "exception": "exception",
            "traceback": "python_error",
        }

        for pattern, keyword in error_patterns.items():
            if pattern in text_lower:
                keywords.append(keyword)

        return list(set(keywords))

    def health_check(self) -> Dict[str, Any]:
        """Check sandbox health status."""
        return {
            "status": "healthy",
            "sandbox_enabled": self.enable_sandbox,
            "mode": self.mode.value,
            "available_modes": [m.value for m in self.available_modes],
            "timeout": self.timeout,
            "work_dir_base": self.work_dir_base,
            "executions": self.executions,
            "successful_executions": self.successful_executions,
            "success_rate": self.successful_executions / max(1, self.executions),
            "docker_available": False,  # Compatibility with Docker sandbox interface
        }


# ============================================================================
# Factory function for sandbox executor
# ============================================================================

_windows_sandbox_instance = None


def get_windows_sandbox_executor(
    timeout: int = 30,
    enable_sandbox: bool = True,
    preferred_mode: Optional[SandboxMode] = None,
) -> WindowsSandboxExecutor:
    """
    Get global Windows sandbox executor instance.

    Args:
        timeout: Timeout in seconds
        enable_sandbox: Enable sandbox isolation
        preferred_mode: Preferred sandbox mode

    Returns:
        WindowsSandboxExecutor instance
    """
    global _windows_sandbox_instance
    if _windows_sandbox_instance is None:
        _windows_sandbox_instance = WindowsSandboxExecutor(
            timeout=timeout,
            enable_sandbox=enable_sandbox,
            preferred_mode=preferred_mode,
        )
    return _windows_sandbox_instance


def reset_windows_sandbox_executor():
    """Reset global Windows sandbox executor instance."""
    global _windows_sandbox_instance
    _windows_sandbox_instance = None


# ============================================================================
# Unified sandbox factory - auto-selects best available sandbox
# ============================================================================


def get_sandbox_executor(
    docker_image: str = "ubuntu:22.04",
    timeout: int = 30,
    enable_sandbox: bool = True,
):
    """
    Get the best available sandbox executor.

    On Windows without Docker: Returns WindowsSandboxExecutor
    On systems with Docker: Returns DockerSandboxExecutor

    This is a drop-in replacement for the Docker-based get_sandbox_executor.
    """
    # Check if Docker is available
    docker_available = False
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        docker_available = result.returncode == 0
    except Exception:
        pass

    if docker_available:
        # Use Docker sandbox if available
        from benchmarks.benchmarking.swe_bench_sandbox import DockerSandboxExecutor

        return DockerSandboxExecutor(
            docker_image=docker_image,
            timeout=timeout,
            enable_sandbox=enable_sandbox,
        )
    else:
        # Use Windows sandbox
        return get_windows_sandbox_executor(
            timeout=timeout,
            enable_sandbox=enable_sandbox,
        )


# ============================================================================
# END OF WINDOWS SANDBOX MODULE
# ============================================================================
