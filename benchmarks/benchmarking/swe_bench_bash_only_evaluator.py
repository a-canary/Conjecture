"""
SWE-Bench Bash-Only Evaluator with ReAct Loop
Implements bash-specific problem solving with temperature=0.0 for reproducibility

=============================================================================
MODIFIED FOR SC-FEAT-001 - TEST BRANCH
Added Docker sandbox integration for secure isolated execution
Generated 2025-12-30
=============================================================================
"""

import os
import subprocess
import tempfile
import shutil
import json
import time
import asyncio
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

from src.endpoint_app import ConjectureEndpoint
from src.config.unified_config import get_config

# SC-FEAT-001: Fixed import to use unified_bridge
from src.processing.unified_bridge import UnifiedLLMBridge, LLMRequest, LLMResponse
from src.processing.simplified_llm_manager import get_simplified_llm_manager

# SC-FEAT-001: Sandbox integration for secure execution
# Use Windows-native sandbox when Docker is not available
try:
    from benchmarks.benchmarking.windows_sandbox import get_sandbox_executor
except ImportError:
    from benchmarks.benchmarking.swe_bench_sandbox import get_sandbox_executor


class Conjecture:
    """
    Wrapper class that bridges ConjectureEndpoint to legacy Conjecture interface
    Maintains compatibility with existing benchmark code
    """

    def __init__(self, config):
        self.config = config
        self.endpoint = ConjectureEndpoint(config)

    async def start_services(self):
        """Initialize services"""
        await self.endpoint.start_services()

    async def stop_services(self):
        """Stop services"""
        await self.endpoint.stop_services()

    async def process_task(self, task):
        """Process a task through Conjecture"""
        # ConjectureEndpoint has process_task method but expects dict format
        # Convert task dict to expected format
        task_dict = {
            "type": "bash_task",
            "content": task.problem_statement,
            "context": {
                "repo": task.repo,
                "base_commit": task.base_commit,
                "hints": task.hints,
                "test_patch": task.test_patch,
            },
        }
        return await self.endpoint.process_task(task_dict)


class EvaluationResult(Enum):
    """Evaluation result status"""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


class ReActState(Enum):
    """ReAct loop state"""

    OBSERVE = "observe"
    DIAGNOSE = "diagnose"
    PATCH = "patch"
    VERIFY = "verify"
    COMPLETE = "complete"


@dataclass
class SWETask:
    """SWE-bench task representation"""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints: Optional[str]
    test_patch: str
    version: str
    environment_setup_commit: Optional[str] = None


@dataclass
class ReActIteration:
    """Single ReAct iteration tracking"""

    iteration: int
    state: ReActState
    observation: str
    diagnosis: str
    action: str
    result: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvaluationOutput:
    """Results from model evaluation"""

    result: EvaluationResult
    execution_time: float
    output: str
    error: Optional[str] = None
    tests_passed: int = 0
    tests_total: int = 0
    generated_patch: Optional[str] = None
    react_iterations: List[ReActIteration] = field(default_factory=list)
    final_bash_commands: List[str] = field(default_factory=list)
    failure_analysis: Optional[Dict[str, Any]] = None


@dataclass
class FailurePattern:
    """Classification of failure patterns"""

    task_id: str
    problem_length: int
    iterations_used: int
    error_type: str  # "syntax", "runtime", "timeout", "logic", "unknown"
    error_keywords: List[str]
    bash_patterns: List[str]  # Patterns found in bash commands
    stderr_output: str
    stdout_output: str


class BashOnlySWEBenchEvaluator:
    """
    SWE-Bench evaluator optimized for bash-only problem solving
    Uses ReAct loop with temperature=0.0 for reproducibility
    """

    def __init__(
        self,
        sandbox_dir: Optional[str] = None,
        max_iterations: int = 4,
        verbose: bool = False,
        use_sandbox: bool = True,  # SC-FEAT-001: Sandbox control
        docker_image: str = "ubuntu:22.04",  # SC-FEAT-001: Docker image
    ):
        self.config = get_config()
        self.sandbox_dir = (
            Path(sandbox_dir)
            if sandbox_dir
            else Path(tempfile.mkdtemp(prefix="swe_bash_"))
        )
        self.sandbox_dir.mkdir(exist_ok=True, parents=True)
        self.conjecture = None
        self.max_iterations = max_iterations
        self.verbose = verbose

        # SC-FEAT-001: Initialize sandbox executor
        self.use_sandbox = use_sandbox
        self.sandbox_executor = get_sandbox_executor(
            docker_image=docker_image,
            timeout=30,
            enable_sandbox=use_sandbox,
        )

        # Metrics tracking
        self.evaluations_completed = 0
        self.total_execution_time = 0.0
        self.successful_evaluations = 0
        self.react_iterations_total = 0

        # Failure analysis tracking
        self.failure_patterns: List[FailurePattern] = []
        self.error_type_counts: Dict[str, int] = defaultdict(int)
        self.error_keyword_counts: Dict[str, int] = defaultdict(int)

        # SC-FEAT-001: Print sandbox status
        sandbox_health = self.sandbox_executor.health_check()
        print(f"[MICROSCOPE] Bash-Only SWE-Bench Evaluator initialized")
        print(f"   Sandbox: {self.sandbox_dir}")
        print(
            f"   Docker Sandbox: {'ENABLED' if sandbox_health['sandbox_enabled'] and sandbox_health['docker_available'] else 'DISABLED (using direct execution)'}"
        )
        print(f"   Max ReAct iterations: {self.max_iterations} (reduced from 5 to 4)")
        print(f"   Temperature: 0.0 (deterministic)")
        print(f"   Early stopping: Enabled (stop on success)")
        print(f"   Verbose logging: {'ENABLED' if verbose else 'disabled'}")

    async def initialize_conjecture(self):
        """Initialize Conjecture system"""
        self.conjecture = Conjecture(self.config)
        await self.conjecture.start_services()
        print("[OK] Conjecture system initialized")

    async def load_swe_tasks(self, num_tasks: int = 500) -> List[SWETask]:
        """
        Load real SWE-bench-lite tasks from HuggingFace
        Loads up to 500 instances for comprehensive evaluation
        """
        try:
            from datasets import load_dataset

            print(
                f"[PACKAGE] Loading {num_tasks} real SWE-bench-lite tasks from HuggingFace..."
            )
            dataset = load_dataset(
                "princeton-nlp/swe-bench_lite",
                split="test",
            )

            tasks = []
            for i, item in enumerate(
                dataset.select(range(min(num_tasks, len(dataset))))
            ):
                task = SWETask(
                    instance_id=item["instance_id"],
                    repo=item["repo"],
                    base_commit=item["base_commit"],
                    problem_statement=item["problem_statement"],
                    hints=item.get("hints_text"),
                    test_patch=item["test_patch"],
                    version=item["version"],
                    environment_setup_commit=item.get("environment_setup_commit"),
                )
                tasks.append(task)

            print(f"[OK] Loaded {len(tasks)} real SWE-bench tasks")
            return tasks

        except Exception as e:
            print(f"[WARN]  Could not load SWE-bench dataset: {e}")
            print("[REFRESH] Creating fallback bash-focused evaluation tasks...")
            return self._create_bash_fallback_tasks(num_tasks)

    def _create_bash_fallback_tasks(self, num_tasks: int) -> List[SWETask]:
        """Create bash-focused fallback tasks"""
        tasks = []

        bash_problems = [
            {
                "name": "file_processing",
                "statement": """
Fix the bash script that processes log files.
The script should:
1. Read all .log files in the current directory
2. Extract lines containing ERROR
3. Count occurrences per file
4. Output summary in format: filename: count

Current script has bugs in file iteration and error handling.
""",
                "hints": "Use find, grep, and wc commands properly",
            },
            {
                "name": "string_manipulation",
                "statement": """
Implement a bash function that:
1. Takes a string as input
2. Converts to uppercase
3. Replaces spaces with underscores
4. Removes special characters
5. Returns the result

Must handle edge cases: empty strings, special chars, unicode
""",
                "hints": "Use tr, sed, and parameter expansion",
            },
            {
                "name": "directory_sync",
                "statement": """
Create a bash script that synchronizes two directories:
1. Copy new files from source to destination
2. Update modified files
3. Delete files in destination not in source
4. Preserve file permissions
5. Log all operations

Must handle spaces in filenames and symlinks correctly.
""",
                "hints": "Use rsync or find with proper quoting",
            },
            {
                "name": "process_monitoring",
                "statement": """
Write a bash script that monitors processes:
1. Check if a process is running (by name)
2. If not running, start it
3. Log start/stop events with timestamps
4. Handle multiple instances correctly
5. Clean up stale lock files

Must be idempotent and handle concurrent calls.
""",
                "hints": "Use pgrep, pidof, and proper locking",
            },
            {
                "name": "config_parser",
                "statement": """
Implement a bash config file parser:
1. Read key=value pairs from config file
2. Handle comments (lines starting with #)
3. Support variable expansion (${VAR})
4. Handle quoted values with spaces
5. Validate required keys

Must handle edge cases: empty values, special chars, missing files
""",
                "hints": "Use grep, sed, and eval carefully",
            },
        ]

        for i in range(num_tasks):
            problem = bash_problems[i % len(bash_problems)]
            task = SWETask(
                instance_id=f"bash_task_{i + 1:04d}_{problem['name']}",
                repo="bash/example",
                base_commit=f"abc{i:03d}",
                problem_statement=problem["statement"],
                hints=problem["hints"],
                test_patch=f"# Test patch for {problem['name']}",
                version="1.0",
            )
            tasks.append(task)

        print(f"[OK] Created {len(tasks)} bash-focused fallback tasks")
        return tasks

    def _build_bash_react_prompt(
        self,
        task: SWETask,
        iteration: int,
        previous_attempts: List[str],
        previous_errors: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Build ReAct prompt for bash-only problem solving with error feedback.
        Temperature 0.0 ensures deterministic output.
        Includes PREVIOUS_ERROR section with specific error keywords and context.
        """
        previous_context = ""
        if previous_attempts:
            previous_context = "\n\nPREVIOUS ATTEMPTS:\n"
            for i, attempt in enumerate(previous_attempts[-2:], 1):  # Last 2 attempts
                previous_context += f"Attempt {i}:\n{attempt}\n\n"

        error_feedback = ""
        if previous_errors:
            error_feedback = "\n\nPREVIOUS_ERROR FEEDBACK:\n"
            for error_info in previous_errors[-2:]:  # Last 2 errors
                keywords = ", ".join(error_info.get("error_keywords", []))
                stderr_sample = error_info.get("stderr", "")[:100]
                error_feedback += f"Iteration {error_info['iteration']}: {keywords}\n"
                if stderr_sample:
                    error_feedback += f"  Error output: {stderr_sample}\n"
            error_feedback += "\nFIX THESE ERRORS IN YOUR NEXT ATTEMPT:\n"
            error_feedback += "- Review the error keywords above\n"
            error_feedback += "- Adjust your bash syntax or logic accordingly\n"
            error_feedback += "- Test edge cases more thoroughly\n"

        return f"""You are an expert bash programmer solving SWE-bench problems.
Use the ReAct (Reason + Act) approach with deterministic thinking (temperature=0.0).

PROBLEM:
{task.problem_statement}

HINTS:
{task.hints or "No hints provided"}

ITERATION: {iteration}/{self.max_iterations}

BASH-SPECIFIC REQUIREMENTS:
1. Use only standard bash (no zsh, fish, etc.)
2. Handle errors explicitly with set -e and error traps
3. Quote all variables properly to handle spaces
4. Use [[ ]] for conditionals, not [ ]
5. Avoid deprecated backticks, use $() instead
6. Handle file paths with spaces and special characters
7. Use proper exit codes (0 for success, non-zero for failure)
8. Add comments explaining complex logic

REACT LOOP STRUCTURE:
1. OBSERVE: Analyze the problem and current state
2. DIAGNOSE: Identify what needs to be fixed
3. PATCH: Write the bash solution
4. VERIFY: Test the solution with edge cases

OUTPUT FORMAT (CRITICAL - MUST FOLLOW EXACTLY):
[OBSERVE]
<Your analysis of the problem>

[DIAGNOSE]
<Root cause and what needs fixing>

[PATCH]
```bash
#!/bin/bash
# Your complete bash solution here
# Must be production-ready with error handling
```

[VERIFY]
<How you would test this solution>

[BASH_COMMANDS]
<List of bash commands to execute for testing>

{previous_context}{error_feedback}

TEMPERATURE: 0.0 (Deterministic - Always produce the same output for same input)
CONTEXT_BUDGET: <500 tokens
MODEL: GraniteTiny optimized

Think step-by-step. Be precise. No explanations outside the format."""

    async def evaluate_bash_react(self, task: SWETask) -> EvaluationOutput:
        """
        Evaluate task using bash-only ReAct loop with error feedback.
        Temperature 0.0 for reproducibility.
        Includes failure analysis and error keyword extraction.
        """
        start_time = time.time()
        task_dir = self.sandbox_dir / f"bash_react_{task.instance_id}"
        task_dir.mkdir(exist_ok=True)

        react_iterations = []
        previous_attempts = []
        previous_errors = []  # Track errors for feedback
        generated_patch = ""
        final_bash_commands = []
        test_result = None

        try:
            if not self.conjecture:
                await self.initialize_conjecture()

            # ReAct loop: max 4 iterations with early stopping
            for iteration in range(1, self.max_iterations + 1):
                print(f"  [REFRESH] ReAct Iteration {iteration}/{self.max_iterations}")

                # Build prompt with previous context AND error feedback
                prompt = self._build_bash_react_prompt(
                    task, iteration, previous_attempts, previous_errors
                )

                # Call LLM with temperature=0.0
                from src.processing.unified_bridge import UnifiedLLMBridge, LLMRequest
                from src.processing.simplified_llm_manager import (
                    get_simplified_llm_manager,
                )

                llm_manager = get_simplified_llm_manager()
                llm_bridge = UnifiedLLMBridge(llm_manager=llm_manager)

                if not llm_bridge.is_available():
                    return EvaluationOutput(
                        result=EvaluationResult.ERROR,
                        execution_time=time.time() - start_time,
                        output="",
                        error="No LLM providers available",
                    )

                request = LLMRequest(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.0,  # CRITICAL: Deterministic output
                    task_type="bash_generation",
                )

                response = llm_bridge.process(request)

                if not response.success:
                    return EvaluationOutput(
                        result=EvaluationResult.ERROR,
                        execution_time=time.time() - start_time,
                        output="",
                        error=f"LLM generation failed: {response.errors}",
                    )

                # Parse ReAct response (with fuzzy matching)
                react_data = self._parse_react_response(response.content)

                # Create iteration record
                iteration_record = ReActIteration(
                    iteration=iteration,
                    state=ReActState.PATCH,
                    observation=react_data.get("observe", ""),
                    diagnosis=react_data.get("diagnose", ""),
                    action=react_data.get("patch", ""),
                    result="",
                )

                # Extract bash commands
                bash_commands = react_data.get("bash_commands", [])
                if bash_commands:
                    final_bash_commands = bash_commands

                # Extract patch
                if react_data.get("patch"):
                    generated_patch = react_data["patch"]

                # Test the solution
                test_result = await self._execute_bash_solution(
                    task_dir, generated_patch, bash_commands
                )

                iteration_record.result = test_result["output"]

                react_iterations.append(iteration_record)
                self.react_iterations_total += 1

                # EARLY STOPPING: Stop if solution passes
                if test_result["success"]:
                    if self.verbose:
                        print(
                            f"    [OK] Solution passed at iteration {iteration} (EARLY STOP)"
                        )
                    self.successful_evaluations += 1
                    break

                # Extract error keywords for next iteration feedback
                error_keywords = test_result.get("error_keywords", [])
                if error_keywords:
                    error_context = {
                        "iteration": iteration,
                        "error_keywords": error_keywords,
                        "stderr": test_result.get("stderr", "")[
                            :200
                        ],  # First 200 chars
                    }
                    previous_errors.append(error_context)

                    if self.verbose:
                        print(
                            f"    [WARN]  Errors detected: {', '.join(error_keywords)}"
                        )

                # Store attempt for next iteration context
                previous_attempts.append(f"Attempt {iteration}:\n{response.content}")

                # Brief pause between iterations
                await asyncio.sleep(0.5)

            # Final evaluation
            execution_time = time.time() - start_time
            self.evaluations_completed += 1
            self.total_execution_time += execution_time

            # Perform failure analysis if task failed
            failure_analysis = None
            if test_result and not test_result["success"]:
                failure_analysis = self._analyze_failure(
                    task, react_iterations, test_result, len(previous_attempts)
                )
                self.failure_patterns.append(failure_analysis)

            return EvaluationOutput(
                result=EvaluationResult.PASSED
                if (test_result and test_result["success"])
                else EvaluationResult.FAILED,
                execution_time=execution_time,
                output=test_result["output"] if test_result else "",
                tests_passed=1 if (test_result and test_result["success"]) else 0,
                tests_total=1,
                generated_patch=generated_patch,
                react_iterations=react_iterations,
                final_bash_commands=final_bash_commands,
                failure_analysis=failure_analysis,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return EvaluationOutput(
                result=EvaluationResult.ERROR,
                execution_time=execution_time,
                output="",
                error=str(e),
                react_iterations=react_iterations,
            )

    def _parse_react_response(self, response: str) -> Dict[str, Any]:
        """
        Parse ReAct structured response with fuzzy matching and fallback extraction.
        Handles:
        - Case-insensitive section headers
        - Missing headers (extracts bash code blocks anyway)
        - Heredoc and multiline scripts
        - Various bash command patterns
        """
        result = {
            "observe": "",
            "diagnose": "",
            "patch": "",
            "verify": "",
            "bash_commands": [],
        }

        # Fuzzy section matching (case-insensitive, flexible)
        section_patterns = {
            r"\[?\s*observe\s*\]?": "observe",
            r"\[?\s*diagnose\s*\]?": "diagnose",
            r"\[?\s*patch\s*\]?": "patch",
            r"\[?\s*verify\s*\]?": "verify",
            r"\[?\s*bash[_\s]*commands?\s*\]?": "bash_commands",
        }

        current_section = None
        current_content = []
        lines = response.split("\n")

        for i, line in enumerate(lines):
            # Try fuzzy section matching
            found_section = False
            for pattern, key in section_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous section
                    if current_section:
                        content = "\n".join(current_content).strip()
                        if current_section == "bash_commands":
                            result[current_section] = self._extract_bash_commands(
                                content
                            )
                        else:
                            result[current_section] = content
                    current_section = key
                    current_content = []
                    found_section = True
                    break

            if not found_section and current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            content = "\n".join(current_content).strip()
            if current_section == "bash_commands":
                result[current_section] = self._extract_bash_commands(content)
            else:
                result[current_section] = content

        # Fallback: Extract bash code blocks if patch is empty
        if not result["patch"]:
            bash_blocks = self._extract_bash_code_blocks(response)
            if bash_blocks:
                result["patch"] = bash_blocks[0]
                if len(bash_blocks) > 1:
                    result["bash_commands"] = bash_blocks[1:]

        # Extract bash code from patch section (handle ```bash blocks)
        if "```bash" in result["patch"]:
            start = result["patch"].find("```bash") + 7
            end = result["patch"].find("```", start)
            if end > start:
                result["patch"] = result["patch"][start:end].strip()
        elif "```" in result["patch"]:
            # Handle generic code blocks
            start = result["patch"].find("```") + 3
            end = result["patch"].find("```", start)
            if end > start:
                result["patch"] = result["patch"][start:end].strip()

        # Fallback: Extract bash commands if section is empty
        if not result["bash_commands"]:
            result["bash_commands"] = self._extract_bash_commands(response)

        if self.verbose:
            print(
                f"  [LIST] Parsed sections: observe={len(result['observe'])}B, "
                f"diagnose={len(result['diagnose'])}B, patch={len(result['patch'])}B, "
                f"bash_commands={len(result['bash_commands'])}"
            )

        return result

    def _extract_bash_code_blocks(self, text: str) -> List[str]:
        """Extract all bash code blocks from text"""
        blocks = []

        # Find ```bash blocks
        pattern = r"```bash\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        blocks.extend(matches)

        # Find generic ``` blocks if no bash blocks found
        if not blocks:
            pattern = r"```\n(.*?)\n```"
            matches = re.findall(pattern, text, re.DOTALL)
            blocks.extend(matches)

        return blocks

    def _extract_bash_commands(self, text: str) -> List[str]:
        """
        Extract bash commands from text.
        Handles:
        - Lines starting with $
        - Lines starting with #!
        - Heredoc patterns
        - Command chains with && and ||
        """
        commands = []

        # Extract lines that look like bash commands
        for line in text.split("\n"):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Extract command lines (starting with $, or common bash patterns)
            if line.startswith("$"):
                commands.append(line[1:].strip())
            elif any(
                line.startswith(cmd)
                for cmd in ["bash", "sh", "python", "npm", "make", "git", "docker"]
            ):
                commands.append(line)
            elif any(op in line for op in ["&&", "||", "|", ">"]):
                # Likely a command chain
                commands.append(line)

        return commands

    async def _execute_bash_solution(
        self, task_dir: Path, bash_script: str, test_commands: List[str]
    ) -> Dict[str, Any]:
        """
        SC-FEAT-001: Execute bash solution with sandbox isolation.

        Uses Docker sandbox for secure isolated execution with 30-second timeout.
        Parses stderr/stdout separately and extracts error keywords for feedback.

        Generated for SC-FEAT-001 test branch.
        """
        try:
            # Save bash script
            script_file = task_dir / "solution.sh"
            script_file.write_text(bash_script)
            script_file.chmod(0o755)

            # SC-FEAT-001: Use sandbox executor instead of direct subprocess
            if self.use_sandbox:
                return await self.sandbox_executor.execute_commands(
                    commands=test_commands[:5],  # Limit to 5 commands
                    work_dir=str(task_dir),
                )

            # Fallback: Direct execution (only if sandbox disabled)
            all_output = []
            all_stderr = []
            all_stdout = []
            success = True
            error_keywords = []

            for cmd in test_commands[:5]:  # Limit to 5 commands
                try:
                    process = await asyncio.create_subprocess_shell(
                        cmd,
                        cwd=str(task_dir),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        shell=True,
                    )

                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(),
                            timeout=30.0,  # 30-second timeout per command
                        )

                        stdout_text = stdout.decode()
                        stderr_text = stderr.decode()

                        all_stdout.append(stdout_text)
                        all_stderr.append(stderr_text)

                        output = stdout_text + stderr_text
                        all_output.append(f"$ {cmd}\n{output}")

                        # Extract error keywords from stderr
                        if stderr_text:
                            keywords = self._extract_error_keywords(stderr_text)
                            error_keywords.extend(keywords)

                        if process.returncode != 0:
                            success = False

                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        all_output.append(f"$ {cmd}\n[TIMEOUT after 30 seconds]")
                        all_stderr.append("[TIMEOUT after 30 seconds]")
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
                "error_keywords": list(set(error_keywords)),  # Unique keywords
                "passed": 1 if success else 0,
                "total": 1,
            }

        except Exception as e:
            return {
                "success": False,
                "output": f"Execution failed: {str(e)}",
                "stdout": "",
                "stderr": str(e),
                "error_keywords": ["execution_error"],
                "passed": 0,
                "total": 1,
            }

    def _extract_error_keywords(self, stderr_text: str) -> List[str]:
        """
        Extract error keywords from stderr output.
        Identifies common bash error patterns.
        """
        keywords = []
        stderr_lower = stderr_text.lower()

        # Common error patterns
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

    def _analyze_failure(
        self,
        task: SWETask,
        react_iterations: List[ReActIteration],
        test_result: Dict[str, Any],
        iterations_used: int,
    ) -> FailurePattern:
        """
        Analyze failure pattern for a task.
        Classifies error type and extracts bash patterns.
        """
        # Determine error type
        error_keywords = test_result.get("error_keywords", [])
        error_type = "unknown"

        if "syntax_error" in error_keywords or "bad_substitution" in error_keywords:
            error_type = "syntax"
        elif (
            "command_not_found" in error_keywords
            or "permission_denied" in error_keywords
        ):
            error_type = "runtime"
        elif "timeout" in error_keywords:
            error_type = "timeout"
        elif (
            "file_not_found" in error_keywords or "undefined_variable" in error_keywords
        ):
            error_type = "logic"
        elif error_keywords:
            error_type = "runtime"

        # Extract bash patterns from generated patches
        bash_patterns = []
        for iteration in react_iterations:
            if iteration.action:
                # Extract common bash patterns
                if "for " in iteration.action:
                    bash_patterns.append("for_loop")
                if "while " in iteration.action:
                    bash_patterns.append("while_loop")
                if "if " in iteration.action or "[[" in iteration.action:
                    bash_patterns.append("conditional")
                if "function " in iteration.action or "() {" in iteration.action:
                    bash_patterns.append("function")
                if "grep" in iteration.action:
                    bash_patterns.append("grep")
                if "sed" in iteration.action:
                    bash_patterns.append("sed")
                if "awk" in iteration.action:
                    bash_patterns.append("awk")
                if "find" in iteration.action:
                    bash_patterns.append("find")

        pattern = FailurePattern(
            task_id=task.instance_id,
            problem_length=len(task.problem_statement),
            iterations_used=iterations_used,
            error_type=error_type,
            error_keywords=error_keywords,
            bash_patterns=list(set(bash_patterns)),
            stderr_output=test_result.get("stderr", "")[:500],
            stdout_output=test_result.get("stdout", "")[:500],
        )

        # Update statistics
        self.error_type_counts[error_type] += 1
        for keyword in error_keywords:
            self.error_keyword_counts[keyword] += 1

        return pattern

    async def evaluate_batch(
        self, tasks: List[SWETask], batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate batch of tasks with bash-only ReAct approach.
        Includes failure analysis and error pattern classification.
        """
        results = []
        passed = 0
        failed = 0

        for i, task in enumerate(tasks[:batch_size]):
            print(
                f"\n[MEMO] Task {i + 1}/{min(batch_size, len(tasks))}: {task.instance_id}"
            )

            result = await self.evaluate_bash_react(task)

            # Convert failure_analysis to dict if it exists
            failure_analysis_dict = None
            if result.failure_analysis:
                fa = result.failure_analysis
                failure_analysis_dict = {
                    "task_id": fa.task_id,
                    "error_type": fa.error_type,
                    "error_keywords": fa.error_keywords,
                    "bash_patterns": fa.bash_patterns,
                    "problem_length": fa.problem_length,
                    "iterations_used": fa.iterations_used,
                }

            results.append(
                {
                    "task_id": task.instance_id,
                    "result": result.result.value,
                    "execution_time": result.execution_time,
                    "react_iterations": len(result.react_iterations),
                    "success": result.result == EvaluationResult.PASSED,
                    "failure_analysis": failure_analysis_dict,
                }
            )

            if result.result == EvaluationResult.PASSED:
                passed += 1
                print(
                    f"  [OK] PASSED ({result.execution_time:.2f}s, {len(result.react_iterations)} iterations)"
                )
            else:
                failed += 1
                print(
                    f"  [FAIL] FAILED ({result.execution_time:.2f}s, {len(result.react_iterations)} iterations)"
                )

                # Print failure analysis if verbose
                if self.verbose and result.failure_analysis:
                    fa = result.failure_analysis
                    print(f"     Error Type: {fa.error_type}")
                    print(f"     Keywords: {', '.join(fa.error_keywords)}")
                    print(f"     Bash Patterns: {', '.join(fa.bash_patterns)}")

        return {
            "results": results,
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / len(results) * 100) if results else 0,
                "average_time": sum(r["execution_time"] for r in results) / len(results)
                if results
                else 0,
                "total_react_iterations": self.react_iterations_total,
                "average_iterations": self.react_iterations_total / len(results)
                if results
                else 0,
            },
            "failure_analysis": self._summarize_failures(),
        }

    async def cleanup(self):
        """Clean up resources"""
        if self.conjecture:
            await self.conjecture.stop_services()

        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)

        print("[CLEANUP] Bash-only evaluator cleaned up")

    def _summarize_failures(self) -> Dict[str, Any]:
        """
        Summarize failure patterns across all evaluated tasks.
        Provides classification and statistics for improvement targeting.
        """
        if not self.failure_patterns:
            return {}

        summary = {
            "total_failures": len(self.failure_patterns),
            "error_type_distribution": dict(self.error_type_counts),
            "top_error_keywords": sorted(
                self.error_keyword_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "average_problem_length": sum(
                f.problem_length for f in self.failure_patterns
            )
            / len(self.failure_patterns),
            "average_iterations_used": sum(
                f.iterations_used for f in self.failure_patterns
            )
            / len(self.failure_patterns),
            "common_bash_patterns": self._get_common_bash_patterns(),
            "failed_tasks": [
                {
                    "task_id": f.task_id,
                    "error_type": f.error_type,
                    "error_keywords": f.error_keywords,
                    "bash_patterns": f.bash_patterns,
                }
                for f in self.failure_patterns[:5]  # Top 5 failures
            ],
        }

        return summary

    def _get_common_bash_patterns(self) -> Dict[str, int]:
        """Get frequency of bash patterns in failed tasks"""
        pattern_counts = defaultdict(int)
        for failure in self.failure_patterns:
            for pattern in failure.bash_patterns:
                pattern_counts[pattern] += 1

        return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "evaluations_completed": self.evaluations_completed,
            "successful_evaluations": self.successful_evaluations,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time
            / max(self.evaluations_completed, 1),
            "success_rate": (
                self.successful_evaluations / self.evaluations_completed * 100
            )
            if self.evaluations_completed > 0
            else 0,
            "total_react_iterations": self.react_iterations_total,
            "average_react_iterations": self.react_iterations_total
            / max(self.evaluations_completed, 1),
            "failure_analysis": self._summarize_failures(),
        }


async def main(verbose: bool = True, batch_size: int = 10):
    """
    Main entry point for bash-only SWE-Bench evaluation.

    Args:
        verbose: Enable verbose logging and failure analysis
        batch_size: Number of tasks to evaluate (default 10)
    """
    print("[ROCKET] SWE-Bench Bash-Only Evaluator with ReAct Loop")
    print("=" * 60)
    print(f"Verbose Mode: {'ENABLED' if verbose else 'disabled'}")
    print(f"Batch Size: {batch_size} tasks")
    print("=" * 60)

    evaluator = BashOnlySWEBenchEvaluator(max_iterations=4, verbose=verbose)

    try:
        # Initialize
        await evaluator.initialize_conjecture()

        # Load tasks (500 instances from HuggingFace)
        tasks = await evaluator.load_swe_tasks(num_tasks=500)

        # Evaluate batch
        print(f"\n[CHART] Starting evaluation of first {batch_size} tasks...")
        results = await evaluator.evaluate_batch(tasks, batch_size=batch_size)

        # Print results
        print("\n" + "=" * 60)
        print("[GRAPH] EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Tasks: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Average Time: {results['summary']['average_time']:.2f}s")
        print(
            f"Average ReAct Iterations: {results['summary']['average_iterations']:.1f}"
        )

        # Print failure analysis if there are failures
        if results["summary"]["failed"] > 0 and results.get("failure_analysis"):
            print("\n" + "=" * 60)
            print("[SEARCH] FAILURE ANALYSIS")
            print("=" * 60)
            fa = results["failure_analysis"]

            if fa.get("error_type_distribution"):
                print("\nError Type Distribution:")
                for error_type, count in fa["error_type_distribution"].items():
                    print(f"  {error_type}: {count}")

            if fa.get("top_error_keywords"):
                print("\nTop Error Keywords:")
                for keyword, count in fa["top_error_keywords"][:5]:
                    print(f"  {keyword}: {count}")

            if fa.get("common_bash_patterns"):
                print("\nCommon Bash Patterns in Failed Tasks:")
                for pattern, count in fa["common_bash_patterns"].items():
                    print(f"  {pattern}: {count}")

            print(
                f"\nAverage Problem Length: {fa.get('average_problem_length', 0):.0f} chars"
            )
            print(
                f"Average Iterations Used: {fa.get('average_iterations_used', 0):.1f}"
            )

            if fa.get("failed_tasks"):
                print("\nSample Failed Tasks:")
                for task in fa["failed_tasks"][:3]:
                    print(f"  {task['task_id']}: {task['error_type']}")

        # Save detailed results
        output_file = Path("swe_bench_bash_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[SAVE] Results saved to {output_file}")

        # Print statistics
        stats = evaluator.get_statistics()
        print("\n[CHART] STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if key == "failure_analysis":
                continue  # Already printed above
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

    finally:
        await evaluator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
