#!/usr/bin/env python
"""
Baseline Evaluator - Direct LLM Without Conjecture
This evaluator tests model performance WITHOUT using Conjecture's context building,
claim management, or evidence tracking systems.

For comparison: swe_bench_bash_only_evaluator.py (WITH Conjecture)
"""

import asyncio
import json
import os
import re
import time
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config.unified_config import get_config

# Use Windows-native sandbox when Docker is not available
try:
    from benchmarks.benchmarking.windows_sandbox import get_sandbox_executor
except ImportError:
    from benchmarks.benchmarking.swe_bench_sandbox import get_sandbox_executor


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
    tokens_used: int = 0  # Token usage tracking


class BaselineSWEBenchEvaluator:
    """
    Baseline SWE-Bench Evaluator WITHOUT Conjecture
    Uses direct LLM calls with simple ReAct loop
    """

    def __init__(
        self,
        max_iterations: int = 4,
        temperature: float = 0.0,
        use_sandbox: bool = True,
        docker_image: str = "ubuntu:22.04",
        model: str = "ibm/granite-4-h-tiny",
    ):
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.use_sandbox = use_sandbox
        self.docker_image = docker_image
        self.model = model
        self.config = None
        self.llm_bridge = None
        self.sandbox = None
        self.tasks: List[SWETask] = []
        self.results: List[EvaluationOutput] = []

    async def initialize(self):
        """Initialize the baseline evaluator (no Conjecture)"""
        from src.processing.unified_bridge import UnifiedLLMBridge, LLMRequest

        # Load config
        self.config = get_config()

        # Create LLM bridge (direct LLM access, no Conjecture)
        # UnifiedLLMBridge takes llm_manager, not config
        self.llm_bridge = UnifiedLLMBridge()

        # Create sandbox executor
        # get_sandbox_executor uses enable_sandbox parameter, not use_sandbox
        self.sandbox = get_sandbox_executor(
            docker_image=self.docker_image, timeout=30, enable_sandbox=self.use_sandbox
        )
        await self.sandbox.initialize()

        print(f"[BASELINE] Initialized (no Conjecture)")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Iterations: {self.max_iterations}")
        print(f"  Sandbox: {self.docker_image if self.use_sandbox else 'disabled'}")

    async def load_swe_tasks(self, num_tasks: int = 10) -> List[SWETask]:
        """
        Load SWE-bench tasks
        Uses fallback synthetic tasks if HuggingFace dataset unavailable
        """
        try:
            from datasets import load_dataset

            print("[BASELINE] Loading SWE-bench-lite from HuggingFace...")
            dataset = load_dataset("princeton-nlp/swe-bench_lite", split="test")
            tasks = []

            for item in dataset:
                if len(tasks) >= num_tasks:
                    break

                # Filter for bash-only tasks (simple heuristics)
                # Look for tasks with simple bash commands
                problem = item["problem_statement"]
                if any(
                    keyword in problem.lower()
                    for keyword in ["bash", "script", "command", "shell"]
                ):
                    task = SWETask(
                        instance_id=item["instance_id"],
                        repo=item["repo"],
                        base_commit=item["base_commit"],
                        problem_statement=problem,
                        hints=item.get("hints_text"),
                        test_patch=item["test_patch"],
                        version=item["version"],
                        environment_setup_commit=item.get("environment_setup_commit"),
                    )
                    tasks.append(task)

            if tasks:
                print(f"[BASELINE] Loaded {len(tasks)} bash-only SWE-bench tasks")
                return tasks

        except ImportError:
            print("[BASELINE] datasets library not installed")
        except Exception as e:
            print(f"[BASELINE] Failed to load SWE-bench dataset: {e}")

        # Fallback to synthetic bash tasks
        print("[BASELINE] Using synthetic bash tasks for evaluation")
        return self._get_synthetic_bash_tasks(num_tasks)

    def _get_synthetic_bash_tasks(self, num_tasks: int) -> List[SWETask]:
        """Generate synthetic bash tasks for testing"""
        synthetic_tasks = [
            {
                "type": "file_processing",
                "problem": "Create a bash script that counts the number of files in /tmp and prints the count",
                "expected_commands": ["find /tmp -type f", "ls -1 /tmp | wc -l"],
            },
            {
                "type": "string_manipulation",
                "problem": "Write a bash script that extracts the first 10 characters from a string and prints them",
                "expected_commands": ["echo", "cut -c1-10", "head -c 10"],
            },
            {
                "type": "directory_sync",
                "problem": "Create a bash script that synchronizes files from /tmp/src to /tmp/dst",
                "expected_commands": ["rsync", "cp -r"],
            },
            {
                "type": "process_monitoring",
                "problem": "Write a bash script that finds all running Python processes and prints their PIDs",
                "expected_commands": ["ps aux | grep python", "pgrep -f python"],
            },
            {
                "type": "config_parser",
                "problem": "Create a bash script that reads a config file and extracts values for a specific key",
                "expected_commands": ["grep", "awk", "sed"],
            },
        ]

        tasks = []
        for i in range(num_tasks):
            template = synthetic_tasks[i % len(synthetic_tasks)]
            task = SWETask(
                instance_id=f"bash_task_{i + 1:04d}_{template['type']}",
                repo="synthetic",
                base_commit="none",
                problem_statement=template["problem"],
                hints=None,
                test_patch="",
                version="1.0",
            )
            tasks.append(task)

        return tasks

    async def evaluate_task(self, task: SWETask) -> EvaluationOutput:
        """
        Evaluate a single task using direct LLM (no Conjecture)
        """
        start_time = time.time()
        iterations: List[ReActIteration] = []
        tokens_used = 0

        # Create workspace
        with tempfile.TemporaryDirectory() as workspace:
            workspace_path = Path(workspace)

            # ReAct loop
            previous_error = ""
            bash_commands = []

            for iteration in range(1, self.max_iterations + 1):
                # Build prompt (no Conjecture context, just ReAct)
                prompt = self._build_baseline_prompt(
                    task, iteration, previous_error, bash_commands
                )

                # Call LLM directly (no Conjecture processing)
                try:
                    llm_request = LLMRequest(
                        prompt=prompt,
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=512,
                    )

                    # UnifiedLLMBridge.process() is synchronous
                    llm_response = self.llm_bridge.process(llm_request)
                    response_text = llm_response.content
                    tokens_used += (
                        llm_response.tokens_used
                        if hasattr(llm_response, "tokens_used")
                        else 0
                    )

                    # Extract bash command from response
                    bash_command = self._extract_bash_command(response_text)
                    bash_commands.append(bash_command)

                    # Execute in sandbox
                    exec_result = await self.sandbox.execute_command(
                        command=bash_command,
                        workdir=str(workspace_path),
                        timeout=30,
                    )

                    # Create iteration record
                    iter_record = ReActIteration(
                        iteration=iteration,
                        state=ReActState.PATCH,
                        observation=task.problem_statement,
                        diagnosis=bash_command,
                        action=bash_command,
                        result=exec_result.stdout + exec_result.stderr,
                    )
                    iterations.append(iter_record)

                    # Check for success
                    if exec_result.exit_code == 0 and self._check_success(
                        exec_result.stdout, exec_result
                    ):
                        execution_time = time.time() - start_time
                        return EvaluationOutput(
                            result=EvaluationResult.PASSED,
                            execution_time=execution_time,
                            output=exec_result.stdout,
                            react_iterations=iterations,
                            final_bash_commands=bash_commands,
                            tokens_used=tokens_used,
                        )

                    # Update error for next iteration
                    previous_error = exec_result.stderr or exec_result.stdout

                except Exception as e:
                    error_msg = str(e)
                    previous_error = f"Error: {error_msg}"

        # Max iterations reached without success
        execution_time = time.time() - start_time
        return EvaluationOutput(
            result=EvaluationResult.FAILED,
            execution_time=execution_time,
            output="",
            error="Max iterations reached",
            react_iterations=iterations,
            final_bash_commands=bash_commands,
            tokens_used=tokens_used,
        )

    def _build_baseline_prompt(
        self,
        task: SWETask,
        iteration: int,
        previous_error: str,
        bash_commands: List[str],
    ) -> str:
        """
        Build baseline prompt WITHOUT Conjecture context
        Simple ReAct structure, no claim management or evidence tracking
        """
        prompt = f"""You are a bash scripting assistant. Solve the following task.

TASK: {task.problem_statement}

"""

        if previous_error:
            prompt += f"""PREVIOUS ATTEMPT FAILED:
{previous_error[:500]}

"""

        if bash_commands:
            prompt += "PREVIOUS COMMANDS:\n"
            for i, cmd in enumerate(bash_commands, 1):
                prompt += f"  {i}. {cmd}\n"
            prompt += "\n"

        prompt += """Provide a single bash command to solve the task.
Output ONLY the bash command, no explanation.

Command:"""

        return prompt

    def _extract_bash_command(self, response: str) -> str:
        """
        Extract bash command from LLM response
        Fuzzy extraction: case-insensitive, supports heredocs
        """
        response = response.strip()

        # Look for code blocks
        block_pattern = re.compile(
            r"```(?:bash|shell)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
        )
        match = block_pattern.search(response)
        if match:
            return match.group(1).strip()

        # Look for commands after "Command:", "$", or similar markers
        marker_patterns = [
            r"Command:\s*(.+)",
            r"\$\s*(.+)",
            r">>>\s*(.+)",
        ]

        for pattern in marker_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: use first non-empty line
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if lines:
            return lines[0]

        return "echo 'No command found'"

    def _check_success(self, output: str, exec_result) -> bool:
        """
        Check if execution was successful
        Uses final bash command exit code and output for accurate success detection
        """
        # Success if exit code is 0 AND output contains expected content
        # (not just checking for absence of error keywords)
        return exec_result.exit_code == 0 and len(exec_result.stdout) > 0

    async def evaluate_batch(
        self, tasks: List[SWETask], batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate a batch of tasks"""
        results = []

        for i, task in enumerate(tasks[:batch_size] if batch_size else tasks, 1):
            print(f"[BASELINE] Evaluating task {i}/{len(tasks)}: {task.instance_id}")
            result = await self.evaluate_task(task)
            results.append(
                {
                    "task_id": task.instance_id,
                    "success": result.result == EvaluationResult.PASSED,
                    "execution_time": result.execution_time,
                    "react_iterations": len(result.react_iterations),
                    "tokens_used": result.tokens_used,
                    "final_command": result.final_bash_commands[-1]
                    if result.final_bash_commands
                    else "",
                }
            )
            print(f"  Result: {result.result.value} ({result.execution_time:.2f}s)")

        # Calculate summary
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        failed = total - passed

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "average_time": sum(r["execution_time"] for r in results) / total,
            "total_iterations": sum(r["react_iterations"] for r in results),
            "average_iterations": sum(r["react_iterations"] for r in results) / total,
            "total_tokens": sum(r["tokens_used"] for r in results),
            "average_tokens": sum(r["tokens_used"] for r in results) / total,
        }

        return {"results": results, "summary": summary}

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from evaluations"""
        total = len(self.results)
        if total == 0:
            return {}

        passed = sum(1 for r in self.results if r.result == EvaluationResult.PASSED)

        return {
            "total_evaluations": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "average_execution_time": sum(r.execution_time for r in self.results)
            / total,
            "average_iterations": sum(len(r.react_iterations) for r in self.results)
            / total,
        }
