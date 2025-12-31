# Mini-SWE-Agent Implementation Guide

**Quick Reference for Building a Bash-Executing ReAct Agent**

---

## 🎯 Quick Start: Copy-Paste Ready Code

### 1. Core Bash Executor (Production-Ready)

```python
# File: src/agent/bash_executor.py
import subprocess
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BashResult:
    """Result of bash command execution."""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    command: str
    execution_time_ms: float
    error: Optional[str] = None

class BashExecutor:
    """Execute bash commands with timeout and error handling."""
    
    def __init__(self, default_timeout: int = 30, max_retries: int = 1):
        self.default_timeout = default_timeout
        self.max_retries = max_retries
    
    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        shell: bool = True
    ) -> BashResult:
        """
        Execute bash command with timeout.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (uses default if None)
            cwd: Working directory
            shell: Use shell execution (default True)
        
        Returns:
            BashResult with execution details
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.utcnow()
        
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return BashResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                command=command,
                execution_time_ms=execution_time
            )
        
        except subprocess.TimeoutExpired:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return BashResult(
                success=False,
                stdout="",
                stderr=f"Command timeout after {timeout}s",
                returncode=-1,
                command=command,
                execution_time_ms=execution_time,
                error=f"Timeout after {timeout}s"
            )
        
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return BashResult(
                success=False,
                stdout="",
                stderr=str(e),
                returncode=-1,
                command=command,
                execution_time_ms=execution_time,
                error=str(e)
            )
```

---

### 2. ReAct Agent with Iteration Limits

```python
# File: src/agent/react_agent.py
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .bash_executor import BashExecutor, BashResult

class AgentStatus(Enum):
    """Agent execution status."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"
    TIMEOUT = "timeout"

@dataclass
class ReActStep:
    """Single step in ReAct loop."""
    step_number: int
    thought: str
    observation: str
    action: str
    action_result: Optional[BashResult] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ReActResult:
    """Final result of ReAct execution."""
    success: bool
    status: AgentStatus
    task: str
    final_answer: Optional[str] = None
    steps: List[ReActStep] = field(default_factory=list)
    total_iterations: int = 0
    total_time_ms: float = 0.0
    error: Optional[str] = None

class ReActAgent:
    """ReAct agent for bash command execution."""
    
    def __init__(
        self,
        max_iterations: int = 10,
        timeout_per_step: int = 30,
        timeout_total: int = 300
    ):
        self.max_iterations = max_iterations
        self.timeout_per_step = timeout_per_step
        self.timeout_total = timeout_total
        self.executor = BashExecutor(default_timeout=timeout_per_step)
        self.steps: List[ReActStep] = []
    
    async def run(self, task: str) -> ReActResult:
        """
        Run ReAct loop for given task.
        
        Args:
            task: Task description
        
        Returns:
            ReActResult with execution details
        """
        start_time = datetime.utcnow()
        step_number = 0
        
        try:
            while step_number < self.max_iterations:
                step_number += 1
                
                # Check total timeout
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed > self.timeout_total:
                    return ReActResult(
                        success=False,
                        status=AgentStatus.TIMEOUT,
                        task=task,
                        steps=self.steps,
                        total_iterations=step_number,
                        total_time_ms=elapsed * 1000,
                        error=f"Total timeout after {self.timeout_total}s"
                    )
                
                # 1. THOUGHT: Generate thought (would call LLM in real implementation)
                thought = await self._generate_thought(task, self.steps)
                
                # Check for finish signal
                if "FINISH" in thought or "finish" in thought.lower():
                    final_answer = self._extract_answer(thought)
                    return ReActResult(
                        success=True,
                        status=AgentStatus.SUCCESS,
                        task=task,
                        final_answer=final_answer,
                        steps=self.steps,
                        total_iterations=step_number,
                        total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                    )
                
                # 2. OBSERVATION: Get current state
                observation = await self._get_observation(task, self.steps)
                
                # 3. ACTION: Extract and execute action
                action = self._extract_action(thought)
                
                if not action:
                    # No action extracted, continue
                    step = ReActStep(
                        step_number=step_number,
                        thought=thought,
                        observation=observation,
                        action="NO_ACTION"
                    )
                    self.steps.append(step)
                    continue
                
                # Execute action
                action_result = self.executor.execute(action)
                
                # Create step record
                step = ReActStep(
                    step_number=step_number,
                    thought=thought,
                    observation=observation,
                    action=action,
                    action_result=action_result
                )
                self.steps.append(step)
                
                # Check for errors
                if not action_result.success:
                    # Could implement retry logic here
                    pass
            
            # Max iterations reached
            return ReActResult(
                success=False,
                status=AgentStatus.MAX_ITERATIONS,
                task=task,
                steps=self.steps,
                total_iterations=step_number,
                total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error=f"Max iterations ({self.max_iterations}) reached"
            )
        
        except Exception as e:
            return ReActResult(
                success=False,
                status=AgentStatus.FAILED,
                task=task,
                steps=self.steps,
                total_iterations=step_number,
                total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error=str(e)
            )
    
    async def _generate_thought(self, task: str, history: List[ReActStep]) -> str:
        """Generate thought (would call LLM in real implementation)."""
        # TODO: Implement LLM call
        return "Analyzing task..."
    
    async def _get_observation(self, task: str, history: List[ReActStep]) -> str:
        """Get current observation."""
        # TODO: Implement observation gathering
        return "Current state: ready"
    
    def _extract_action(self, thought: str) -> Optional[str]:
        """Extract bash command from thought."""
        # Simple extraction - look for code blocks or commands
        if "```bash" in thought:
            start = thought.find("```bash") + 7
            end = thought.find("```", start)
            if end > start:
                return thought[start:end].strip()
        
        # Look for ACTION: pattern
        if "ACTION:" in thought:
            start = thought.find("ACTION:") + 7
            end = thought.find("\n", start)
            if end > start:
                return thought[start:end].strip()
        
        return None
    
    def _extract_answer(self, thought: str) -> str:
        """Extract final answer from thought."""
        if "ANSWER:" in thought:
            start = thought.find("ANSWER:") + 7
            return thought[start:].strip()
        return thought
```

---

### 3. Integration with Conjecture

```python
# File: src/endpoint/conjecture_endpoint.py (add method)

async def execute_bash_task(
    self,
    task: str,
    max_iterations: int = 10,
    timeout_per_step: int = 30
) -> Dict[str, Any]:
    """
    Execute a bash task using ReAct agent.
    
    Args:
        task: Task description
        max_iterations: Maximum iterations
        timeout_per_step: Timeout per step in seconds
    
    Returns:
        Execution result with steps and final answer
    """
    from .agent.react_agent import ReActAgent
    
    agent = ReActAgent(
        max_iterations=max_iterations,
        timeout_per_step=timeout_per_step
    )
    
    result = await agent.run(task)
    
    return {
        "success": result.success,
        "status": result.status.value,
        "task": result.task,
        "final_answer": result.final_answer,
        "steps": [
            {
                "step": step.step_number,
                "thought": step.thought,
                "observation": step.observation,
                "action": step.action,
                "result": {
                    "success": step.action_result.success if step.action_result else None,
                    "stdout": step.action_result.stdout if step.action_result else None,
                    "stderr": step.action_result.stderr if step.action_result else None,
                    "execution_time_ms": step.action_result.execution_time_ms if step.action_result else None
                } if step.action_result else None
            }
            for step in result.steps
        ],
        "total_iterations": result.total_iterations,
        "total_time_ms": result.total_time_ms,
        "error": result.error
    }
```

---

## 📋 Implementation Checklist

- [ ] Copy `BashExecutor` class to `src/agent/bash_executor.py`
- [ ] Copy `ReActAgent` class to `src/agent/react_agent.py`
- [ ] Add `execute_bash_task` method to `ConjectureEndpoint`
- [ ] Implement `_generate_thought` with actual LLM call
- [ ] Implement `_get_observation` with state gathering
- [ ] Add tests in `tests/test_react_agent.py`
- [ ] Add CLI command: `python conjecture bash "task description"`
- [ ] Document in `docs/bash_agent.md`

---

## 🔗 Key Patterns from Codebase

### From `improvement_cycle_agent.py`:
- ✅ Subprocess timeout handling (line 585)
- ✅ Output parsing (line 596-598)
- ✅ Error handling (line 603-606)

### From `agent_harness.py`:
- ✅ Session state management (line 41-52)
- ✅ Iteration limits (line 61)
- ✅ Timeout tracking (line 62)

### From `agent_coordination.py`:
- ✅ Pure function coordination (line 49-60)
- ✅ Result dataclasses (line 34-45)
- ✅ Error aggregation (line 44)

---

## 🚀 Testing Template

```python
# File: tests/test_react_agent.py
import pytest
import asyncio
from src.agent.react_agent import ReActAgent, AgentStatus

@pytest.mark.asyncio
async def test_react_agent_simple_task():
    """Test ReAct agent with simple bash task."""
    agent = ReActAgent(max_iterations=5, timeout_per_step=10)
    
    result = await agent.run("List files in current directory")
    
    assert result.status in [AgentStatus.SUCCESS, AgentStatus.MAX_ITERATIONS]
    assert len(result.steps) > 0
    assert result.total_iterations > 0

@pytest.mark.asyncio
async def test_react_agent_timeout():
    """Test ReAct agent timeout handling."""
    agent = ReActAgent(max_iterations=2, timeout_per_step=1)
    
    result = await agent.run("Sleep for 10 seconds")
    
    # Should timeout
    assert result.status in [AgentStatus.TIMEOUT, AgentStatus.FAILED]

@pytest.mark.asyncio
async def test_bash_executor():
    """Test bash executor directly."""
    from src.agent.bash_executor import BashExecutor
    
    executor = BashExecutor(default_timeout=5)
    result = executor.execute("echo 'hello world'")
    
    assert result.success
    assert "hello world" in result.stdout
    assert result.returncode == 0
```

---

## 📚 Additional Resources

- **Subprocess docs**: https://docs.python.org/3/library/subprocess.html
- **Async/await**: https://docs.python.org/3/library/asyncio.html
- **Dataclasses**: https://docs.python.org/3/library/dataclasses.html
- **ReAct paper**: https://arxiv.org/abs/2210.03629

---

**Implementation Time Estimate**: 2-3 hours  
**Complexity**: Medium (straightforward, well-documented patterns)  
**Testing**: Comprehensive test suite recommended
