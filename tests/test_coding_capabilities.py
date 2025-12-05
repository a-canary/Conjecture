#!/usr/bin/env python3
"""
Comprehensive Coding Capabilities Evaluation Framework

This framework evaluates AI agent coding capabilities across multiple dimensions:
- Code generation and correctness
- Algorithm design and efficiency
- System architecture and scalability
- Debugging and error resolution
- Code quality and best practices
- Security and performance optimization
"""

import asyncio
import json
import time
import uuid
import subprocess
import tempfile
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import statistics
import ast
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use mock implementation for testing to avoid import issues
try:
    from core.models import Claim, ClaimState, ClaimType
    from processing.llm.llm_manager import LLMManager
    from config.common import ProviderConfig
    CONJECTURE_AVAILABLE = True
except ImportError as e:
    CONJECTURE_AVAILABLE = False
    print(f"Note: Conjecture components not available ({e}), using mock implementation")
    
    # Mock classes for testing
    class Claim:
        pass
    class ClaimState:
        pass
    class ClaimType:
        pass
    class LLMManager:
        pass
    class ProviderConfig:
        pass


@dataclass
class CodingEvaluationCriteria:
    """Evaluation criteria for coding tasks"""
    
    # Primary metrics (weighted)
    correctness_score: float = 0.0          # Weight: 2.0 - Code works as specified
    efficiency_score: float = 0.0           # Weight: 1.5 - Optimal algorithms and performance
    architecture_score: float = 0.0         # Weight: 1.5 - System design and scalability
    security_score: float = 0.0            # Weight: 1.0 - Security best practices
    maintainability_score: float = 0.0       # Weight: 1.0 - Code readability and maintenance
    completeness_score: float = 0.0          # Weight: 1.0 - All requirements addressed
    
    # Secondary metrics
    innovation_score: float = 0.0             # Creative solutions and novel approaches
    documentation_score: float = 0.0           # Code documentation and comments
    testing_score: float = 0.0                # Test coverage and quality
    error_handling_score: float = 0.0          # Robust error handling
    
    # Overall scores
    weighted_average: float = 0.0
    confidence_level: float = 0.0
    
    # Metadata
    evaluation_timestamp: datetime = None
    evaluator_model: str = ""
    task_complexity: str = ""
    execution_time: float = 0.0


@dataclass
class CodeExecutionResult:
    """Result from code execution"""
    
    success: bool
    output: str
    error_message: str = ""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    exit_code: int = 0
    security_issues: List[str] = None
    performance_metrics: Dict[str, float] = None


@dataclass
class CodingTestResult:
    """Complete result for a coding task evaluation"""
    
    test_id: str
    task_category: str
    task_complexity: str
    task_description: str
    
    # Generated code
    generated_code: str
    code_language: str
    code_length: int
    line_count: int
    
    # Execution results
    execution_result: CodeExecutionResult
    
    # Evaluation metrics
    evaluation_criteria: CodingEvaluationCriteria
    
    # LLM evaluation
    llm_judge_score: float
    llm_judge_feedback: str
    
    # Metadata
    timestamp: datetime
    model_used: str
    approach_used: str


class CodingCapabilitiesEvaluator:
    """Comprehensive evaluator for AI coding capabilities"""
    
    def __init__(self, llm_manager: LLMManager = None, judge_model: str = "zai-org/GLM-4.6", use_mock: bool = False):
        self.use_mock = use_mock or llm_manager is None
        self.llm_manager = llm_manager
        self.judge_model = judge_model
        
        # Directory setup
        self.results_dir = Path("tests/results/coding_evaluation")
        self.code_dir = Path("tests/generated_code")
        self.reports_dir = Path("tests/reports/coding_evaluation")
        
        for dir_path in [self.results_dir, self.code_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = self._setup_logging()
        
        # Evaluation weights
        self.weights = {
            "correctness": 2.0,
            "efficiency": 1.5,
            "architecture": 1.5,
            "security": 1.0,
            "maintainability": 1.0,
            "completeness": 1.0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the evaluator"""
        logger = logging.getLogger("coding_capabilities_evaluator")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.results_dir / "evaluation.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    async def evaluate_coding_task(
        self,
        task_case: Dict[str, Any],
        model: str = None,
        approach: str = "conjecture"
    ) -> CodingTestResult:
        """Evaluate a single coding task"""
        
        start_time = time.time()
        
        # Mock implementation for testing
        if self.use_mock:
            return self._mock_evaluation(task_case, model, approach)
        
        try:
            self.logger.info(f"Evaluating coding task {task_case['id']} with model {model}")
            
            # Step 1: Generate code using the specified model and approach
            generated_code = await self._generate_code(task_case, model, approach)
            
            # Step 2: Execute the code and collect performance metrics
            execution_result = await self._execute_code(generated_code, task_case)
            
            # Step 3: Evaluate code quality using automated analysis
            automated_scores = self._analyze_code_quality(generated_code, task_case)
            
            # Step 4: Get LLM judge evaluation
            llm_judge_result = await self._get_llm_judge_evaluation(
                task_case, generated_code, execution_result
            )
            
            # Step 5: Compile final evaluation criteria
            evaluation_criteria = self._compile_evaluation_criteria(
                automated_scores, llm_judge_result, execution_result, task_case
            )
            
            # Create result object
            result = CodingTestResult(
                test_id=task_case["id"],
                task_category=task_case.get("category", "coding_tasks"),
                task_complexity=task_case.get("difficulty", "medium"),
                task_description=task_case.get("task", ""),
                generated_code=generated_code,
                code_language=task_case.get("language", "python"),
                code_length=len(generated_code),
                line_count=len(generated_code.split('\n')),
                execution_result=execution_result,
                evaluation_criteria=evaluation_criteria,
                llm_judge_score=llm_judge_result.get("overall_score", 0.0),
                llm_judge_feedback=llm_judge_result.get("feedback", ""),
                timestamp=datetime.utcnow(),
                model_used=model,
                approach_used=approach
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"Completed evaluation for {task_case['id']} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating coding task {task_case.get('id', 'unknown')}: {e}")
            
            # Return error result
            return CodingTestResult(
                test_id=task_case.get("id", "error"),
                task_category=task_case.get("category", "coding_tasks"),
                task_complexity=task_case.get("difficulty", "medium"),
                task_description=task_case.get("task", ""),
                generated_code="",
                code_language="",
                code_length=0,
                line_count=0,
                execution_result=CodeExecutionResult(success=False, error_message=str(e)),
                evaluation_criteria=CodingEvaluationCriteria(),
                llm_judge_score=0.0,
                llm_judge_feedback=f"Evaluation failed: {e}",
                timestamp=datetime.utcnow(),
                model_used=model,
                approach_used=approach
            )
    
    def _mock_evaluation(self, task_case: Dict[str, Any], model: str, approach: str) -> CodingTestResult:
        """Mock evaluation for testing purposes"""
        
        # Create mock evaluation criteria
        evaluation_criteria = CodingEvaluationCriteria(
            correctness_score=0.85,
            efficiency_score=0.75,
            architecture_score=0.80,
            security_score=0.70,
            maintainability_score=0.75,
            completeness_score=0.90,
            innovation_score=0.65,
            documentation_score=0.70,
            testing_score=0.60,
            error_handling_score=0.75,
            weighted_average=0.78,
            confidence_level=0.85,
            evaluation_timestamp=datetime.utcnow(),
            evaluator_model="mock_judge",
            task_complexity=task_case.get("complexity", "medium"),
            execution_time=1.5
        )
        
        # Create mock execution result
        execution_result = CodeExecutionResult(
            success=True,
            output="Mock execution completed successfully",
            execution_time=0.5,
            memory_usage=1024.0,
            cpu_usage=25.0,
            exit_code=0,
            security_issues=[],
            performance_metrics={"fps": 60, "latency_ms": 16}
        )
        
        # Create and return mock result
        return CodingTestResult(
            test_id=task_case.get("id", "mock_test"),
            task_category=task_case.get("category", "code_generation"),
            task_complexity=task_case.get("complexity", "medium"),
            task_description=task_case.get("task", "Mock coding task"),
            generated_code="# Mock generated code\ndef mock_function():\n    return 'mock_result'",
            code_language=task_case.get("language", "python"),
            code_length=100,
            line_count=5,
            execution_result=execution_result,
            evaluation_criteria=evaluation_criteria,
            llm_judge_score=0.82,
            llm_judge_feedback="Mock evaluation: Code appears functional and follows basic best practices.",
            timestamp=datetime.utcnow(),
            model_used=model or "mock_model",
            approach_used=approach
        )
    
    async def _generate_code(self, task_case: Dict[str, Any], model: str, approach: str) -> str:
        """Generate code using the specified model and approach"""
        
        task_description = task_case.get("task", "")
        specification = task_case.get("specification", "")
        language = task_case.get("language", "python")
        
        if approach == "direct":
            prompt = f"""
Please write {language} code to accomplish the following task:

Task: {task_description}

Requirements: {specification}

Provide only the complete, working code solution without explanations.
"""
        elif approach == "conjecture":
            prompt = f"""
You are an expert software engineer using Conjecture methods for systematic problem-solving.

TASK: {task_description}

SPECIFICATION: {specification}

Approach this systematically:
1. **Decompose** the problem into smaller, manageable components
2. **Plan** the architecture and identify key requirements  
3. **Research** best practices and patterns for this type of problem
4. **Work** through implementation step by step
5. **Validate** each component thoroughly
6. **Integrate** components into a complete solution

Provide the complete {language} code with comments explaining your systematic approach.
"""
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                model=model,
                temperature=0.1,
                max_tokens=4000
            )
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            raise
    
    async def _execute_code(self, code: str, task_case: Dict[str, Any]) -> CodeExecutionResult:
        """Execute code and collect performance metrics"""
        
        language = task_case.get("language", "python")
        
        try:
            # Create temporary file for execution
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file_path = f.name
            
            # Execute based on language
            if language == "python":
                return await self._execute_python_code(temp_file_path, code)
            elif language == "javascript":
                return await self._execute_javascript_code(temp_file_path, code)
            elif language == "solidity":
                return await self._execute_solidity_code(temp_file_path, code)
            else:
                return CodeExecutionResult(
                    success=False,
                    error_message=f"Unsupported language: {language}"
                )
                
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error_message=f"Code execution failed: {e}"
            )
        finally:
            # Clean up temporary file
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
            except:
                pass
    
    async def _execute_python_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """Execute Python code and collect metrics"""
        
        try:
            start_time = time.time()
            
            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(file_path)
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                return CodeExecutionResult(
                    success=True,
                    output=result.stdout,
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
            else:
                return CodeExecutionResult(
                    success=False,
                    output=result.stdout,
                    error_message=result.stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
                
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                success=False,
                error_message="Code execution timed out (30s limit)"
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error_message=f"Python execution error: {e}"
            )
    
    async def _execute_javascript_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """Execute JavaScript code using Node.js"""
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ["node", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.dirname(file_path)
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return CodeExecutionResult(
                    success=True,
                    output=result.stdout,
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
            else:
                return CodeExecutionResult(
                    success=False,
                    output=result.stdout,
                    error_message=result.stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
                
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                success=False,
                error_message="JavaScript execution timed out (30s limit)"
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error_message=f"JavaScript execution error: {e}"
            )
    
    async def _execute_solidity_code(self, file_path: str, code: str) -> CodeExecutionResult:
        """Execute Solidity code using compilation check"""
        
        try:
            start_time = time.time()
            
            # For Solidity, we'll do syntax checking using solc
            result = subprocess.run(
                ["solc", "--bin", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return CodeExecutionResult(
                    success=True,
                    output="Solidity compilation successful",
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
            else:
                return CodeExecutionResult(
                    success=False,
                    output="",
                    error_message=result.stderr,
                    execution_time=execution_time,
                    exit_code=result.returncode
                )
                
        except subprocess.TimeoutExpired:
            return CodeExecutionResult(
                success=False,
                error_message="Solidity compilation timed out (30s limit)"
            )
        except Exception as e:
            return CodeExecutionResult(
                success=False,
                error_message=f"Solidity compilation error: {e}"
            )
    
    def _analyze_code_quality(self, code: str, task_case: Dict[str, Any]) -> Dict[str, float]:
        """Analyze code quality using automated metrics"""
        
        language = task_case.get("language", "python")
        scores = {}
        
        try:
            if language == "python":
                scores.update(self._analyze_python_code(code))
            elif language == "javascript":
                scores.update(self._analyze_javascript_code(code))
            elif language == "solidity":
                scores.update(self._analyze_solidity_code(code))
            
            # Generic analysis
            scores.update(self._analyze_generic_code_quality(code))
            
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {e}")
            scores = {"error": 0.0}
        
        return scores
    
    def _analyze_python_code(self, code: str) -> Dict[str, float]:
        """Analyze Python-specific code quality"""
        
        scores = {}
        
        try:
            # Parse AST for structural analysis
            tree = ast.parse(code)
            
            # Count different statement types
            function_defs = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            class_defs = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            import_stmts = len([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
            
            # Calculate complexity metrics
            scores["structure_score"] = min(1.0, (function_defs + class_defs) / 10.0)
            scores["modularity_score"] = min(1.0, function_defs / 5.0) if function_defs > 0 else 0.5
            
            # Check for Python best practices
            scores["naming_conventions"] = 0.8 if self._check_python_naming(tree) else 0.4
            scores["docstring_coverage"] = min(1.0, self._count_docstrings(tree) / max(function_defs, 1))
            
        except SyntaxError:
            scores["syntax_validity"] = 0.0
        except Exception as e:
            self.logger.error(f"Python analysis error: {e}")
        
        return scores
    
    def _analyze_javascript_code(self, code: str) -> Dict[str, float]:
        """Analyze JavaScript-specific code quality"""
        
        scores = {}
        
        # Check for modern JS features
        modern_features = ["const", "let", "arrow functions", "async/await"]
        scores["modern_syntax"] = min(1.0, sum(1 for feature in modern_features if feature in code) / len(modern_features))
        
        # Check for error handling
        scores["error_handling"] = 0.8 if "try" in code and "catch" in code else 0.3
        
        # Check for module patterns
        scores["modularity"] = 0.7 if "export" in code or "module.exports" in code else 0.4
        
        return scores
    
    def _analyze_solidity_code(self, code: str) -> Dict[str, float]:
        """Analyze Solidity-specific code quality"""
        
        scores = {}
        
        # Check for security patterns
        security_patterns = ["require", "SafeMath", "ReentrancyGuard"]
        scores["security_practices"] = min(1.0, sum(1 for pattern in security_patterns if pattern in code) / len(security_patterns))
        
        # Check for gas optimization
        scores["gas_optimization"] = 0.6 if "gas" in code.lower() or "optimization" in code.lower() else 0.3
        
        # Check for event patterns
        scores["event_handling"] = 0.7 if "event" in code and "emit" in code else 0.4
        
        return scores
    
    def _analyze_generic_code_quality(self, code: str) -> Dict[str, float]:
        """Analyze generic code quality metrics"""
        
        scores = {}
        
        # Code length and complexity
        lines = code.split('\n')
        scores["conciseness"] = min(1.0, 100.0 / max(len(lines), 1))  # Prefer concise code
        scores["readability"] = min(1.0, self._calculate_readability_score(code))
        
        # Comment coverage
        comment_lines = len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')])
        scores["comment_coverage"] = min(1.0, comment_lines / max(len(lines), 1))
        
        # Error handling patterns
        error_patterns = ["try", "except", "catch", "throw", "raise"]
        scores["error_handling"] = min(1.0, sum(1 for pattern in error_patterns if pattern in code) / len(error_patterns))
        
        return scores
    
    def _check_python_naming(self, tree: ast.AST) -> bool:
        """Check if Python naming conventions are followed"""
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                if not name.islower() and not name.isupper():
                    return False
        return True
    
    def _count_docstrings(self, tree: ast.AST) -> int:
        """Count docstrings in Python AST"""
        
        docstring_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                if (ast.get_docstring(node) is not None and 
                    ast.get_docstring(node).strip()):
                    docstring_count += 1
        return docstring_count
    
    def _calculate_readability_score(self, code: str) -> float:
        """Calculate code readability score"""
        
        # Simple heuristic based on line length and nesting
        lines = code.split('\n')
        long_lines = len([line for line in lines if len(line) > 80])
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        
        # Score based on line length and consistency
        length_score = max(0.0, 1.0 - (long_lines / len(lines)))
        consistency_score = 1.0 - (abs(avg_line_length - 40) / 40)
        
        return (length_score + consistency_score) / 2
    
    async def _get_llm_judge_evaluation(
        self, 
        task_case: Dict[str, Any], 
        generated_code: str, 
        execution_result: CodeExecutionResult
    ) -> Dict[str, Any]:
        """Get evaluation from LLM judge"""
        
        evaluation_prompt = f"""
You are an expert software engineering evaluator. Evaluate the following code solution:

TASK: {task_case.get('task', '')}
SPECIFICATION: {task_case.get('specification', '')}
LANGUAGE: {task_case.get('language', 'python')}

GENERATED CODE:
```{generated_code}
```

EXECUTION RESULT:
- Success: {execution_result.success}
- Output: {execution_result.output[:500]}...
- Error: {execution_result.error_message}
- Execution Time: {execution_result.execution_time:.2f}s

Evaluate the code on these criteria (0.0-1.0 scale):

1. **Correctness** (Weight: 2.0): Does the code correctly solve the task?
2. **Efficiency** (Weight: 1.5): Are algorithms optimal and performant?
3. **Architecture** (Weight: 1.5): Is the system design sound and scalable?
4. **Security** (Weight: 1.0): Are security best practices followed?
5. **Maintainability** (Weight: 1.0): Is code readable and maintainable?
6. **Completeness** (Weight: 1.0): Are all requirements addressed?

Provide:
- Overall score (0.0-1.0)
- Individual criterion scores
- Specific feedback for improvement
- Confidence level in your evaluation

Format as JSON:
{{
  "overall_score": 0.85,
  "criterion_scores": {{
    "correctness": 0.9,
    "efficiency": 0.8,
    "architecture": 0.85,
    "security": 0.9,
    "maintainability": 0.8,
    "completeness": 0.9
  }},
  "feedback": "Detailed feedback here...",
  "confidence": 0.9
}}
"""
        
        try:
            response = await self.llm_manager.generate_response(
                prompt=evaluation_prompt,
                model=self.judge_model,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse JSON response
            try:
                evaluation = json.loads(response.content.strip())
                return evaluation
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "overall_score": 0.5,
                    "criterion_scores": {
                        "correctness": 0.5,
                        "efficiency": 0.5,
                        "architecture": 0.5,
                        "security": 0.5,
                        "maintainability": 0.5,
                        "completeness": 0.5
                    },
                    "feedback": response.content,
                    "confidence": 0.5
                }
                
        except Exception as e:
            self.logger.error(f"LLM judge evaluation failed: {e}")
            return {
                "overall_score": 0.0,
                "criterion_scores": {},
                "feedback": f"Evaluation failed: {e}",
                "confidence": 0.0
            }
    
    def _compile_evaluation_criteria(
        self,
        automated_scores: Dict[str, float],
        llm_judge_result: Dict[str, Any],
        execution_result: CodeExecutionResult,
        task_case: Dict[str, Any]
    ) -> CodingEvaluationCriteria:
        """Compile final evaluation criteria from all sources"""
        
        # Get LLM judge scores
        criterion_scores = llm_judge_result.get("criterion_scores", {})
        
        # Create evaluation criteria object
        criteria = CodingEvaluationCriteria(
            correctness_score=criterion_scores.get("correctness", 0.0),
            efficiency_score=criterion_scores.get("efficiency", 0.0),
            architecture_score=criterion_scores.get("architecture", 0.0),
            security_score=criterion_scores.get("security", 0.0),
            maintainability_score=criterion_scores.get("maintainability", 0.0),
            completeness_score=criterion_scores.get("completeness", 0.0),
            innovation_score=automated_scores.get("innovation", 0.0),
            documentation_score=automated_scores.get("docstring_coverage", 0.0),
            testing_score=automated_scores.get("testing", 0.0),
            error_handling_score=automated_scores.get("error_handling", 0.0),
            evaluation_timestamp=datetime.utcnow(),
            evaluator_model=self.judge_model,
            task_complexity=task_case.get("difficulty", "medium"),
            execution_time=execution_result.execution_time
        )
        
        # Calculate weighted average
        total_weight = sum(self.weights.values())
        weighted_sum = (
            criteria.correctness_score * self.weights["correctness"] +
            criteria.efficiency_score * self.weights["efficiency"] +
            criteria.architecture_score * self.weights["architecture"] +
            criteria.security_score * self.weights["security"] +
            criteria.maintainability_score * self.weights["maintainability"] +
            criteria.completeness_score * self.weights["completeness"]
        )
        
        criteria.weighted_average = weighted_sum / total_weight
        criteria.confidence_level = llm_judge_result.get("confidence", 0.5)
        
        return criteria
    
    async def evaluate_coding_capabilities(
        self,
        test_cases: List[Dict[str, Any]],
        models: List[str],
        approaches: List[str] = ["direct", "conjecture"]
    ) -> List[CodingTestResult]:
        """Evaluate coding capabilities across multiple test cases"""
        
        self.logger.info(f"Starting coding capabilities evaluation with {len(test_cases)} test cases")
        
        all_results = []
        
        for i, test_case in enumerate(test_cases):
            self.logger.info(f"Processing test case {i+1}/{len(test_cases)}: {test_case['id']}")
            
            for model in models:
                for approach in approaches:
                    result = await self.evaluate_coding_task(test_case, model, approach)
                    all_results.append(result)
                    
                    # Save individual result
                    await self._save_test_result(result)
        
        self.logger.info(f"Completed {len(all_results)} coding evaluations")
        return all_results
    
    async def _save_test_result(self, result: CodingTestResult):
        """Save individual test result"""
        
        result_file = self.results_dir / f"coding_result_{result.test_id}_{result.model_used}_{result.approach_used}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save generated code
        code_file = self.code_dir / f"{result.test_id}_{result.model_used}_{result.approach_used}.{result.code_language}"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(result.generated_code)
        
        # Save result data
        result_data = asdict(result)
        result_data["evaluation_timestamp"] = result.evaluation_timestamp.isoformat()
        result_data["timestamp"] = result.timestamp.isoformat()
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load coding test cases from JSON files"""
        
        test_cases = []
        
        # Define possible test case file paths
        test_case_paths = [
            "experiments/experiments/test_cases/coding_tasks_simple.json",
            "experiments/experiments/test_cases/coding_tasks_medium.json",
            "experiments/experiments/test_cases/coding_tasks_complex.json",
            "experiments/experiments/test_cases/coding_tasks_debugging.json",
            "experiments/experiments/test_cases/coding_tasks_algorithms.json"
        ]
        
        for file_path in test_case_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cases = json.load(f)
                        if isinstance(cases, list):
                            test_cases.extend(cases)
                            print(f"Loaded {len(cases)} test cases from {file_path}")
                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        # If no test cases found, create some mock ones for testing
        if not test_cases:
            print("No test case files found, creating mock test cases...")
            test_cases = [
                {
                    "id": "mock_001",
                    "task": "Write a function that adds two numbers",
                    "expected_output": "def add(a, b): return a + b",
                    "category": "code_generation",
                    "complexity": "simple",
                    "language": "python"
                },
                {
                    "id": "mock_002",
                    "task": "Write a function that sorts a list of numbers",
                    "expected_output": "def sort_list(numbers): return sorted(numbers)",
                    "category": "algorithms",
                    "complexity": "medium",
                    "language": "python"
                },
                {
                    "id": "mock_003",
                    "task": "Fix the bug in this function: def divide(a, b): return a / b",
                    "expected_output": "def divide(a, b): return a / b if b != 0 else None",
                    "category": "debugging",
                    "complexity": "simple",
                    "language": "python"
                }
            ]
        
        return test_cases
    
    async def generate_comprehensive_report(
        self, 
        results: List[CodingTestResult]
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        self.logger.info("Generating comprehensive coding capabilities report")
        
        # Group results by model and approach
        model_results = {}
        approach_results = {}
        
        for result in results:
            model_key = result.model_used
            approach_key = result.approach_used
            
            if model_key not in model_results:
                model_results[model_key] = []
            if approach_key not in approach_results:
                approach_results[approach_key] = []
            
            model_results[model_key].append(result)
            approach_results[approach_key].append(result)
        
        # Calculate statistics
        report_lines = [
            "# Coding Capabilities Evaluation Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Total Test Cases Evaluated**: {len(results)}",
            f"**Models Tested**: {', '.join(model_results.keys())}",
            f"**Approaches Evaluated**: {', '.join(approach_results.keys())}",
            "",
            "## Performance by Model",
            ""
        ]
        
        # Model performance summary
        for model, model_results_list in model_results.items():
            if model_results_list:
                avg_score = statistics.mean([r.evaluation_criteria.weighted_average for r in model_results_list])
                success_rate = len([r for r in model_results_list if r.execution_result.success]) / len(model_results_list)
                
                report_lines.extend([
                    f"### {model}",
                    f"- **Average Weighted Score**: {avg_score:.3f}",
                    f"- **Success Rate**: {success_rate:.1%}",
                    f"- **Total Evaluations**: {len(model_results_list)}",
                    ""
                ])
        
        # Approach comparison
        report_lines.extend([
            "## Approach Comparison (Direct vs Conjecture)",
            ""
        ])
        
        for approach, approach_results_list in approach_results.items():
            if approach_results_list:
                avg_score = statistics.mean([r.evaluation_criteria.weighted_average for r in approach_results_list])
                success_rate = len([r for r in approach_results_list if r.execution_result.success]) / len(approach_results_list)
                
                report_lines.extend([
                    f"### {approach.title()} Approach",
                    f"- **Average Weighted Score**: {avg_score:.3f}",
                    f"- **Success Rate**: {success_rate:.1%}",
                    f"- **Total Evaluations**: {len(approach_results_list)}",
                    ""
                ])
        
        # Detailed analysis by complexity
        report_lines.extend([
            "## Analysis by Task Complexity",
            ""
        ])
        
        complexity_results = {"easy": [], "medium": [], "hard": []}
        for result in results:
            complexity = result.task_complexity
            if complexity in complexity_results:
                complexity_results[complexity].append(result)
        
        for complexity, complexity_list in complexity_results.items():
            if complexity_list:
                avg_score = statistics.mean([r.evaluation_criteria.weighted_average for r in complexity_list])
                success_rate = len([r for r in complexity_list if r.execution_result.success]) / len(complexity_list)
                
                report_lines.extend([
                    f"### {complexity.title()} Tasks",
                    f"- **Average Weighted Score**: {avg_score:.3f}",
                    f"- **Success Rate**: {success_rate:.1%}",
                    f"- **Total Tasks**: {len(complexity_list)}",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### For Model Improvement",
            "1. **Code Generation**: Focus on improving correctness for complex tasks",
            "2. **Algorithm Design**: Enhance efficiency and optimization awareness",
            "3. **Security Practices**: Strengthen security implementation patterns",
            "",
            "### For Conjecture Method Enhancement",
            "1. **Task Decomposition**: Improve systematic breakdown of complex problems",
            "2. **Planning Phase**: Enhance architecture design capabilities",
            "3. **Validation**: Strengthen testing and validation approaches",
            "",
            "## Statistical Analysis",
            ""
        ])
        
        # Add statistical significance testing if we have enough data
        if len(results) >= 20:
            direct_results = [r for r in results if r.approach_used == "direct"]
            conjecture_results = [r for r in results if r.approach_used == "conjecture"]
            
            if direct_results and conjecture_results:
                from scipy import stats
                
                direct_scores = [r.evaluation_criteria.weighted_average for r in direct_results]
                conjecture_scores = [r.evaluation_criteria.weighted_average for r in conjecture_results]
                
                t_stat, p_value = stats.ttest_ind(direct_scores, conjecture_scores)
                effect_size = (statistics.mean(conjecture_scores) - statistics.mean(direct_scores)) / statistics.stdev(direct_scores)
                
                report_lines.extend([
                    f"**Statistical Comparison** (Direct vs Conjecture):",
                    f"- **T-statistic**: {t_stat:.3f}",
                    f"- **P-value**: {p_value:.4f}",
                    f"- **Effect Size**: {effect_size:.3f}",
                    f"- **Significance**: {'✅ Significant (p<0.05)' if p_value < 0.05 else '❌ Not significant'}",
                    ""
                ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.reports_dir / f"coding_capabilities_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Comprehensive report saved to: {report_file}")
        return report_content


async def main():
    """Main function to run coding capabilities evaluation"""
    
    # Use mock implementation for testing
    print("Using mock implementation for coding capabilities evaluation...")
    
    # Initialize evaluator with mock
    evaluator = CodingCapabilitiesEvaluator(use_mock=True)
    
    # Load coding test cases
    test_cases = []
    test_case_files = [
        "experiments/experiments/test_cases/coding_tasks_simple.json",
        "experiments/experiments/test_cases/coding_tasks_medium.json",
        "experiments/experiments/test_cases/coding_tasks_complex.json",
        "experiments/experiments/test_cases/coding_tasks_debugging.json",
        "experiments/experiments/test_cases/coding_tasks_algorithms.json"
    ]
    
    for file_path in test_case_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                cases = json.load(f)
                test_cases.extend(cases)
                print(f"Loaded {len(cases)} test cases from {file_path}")
    
    print(f"Total test cases loaded: {len(test_cases)}")
    
    # Run evaluation
    models = ["ibm/granite-4-h-tiny", "zai-org/GLM-4.6"]
    approaches = ["direct", "conjecture"]
    
    results = await evaluator.evaluate_coding_capabilities(test_cases, models, approaches)
    
    # Generate report
    report = await evaluator.generate_comprehensive_report(results)
    print("\n" + "="*60)
    print("CODING CAPABILITIES EVALUATION COMPLETE")
    print("="*60)
    print(f"Total Evaluations: {len(results)}")
    print(f"Reports saved to: {evaluator.reports_dir}")
    print(f"Results saved to: {evaluator.results_dir}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())