"""
Async Claim Evaluation Service
Confidence-driven claim processing with priority queue management and dirty claim handling.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import heapq

from ..core.models import Claim, ClaimState, ClaimType, ToolCall, ExecutionResult
from .bridge import LLMBridge, LLMRequest
from .context_collector import ContextCollector
from .response_parser import ResponseParser
from .tool_executor import ToolExecutor
from ..data.data_manager import DataManager


class ClaimScope(Enum):
    """Claim scope enumeration"""

    SESSION = "session"
    USER = "user"
    PROJECT = "project"
    TEAM = "team"
    GLOBAL = "global"


@dataclass
class EvaluationEvent:
    """Event emitted during claim evaluation"""

    claim_id: str
    event_type: str  # started, confidence_updated, tool_called, completed, error
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationTask:
    """Task for claim evaluation with priority"""

    priority: int
    claim_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    attempts: int = 0
    max_attempts: int = 3
    last_attempt: Optional[datetime] = None
    is_dirty: bool = False

    def __lt__(self, other):
        return self.priority < other.priority


class AsyncClaimEvaluationService:
    """
    Async Claim Evaluation Service
    Continuously processes claims based on priority and confidence.
    Integrates dirty claim processing for batch updates.
    """

    def __init__(
        self,
        llm_bridge: LLMBridge,
        context_collector: ContextCollector,
        data_manager: DataManager,
        tool_executor: ToolExecutor,
        max_concurrent_evaluations: int = 5,
        evaluation_timeout: int = 300,
    ):
        self.llm_bridge = llm_bridge
        self.context_collector = context_collector
        self.data_manager = data_manager
        self.tool_executor = tool_executor
        self.response_parser = ResponseParser()

        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.evaluation_timeout = evaluation_timeout

        # Priority queue for evaluation tasks
        self._evaluation_queue: List[EvaluationTask] = []
        self._queue_lock = asyncio.Lock()

        # Active evaluations tracking
        self._active_evaluations: Set[str] = set()
        self._evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)

        # Event system
        self._event_subscribers: List[callable] = []
        self._event_queue = asyncio.Queue()

        # Service state
        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "evaluations_started": 0,
            "evaluations_completed": 0,
            "evaluations_failed": 0,
            "average_evaluation_time": 0.0,
            "queue_depth": 0,
            "dirty_claims_processed": 0,
        }

        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the evaluation service"""
        if self._running:
            return

        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        self._event_task = asyncio.create_task(self._event_loop())

        self.logger.info("Async Claim Evaluation Service started")

    async def stop(self):
        """Stop the evaluation service"""
        self._running = False

        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Async Claim Evaluation Service stopped")

    async def submit_claim(
        self, claim: Claim, priority_boost: int = 0, is_dirty: bool = False
    ):
        """Submit a claim for evaluation"""
        if not claim.id:
            raise ValueError("Claim must have an ID")

        # Calculate priority based on multiple factors
        base_priority = self._calculate_priority(claim)
        adjusted_priority = base_priority + priority_boost

        # Dirty claims get higher priority
        if is_dirty:
            adjusted_priority -= 500

        task = EvaluationTask(
            priority=adjusted_priority, claim_id=claim.id, is_dirty=is_dirty
        )

        async with self._queue_lock:
            heapq.heappush(self._evaluation_queue, task)
            self._stats["queue_depth"] = len(self._evaluation_queue)

        self.logger.info(
            f"Submitted claim {claim.id} for evaluation with priority {adjusted_priority} (dirty={is_dirty})"
        )

    async def process_dirty_claims_batch(self):
        """
        Scan for and submit dirty claims for evaluation.
        Replaces the standalone DirtyEvaluator functionality.
        """
        try:
            # Find dirty claims using DataManager
            # Assuming DataManager has a method to filter by dirty status or we iterate
            # For now, we'll assume a filter method exists or we implement a scan
            # This is a placeholder for the actual data query
            dirty_claims = await self.data_manager.get_dirty_claims()

            count = 0
            for claim in dirty_claims:
                await self.submit_claim(claim, is_dirty=True)
                count += 1

            self._stats["dirty_claims_processed"] += count
            self.logger.info(f"Queued {count} dirty claims for evaluation")

        except Exception as e:
            self.logger.error(f"Error processing dirty claims batch: {e}")

    def subscribe_to_events(self, callback: callable):
        """Subscribe to evaluation events"""
        self._event_subscribers.append(callback)

    async def _evaluation_loop(self):
        """Main evaluation loop"""
        while self._running:
            try:
                # Get next evaluation task
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(1)  # No tasks, wait briefly
                    continue

                # Check if already being evaluated
                if task.claim_id in self._active_evaluations:
                    continue

                # Evaluate claim with semaphore control
                async with self._evaluation_semaphore:
                    # Create a task to run evaluation so we don't block the loop
                    asyncio.create_task(self._evaluate_claim_wrapper(task))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _evaluate_claim_wrapper(self, task: EvaluationTask):
        """Wrapper to handle evaluation and semaphore release implicitly via task completion"""
        try:
            await self._evaluate_claim(task)
        except Exception as e:
            self.logger.error(f"Unhandled error in evaluation wrapper: {e}")

    async def _event_loop(self):
        """Event broadcasting loop"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._broadcast_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in event loop: {e}")

    async def _get_next_task(self) -> Optional[EvaluationTask]:
        """Get next evaluation task from queue"""
        async with self._queue_lock:
            while self._evaluation_queue:
                task = heapq.heappop(self._evaluation_queue)
                self._stats["queue_depth"] = len(self._evaluation_queue)

                # Skip if already being evaluated
                if task.claim_id not in self._active_evaluations:
                    return task

                # If already active, we might want to re-queue it if it's a new request,
                # but for now we just drop it as the active one will handle it
                # or we could put it back? Let's just skip.

            return None

    async def _evaluate_claim(self, task: EvaluationTask):
        """Evaluate a single claim"""
        claim_id = task.claim_id
        self._active_evaluations.add(claim_id)

        start_time = time.time()

        try:
            # Emit started event
            await self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id,
                    event_type="started",
                    data={"priority": task.priority, "is_dirty": task.is_dirty},
                )
            )

            # Get claim
            claim = await self.data_manager.get_claim(claim_id)
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")

            # Build context
            context_result = await self.context_collector.collect_context_for_claim(
                claim.content, {}
            )
            # Convert context result to list of claims/strings for LLM
            # The context_collector returns a dict with 'skills', 'samples'
            # We need to format this for the LLM

            # Evaluate with confidence-driven continuation
            updated_claim = await self._confidence_driven_evaluation(
                claim, context_result
            )

            # Clear dirty flag if it was dirty
            if task.is_dirty:
                updated_claim.dirty = False
                updated_claim.dirty_reasons = []

            # Update claim in data layer
            await self.data_manager.update_claim(updated_claim)

            # Update statistics
            evaluation_time = time.time() - start_time
            self._update_stats(evaluation_time, success=True)

            # Emit completed event
            await self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id,
                    event_type="completed",
                    data={
                        "final_confidence": updated_claim.confidence,
                        "evaluation_time": evaluation_time,
                    },
                )
            )

            self.logger.info(
                f"Completed evaluation of claim {claim_id} in {evaluation_time:.2f}s"
            )

        except Exception as e:
            evaluation_time = time.time() - start_time
            self._update_stats(evaluation_time, success=False)

            # Emit error event
            await self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id,
                    event_type="error",
                    data={"error": str(e), "evaluation_time": evaluation_time},
                )
            )

            self.logger.error(f"Failed to evaluate claim {claim_id}: {e}")

            # Retry logic
            task.attempts += 1
            task.last_attempt = datetime.utcnow()

            if task.attempts < task.max_attempts:
                # Re-queue with lower priority
                task.priority += 100  # Lower priority for retries
                async with self._queue_lock:
                    heapq.heappush(self._evaluation_queue, task)
                    self._stats["queue_depth"] = len(self._evaluation_queue)

        finally:
            self._active_evaluations.discard(claim_id)

    async def _confidence_driven_evaluation(
        self, claim: Claim, context_result: Dict[str, Any]
    ) -> Claim:
        """
        Confidence-driven evaluation where LLM decides when to stop exploring
        """
        max_iterations = 5  # Prevent infinite loops
        iteration = 0

        current_claim = claim

        while (
            iteration < max_iterations and current_claim.state != ClaimState.VALIDATED
        ):
            iteration += 1

            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(
                current_claim, context_result, iteration
            )

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7,
                task_type="evaluate_claim",
            )

            # Process with LLM
            response = self.llm_bridge.process(llm_request)

            if not response.success:
                raise Exception(f"LLM processing failed: {response.errors}")

            # Parse response
            parsed_response = self.response_parser.parse_response(response.content)

            # Execute tool calls if any
            if parsed_response.tool_calls:
                await self._execute_tool_calls(
                    parsed_response.tool_calls, current_claim.id
                )
                # In a real loop, we would feed the tool results back into the next iteration
                # For now, we just continue, assuming the tool execution might have side effects
                # or we'd append results to context in a more complex version.
                # To keep it simple, we assume tool calls are for information gathering
                # and we might need to re-prompt with results.
                # TODO: Append tool results to context for next iteration
                continue

            # Check for confidence updates in text content (heuristic or structured)
            # The ResponseParser mainly handles tool calls.
            # We might need a specific parser for confidence or rely on the LLM to output a tool call for it.
            # Let's assume the LLM uses a specific tool for updating confidence/claim.

            # If no tool calls, we check if we can extract confidence/validation
            # This part depends on the prompt instructions.
            # If the LLM outputs "Confidence: 0.9", we parse it.

            confidence_update = self._extract_confidence(parsed_response.text_content)
            if confidence_update is not None:
                current_claim.confidence = confidence_update
                if confidence_update >= 0.8:
                    current_claim.state = ClaimState.VALIDATED
                    break

            # If we have no tool calls and no explicit confidence update,
            # but we have text, we might assume it's an analysis.
            # We stop if we run out of iterations.

        return current_claim

    def _build_evaluation_prompt(
        self, claim: Claim, context_result: Dict[str, Any], iteration: int
    ) -> str:
        """Build evaluation prompt for LLM"""
        context_str = self.context_collector.build_llm_context_string(context_result)

        return f"""You are evaluating the following claim (iteration {iteration}):

CLAIM: {claim.content}
CURRENT CONFIDENCE: {claim.confidence}
TYPE: {claim.type[0].value if claim.type else "unknown"}

CONTEXT:
{context_str}

EVALUATION INSTRUCTIONS:
1. Assess the claim's accuracy based on the context.
2. If you need more information, make tool calls.
3. If you are satisfied, provide a confidence score in the format "Confidence: X.X".
4. Mark evaluation as complete when confidence >= 0.8.

RESPONSE FORMAT:
- Use the available tools if needed.
- Provide reasoning in plain text.
- End with "Confidence: 0.X" if you can assess it.
"""

    def _extract_confidence(self, text: str) -> Optional[float]:
        """Extract confidence score from text"""
        import re

        match = re.search(r"Confidence:\s*(0\.\d+|1\.0|0|1)", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _calculate_priority(self, claim: Claim) -> int:
        """Calculate evaluation priority for a claim"""
        priority = 1000  # Base priority

        # Higher priority for higher confidence claims
        priority -= int(claim.confidence * 200)

        # Higher priority for certain types
        if claim.type and ClaimType.CONCEPT in claim.type:
            priority -= 100
        elif claim.type and ClaimType.THESIS in claim.type:
            priority -= 50

        # Lower priority for older claims
        if claim.created:
            age_hours = (datetime.utcnow() - claim.created).total_seconds() / 3600
            priority += int(age_hours * 2)

        return priority

    async def _emit_event(self, event: EvaluationEvent):
        """Emit an evaluation event"""
        await self._event_queue.put(event)

    async def _broadcast_event(self, event: EvaluationEvent):
        """Broadcast event to subscribers"""
        for callback in self._event_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")

    def _update_stats(self, evaluation_time: float, success: bool):
        """Update evaluation statistics"""
        if success:
            self._stats["evaluations_completed"] += 1
        else:
            self._stats["evaluations_failed"] += 1

        # Update average time
        total_evaluations = (
            self._stats["evaluations_completed"] + self._stats["evaluations_failed"]
        )
        if total_evaluations > 0:
            current_avg = self._stats["average_evaluation_time"]
            self._stats["average_evaluation_time"] = (
                current_avg * (total_evaluations - 1) + evaluation_time
            ) / total_evaluations

    async def _execute_tool_calls(self, tool_calls: List[ToolCall], claim_id: str):
        """Execute tool calls and emit events"""
        for tool_call in tool_calls:
            await self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id, event_type="tool_called", data=tool_call.__dict__
                )
            )

            # Execute tool
            result = await self.tool_executor.execute_tool(
                tool_name=tool_call.name, params=tool_call.parameters
            )

            # Log result or handle it
            # In a full implementation, we'd return this to the loop
            self.logger.info(f"Tool execution result: {result}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            "active_evaluations": len(self._active_evaluations),
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "service_running": self._running,
        }
