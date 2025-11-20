"""
Async Claim Evaluation Service
Confidence-driven claim processing with priority queue management
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

from core.models import Claim, ClaimState, ClaimType
from processing.llm_bridge import LLMBridge, LLMRequest
from processing.context_collector import ContextCollector


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

    def __lt__(self, other):
        return self.priority < other.priority


class AsyncClaimEvaluationService:
    """
    Async Claim Evaluation Service
    Continuously processes claims based on priority and confidence
    """

    def __init__(
        self,
        llm_bridge: LLMBridge,
        context_collector: ContextCollector,
        max_concurrent_evaluations: int = 5,
        evaluation_timeout: int = 300,
    ):
        self.llm_bridge = llm_bridge
        self.context_collector = context_collector
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

        # Statistics
        self._stats = {
            "evaluations_started": 0,
            "evaluations_completed": 0,
            "evaluations_failed": 0,
            "average_evaluation_time": 0.0,
            "queue_depth": 0,
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

        if hasattr(self, "_event_task"):
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Async Claim Evaluation Service stopped")

    async def submit_claim(self, claim: Claim, priority_boost: int = 0):
        """Submit a claim for evaluation"""
        if not claim.id:
            raise ValueError("Claim must have an ID")

        # Calculate priority based on multiple factors
        base_priority = self._calculate_priority(claim)
        adjusted_priority = base_priority + priority_boost

        task = EvaluationTask(priority=adjusted_priority, claim_id=claim.id)

        async with self._queue_lock:
            heapq.heappush(self._evaluation_queue, task)
            self._stats["queue_depth"] = len(self._evaluation_queue)

        self.logger.info(
            f"Submitted claim {claim.id} for evaluation with priority {adjusted_priority}"
        )

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
                    await self._evaluate_claim(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in evaluation loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

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
                    data={"priority": task.priority},
                )
            )

            # Get claim (this would come from data layer)
            claim = await self._get_claim(claim_id)
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")

            # Build context from existing claims
            context = await self.context_collector.build_context(claim)

            # Evaluate with confidence-driven continuation
            updated_claim = await self._confidence_driven_evaluation(claim, context)

            # Update claim in data layer
            await self._save_claim(updated_claim)

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
        self, claim: Claim, context: List[Claim]
    ) -> Claim:
        """
        Confidence-driven evaluation where LLM decides when to stop exploring
        """
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        current_claim = claim

        while (
            iteration < max_iterations and current_claim.state != ClaimState.VALIDATED
        ):
            iteration += 1

            # Build evaluation prompt
            prompt = self._build_evaluation_prompt(current_claim, context, iteration)

            # Create LLM request
            llm_request = LLMRequest(
                prompt=prompt,
                context_claims=context,
                max_tokens=2048,
                temperature=0.7,
                task_type="evaluate_claim",
            )

            # Process with LLM
            response = self.llm_bridge.process(llm_request)

            if not response.success:
                raise Exception(f"LLM processing failed: {response.errors}")

            # Parse response for tool calls, new claims, and confidence updates
            tool_calls = self._parse_tool_calls(response.content)
            new_claims = self._parse_new_claims(response.content)
            confidence_update = self._parse_confidence_update(response.content)

            # Execute tool calls if any
            if tool_calls:
                await self._execute_tool_calls(tool_calls, current_claim.id)
                # Continue loop after tool execution
                continue

            # Create new claims if any
            if new_claims:
                for new_claim_data in new_claims:
                    new_claim = Claim(**new_claim_data)
                    await self._save_claim(new_claim)
                # Continue loop after creating claims
                continue

            # Update confidence and mark as validated if LLM is confident
            if confidence_update is not None:
                current_claim.confidence = confidence_update
                current_claim.updated = datetime.utcnow()

                # Mark as validated if confidence is high enough
                if confidence_update >= 0.8:
                    current_claim.state = ClaimState.VALIDATED
                    break

            # If no tool calls, new claims, or confidence updates, mark as validated
            current_claim.state = ClaimState.VALIDATED
            break

        return current_claim

    def _build_evaluation_prompt(
        self, claim: Claim, context: List[Claim], iteration: int
    ) -> str:
        """Build evaluation prompt for LLM"""
        context_text = "\n".join(
            [c.format_for_context() for c in context[:10]]
        )  # Limit context

        return f"""You are evaluating the following claim (iteration {iteration}):

CLAIM: {claim.content}
CURRENT CONFIDENCE: {claim.confidence}
TYPE: {claim.type[0].value if claim.type else "unknown"}

CONTEXT:
{context_text}

EVALUATION INSTRUCTIONS:
1. Assess the claim's accuracy based on the context
2. If you need more information, make tool calls (max 1 tool call per response)
3. If you discover related information, create new claims
4. When satisfied, set a confidence level (0.0-1.0)
5. Mark evaluation as complete when confidence >= 0.8 or no more exploration needed

RESPONSE FORMAT:
- For tool calls: ToolCall(tool_name, parameters)
- For new claims: NewClaim(content, confidence, type)
- For confidence: Confidence(value)
- To complete: Complete()

Current claim needs evaluation. Proceed with analysis."""

    def _calculate_priority(self, claim: Claim) -> int:
        """Calculate evaluation priority for a claim"""
        priority = 1000  # Base priority

        # Higher priority for higher confidence claims
        priority -= int(claim.confidence * 200)

        # Higher priority for certain types
        if ClaimType.CONCEPT in claim.type:
            priority -= 100
        elif ClaimType.THESIS in claim.type:
            priority -= 50

        # Lower priority for older claims
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

    async def _get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get claim from data layer (mock implementation)"""
        # TODO: Implement actual data layer integration
        return None

    async def _save_claim(self, claim: Claim):
        """Save claim to data layer (mock implementation)"""
        # TODO: Implement actual data layer integration
        pass

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response"""
        # TODO: Implement proper parsing
        return []

    def _parse_new_claims(self, response: str) -> List[Dict[str, Any]]:
        """Parse new claims from LLM response"""
        # TODO: Implement proper parsing
        return []

    def _parse_confidence_update(self, response: str) -> Optional[float]:
        """Parse confidence update from LLM response"""
        # TODO: Implement proper parsing
        return None

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], claim_id: str
    ):
        """Execute tool calls and emit events"""
        for tool_call in tool_calls:
            await self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id, event_type="tool_called", data=tool_call
                )
            )
            # TODO: Implement actual tool execution

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            "active_evaluations": len(self._active_evaluations),
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "service_running": self._running,
        }
