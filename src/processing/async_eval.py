"""
Async Claim Evaluation Service
Confidence-driven claim processing with priority queue management and dirty claim handling.
OPTIMIZED: Parallel processing, batch operations, intelligent caching
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
import hashlib

from ..core.models import Claim, ClaimState, ToolCall, ExecutionResult
from .unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
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

        # OPTIMIZATION: Performance monitoring and caching
        self._performance_stats = {
            "context_collection_time": [],
            "llm_processing_time": [],
            "tool_execution_time": [],
            "database_update_time": [],
            "total_evaluation_time": [],
        }
        
        # OPTIMIZATION: Intelligent caching for context and results
        self._context_cache = {}
        self._evaluation_result_cache = {}
        self._cache_ttl = 600  # 10 minutes
        self._max_cache_size = 200

        # OPTIMIZATION: Batch processing
        self._batch_size = 5
        self._batch_timeout = 2.0  # seconds
        self._pending_batch: List[EvaluationTask] = []
        self._batch_timer: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "evaluations_started": 0,
            "evaluations_completed": 0,
            "evaluations_failed": 0,
            "average_evaluation_time": 0.0,
            "queue_depth": 0,
            "dirty_claims_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_evaluations": 0,
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
        OPTIMIZATION: Batch process dirty claims for evaluation
        """
        try:
            # Find dirty claims using DataManager
            dirty_claims = await self.data_manager.get_dirty_claims()

            if not dirty_claims:
                return

            # OPTIMIZATION: Batch submit dirty claims
            batch_tasks = []
            for claim in dirty_claims:
                batch_tasks.append(self.submit_claim(claim, is_dirty=True))

            # Execute all submissions in parallel
            await asyncio.gather(*batch_tasks, return_exceptions=True)

            self._stats["dirty_claims_processed"] += len(dirty_claims)
            self._stats["batch_evaluations"] += 1
            self.logger.info(f"Batch queued {len(dirty_claims)} dirty claims for evaluation")

        except Exception as e:
            self.logger.error(f"Error processing dirty claims batch: {e}")

    def subscribe_to_events(self, callback: callable):
        """Subscribe to evaluation events"""
        self._event_subscribers.append(callback)

    async def _evaluation_loop(self):
        """OPTIMIZATION: Enhanced evaluation loop with batch processing"""
        while self._running:
            try:
                # OPTIMIZATION: Try to collect batch first
                batch_tasks = await self._collect_evaluation_batch()
                
                if batch_tasks:
                    # Process batch in parallel
                    await self._process_evaluation_batch(batch_tasks)
                    continue
                
                # Fallback to individual processing
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(0.1)  # Reduced wait time for better responsiveness
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
                await asyncio.sleep(1)  # Reduced pause time for better recovery

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

    async def _collect_evaluation_batch(self) -> List[EvaluationTask]:
        """OPTIMIZATION: Collect a batch of evaluation tasks"""
        batch = []
        
        async with self._queue_lock:
            # Try to collect up to batch_size tasks
            while len(batch) < self._batch_size and self._evaluation_queue:
                task = heapq.heappop(self._evaluation_queue)
                
                # Skip if already being evaluated
                if task.claim_id not in self._active_evaluations:
                    batch.append(task)
                    self._active_evaluations.add(task.claim_id)
        
        if batch:
            self._stats["batch_evaluations"] += 1
            
        return batch

    async def _process_evaluation_batch(self, tasks: List[EvaluationTask]):
        """OPTIMIZATION: Process a batch of evaluation tasks in parallel"""
        if not tasks:
            return
            
        # Create semaphore for batch control
        batch_semaphore = asyncio.Semaphore(min(len(tasks), self.max_concurrent_evaluations))
        
        async def evaluate_with_semaphore(task):
            async with batch_semaphore:
                try:
                    await self._evaluate_claim_optimized(task)
                finally:
                    self._active_evaluations.discard(task.claim_id)
        
        # Execute all tasks in parallel
        await asyncio.gather(
            *[evaluate_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

    async def _evaluate_claim(self, task: EvaluationTask):
        """Evaluate a single claim (legacy method)"""
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

    async def _evaluate_claim_optimized(self, task: EvaluationTask):
        """OPTIMIZATION: Optimized claim evaluation with caching and parallel processing"""
        claim_id = task.claim_id
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

            # OPTIMIZATION: Parallel claim retrieval and context collection
            claim_future = self.data_manager.get_claim(claim_id)
            context_future = self._get_cached_context(claim_id)
            
            # Wait for both operations in parallel
            claim, context_result = await asyncio.gather(claim_future, context_future)
            
            if not claim:
                raise ValueError(f"Claim {claim_id} not found")

            # OPTIMIZATION: Check evaluation cache
            cache_key = self._generate_evaluation_cache_key(claim, context_result)
            cached_result = self._get_from_cache(cache_key, "evaluation")
            
            if cached_result:
                self._stats["cache_hits"] += 1
                updated_claim = cached_result
            else:
                self._stats["cache_misses"] += 1
                
                # Evaluate with confidence-driven continuation
                updated_claim = await self._confidence_driven_evaluation_optimized(
                    claim, context_result
                )
                
                # Cache the result
                self._add_to_cache(cache_key, updated_claim, "evaluation")

            # Clear dirty flag if it was dirty
            if task.is_dirty:
                updated_claim.dirty = False
                updated_claim.dirty_reasons = []

            # OPTIMIZATION: Parallel database update and event emission
            db_update_future = self.data_manager.update_claim(updated_claim)
            
            # Update statistics
            evaluation_time = time.time() - start_time
            self._performance_stats["total_evaluation_time"].append(evaluation_time)
            self._update_stats(evaluation_time, success=True)

            # Emit completed event
            event_future = self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id,
                    event_type="completed",
                    data={
                        "final_confidence": updated_claim.confidence,
                        "evaluation_time": evaluation_time,
                    },
                )
            )

            # Wait for database update and event emission in parallel
            await asyncio.gather(db_update_future, event_future)

            self.logger.info(
                f"Completed optimized evaluation of claim {claim_id} in {evaluation_time:.2f}s"
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

    async def _get_cached_context(self, claim_id: str) -> Dict[str, Any]:
        """OPTIMIZATION: Get cached context for claim"""
        # First try to get claim content for context generation
        claim = await self.data_manager.get_claim(claim_id)
        if not claim:
            return {}
        
        # Generate cache key for context
        cache_key = self._generate_context_cache_key(claim.content)
        cached_context = self._get_from_cache(cache_key, "context")
        
        if cached_context:
            return cached_context
        
        # Collect context and cache it
        context_start = time.time()
        context_result = await self.context_collector.collect_context_for_claim(
            claim.content, {}
        )
        context_time = time.time() - context_start
        self._performance_stats["context_collection_time"].append(context_time)
        
        # Cache the context
        self._add_to_cache(cache_key, context_result, "context")
        
        return context_result

    def _generate_context_cache_key(self, content: str) -> str:
        """Generate cache key for context"""
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_evaluation_cache_key(self, claim: Claim, context: Dict[str, Any]) -> str:
        """Generate cache key for evaluation result"""
        key_data = f"{claim.id}:{claim.confidence}:{hash(str(context))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """OPTIMIZATION: Get item from cache with TTL check"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        
        if cache_key in cache:
            cached_item = cache[cache_key]
            timestamp = cached_item.get("timestamp", 0)
            
            # Check if cache is still valid
            if time.time() - timestamp < self._cache_ttl:
                return cached_item["data"]
            else:
                # Remove expired cache item
                del cache[cache_key]
        
        return None

    def _add_to_cache(self, cache_key: str, data: Any, cache_type: str) -> None:
        """OPTIMIZATION: Add item to cache with timestamp and size management"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        cache[cache_key] = {
            "data": data,
            "timestamp": time.time(),
        }
        
        # Maintain cache size
        if len(cache) > self._max_cache_size:
            # Remove oldest items
            oldest_keys = sorted(
                cache.keys(),
                key=lambda k: cache[k]["timestamp"],
            )[:self._max_cache_size // 4]  # Remove 25% of items
            
            for key in oldest_keys:
                del cache[key]

    async def _confidence_driven_evaluation_optimized(
        self, claim: Claim, context_result: Dict[str, Any]
    ) -> Claim:
        """
        OPTIMIZATION: Optimized confidence-driven evaluation with parallel processing
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

            # OPTIMIZATION: Time LLM processing
            llm_start = time.time()
            response = self.llm_bridge.process(llm_request)
            llm_time = time.time() - llm_start
            self._performance_stats["llm_processing_time"].append(llm_time)

            if not response.success:
                raise Exception(f"LLM processing failed: {response.errors}")

            # Parse response
            parsed_response = self.response_parser.parse_response(response.content)

            # OPTIMIZATION: Parallel tool execution if any
            if parsed_response.tool_calls:
                tool_start = time.time()
                await self._execute_tool_calls_parallel(
                    parsed_response.tool_calls, current_claim.id
                )
                tool_time = time.time() - tool_start
                self._performance_stats["tool_execution_time"].append(tool_time)
                
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

    async def _execute_tool_calls_parallel(self, tool_calls: List[ToolCall], claim_id: str):
        """OPTIMIZATION: Execute tool calls in parallel"""
        if not tool_calls:
            return
        
        # Create tasks for parallel execution
        tool_tasks = []
        for tool_call in tool_calls:
            # Emit event for each tool call
            event_task = self._emit_event(
                EvaluationEvent(
                    claim_id=claim_id, event_type="tool_called", data=tool_call.__dict__
                )
            )
            tool_tasks.append(event_task)
            
            # Create tool execution task
            exec_task = self.tool_executor.execute_tool(
                tool_name=tool_call.name, params=tool_call.parameters
            )
            tool_tasks.append(exec_task)
        
        # Execute all tool calls and events in parallel
        results = await asyncio.gather(*tool_tasks, return_exceptions=True)
        
        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Tool execution error: {result}")
            else:
                self.logger.info(f"Tool execution result: {result}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """OPTIMIZATION: Get detailed performance statistics"""
        stats = {
            "cache_performance": {
                "hits": self._stats["cache_hits"],
                "misses": self._stats["cache_misses"],
                "hit_rate": (
                    self._stats["cache_hits"] /
                    max(1, self._stats["cache_hits"] + self._stats["cache_misses"])
                ) * 100
            },
            "batch_performance": {
                "batch_evaluations": self._stats["batch_evaluations"],
                "batch_size": self._batch_size,
            },
            "timing_breakdown": {}
        }
        
        # Calculate averages for performance metrics
        for metric, times in self._performance_stats.items():
            if times:
                stats["timing_breakdown"][metric] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        
        return stats

    def _build_evaluation_prompt(
        self, claim: Claim, context_result: Dict[str, Any], iteration: int
    ) -> str:
        """Build evaluation prompt for LLM"""
        context_str = self.context_collector.build_llm_context_string(context_result)

        return f"""You are evaluating the following claim (iteration {iteration}):

CLAIM: {claim.content}
CURRENT CONFIDENCE: {claim.confidence}
TAGS: {",".join(claim.tags) if claim.tags else "none"}

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

        # Higher priority for certain tags
        if claim.tags and "concept" in claim.tags:
            priority -= 100
        elif claim.tags and "thesis" in claim.tags:
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
