"""
Conjecture: Async Evidence-Based AI Reasoning System
OPTIMIZED: Enhanced with comprehensive performance monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from functools import lru_cache
import hashlib

from src.core.models import Claim, ClaimState
from src.config.unified_config import UnifiedConfig as Config
from src.processing.unified_bridge import UnifiedLLMBridge as LLMBridge, LLMRequest
from src.processing.simplified_llm_manager import get_simplified_llm_manager
from src.processing.async_eval import AsyncClaimEvaluationService
from src.processing.context_collector import ContextCollector
from src.processing.tool_manager import DynamicToolCreator
from src.data.repositories import get_data_manager, RepositoryFactory
from src.monitoring import get_performance_monitor, monitor_performance


class Conjecture:
    """
    Enhanced Conjecture with Async Claim Evaluation and Dynamic Tool Creation
    Implements the full architecture described in the specifications
    """

    def __init__(self, config: Optional[Config] = None):
        """OPTIMIZED: Initialize Enhanced Conjecture with performance monitoring"""
        self.config = config or Config()

        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()

        # Initialize data layer with repository pattern
        self.data_manager = get_data_manager(use_mock_embeddings=False)
        self.claim_repository = RepositoryFactory.create_claim_repository(
            self.data_manager
        )

        # Initialize LLM bridge
        self._initialize_llm_bridge()

        # Initialize processing components
        self.context_collector = ContextCollector(self.data_manager)
        from .processing.tool_executor import ToolExecutor

        tool_executor = ToolExecutor()
        self.async_evaluation = AsyncClaimEvaluationService(
            llm_bridge=self.llm_bridge,
            context_collector=self.context_collector,
            data_manager=self.data_manager,
            tool_executor=tool_executor,
        )
        self.tool_creator = DynamicToolCreator(
            llm_bridge=self.llm_bridge, tools_dir="tools"
        )
        
        # Initialize enhanced template manager for XML optimization
        from .processing.llm_prompts.xml_optimized_templates import XMLOptimizedTemplateManager
        self.enhanced_template_manager = XMLOptimizedTemplateManager()
        
        # Service state
        self._services_started = False

        # Performance optimization: Caching with memory management
        self._claim_generation_cache = {}
        self._context_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._max_cache_size = 50  # Maximum items per cache
        self._cache_cleanup_interval = 60  # Cleanup every 60 seconds
        self._last_cache_cleanup = time.time()

        # Performance monitoring
        self._performance_stats = {
            "claim_generation_time": [],
            "context_collection_time": [],
            "claim_storage_time": [],
            "evaluation_time": [],
            "total_pipeline_time": [],
        }

        # Statistics
        self._stats = {
            "claims_processed": 0,
            "tools_created": 0,
            "evaluation_time_total": 0.0,
            "session_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self.logger = logging.getLogger(__name__)

        print(f"OPTIMIZED Conjecture initialized with performance monitoring")

    async def start_services(self):
        """Start background services"""
        if self._services_started:
            return

        # Initialize data manager first
        await self.data_manager.initialize()
        
        await self.async_evaluation.start()
        self._services_started = True

        self.logger.info("Enhanced Conjecture services started")

    async def stop_services(self):
        """Stop background services and cleanup resources"""
        if not self._services_started:
            return

        await self.async_evaluation.stop()
        
        # Clear caches to prevent memory leaks during shutdown
        self.clear_all_caches()
        
        self._services_started = False

        self.logger.info("Enhanced Conjecture services stopped and resources cleaned up")

    def _initialize_llm_bridge(self):
        """Initialize LLM bridge with simplified manager"""
        try:
            llm_manager = get_simplified_llm_manager()
            self.llm_bridge = LLMBridge(llm_manager=llm_manager)

            if self.llm_bridge.is_available():
                print(f"LLM Bridge: Simplified manager connected")
            else:
                print("LLM Bridge: No providers available, using mock mode")

        except Exception as e:
            print(f"LLM Bridge initialization failed: {e}")
            self.llm_bridge = LLMBridge()

    # @monitor_performance("explore", {"component": "conjecture"})  # Temporarily disabled for testing
    async def explore(
        self,
        query: str,
        max_claims: int = 10,
        claim_types: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        auto_evaluate: bool = True,
    ) -> "ExplorationResult":
        """
        OPTIMIZED: Enhanced exploration with performance monitoring and parallel processing

        Args:
            query: Research question or topic
            max_claims: Maximum claims to return
            claim_types: Specific claim types to include
            confidence_threshold: Minimum confidence level
            auto_evaluate: Whether to automatically evaluate created claims

        Returns:
            ExplorationResult with claims and evaluation status
        """
        start_time = time.time()

        if not query or len(query.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")

        confidence_threshold = confidence_threshold or self.config.confidence_threshold

        print(f"OPTIMIZED exploration: '{query}'")

        try:
            # Start services if not already running
            if not self._services_started:
                await self.start_services()

            # OPTIMIZATION: Parallel claim generation and context collection
            claim_gen_start = time.time()
            
            # Generate initial claims using LLM with caching
            initial_claims = await self._generate_initial_claims_cached(query, max_claims)
            
            claim_gen_time = time.time() - claim_gen_start
            self._performance_stats["claim_generation_time"].append(claim_gen_time)
            self.performance_monitor.record_timing("claim_generation", claim_gen_time)

            # Filter by confidence threshold
            filtered_claims = [
                claim
                for claim in initial_claims
                if claim.confidence >= confidence_threshold
            ]

            # OPTIMIZATION: Parallel claim storage and evaluation submission
            storage_start = time.time()
            
            # Prepare claim data for batch operations
            claims_data = []
            for claim in filtered_claims:
                claim_data = {
                    "content": claim.content,
                    "confidence": claim.confidence,
                    "tags": claim.tags,
                    "state": ClaimState.EXPLORE,
                }
                claims_data.append(claim_data)

            # Batch store claims
            stored_claims = await self._batch_create_claims(claims_data)
            
            storage_time = time.time() - storage_start
            self._performance_stats["claim_storage_time"].append(storage_time)
            self.performance_monitor.record_timing("claim_storage", storage_time)

            # OPTIMIZATION: Parallel evaluation submission and tool need checking
            if auto_evaluate or stored_claims:
                parallel_tasks = []
                
                # Submit claims for evaluation in parallel
                if auto_evaluate:
                    for claim in stored_claims:
                        parallel_tasks.append(
                            self.async_evaluation.submit_claim(claim)
                        )
                
                # Check for tool creation opportunities
                parallel_tasks.append(
                    self._check_tool_needs(stored_claims)
                )
                
                # Execute all tasks in parallel
                if parallel_tasks:
                    await asyncio.gather(*parallel_tasks, return_exceptions=True)

            processing_time = time.time() - start_time
            self._performance_stats["total_pipeline_time"].append(processing_time)
            self.performance_monitor.record_timing("total_explore_pipeline", processing_time)
            self._update_stats(processing_time, len(stored_claims))

            result = ExplorationResult(
                query=query,
                claims=stored_claims,
                total_found=len(filtered_claims),
                search_time=processing_time,
                confidence_threshold=confidence_threshold,
                max_claims=max_claims,
                evaluation_pending=auto_evaluate,
                tools_created=len(self.tool_creator.get_created_tools()),
            )

            print(
                f"OPTIMIZED exploration completed: {len(result.claims)} claims in {result.search_time:.2f}s"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in optimized exploration: {e}")
            self.performance_monitor.increment_counter("explore_errors")
            raise

    async def _generate_initial_claims_cached(
        self, query: str, max_claims: int
    ) -> List[Claim]:
        """Generate initial claims using LLM with caching"""
        # Check cache first
        cache_key = self._generate_cache_key(query, max_claims)
        cached_result = self._get_from_cache(cache_key, "claim_generation")
        
        if cached_result:
            self._stats["cache_hits"] += 1
            return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Generate claims
        claims = await self._generate_initial_claims(query, max_claims)
        
        # Cache the result
        self._add_to_cache(cache_key, claims, "claim_generation")
        
        return claims

    async def _generate_initial_claims(
        self, query: str, max_claims: int
    ) -> List[Claim]:
        """Generate initial claims using LLM with XML-optimized prompts"""
        try:
            # OPTIMIZATION: Parallel context collection
            context_start = time.time()
            
            # Get relevant context for query with caching and XML optimization
            context_claims = await self._collect_context_cached(
                query, {"task": "exploration"}, max_skills=3, max_samples=5
            )
            
            context_time = time.time() - context_start
            self._performance_stats["context_collection_time"].append(context_time)

            # Build context string from collected claims
            context_string = ""
            if context_claims:
                context_parts = []
                for claim in context_claims:
                    context_parts.append(f"- {claim.content} (confidence: {claim.confidence:.2f})")
                context_string = "\n".join(context_parts)
            else:
                context_string = "No relevant context available."
            
            # Get enhanced XML template for claim creation with chain-of-thought and confidence calibration
            xml_template = self.enhanced_template_manager.get_template("research_enhanced_xml")
            
            if not xml_template:
                # Fallback to basic prompt if XML template not available
                prompt = f"""Generate up to {max_claims} high-quality claims about: {query}

Requirements:
- Use XML format: <claim type="[fact|concept|example|goal|reference|hypothesis]" confidence="[0.0-1.0]">content</claim>
- Include clear, specific statements
- Provide realistic confidence scores
- Cover different aspects: facts, concepts, examples, goals

Context:
{context_string}

Generate claims using this XML structure:
<claims>
  <claim type="fact" confidence="0.9">Your factual claim here</claim>
  <claim type="concept" confidence="0.8">Your conceptual claim here</claim>
  <!-- Add more claims as needed -->
</claims>"""
            else:
                # Use enhanced XML template with proper variable substitution and chain-of-thought guidance
                try:
                    prompt = xml_template.template_content.format(
                        user_query=query,
                        relevant_context=context_string
                    )
                    
                    # Add specific claim count instruction with enhanced guidance
                    prompt += f"\n\nGenerate exactly {max_claims} high-quality claims using the enhanced XML format specified above."
                    prompt += "\nFollow the 6-step chain-of-thought research process for each claim."
                    prompt += "\nApply the confidence calibration guidelines to ensure accurate confidence scoring."
                    prompt += f"\nTarget {max_claims} claims with diverse types (fact, concept, example, goal, reference, hypothesis)."
                    
                except KeyError as e:
                    self.logger.warning(f"Template variable substitution failed: {e}")
                    # Fallback to basic format rendering
                    prompt = xml_template.template_content.replace("{{user_query}}", query).replace("{{relevant_context}}", context_string)
                    prompt += f"\n\nGenerate exactly {max_claims} high-quality claims using the XML format specified above."

            llm_request = LLMRequest(
                prompt=prompt, max_tokens=3000, temperature=0.7, task_type="explore"
            )

            response = self.llm_bridge.process(llm_request)

            if response.success:
                return self._parse_claims_from_response(response.content)
            else:
                raise Exception(f"LLM processing failed: {response.errors}")

        except Exception as e:
            self.logger.error(f"Error generating initial claims: {e}")
            return []

    def _parse_claims_from_response(self, response: str) -> List[Claim]:
        """Parse claims from LLM response using JSON frontmatter parser"""
        try:
            from .processing.json_frontmatter_parser import parse_response_with_json_frontmatter
            result = parse_response_with_json_frontmatter(response)
            if result.success:
                # Add exploration-specific tags to parsed claims
                for claim in result.claims:
                    if "exploration" not in claim.tags:
                        claim.tags.append("exploration")
                    if "auto_generated" not in claim.tags:
                        claim.tags.append("auto_generated")
                
                self.logger.info(f"Successfully parsed {len(result.claims)} claims using JSON frontmatter parser")
                return result.claims[:10]  # Limit to 10 claims
            else:
                # Fallback to unified parser if JSON frontmatter fails
                from .processing.unified_claim_parser import parse_claims_from_response
                claims = parse_claims_from_response(response)
                
                # Add exploration-specific tags to parsed claims
                for claim in claims:
                    if "exploration" not in claim.tags:
                        claim.tags.append("exploration")
                    if "auto_generated" not in claim.tags:
                        claim.tags.append("auto_generated")
                
                self.logger.info(f"Successfully parsed {len(claims)} claims using fallback unified parser")
                return claims[:10]  # Limit to 10 claims
            
        except ImportError as e:
            # Fallback to basic parsing if parsers not available
            self.logger.warning(f"JSON frontmatter parser not available, using fallback: {e}")
            return self._fallback_parse_claims(response)
        except Exception as e:
            self.logger.error(f"Error parsing claims from response: {e}")
            return []
    
    def _fallback_parse_claims(self, response: str) -> List[Claim]:
        """Fallback claim parsing method"""
        claims = []
        
        try:
            # Simple fallback - create claims from substantial lines
            lines = response.split("\n")
            for i, line in enumerate(lines):
                if line.strip() and len(line.strip()) > 20:
                    claim = Claim(
                        id=f"exploration_{int(time.time())}_{i}",
                        content=line.strip(),
                        confidence=0.7,  # Default confidence
                        tags=["exploration", "auto_generated", "fallback"],
                        state=ClaimState.EXPLORE,
                    )
                    claims.append(claim)
        except Exception as e:
            self.logger.error(f"Error in fallback parsing: {e}")
        
        return claims[:10]  # Limit to 10 claims

    async def _check_tool_needs(self, claims: List[Claim]):
        """Check if any claims indicate need for new tools"""
        for claim in claims:
            try:
                tool_need = await self.tool_creator.discover_tool_need(claim)
                if tool_need:
                    print(f"Tool need detected: {tool_need[:100]}...")

                    # Search for implementation methods
                    methods = await self.tool_creator.websearch_tool_methods(tool_need)

                    if methods:
                        # Create tool
                        tool_name = f"tool_{claim.id[:8]}"
                        tool_path = await self.tool_creator.create_tool_file(
                            tool_name, tool_need, methods
                        )

                        if tool_path:
                            # Create skill and sample claims
                            skill_claim = await self.tool_creator.create_skill_claim(
                                tool_name, tool_need, tool_path
                            )
                            sample_claim = await self.tool_creator.create_sample_claim(
                                tool_name, tool_path
                            )

                            # Store skill and sample claims
                            await self.data_manager.create_claim(
                                content=skill_claim.content,
                                confidence=skill_claim.confidence,
                                tags=skill_claim.tags,
                                state=skill_claim.state,
                            )

                            await self.data_manager.create_claim(
                                content=sample_claim.content,
                                confidence=sample_claim.confidence,
                                tags=sample_claim.tags,
                                state=sample_claim.state,
                            )

                            self._stats["tools_created"] += 1
                            print(f"Created new tool: {tool_name}")

            except Exception as e:
                self.logger.error(
                    f"Error checking tool needs for claim {claim.id}: {e}"
                )

    async def add_claim(
        self,
        content: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        auto_evaluate: bool = True,
        **kwargs,
    ) -> Claim:
        """
        Enhanced claim creation with automatic evaluation

        Args:
            content: Claim content
            confidence: Initial confidence score
            tags: Optional tags
            auto_evaluate: Whether to submit for evaluation
            **kwargs: Additional claim attributes

        Returns:
            Created claim
        """
        # Validate inputs
        if len(content.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Create claim using repository
        claim_data = {
            "content": content.strip(),
            "confidence": confidence,
            "tags": tags or [],
            "state": ClaimState.EXPLORE,
            **kwargs,
        }
        claim = await self.claim_repository.create(claim_data)

        # Submit for evaluation if enabled
        if auto_evaluate and self._services_started:
            await self.async_evaluation.submit_claim(claim)

        self._stats["claims_processed"] += 1
        print(f"Created claim: {claim.id}")

        return claim

    async def get_evaluation_status(self, claim_id: str) -> Dict[str, Any]:
        """Get evaluation status for a specific claim"""
        try:
            claim = await self.claim_repository.get_by_id(claim_id)
            if not claim:
                return {"error": "Claim not found"}

            return {
                "claim_id": claim_id,
                "state": claim.state.value,
                "confidence": claim.confidence,
                "last_updated": claim.updated.isoformat(),
                "evaluation_active": claim_id
                in self.async_evaluation._active_evaluations,
            }
        except Exception as e:
            return {"error": str(e)}

    async def wait_for_evaluation(
        self, claim_id: str, timeout: int = 60
    ) -> Dict[str, Any]:
        """Wait for claim evaluation to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = await self.get_evaluation_status(claim_id)

            if "error" in status:
                return status

            if status["state"] == ClaimState.VALIDATED.value:
                return {"success": True, "claim": status}

            if not status["evaluation_active"]:
                return {
                    "success": False,
                    "reason": "Evaluation not active",
                    "claim": status,
                }

            await asyncio.sleep(2)

        return {"success": False, "reason": "Timeout", "claim": status}

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task through the full Conjecture pipeline with 4 stages:
        1. Task Decomposition (via LLM)
        2. Context Collection (from existing skills/samples)
        3. Claims Evaluation (generate and evaluate claims for each sub-task)
        4. Final Synthesis (via LLM)
        
        Args:
            task: Task dictionary containing instructions and parameters
            
        Returns:
            Result of task processing with comprehensive response
        """
        start_time = time.time()
        
        try:
            # Extract task details
            task_type = task.get("type", "full_pipeline")
            content = task.get("content", "")
            
            # Handle legacy task types for backward compatibility
            if task_type == "explore":
                # Use the explore method for exploration tasks
                result = await self.explore(content, max_claims=task.get("max_claims", 10))
                return {
                    "success": True,
                    "type": "exploration",
                    "result": result.to_dict(),
                    "content": result.summary()
                }
            elif task_type == "create_claim":
                # Use the add_claim method for claim creation
                confidence = task.get("confidence", 0.8)
                claim = await self.add_claim(content, confidence)
                return {
                    "success": True,
                    "type": "claim_creation",
                    "result": claim.to_dict(),
                    "claim_id": claim.id
                }
            elif task_type == "analyze":
                # Use the analyze method for claim analysis
                claim_id = task.get("claim_id")
                if claim_id:
                    claim = await self.claim_repository.get_by_id(claim_id)
                    if claim:
                        result = await self.async_evaluation.submit_claim(claim)
                        return {
                            "success": True,
                            "type": "analysis",
                            "result": result,
                            "claim_id": claim_id
                        }
                return {"success": False, "error": "Claim not found"}
            
            # Full pipeline processing for complex tasks
            elif task_type == "full_pipeline" or task_type == "task":
                print(f"🚀 Starting full pipeline processing: '{content[:100]}...'")
                
                # Stage 1: Task Decomposition
                decomposition_result = await self._decompose_task(content)
                if not decomposition_result["success"]:
                    return decomposition_result
                
                subtasks = decomposition_result["subtasks"]
                print(f"📋 Task decomposed into {len(subtasks)} subtasks")
                
                # Stage 2: Context Collection for each subtask
                context_results = []
                for i, subtask in enumerate(subtasks):
                    print(f"🔍 Collecting context for subtask {i+1}/{len(subtasks)}")
                    context = await self._collect_context_cached(
                        subtask, {"task": "pipeline"}, max_skills=3, max_samples=5
                    )
                    context_results.append(context)
                
                print(f"📚 Context collected for all {len(subtasks)} subtasks")
                
                # Stage 3: Claims Generation and Evaluation
                claims_results = []
                for i, (subtask, context) in enumerate(zip(subtasks, context_results)):
                    print(f"🧠 Processing claims for subtask {i+1}/{len(subtasks)}")
                    
                    # Generate claims for this subtask
                    claims = await self._generate_claims_for_subtask(subtask, context)
                    
                    # Evaluate claims
                    evaluated_claims = []
                    for claim in claims:
                        try:
                            # Submit claim for evaluation
                            await self.async_evaluation.submit_claim(claim)
                            evaluated_claims.append(claim)
                        except Exception as e:
                            self.logger.warning(f"Failed to evaluate claim {claim.id}: {e}")
                    
                    claims_results.append({
                        "subtask": subtask,
                        "context": context,
                        "claims": evaluated_claims,
                        "claims_count": len(evaluated_claims)
                    })
                
                total_claims = sum(result["claims_count"] for result in claims_results)
                print(f"✅ Generated and evaluated {total_claims} claims across all subtasks")
                
                # Stage 4: Final Synthesis
                print(f"🔗 Performing final synthesis...")
                synthesis_result = await self._synthesize_final_answer(
                    original_task=content,
                    decomposition=decomposition_result,
                    contexts=context_results,
                    claims_results=claims_results
                )
                
                processing_time = time.time() - start_time
                
                result = {
                    "success": True,
                    "type": "full_pipeline",
                    "processing_time": processing_time,
                    "stages_completed": {
                        "task_decomposition": True,
                        "context_collection": True,
                        "claims_evaluation": True,
                        "final_synthesis": True
                    },
                    "pipeline_results": {
                        "decomposition": decomposition_result,
                        "contexts": context_results,
                        "claims": claims_results,
                        "synthesis": synthesis_result
                    },
                    "summary": {
                        "subtasks_count": len(subtasks),
                        "total_claims": total_claims,
                        "processing_time": processing_time
                    },
                    "final_answer": synthesis_result.get("answer", ""),
                    "confidence": synthesis_result.get("confidence", 0.8)
                }
                
                print(f"🎉 Full pipeline completed in {processing_time:.2f}s")
                return result
                
            else:
                return {"success": False, "error": f"Unsupported task type: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"Task processing failed: {e}")
            return {"success": False, "error": str(e)}

    async def _decompose_task(self, task: str) -> Dict[str, Any]:
        """
        Stage 1: Decompose complex task into manageable subtasks using LLM
        
        Args:
            task: The original task to decompose
            
        Returns:
            Dictionary with success status and list of subtasks
        """
        try:
            # Build task decomposition prompt
            prompt = f"""You are tasked with breaking down the following complex task into manageable subtasks:

TASK: {task}

Please break this task into 3-5 logical subtasks that:
1. Are specific and actionable
2. Cover all aspects of the original task
3. Can be addressed independently
4. Follow a logical sequence

Format your response as a numbered list of subtasks:
1. [First subtask]
2. [Second subtask]
3. [Third subtask]
etc.

Focus on creating clear, specific subtasks that together address the original task comprehensively."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3,
                task_type="task_decomposition"
            )

            response = self.llm_bridge.process(llm_request)
            
            if response.success:
                # Parse subtasks from response
                subtasks = self._parse_subtasks_from_response(response.content)
                
                return {
                    "success": True,
                    "subtasks": subtasks,
                    "decomposition": response.content
                }
            else:
                raise Exception(f"LLM decomposition failed: {response.errors}")
                
        except Exception as e:
            self.logger.error(f"Task decomposition failed: {e}")
            return {"success": False, "error": str(e)}

    def _parse_subtasks_from_response(self, response: str) -> List[str]:
        """Parse numbered list of subtasks from LLM response"""
        subtasks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., etc.)
            if line and (line[0].isdigit() or (len(line) > 1 and line[0].isdigit() and line[1] == '.')):
                # Remove the number and any leading punctuation
                task_text = line
                if '.' in task_text:
                    task_text = task_text.split('.', 1)[1].strip()
                elif ')' in task_text:
                    task_text = task_text.split(')', 1)[1].strip()
                elif '-' in task_text:
                    task_text = task_text.split('-', 1)[1].strip()
                
                if task_text:
                    subtasks.append(task_text)
        
        # Fallback: if no numbered items found, split by lines
        if not subtasks:
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Only substantial lines
                    subtasks.append(line)
        
        return subtasks[:5]  # Limit to 5 subtasks

    async def _generate_claims_for_subtask(self, subtask: str, context: Dict[str, Any]) -> List[Claim]:
        """
        Generate claims for a specific subtask using LLM and context
        
        Args:
            subtask: The subtask to generate claims for
            context: Relevant context from ContextCollector
            
        Returns:
            List of generated claims
        """
        try:
            # Build context string
            context_string = self.context_collector.build_llm_context_string(context)
            
            # Build claims generation prompt
            prompt = f"""Generate evidence-based claims for the following subtask:

SUBTASK: {subtask}

RELEVANT CONTEXT:
{context_string}

            Generate 3-5 specific, evidence-based claims that address this subtask. For each claim:
            - Use claim IDs in format 'c1', 'c2', etc.
            - Include clear, specific statements
            - Provide confidence scores (0.0-1.0)
            - Use appropriate claim types: fact, concept, example, goal, reference, assertion
            - Focus on accuracy and verifiability

            Format each claim as:
            [cID] [Type] [Confidence]: Claim statement

            Example:
            [c1] [fact] [0.9]: The Earth's average temperature has increased by 1.1C since pre-industrial times.
            [c2] [concept] [0.8]: Climate change refers to long-term shifts in global weather patterns."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.5,
                task_type="claims_generation"
            )

            response = self.llm_bridge.process(llm_request)
            
            if response.success:
                claims = self._parse_claims_from_response(response.content)
                return claims
            else:
                raise Exception(f"Claims generation failed: {response.errors}")
                
        except Exception as e:
            self.logger.error(f"Claims generation failed for subtask '{subtask}': {e}")
            return []

    async def _synthesize_final_answer(
        self,
        original_task: str,
        decomposition: Dict[str, Any],
        contexts: List[Dict[str, Any]],
        claims_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Stage 4: Synthesize final comprehensive answer using LLM
        
        Args:
            original_task: The original task
            decomposition: Task decomposition result
            contexts: Context results for each subtask
            claims_results: Claims and evaluation results
            
        Returns:
            Dictionary with final synthesized answer and confidence
        """
        try:
            # Build comprehensive context for synthesis
            synthesis_context = f"""ORIGINAL TASK: {original_task}

TASK DECOMPOSITION:
{decomposition.get('decomposition', 'N/A')}

CLAIMS AND EVIDENCE:
"""
            
            for i, claim_result in enumerate(claims_results, 1):
                synthesis_context += f"\nSUBTASK {i}: {claim_result['subtask']}\n"
                
                for claim in claim_result['claims']:
                    synthesis_context += f"- {claim.content} (confidence: {claim.confidence:.2f})\n"
            
            # Build synthesis prompt
            prompt = f"""Based on the comprehensive analysis below, provide a complete, well-reasoned answer to the original task.

{synthesis_context}

Synthesize this information into a comprehensive answer that:
1. Directly addresses the original task
2. Integrates insights from all subtasks and claims
3. Provides a clear, structured response
4. Includes appropriate confidence level
5. Highlights key findings and conclusions

Format your response as:
[ANSWER]
Your comprehensive answer here...

[CONFIDENCE]
Overall confidence in this answer: X.XX

[KEY_FINDINGS]
- Key finding 1
- Key finding 2
- etc."""

            llm_request = LLMRequest(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.4,
                task_type="synthesis"
            )

            response = self.llm_bridge.process(llm_request)
            
            if response.success:
                # Parse synthesis response
                parsed = self._parse_synthesis_response(response.content)
                
                return {
                    "success": True,
                    "answer": parsed.get("answer", response.content),
                    "confidence": parsed.get("confidence", 0.8),
                    "key_findings": parsed.get("key_findings", []),
                    "raw_response": response.content
                }
            else:
                raise Exception(f"Synthesis failed: {response.errors}")
                
        except Exception as e:
            self.logger.error(f"Final synthesis failed: {e}")
            return {
                "success": False,
                "answer": f"Error during synthesis: {str(e)}",
                "confidence": 0.0,
                "key_findings": [],
                "error": str(e)
            }

    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse structured synthesis response from LLM"""
        result = {
            "answer": "",
            "confidence": 0.8,
            "key_findings": []
        }
        
        lines = response.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('[ANSWER]'):
                current_section = 'answer'
                current_content = []
            elif line.startswith('[CONFIDENCE]'):
                if current_section == 'answer':
                    result['answer'] = '\n'.join(current_content).strip()
                current_section = 'confidence'
            elif line.startswith('[KEY_FINDINGS]'):
                if current_section == 'confidence':
                    result['confidence'] = '\n'.join(current_content).strip()
                current_section = 'findings'
                current_content = []
            elif line.startswith('[') and current_section:
                # End of current section
                if current_section == 'answer':
                    result['answer'] = '\n'.join(current_content).strip()
                elif current_section == 'confidence':
                    # Extract confidence value
                    confidence_text = '\n'.join(current_content).strip()
                    try:
                        # Look for numeric confidence
                        import re
                        match = re.search(r'(\d+\.?\d*)', confidence_text)
                        if match:
                            result['confidence'] = float(match.group(1))
                    except:
                        pass
                elif current_section == 'findings':
                    result['key_findings'] = [f.strip() for f in current_content if f.strip()]
                current_section = None
            elif current_section:
                current_content.append(line)
        
        # Handle final section if no closing tag
        if current_section == 'answer':
            result['answer'] = '\n'.join(current_content).strip()
        elif current_section == 'confidence':
            confidence_text = '\n'.join(current_content).strip()
            try:
                import re
                match = re.search(r'(\d+\.?\d*)', confidence_text)
                if match:
                    result['confidence'] = float(match.group(1))
            except:
                pass
        elif current_section == 'findings':
            result['key_findings'] = [f.strip() for f in current_content if f.strip()]
        
        # Fallback: if no structured parsing worked, use entire response as answer
        if not result['answer']:
            result['answer'] = response.strip()
        
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        from src.tools.registry import ToolRegistry

        # Get tool counts
        registry = ToolRegistry()
        available_tools = len(registry.core_tools) + len(registry.optional_tools)

        base_stats = {
            "config": self.config.to_dict(),
            "services_running": self._services_started,
            "claims_processed": self._stats["claims_processed"],
            "tools_created": self._stats["tools_created"],
            "total_evaluation_time": self._stats["evaluation_time_total"],
            "session_count": self._stats["session_count"],
            "available_tools": available_tools,
            "available_skills": 4,  # Research, Code, Test, Evaluate
            "total_claims": 0,  # Would need to query database for actual count
        }

        # Add async evaluation stats
        if self._services_started:
            eval_stats = self.async_evaluation.get_statistics()
            base_stats.update({"evaluation_service": eval_stats})

        # Add tool creator stats
        tool_stats = self.tool_creator.get_created_tools()
        base_stats.update({"created_tools": tool_stats})

        return base_stats

    async def _batch_create_claims(self, claims_data: List[Dict[str, Any]]) -> List[Claim]:
        """Batch create claims for better performance"""
        try:
            # Use batch operation if available
            if hasattr(self.data_manager, 'batch_create_claims'):
                batch_result = await self.data_manager.batch_create_claims(claims_data)
                # Extract claims from batch result
                claims = []
                for result in batch_result.results:
                    if result.success:
                        # This is a simplified approach - in practice, we'd need
                        # to get the actual created claims
                        for claim_data in claims_data:
                            claim = await self.claim_repository.create(claim_data)
                            claims.append(claim)
                return claims
            else:
                # Fallback to individual creation
                claims = []
                for claim_data in claims_data:
                    claim = await self.claim_repository.create(claim_data)
                    claims.append(claim)
                return claims
        except Exception as e:
            self.logger.error(f"Error in batch claim creation: {e}")
            # Fallback to individual creation
            claims = []
            for claim_data in claims_data:
                try:
                    claim = await self.claim_repository.create(claim_data)
                    claims.append(claim)
                except Exception as claim_error:
                    self.logger.error(f"Failed to create individual claim: {claim_error}")
            return claims

    async def _collect_context_cached(
        self, query: str, context: Dict[str, Any], max_skills: int = 5, max_samples: int = 10
    ) -> Dict[str, Any]:
        """Collect context with caching"""
        cache_key = self._generate_cache_key(query, max_skills, max_samples)
        cached_result = self._get_from_cache(cache_key, "context_collection")
        
        if cached_result:
            self._stats["cache_hits"] += 1
            return cached_result
        
        self._stats["cache_misses"] += 1
        
        # Collect context
        context_result = await self.context_collector.collect_context_for_claim(
            query, context, max_skills, max_samples
        )
        
        # Cache the result
        self._add_to_cache(cache_key, context_result, "context_collection")
        
        return context_result

    def _generate_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """Get item from cache with TTL check and periodic cleanup"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        
        # Periodic cleanup to prevent memory leaks
        current_time = time.time()
        if current_time - self._last_cache_cleanup > self._cache_cleanup_interval:
            self._cleanup_expired_cache(cache_type)
            self._last_cache_cleanup = current_time
        
        if cache_key in cache:
            cached_item = cache[cache_key]
            timestamp = cached_item.get("timestamp", 0)
            
            # Check if cache is still valid
            if current_time - timestamp < self._cache_ttl:
                return cached_item["data"]
            else:
                # Remove expired cache item
                del cache[cache_key]
        
        return None

    def _add_to_cache(self, cache_key: str, data: Any, cache_type: str) -> None:
        """Add item to cache with timestamp and size management"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        
        # Enforce cache size limit to prevent memory leaks
        if len(cache) >= self._max_cache_size:
            self._enforce_cache_size_limit(cache_type)
        
        cache[cache_key] = {
            "data": data,
            "timestamp": time.time(),
        }
    
    def _cleanup_expired_cache(self, cache_type: str) -> None:
        """Remove expired items from cache to prevent memory leaks"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        current_time = time.time()
        
        expired_keys = [
            key for key, item in cache.items()
            if current_time - item.get("timestamp", 0) >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired items from {cache_type} cache")
    
    def _enforce_cache_size_limit(self, cache_type: str) -> None:
        """Enforce maximum cache size to prevent memory leaks"""
        cache = getattr(self, f"_{cache_type}_cache", {})
        
        if len(cache) <= self._max_cache_size:
            return
        
        # Remove oldest items to maintain size limit
        items_by_age = sorted(
            cache.items(),
            key=lambda item: item[1].get("timestamp", 0)
        )
        
        # Remove oldest 25% of items
        items_to_remove = max(1, len(items_by_age) // 4)
        
        for i in range(items_to_remove):
            key = items_by_age[i][0]
            del cache[key]
        
        self.logger.debug(f"Removed {items_to_remove} old items from {cache_type} cache to enforce size limit")
    
    def clear_all_caches(self) -> None:
        """Clear all caches to free memory - call this during shutdown or memory pressure"""
        self._claim_generation_cache.clear()
        self._context_cache.clear()
        self.logger.info("All caches cleared to free memory")

    def _update_stats(self, processing_time: float, claims_count: int):
        """Update internal statistics"""
        self._stats["evaluation_time_total"] += processing_time
        self._stats["claims_processed"] += claims_count

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "performance": {},
            "cache_stats": {
                "hits": self._stats["cache_hits"],
                "misses": self._stats["cache_misses"],
                "hit_rate": (
                    self._stats["cache_hits"] /
                    max(1, self._stats["cache_hits"] + self._stats["cache_misses"])
                ) * 100
            }
        }
        
        # Calculate averages for performance metrics
        for metric, times in self._performance_stats.items():
            if times:
                stats["performance"][metric] = {
                    "count": len(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }
        
        return stats

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_services()


class ExplorationResult:
    """Enhanced exploration result with evaluation information"""

    def __init__(
        self,
        query: str,
        claims: List[Claim],
        total_found: int,
        search_time: float,
        confidence_threshold: float,
        max_claims: int,
        evaluation_pending: bool = False,
        tools_created: int = 0,
    ):
        self.query = query
        self.claims = claims
        self.total_found = total_found
        self.search_time = search_time
        self.confidence_threshold = confidence_threshold
        self.max_claims = max_claims
        self.evaluation_pending = evaluation_pending
        self.tools_created = tools_created
        self.timestamp = datetime.utcnow()

    def summary(self) -> str:
        """Provide enhanced summary"""
        if not self.claims:
            return f"No claims found for '{self.query}' above confidence threshold {self.confidence_threshold}"

        lines = [
            f"🎯 Enhanced Exploration: '{self.query}'",
            f"📊 Found: {len(self.claims)} claims (of {self.total_found} total)",
            f"⏱️  Time: {self.search_time:.2f}s",
            f"🎚️  Confidence: ≥{self.confidence_threshold}",
            f"🔄 Evaluation: {'Pending' if self.evaluation_pending else 'Disabled'}",
            f"🔧 Tools Created: {self.tools_created}",
            "",
            "📋 Top Claims:",
        ]

        for i, claim in enumerate(self.claims[:5], 1):
            tags_str = ",".join(claim.tags) if claim.tags else "none"
            lines.append(
                f"  {i}. [{claim.confidence:.2f}, {tags_str}, {claim.state.value}] {claim.content[:100]}{'...' if len(claim.content) > 100 else ''}"
            )

        if len(self.claims) > 5:
            lines.append(f"  ... and {len(self.claims) - 5} more claims")

        return "\n".join(lines)


# Convenience functions
async def explore(query: str, **kwargs) -> ExplorationResult:
    """Quick exploration function"""
    async with Conjecture() as cf:
        return await cf.explore(query, **kwargs)


async def add_claim(
    content: str, confidence: float, **kwargs
) -> Claim:
    """Quick claim creation function"""
    async with Conjecture() as cf:
        return await cf.add_claim(content, confidence, **kwargs)


if __name__ == "__main__":

    async def test_enhanced_conjecture():
        print("🧪 Testing Enhanced Conjecture")
        print("=" * 40)

        async with Conjecture() as cf:
            # Test enhanced exploration
            print("\n🔍 Testing enhanced exploration...")
            result = await cf.explore("quantum computing applications", max_claims=3)
            print(result.summary())

            # Test claim creation
            print("\n➕ Testing enhanced claim creation...")
            claim = await cf.add_claim(
                content="Enhanced Conjecture provides async evaluation and dynamic tool creation",
                confidence=0.9,
                tags=["concept", "architecture", "enhancement"],
            )
            print(f"✅ Created claim: {claim.id}")

            # Wait for evaluation
            print("\n⏳ Waiting for evaluation...")
            eval_result = await cf.wait_for_evaluation(claim.id, timeout=10)
            print(f"Evaluation result: {eval_result}")

            # Test statistics
            print("\n📊 Testing enhanced statistics...")
            stats = cf.get_statistics()
            print(f"Enhanced stats: {stats}")

        print("\n🎉 Enhanced Conjecture tests completed!")

    asyncio.run(test_enhanced_conjecture())
