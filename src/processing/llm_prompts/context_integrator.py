"""
Context Integrator for LLM Prompt Management System
Integrates collected context into prompts with optimization
"""

import re
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .models import (
    IntegratedPrompt, OptimizedPrompt, TokenUsage
)
from ..support_systems.models import ContextResult, ContextItem
from ..support_systems.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

class ContextIntegrator:
    """
    Integrates context into prompts with optimization and formatting
    """

    def __init__(self, context_builder: ContextBuilder, 
                 default_token_limit: int = 4000,
                 token_ratio: float = 4.0):  # characters per token
        
        self.context_builder = context_builder
        self.default_token_limit = default_token_limit
        self.token_ratio = token_ratio
        
        # Integration strategies
        self.integration_strategies = {
            'sequential': self._integrate_sequential,
            'grouped': self._integrate_grouped,
            'prio_relevance': self._integrate_priority_relevance,
            'balanced': self._integrate_balanced
        }
        
        # Formatting options
        self.formatters = {
            'default': self._format_default,
            'compact': self._format_compact,
            'detailed': self._format_detailed,
            'structured': self._format_structured
        }
        
        # Statistics
        self.integration_stats = {
            'total_integrations': 0,
            'average_token_reduction': 0.0,
            'optimization_success_rate': 0.0
        }

    async def integrate_context(self, template_content: str, 
                              context_result: ContextResult,
                              token_limit: Optional[int] = None,
                              integration_strategy: str = 'sequential',
                              format_type: str = 'default') -> IntegratedPrompt:
        """
        Integrate context into template content
        
        Args:
            template_content: Base template content
            context_result: Collected context
            token_limit: Optional token limit
            integration_strategy: Strategy for context integration
            format_type: Formatting style for context
            
        Returns:
            Integrated prompt
        """
        start_time = time.time()
        token_limit = token_limit or self.default_token_limit
        
        try:
            logger.debug(f"Integrating context: {len(context_result.context_items)} items")
            
            # Choose integration strategy
            integrator = self.integration_strategies.get(
                integration_strategy, 
                self.integration_strategies['sequential']
            )
            
            # Choose formatter
            formatter = self.formatters.get(format_type, self._format_default)
            
            # Integrate context
            integrated_content, used_items = await integrator(
                template_content, context_result, formatter
            )
            
            # Estimate token usage
            token_count = self._estimate_tokens(integrated_content)
            
            # Optimize if over limit
            if token_count > token_limit:
                optimized = await self.optimize_for_tokens(
                    integrated_content, token_limit
                )
                integrated_content = optimized.optimized_prompt
                used_items = optimized.changes_made  # This would need to be context items
                token_count = optimized.token_count
            
            # Create integrated prompt
            integrated_prompt = IntegratedPrompt(
                template_id=f"integrated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rendered_prompt=integrated_content,
                variables_used={
                    'context_items': len(context_result.context_items),
                    'used_items': len(used_items),
                    'integration_strategy': integration_strategy,
                    'format_type': format_type
                },
                context_items_used=[item.id for item in used_items],
                token_count=token_count,
                metadata={
                    'original_template_length': len(template_content),
                    'context_processing_time_ms': context_result.processing_time_ms,
                    'integration_time_ms': int((time.time() - start_time) * 1000),
                    'optimization_applied': token_count > self.default_token_limit
                }
            )
            
            # Update statistics
            self.integration_stats['total_integrations'] += 1
            
            logger.debug(f"Context integrated: {token_count} tokens in {integrated_prompt.metadata['integration_time_ms']}ms")
            return integrated_prompt

        except Exception as e:
            logger.error(f"Failed to integrate context: {e}")
            # Return basic prompt without context
            fallback_prompt = IntegratedPrompt(
                template_id="fallback",
                rendered_prompt=template_content,
                variables_used={'error': str(e)},
                token_count=self._estimate_tokens(template_content),
                metadata={'error': True, 'context_integration_failed': True}
            )
            return fallback_prompt

    async def optimize_for_tokens(self, prompt: str, limit: int) -> OptimizedPrompt:
        """
        Optimize prompt to fit within token limit
        
        Args:
            prompt: Original prompt
            limit: Token limit
            
        Returns:
            Optimized prompt with changes
        """
        try:
            current_tokens = self._estimate_tokens(prompt)
            
            if current_tokens <= limit:
                return OptimizedPrompt(
                    original_prompt=prompt,
                    optimized_prompt=prompt,
                    optimization_strategy="none",
                    token_reduction=0,
                    optimization_score=1.0,
                    changes_made=[]
                )
            
            optimization_score = 0.0
            optimized_prompt = prompt
            changes_made = []
            
            # Strategy 1: Remove redundant whitespace
            optimized_prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', optimized_prompt)
            optimized_prompt = re.sub(r'[ \t]+', ' ', optimized_prompt)
            if optimized_prompt != prompt:
                changes_made.append("Removed redundant whitespace")
            
            # Strategy 2: Shorten common patterns
            replacements = {
                r'\b(You are|You\'re) a\b': 'As',
                r'\b(Please|Could you)\b': '',
                r'\b(in order to|so as to)\b': 'to',
                r'\b(the following|the below)\b': '',
                r'\b(We need to|We should)\b': '',
                r'\b(make sure|ensure|verify)\b': '',
                r'\b(It is important to|It is necessary)\b': '',
            }
            
            for pattern, replacement in replacements.items():
                old_prompt = optimized_prompt
                optimized_prompt = re.sub(pattern, replacement, optimized_prompt, flags=re.IGNORECASE)
                if optimized_prompt != old_prompt:
                    changes_made.append(f"Pattern: {pattern}")
            
            # Strategy 3: Truncate context sections if needed
            if self._estimate_tokens(optimized_prompt) > limit:
                optimized_prompt = await self._truncate_context_sections(optimized_prompt, limit)
                changes_made.append("Truncated least important sections")
            
            # Calculate optimization metrics
            new_tokens = self._estimate_tokens(optimized_prompt)
            token_reduction = current_tokens - new_tokens
            optimization_score = min(1.0, new_tokens / limit)
            
            # Log optimization results
            logger.info(f"Prompt optimization: {current_tokens} -> {new_tokens} tokens ({token_reduction} saved)")
            
            return OptimizedPrompt(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                optimization_strategy="multi_stage",
                token_reduction=token_reduction,
                optimization_score=optimization_score,
                changes_made=changes_made
            )

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            # Return original prompt if optimization fails
            return OptimizedPrompt(
                original_prompt=prompt,
                optimized_prompt=prompt,
                optimization_strategy="failed",
                token_reduction=0,
                optimization_score=0.0,
                changes_made=["Optimization failed"]
            )

    async def format_context_item(self, item: ContextItem, format_type: str = "default") -> str:
        """
        Format a single context item
        
        Args:
            item: Context item to format
            format_type: Format style
            
        Returns:
            Formatted context string
        """
        formatter = self.formatters.get(format_type, self._format_default)
        return await formatter(item)

    async def calculate_token_usage(self, text: str) -> TokenUsage:
        """
        Calculate token usage for text
        
        Args:
            text: Input text
            
        Returns:
            Token usage information
        """
        try:
            # Rough token estimation
            char_count = len(text)
            estimated_tokens = math.ceil(char_count / self.token_ratio)
            
            return TokenUsage(
                context_tokens=estimated_tokens,
                total_tokens=estimated_tokens
            )

        except Exception as e:
            logger.error(f"Token calculation failed: {e}")
            return TokenUsage()

    async def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get integration statistics
        
        Returns:
            Integration statistics
        """
        return {
            **self.integration_stats,
            'available_strategies': list(self.integration_strategies.keys()),
            'available_formatters': list(self.formatters.keys()),
            'default_token_limit': self.default_token_limit
        }

    async def _integrate_sequential(self, template_content: str, 
                                  context_result: ContextResult,
                                  formatter) -> Tuple[str, List[ContextItem]]:
        """Sequential integration strategy"""
        context_sections = []
        used_items = []
        
        for item in context_result.context_items:
            formatted_item = await formatter(item)
            context_sections.append(formatted_item)
            used_items.append(item)
        
        context_text = "\n\n".join(context_sections)
        
        # Find context placeholder in template
        if "{{context}}" in template_content:
            integrated = template_content.replace("{{context}}", context_text)
        else:
            # Insert context before the end
            integrated = f"{template_content}\n\n---\n\nCONTEXT:\n{context_text}"
        
        return integrated, used_items

    async def _integrate_grouped(self, template_content: str, 
                               context_result: ContextResult,
                               formatter) -> Tuple[str, List[ContextItem]]:
        """Grouped integration by item type"""
        # Group items by type
        grouped_items = {}
        for item in context_result.context_items:
            item_type = item.item_type.value
            if item_type not in grouped_items:
                grouped_items[item_type] = []
            grouped_items[item_type].append(item)
        
        context_sections = []
        used_items = []
        
        # Format each group
        for item_type, items in grouped_items.items():
            if not items:
                continue
            
            group_header = f"{item_type.upper()}:"
            item_sections = []
            
            for item in items:
                formatted_item = await formatter(item)
                item_sections.append(formatted_item)
                used_items.append(item)
            
            group_text = f"{group_header}\n" + "\n".join(f"- {section}" for section in item_sections)
            context_sections.append(group_text)
        
        context_text = "\n\n".join(context_sections)
        
        # Integrate with template
        if "{{context}}" in template_content:
            integrated = template_content.replace("{{context}}", context_text)
        else:
            integrated = f"{template_content}\n\n---\n\nCONTEXT:\n{context_text}"
        
        return integrated, used_items

    async def _integrate_priority_relevance(self, template_content: str,
                                          context_result: ContextResult,
                                          formatter) -> Tuple[str, List[ContextItem]]:
        """Priority-based integration by relevance score"""
        # Sort by relevance (already sorted in context_result)
        priority_items = context_result.context_items
        
        # Use top items until token limit would be exceeded
        context_sections = []
        used_items = []
        current_tokens = self._estimate_tokens(template_content)
        max_tokens = self.default_token_limit * 0.8  # Leave buffer
        
        for item in priority_items:
            formatted_item = await formatter(item)
            item_tokens = len(formatted_item) // self.token_ratio
            
            if current_tokens + item_tokens <= max_tokens:
                context_sections.append(formatted_item)
                used_items.append(item)
                current_tokens += item_tokens
            else:
                break
        
        context_text = "\n\n".join(context_sections)
        
        if "{{context}}" in template_content:
            integrated = template_content.replace("{{context}}", context_text)
        else:
            integrated = f"{template_content}\n\n---\n\nCONTEXT:\n{context_text}"
        
        return integrated, used_items

    async def _integrate_balanced(self, template_content: str,
                                context_result: ContextResult,
                                formatter) -> Tuple[str, List[ContextItem]]:
        """Balanced integration considering multiple factors"""
        # Start with all items
        all_items = context_result.context_items
        
        # If under token budget, use all
        total_context_tokens = sum(len(await formatter(item)) for item in all_items) // self.token_ratio
        
        if total_context_tokens <= self.default_token_limit * 0.6:
            return await self._integrate_sequential(template_content, context_result, formatter)
        
        # Otherwise use priority approach
        return await self._integrate_priority_relevance(template_content, context_result, formatter)

    async def _format_default(self, item: ContextItem) -> str:
        """Default formatting"""
        return f"[{item.item_type.value.upper()}] {item.content}"

    async def _format_compact(self, item: ContextItem) -> str:
        """Compact formatting"""
        return f"{item.content} ({item.item_type.value})"

    async def _format_detailed(self, item: ContextItem) -> str:
        """Detailed formatting"""
        metadata_str = ", ".join([f"{k}={v}" for k, v in item.metadata.items() if k in ['confidence', 'source']])
        parts = [
            f"Type: {item.item_type.value}",
            f"Content: {item.content}",
            f"Relevance: {item.relevance_score:.2f}"
        ]
        if metadata_str:
            parts.append(f"Info: {metadata_str}")
        
        return " | ".join(parts)

    async def _format_structured(self, item: ContextItem) -> str:
        """Structured formatting"""
        return f"{item.item_type.value.upper()}: {item.content}\n  Relevance: {item.relevance_score:.2f}"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return math.ceil(len(text) / self.token_ratio)

    async def _truncate_context_sections(self, prompt: str, limit: int) -> str:
        """Truncate context sections to fit token limit"""
        try:
            lines = prompt.split('\n')
            current_tokens = self._estimate_tokens(prompt)
            
            if current_tokens <= limit:
                return prompt
            
            # Try to preserve structure while truncating
            # Keep beginning, systematically remove from middle, keep end
            
            # Identify context section
            context_start = None
            context_end = None
            
            for i, line in enumerate(lines):
                if "CONTEXT:" in line or "RELEVANT:" in line:
                    context_start = i
                elif context_start is not None and line.strip() == "":
                    # Potential end of context
                    context_end = i
                    break
            
            if context_start is None:
                # No clear context section, just truncate from middle
                total_chars = len(prompt)
                target_chars = limit * self.token_ratio
                keep_per_side = target_chars // 2
                
                return (
                    prompt[:keep_per_side] + 
                    "\n\n[... content truncated due to token limit ...]\n\n" + 
                    prompt[-keep_per_side:]
                )
            
            # Truncate context section while preserving intro and outro
            intro = '\n'.join(lines[:context_start])
            context_lines = lines[context_start:context_end or context_start]
            outro = '\n'.join(lines[context_end:]) if context_end else ""
            
            # Calculate target context length
            intro_tokens = self._estimate_tokens(intro)
            outro_tokens = self._estimate_tokens(outro)
            available_for_context = limit - intro_tokens - outro_tokens - 50  # buffer
            
            if available_for_context < 100:
                # Not much space, keep minimal context
                context_part = "[CONTEXT TRUNCATED]"
            else:
                # Gradually include context items until limit
                context_part = ""
                current_context_tokens = 0
                
                for line in context_lines:
                    line_tokens = len(line) // self.token_ratio
                    if current_context_tokens + line_tokens <= available_for_context:
                        context_part += line + '\n'
                        current_context_tokens += line_tokens
                    else:
                        context_part += "[... additional context truncated ...]\n"
                        break
            
            return intro + '\n' + context_part + outro

        except Exception as e:
            logger.error(f"Context truncation failed: {e}")
            # Fallback: simply truncate end
            target_chars = limit * self.token_ratio
            return prompt[:target_chars] + "\n\n[... content truncated ...]"