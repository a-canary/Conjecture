"""
Tiny Model Processor for IBM Granite Tiny
Specialized handling for tiny models to achieve SOTA reasoning with Conjecture methods
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.core.models import Claim, ClaimState, ClaimType
from src.config.tiny_model_config import (
    TinyModelConfig, 
    TinyModelPromptTemplates, 
    is_tiny_model,
    get_tiny_model_config,
    optimize_prompt_for_tiny_model
)
from src.processing.json_frontmatter_parser import (
    JSONFrontmatterParser,
    ParseResult,
    ResponseType,
    create_json_frontmatter_prompt_template
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

class TinyModelProcessor:
    """
    Specialized processor for tiny models like IBM Granite Tiny.
    
    Optimizes prompts, context, and processing parameters for small LLMs
    to achieve maximum reasoning performance.
    """
    
    def __init__(self, config: Optional[TinyModelConfig] = None):
        self.config = config or TinyModelConfig()
        self.json_parser = JSONFrontmatterParser()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'json_frontmatter_success': 0,
            'fallback_success': 0
        }
    
    def generate_claims(
        self, 
        topic: str, 
        context_claims: List[Claim] = None,
        max_claims: int = 5,
        llm_generator = None
    ) -> Tuple[List[Claim], Dict[str, Any]]:
        """
        Generate claims using tiny model with optimized prompts and parameters.
        
        Args:
            topic: Research topic
            context_claims: Optional context claims
            max_claims: Maximum number of claims to generate
            llm_generator: LLM generation function
            
        Returns:
            Tuple of (claims, metadata)
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Optimize context for tiny models
            optimized_context = self._optimize_context(context_claims or [])
            
            # Generate optimized prompt
            prompt = self._create_optimized_prompt(topic, optimized_context, max_claims)
            
            # Generate response
            response = llm_generator(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            response_text = response.get('content', '')
            
            # Parse response with JSON frontmatter
            claims = self._parse_response(response_text)
            
            # Apply tiny model post-processing
            claims = self._post_process_claims(claims, topic)
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self.stats['avg_response_time'] = self._update_avg_time(response_time)
            
            metadata = {
                'processor': 'tiny_model',
                'model': self.config.model_name,
                'response_time': response_time,
                'claims_generated': len(claims),
                'context_used': len(optimized_context),
                'optimizations_applied': self._get_optimizations_applied()
            }
            
            return claims, metadata
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Tiny model claim generation failed: {e}")
            return [], {'error': str(e)}
    
    def analyze_claims(
        self, 
        claims: List[Claim], 
        llm_generator = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze claims using tiny model with optimized analysis prompts.
        
        Args:
            claims: Claims to analyze
            llm_generator: LLM generation function
            
        Returns:
            Tuple of (analysis_result, metadata)
        """
        start_time = time.time()
        
        try:
            # Create optimized analysis prompt
            prompt = self._create_analysis_prompt(claims)
            
            # Generate analysis
            response = llm_generator(
                prompt=prompt,
                max_tokens=self.config.max_tokens // 2,  # Shorter for analysis
                temperature=self.config.temperature
            )
            
            response_text = response.get('content', '')
            
            # Parse analysis
            analysis = self._parse_analysis(response_text)
            
            response_time = time.time() - start_time
            
            metadata = {
                'processor': 'tiny_model_analysis',
                'model': self.config.model_name,
                'response_time': response_time,
                'claims_analyzed': len(claims)
            }
            
            return analysis, metadata
            
        except Exception as e:
            logger.error(f"Tiny model analysis failed: {e}")
            return {'error': str(e)}, {'error': str(e)}
    
    def validate_claims(
        self, 
        claims: List[Claim], 
        llm_generator = None
    ) -> Tuple[List[Claim], Dict[str, Any]]:
        """
        Validate claims using tiny model with confidence boosting.
        
        Args:
            claims: Claims to validate
            llm_generator: LLM generation function
            
        Returns:
            Tuple of (validated_claims, metadata)
        """
        start_time = time.time()
        validated_claims = []
        
        try:
            for claim in claims:
                # Create validation prompt for single claim
                prompt = self._create_validation_prompt(claim)
                
                # Generate validation
                response = llm_generator(
                    prompt=prompt,
                    max_tokens=200,  # Very short for validation
                    temperature=0.2  # Lower temperature for validation
                )
                
                response_text = response.get('content', '')
                
                # Parse validation result
                validation_result = self._parse_validation(response_text)
                
                # Update claim based on validation
                if validation_result.get('is_valid', False):
                    updated_claim = self._apply_validation(claim, validation_result)
                    validated_claims.append(updated_claim)
            
            response_time = time.time() - start_time
            
            metadata = {
                'processor': 'tiny_model_validation',
                'model': self.config.model_name,
                'response_time': response_time,
                'claims_validated': len(validated_claims),
                'validation_rate': len(validated_claims) / len(claims) if claims else 0
            }
            
            return validated_claims, metadata
            
        except Exception as e:
            logger.error(f"Tiny model validation failed: {e}")
            return claims, {'error': str(e)}
    
    def _optimize_context(self, claims: List[Claim]) -> List[Claim]:
        """Optimize context for tiny models by reducing size and complexity"""
        if not claims:
            return []
        
        # Sort by confidence and take top claims
        sorted_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)
        
        # Limit context size
        max_context = min(len(sorted_claims), self.config.max_context_size)
        context_claims = sorted_claims[:max_context]
        
        # Simplify claim content for tiny models
        optimized_claims = []
        for claim in context_claims:
            # Truncate very long claims
            if len(claim.content) > 200:
                simplified_content = claim.content[:197] + "..."
            else:
                simplified_content = claim.content
            
            # Create optimized claim
            optimized_claim = Claim(
                id=claim.id,
                content=simplified_content,
                confidence=claim.confidence,
                state=claim.state,
                tags=claim.tags + ["context_optimized"],
                created=claim.created,
                updated=claim.updated
            )
            optimized_claims.append(optimized_claim)
        
        return optimized_claims
    
    def _create_optimized_prompt(self, topic: str, context: List[Claim], max_claims: int) -> str:
        """Create optimized prompt for tiny models"""
        if self.config.use_json_frontmatter:
            # Use JSON frontmatter for better parsing
            template = create_json_frontmatter_prompt_template(ResponseType.CLAIMS)
            
            # Add context
            context_str = ""
            if context:
                context_str = "\nContext:\n" + "\n".join([
                    f"[{c.id} | {c.content} | / {c.confidence:.2f}]" 
                    for c in context
                ])
            
            # Add topic and constraints
            prompt = f"""{template}

Topic: {topic}
{context_str}

Generate {min(max_claims, 3)} clear, specific claims using JSON frontmatter format.

Focus on:
- Factual accuracy
- Clear, concise statements
- Appropriate confidence scores
- Simple claim types: fact, concept, example"""
        else:
            # Use simplified template
            template = TinyModelPromptTemplates.get_simplified_claim_prompt()
            
            context_str = ""
            if context:
                context_str = "\nContext:\n" + "\n".join([
                    f"[{c.id} | {c.content} | / {c.confidence:.2f}]" 
                    for c in context[:3]  # Even more limited for text format
                ])
            
            prompt = template.format(
                topic=topic,
                context=context_str
            )
        
        # Optimize prompt for tiny models
        return optimize_prompt_for_tiny_model(prompt, self.config.model_name)
    
    def _create_analysis_prompt(self, claims: List[Claim]) -> str:
        """Create optimized analysis prompt for tiny models"""
        claims_text = "\n".join([
            f"[{c.id} | {c.content} | / {c.confidence:.2f}]" 
            for c in claims[:5]  # Limit claims for analysis
        ])
        
        template = TinyModelPromptTemplates.get_analysis_prompt()
        prompt = template.format(claims=claims_text)
        
        return optimize_prompt_for_tiny_model(prompt, self.config.model_name)
    
    def _create_validation_prompt(self, claim: Claim) -> str:
        """Create optimized validation prompt for tiny models"""
        template = TinyModelPromptTemplates.get_validation_prompt()
        prompt = template.format(claim=claim.content)
        
        return optimize_prompt_for_tiny_model(prompt, self.config.model_name)
    
    def _parse_response(self, response_text: str) -> List[Claim]:
        """Parse LLM response with JSON frontmatter support"""
        try:
            # Try JSON frontmatter first
            result = self.json_parser.parse_response(response_text)
            if result.success:
                self.stats['json_frontmatter_success'] += 1
                return result.claims
        except Exception as e:
            logger.debug(f"JSON frontmatter parsing failed: {e}")
        
        # Fallback to simple parsing
        try:
            claims = self._parse_simple_claims(response_text)
            self.stats['fallback_success'] += 1
            return claims
        except Exception as e:
            logger.error(f"All parsing methods failed: {e}")
            return []
    
    def _parse_simple_claims(self, response_text: str) -> List[Claim]:
        """Parse claims from simple text format"""
        import re
        
        # Look for [c1 | content | / confidence] format
        pattern = r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]'
        matches = re.findall(pattern, response_text)
        
        claims = []
        for claim_id, content, confidence in matches:
            try:
                claim = Claim(
                    id=f"c{claim_id}",
                    content=content.strip(),
                    confidence=float(confidence),
                    state=ClaimState.EXPLORE,
                    tags=["simple_parsed", "auto_generated"],
                    created=datetime.utcnow(),
                    updated=datetime.utcnow()
                )
                claims.append(claim)
            except Exception as e:
                logger.warning(f"Failed to parse claim {claim_id}: {e}")
        
        return claims
    
    def _parse_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse analysis response"""
        # Simple analysis parsing for tiny models
        analysis = {
            'relationships': [],
            'confidence_assessment': {},
            'recommendations': [],
            'raw_response': response_text
        }
        
        # Try to extract key insights
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if 'relationship' in line.lower():
                analysis['relationships'].append(line)
            elif 'confidence' in line.lower():
                analysis['confidence_assessment']['summary'] = line
            elif 'recommend' in line.lower():
                analysis['recommendations'].append(line)
        
        return analysis
    
    def _parse_validation(self, response_text: str) -> Dict[str, Any]:
        """Parse validation response"""
        # Simple validation parsing
        validation = {
            'is_valid': True,  # Default to valid for tiny models
            'confidence_adjustment': 0.0,
            'reasoning': response_text
        }
        
        # Look for validation indicators
        text_lower = response_text.lower()
        if 'invalid' in text_lower or 'false' in text_lower:
            validation['is_valid'] = False
        elif 'valid' in text_lower or 'true' in text_lower:
            validation['is_valid'] = True
        
        # Extract confidence adjustment
        import re
        confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', text_lower)
        if confidence_match:
            try:
                validation['confidence_adjustment'] = float(confidence_match.group(1)) - 0.8  # Relative to baseline
            except:
                pass
        
        return validation
    
    def _post_process_claims(self, claims: List[Claim], topic: str) -> List[Claim]:
        """Post-process claims for tiny model optimization"""
        processed_claims = []
        
        for claim in claims:
            # Apply confidence boosting for tiny models
            if self.config.enable_confidence_boosting:
                boosted_confidence = min(1.0, claim.confidence + 0.1)
                claim.confidence = boosted_confidence
            
            # Add topic tag
            if 'topic' not in claim.tags:
                claim.tags.append(f"topic:{topic.lower().replace(' ', '_')}")
            
            # Add tiny model specific tags
            claim.tags.extend(['tiny_model_generated', 'optimized'])
            
            # Ensure claim ID format
            if not claim.id.startswith('c'):
                claim.id = f"c{claim.id}"
            
            processed_claims.append(claim)
        
        return processed_claims
    
    def _apply_validation(self, claim: Claim, validation: Dict[str, Any]) -> Claim:
        """Apply validation results to claim"""
        if validation.get('is_valid', True):
            # Adjust confidence based on validation
            adjustment = validation.get('confidence_adjustment', 0.0)
            new_confidence = max(0.0, min(1.0, claim.confidence + adjustment))
            
            # Create updated claim
            updated_claim = Claim(
                id=claim.id,
                content=claim.content,
                confidence=new_confidence,
                state=claim.state,
                tags=claim.tags + ['validated'],
                created=claim.created,
                updated=datetime.utcnow()
            )
            
            return updated_claim
        
        return claim
    
    def _update_avg_time(self, new_time: float) -> float:
        """Update average response time"""
        total = self.stats['total_requests']
        current_avg = self.stats['avg_response_time']
        return (current_avg * (total - 1) + new_time) / total
    
    def _get_optimizations_applied(self) -> List[str]:
        """Get list of optimizations applied"""
        optimizations = []
        
        if self.config.use_simplified_prompts:
            optimizations.append('simplified_prompts')
        if self.config.use_json_frontmatter:
            optimizations.append('json_frontmatter')
        if self.config.enable_confidence_boosting:
            optimizations.append('confidence_boosting')
        if self.config.enable_two_step_processing:
            optimizations.append('two_step_processing')
        
        return optimizations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = self.stats['total_requests']
        if total == 0:
            return self.stats.copy()
        
        return {
            **self.stats,
            'success_rate': self.stats['successful_requests'] / total,
            'failure_rate': self.stats['failed_requests'] / total,
            'json_frontmatter_rate': self.stats['json_frontmatter_success'] / total,
            'fallback_rate': self.stats['fallback_success'] / total
        }

def create_tiny_model_processor(model_name: str) -> Optional[TinyModelProcessor]:
    """Create tiny model processor if model is a tiny model"""
    if is_tiny_model(model_name):
        config = get_tiny_model_config(model_name)
        return TinyModelProcessor(config)
    return None