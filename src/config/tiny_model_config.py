"""
Tiny Model Configuration for IBM Granite Tiny
Optimized settings for small LLMs to achieve SOTA reasoning with Conjecture methods
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TinyModelConfig:
    """Configuration optimized for tiny models like IBM Granite Tiny"""
    
    # Model-specific parameters
    model_name: str = "ibm/granite-4-h-tiny"
    max_tokens: int = 42000  # Updated for granite-4-h-tiny proper evaluation
    temperature: float = 0.3  # Lower for more consistent reasoning
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Context optimization
    max_context_size: int = 5  # Reduced from 10
    max_context_concepts: int = 3  # Reduced from 10
    max_context_references: int = 2  # Reduced from 8
    max_context_goals: int = 1  # Reduced from 3
    
    # Processing parameters
    batch_size: int = 3  # Reduced from 10
    confidence_threshold: float = 0.90  # Slightly lower for tiny models
    confident_threshold: float = 0.75  # Reduced from 0.8
    
    # Prompt optimization
    use_simplified_prompts: bool = True
    include_examples: bool = True
    max_examples: int = 2  # Reduced context
    
    # Error handling
    max_retries: int = 2  # Reduced from 3
    retry_delay: float = 0.5  # Faster retry
    timeout: int = 15  # Shorter timeout
    
    # Special handling for tiny models
    enable_two_step_processing: bool = True
    use_json_frontmatter: bool = True
    enable_confidence_boosting: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for configuration"""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_context_size": self.max_context_size,
            "max_context_concepts": self.max_context_concepts,
            "max_context_references": self.max_context_references,
            "max_context_goals": self.max_context_goals,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "confident_threshold": self.confident_threshold,
            "use_simplified_prompts": self.use_simplified_prompts,
            "include_examples": self.include_examples,
            "max_examples": self.max_examples,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "enable_two_step_processing": self.enable_two_step_processing,
            "use_json_frontmatter": self.use_json_frontmatter,
            "enable_confidence_boosting": self.enable_confidence_boosting
        }

class TinyModelPromptTemplates:
    """Optimized prompt templates for tiny models"""
    
    @staticmethod
    def get_simplified_claim_prompt() -> str:
        """Simplified claim generation prompt for tiny models"""
        return """Generate 3-5 clear claims about this topic.

Requirements:
- Use format: [c1 | claim text | / confidence]
- Confidence between 0.0-1.0
- Focus on factual accuracy
- Keep claims concise and specific

Topic: {topic}

Claims:"""

    @staticmethod
    def get_json_frontmatter_prompt() -> str:
        """JSON frontmatter prompt optimized for tiny models"""
        return """Format response as JSON frontmatter.

```json
---
{
  "type": "claims",
  "confidence": 0.90,
  "claims": [
    {
      "id": "c1",
      "content": "Clear, specific claim",
      "confidence": 0.90,
      "type": "fact"
    }
  ]
}
---
```

Topic: {topic}

Generate 3-5 claims:"""

    @staticmethod
    def get_analysis_prompt() -> str:
        """Simplified analysis prompt for tiny models"""
        return """Analyze these claims:

{claims}

Provide brief analysis:
1. Key relationships
2. Confidence assessment
3. Recommendations

Keep response concise."""

    @staticmethod
    def get_validation_prompt() -> str:
        """Simplified validation prompt for tiny models"""
        return """Validate claim: {claim}

Check:
- Factual accuracy
- Logical consistency
- Confidence score

Respond with validation result."""

# Default tiny model configuration
DEFAULT_TINY_MODEL_CONFIG = TinyModelConfig()

# LM Studio specific configuration
LM_STUDIO_CONFIG = {
    "url": "http://localhost:1234/v1",
    "api": "",
    "model": "ibm/granite-4-h-tiny",
    "name": "lm_studio",
    "priority": 1,
    "is_local": True,
    "description": "IBM Granite Tiny model for SOTA reasoning research",
    "max_tokens": 42000,
    "temperature": 0.3,
    "supports_json_frontmatter": True,
    "optimized_for_tiny_models": True
}

def is_tiny_model(model_name: str) -> bool:
    """Check if model is a tiny model requiring special handling"""
    tiny_model_patterns = [
        "tiny", "small", "mini", "1b", "3b", "0.5b", "granite-4-h-tiny"
    ]
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in tiny_model_patterns)

def get_tiny_model_config(model_name: str) -> Optional[TinyModelConfig]:
    """Get optimized configuration for tiny models"""
    if is_tiny_model(model_name):
        return DEFAULT_TINY_MODEL_CONFIG
    return None

def optimize_prompt_for_tiny_model(prompt: str, model_name: str) -> str:
    """Optimize prompt for tiny models"""
    if not is_tiny_model(model_name):
        return prompt
    
    # Simplify prompt for tiny models
    optimized = prompt
    
    # Reduce length
    if len(optimized) > 1000:
        optimized = optimized[:1000] + "..."
    
    # Simplify language
    replacements = {
        "Please": "",
        "Could you": "",
        "Would you": "",
        "I would like you to": "",
        "comprehensive": "key",
        "detailed": "brief",
        "extensive": "focused",
        "thorough": "quick",
        "elaborate": "explain",
        "demonstrate": "show",
        "illustrate": "show"
    }
    
    for old, new in replacements.items():
        optimized = optimized.replace(old, new)
    
    return optimized.strip()