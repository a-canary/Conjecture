# Chutes API Documentation Summary

## API Overview
- **Base URL**: `https://llm.chutes.ai/v1`
- **Compatibility**: OpenAI-compatible `/chat/completions` endpoint
- **Authentication**: Bearer token via `CHUTES_API_KEY`
- **Total Models**: 52 available models

## Authentication
```bash
# Environment variable
CHUTES_API_KEY=your-api-key

# HTTP Header
Authorization: Bearer cpk_your-api-key
```

## Request Format (OpenAI Compatible)
```json
{
  "model": "chutes/zai-org/GLM-4.6",
  "messages": [
    {"role": "user", "content": "Your prompt here"}
  ],
  "max_tokens": 1500,
  "temperature": 0.3
}
```

## Response Format
```json
{
  "id": "chat-completion-id",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "chutes/zai-org/GLM-4.6",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text",
        "reasoning_content": "Reasoning text (may be null)"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

## Key Models for Research

### High-Quality Models
| Model | Context | Input $/1M | Output $/1M | Features |
|-------|---------|------------|-------------|----------|
| `chutes/zai-org/GLM-4.6` | 203K | $0.40 | $2.00 | Tools, 203K context |
| `chutes/deepseek-ai/DeepSeek-R1` | 164K | $0.30 | $1.00 | Tools, 164K context |
| `chutes/deepseek-ai/DeepSeek-V3` | 164K | $0.30 | $1.00 | Tools, 164K context |
| `chutes/Qwen/Qwen2.5-72B-Instruct` | 33K | $0.07 | $0.26 | Tools, 33K context |
| `chutes/openai/gpt-oss-20b` | 131K | $0.05 | $0.22 | Tools, 131K context |

### Model Naming Convention
- **Format**: `chutes/{provider}/{model-name}`
- **Example**: `chutes/zai-org/GLM-4.6`
- **Alternative**: Direct model name also works: `zai-org/GLM-4.6`

## Special Response Format
Chutes API may return responses with:
- `content`: Standard response content (may be null)
- `reasoning_content`: Alternative content field (may contain actual response)

**Important**: Check both fields and use whichever is not null.

## Usage Examples

### Python (requests)
```python
import requests

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "chutes/zai-org/GLM-4.6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
}

response = requests.post(
    "https://llm.chutes.ai/v1/chat/completions",
    headers=headers,
    json=data
)
```

### JavaScript
```javascript
const response = await fetch('https://llm.chutes.ai/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'chutes/zai-org/GLM-4.6',
    messages: [{ role: 'user', content: 'Hello' }],
    max_tokens: 100
  })
});
```

## Research Recommendations

### Best Models for Conjecture Research
1. **GLM-4.6**: Highest quality, 203K context, excellent for complex reasoning
2. **DeepSeek-R1**: Good balance of quality and cost, 164K context
3. **GPT-OSS-20B**: Fast and cost-effective, 131K context

### Cost Considerations
- GLM-4.6: Most expensive but highest quality
- GPT-OSS-20B: Best value for speed and quality
- DeepSeek models: Good middle ground

### Context Lengths
- GLM-4.6: 203K tokens (largest)
- DeepSeek models: 164K tokens
- GPT-OSS-20B: 131K tokens
- Qwen models: 33K tokens (smallest)

## Error Handling
Common errors and solutions:
- `model not found`: Check model name spelling
- `invalid API key`: Ensure key starts with `cpk_`
- `timeout`: Increase timeout for large models
- `rate limit`: Implement retry logic