# LM Studio Integration Guide for Conjecture

## Overview

Conjecture now supports LM Studio as a local LLM provider. This allows you to use local language models without requiring external API services, providing privacy and cost benefits.

## Supported Models

The system is configured to work with the `ibm/granite-4-h-tiny` model by default, but you can use any model compatible with LM Studio.

## Configuration

### Environment Variables

To use LM Studio as your LLM provider, set the following environment variables:

```bash
# Set the LLM provider to LM Studio
Conjecture_LLM_PROVIDER=lm_studio

# Set the API URL to your LM Studio instance (without /v1 suffix)
Conjecture_LLM_API_URL=http://localhost:1234

# Set the model name (optional, defaults to ibm/granite-4-h-tiny)
Conjecture_LLM_MODEL=ibm/granite-4-h-tiny

# Enable LLM features
# Note: LM Studio doesn't require an API key
```

### Using the unified provider configuration (recommended):

Alternatively, you can use the unified provider configuration which takes precedence:

```bash
# Provider configuration (takes precedence over Conjecture_ prefixed variables)
PROVIDER_API_URL=http://localhost:1234
PROVIDER_MODEL=ibm/granite-4-h-tiny
# PROVIDER_API_KEY is not needed for LM Studio
```

### Example .env file configuration:

```bash
# LM Studio Configuration
Conjecture_LLM_PROVIDER=lm_studio
Conjecture_LLM_API_URL=http://127.0.0.1:1234
Conjecture_LLM_MODEL=ibm/granite-4-h-tiny

# Optional: Set confidence threshold and other parameters
Conjecture_CONFIDENCE=0.90
Conjecture_BATCH_SIZE=5
```

### Using the unified provider configuration:

Alternatively, you can use the unified provider configuration:

```bash
# Provider configuration
PROVIDER_API_URL=http://localhost:1234/v1
PROVIDER_MODEL=ibm/granite-4-h-tiny
```

## Setup Instructions

1. **Install and run LM Studio**
   - Download LM Studio from https://lmstudio.ai/
   - Install and launch the application
   - Load your preferred model (e.g., `ibm/granite-4-h-tiny`)
   - Ensure the "Enable API" option is checked in the local server settings
   - Note the server address (default: http://localhost:1234)

2. **Configure Conjecture**
   - Set the environment variables as shown above
   - Or update your `.env` file with the configuration

3. **Verify the setup**
   - Initialize Conjecture and check that it connects to LM Studio
   - The system should report "LLM Bridge: lm_studio connected (primary)"

## Usage

Once configured, you can use Conjecture normally and it will route LLM requests to your local LM Studio instance:

```python
from src.contextflow import Conjecture

# Initialize with LM Studio configuration
cf = Conjecture()

# Use exploration features - will use LM Studio
result = cf.explore("machine learning", max_claims=5)

# Use claim validation - will use LM Studio
claim = cf.add_claim(
    content="Machine learning algorithms require substantial training data",
    confidence=0.85,
    claim_type="concept"
)
```

## Fallback Configuration

The system is configured with fallback behavior:
- If LM Studio is primary but unavailable, it will try Chutes.ai as fallback (if configured)
- If Chutes.ai is primary but unavailable, it will try LM Studio as fallback

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure LM Studio server is running and accessible at the configured URL
2. **Model not found**: Verify that the configured model is loaded in LM Studio
3. **API endpoint error**: Make sure the API endpoint in LM Studio is enabled

### Health Check
To verify your LM Studio configuration is working:

```python
from src.contextflow import Conjecture

cf = Conjecture()
status = cf.llm_bridge.get_status()
print(f"Primary available: {status['primary_available']}")
print(f"Fallback available: {status['fallback_available']}")
```

## Performance Considerations

- Local models may be slower than cloud-based services
- Performance depends on your hardware capabilities
- Memory requirements vary by model size
- Consider model size vs. performance trade-offs for your use case

## Security Notes

- LM Studio runs locally, so no data leaves your machine
- No API keys required for local models
- Full control over the model and its outputs

## Troubleshooting and Performance

### Common Issues

1. **Connection refused**: Ensure LM Studio server is running and accessible at the configured URL
2. **Model not found**: Verify that the configured model is loaded in LM Studio
3. **API endpoint error**: Make sure the API endpoint in LM Studio is enabled
4. **Unicode encoding errors**: These may occur due to special characters in model responses; the system handles these gracefully

### Performance Considerations

- Local models typically slower than cloud-based services
- Performance depends on hardware capabilities (CPU/GPU)
- Memory requirements vary by model size
- Consider model size vs. performance trade-offs for your use case
- The ibm/granite-4-h-tiny model is optimized for efficiency

### Logging and Monitoring

The system provides basic logging of:
- Provider connection status
- Request/response timing
- Error conditions and fallback usage
- Available models detection

### Resource Management

- The system properly handles asyncio session lifecycle
- Connections are closed appropriately when no longer needed
- Memory usage remains stable during extended operations