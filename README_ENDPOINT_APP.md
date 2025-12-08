# Conjecture EndPoint App

A lightweight FastAPI application that provides direct REST API access to Conjecture's ProcessingInterface capabilities with clean architectural separation.

## Features

### Core API Endpoints
- **Claims Management**: Create, read, update, delete, and search claims
- **Evaluation**: Evaluate claims using LLM with context
- **Context Building**: Build context for claim processing
- **Session Management**: Create and resume processing sessions
- **Tool Execution**: Execute available tools with parameters
- **Batch Operations**: Batch create and evaluate claims for performance

### Real-time Features
- **Server-Sent Events**: Stream processing events in real-time
- **WebSocket Support**: Bidirectional real-time communication
- **Event Filtering**: Filter events by type, session, or timestamp

### Integration Features
- **ProcessingInterface Integration**: Direct use of ProcessingInterface for clean architecture
- **Unified Configuration**: Uses existing Conjecture configuration system
- **Provider Routing**: Support for multiple LLM providers with failover
- **Health Monitoring**: Comprehensive health checks and statistics

## Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements_endpoint.txt

# Ensure Conjecture is configured
python conjecture config
python conjecture setup
```

### Running the Server

#### Development Mode
```bash
# Start with auto-reload and debug logging
python run_endpoint_app.py --dev

# Or directly
python src/endpoint_app.py --reload --log-level debug
```

#### Production Mode
```bash
# Start production server
python run_endpoint_app.py --prod

# Or with custom settings
python run_endpoint_app.py --host 0.0.0.0 --port 8080
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Configuration**: http://localhost:8000/config

## API Usage Examples

### Create a Claim
```bash
curl -X POST "http://localhost:8000/claims" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "FastAPI provides clean integration with ProcessingInterface",
    "confidence": 0.9,
    "tags": ["api", "integration"]
  }'
```

### Search Claims
```bash
curl "http://localhost:8000/claims/search?query=FastAPI&limit=5"
```

### Evaluate a Claim
```bash
curl -X POST "http://localhost:8000/evaluate?claim_id=c1234567" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session_1"}'
```

### Get Context
```bash
curl -X GET "http://localhost:8000/context" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_ids": ["c1234567", "c7654321"],
    "max_skills": 5,
    "max_samples": 10
  }'
```

### Stream Events (Server-Sent Events)
```bash
curl -N "http://localhost:8000/events/stream?session_id=session_1"
```

### WebSocket Events
```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8000/events/ws?session_id=session_1');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Event:', data);
};
```

## API Endpoints Reference

### Claims
| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/claims` | Create new claim |
| GET | `/claims/{claim_id}` | Get specific claim |
| PUT | `/claims/{claim_id}` | Update existing claim |
| DELETE | `/claims/{claim_id}` | Delete claim |
| GET | `/claims/search` | Search claims with filters |
| POST | `/claims/batch` | Batch create claims |

### Evaluation & Context
| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/evaluate` | Evaluate claim |
| GET | `/context` | Build context |

### Sessions
| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/sessions` | Create session |
| POST | `/sessions/{session_id}/resume` | Resume session |

### Tools
| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/tools` | List available tools |
| POST | `/tools/execute` | Execute tool |

### Events
| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/events/stream` | Server-Sent Events stream |
| WS | `/events/ws` | WebSocket endpoint |

### System
| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Health status |
| GET | `/stats` | System statistics |
| GET | `/config` | Configuration info |
| GET | `/` | API information |

## Testing

### Run Comprehensive Tests
```bash
# Test against running server
python test_endpoint_app.py

# Test against specific URL
python test_endpoint_app.py --url http://localhost:8080

# Run specific tests
python test_endpoint_app.py --tests health claims search
```

### Test Coverage
The test suite validates:
- ✅ Health checks and system status
- ✅ Configuration integration
- ✅ Session management
- ✅ CRUD operations for claims
- ✅ Search and filtering
- ✅ Claim evaluation
- ✅ Context building
- ✅ Tool execution
- ✅ Batch operations
- ✅ Error handling (400, 404, 500)
- ✅ Server-Sent Events streaming
- ✅ WebSocket real-time communication

## Architecture

### Clean Separation
The EndPoint App maintains clean architectural separation by:
- **Direct ProcessingInterface Usage**: No additional abstraction layers
- **Minimal Overhead**: Lightweight FastAPI implementation
- **Event-Driven**: Real-time event streaming and notifications
- **Configuration Integration**: Uses existing unified config system

### Provider Support
The app supports multiple LLM providers:
- **LM Studio**: Local model hosting
- **OpenRouter**: Multi-provider routing
- **z.ai**: Cloud-based AI services
- **Failover**: Automatic provider switching on failures

### Performance Features
- **Async Operations**: All endpoints use async ProcessingInterface methods
- **Batch Processing**: Optimized batch operations for better performance
- **Caching**: Leverages Conjecture's internal caching
- **Monitoring**: Built-in performance metrics and health checks

## Configuration

### Environment Variables
```bash
# Optional: Override default configuration
export CONJECTURE_CONFIG_PATH="/path/to/config.json"
export CONJECTURE_LOG_LEVEL="debug"
export CONJECTURE_PROVIDERS_URL="http://localhost:11434"
```

### Provider Configuration
The app uses the same configuration as Conjecture CLI:
```json
{
  "providers": [
    {
      "url": "https://llm.chutes.ai/v1",
      "api": "your-api-key",
      "model": "zai-org/GLM-4.6-FP8",
      "name": "chutes"
    },
    {
      "url": "http://localhost:11434",
      "api": "",
      "model": "llama2",
      "name": "ollama"
    }
  ],
  "confidence_threshold": 0.95,
  "max_context_size": 10
}
```

## Development

### Project Structure
```
src/
├── endpoint_app.py          # Main FastAPI application
├── conjecture.py           # ProcessingInterface implementation
├── interfaces/
│   └── processing_interface.py  # Abstract interface definition
├── config/
│   └── unified_config.py       # Configuration system
└── core/
    └── models.py              # Data models

tests/
├── test_endpoint_app.py    # Comprehensive test suite
└── endpoint_test_results.json  # Test output

docs/
└── README_ENDPOINT_APP.md   # This documentation
```

### Adding New Endpoints
1. Define request/response models in `endpoint_app.py`
2. Implement endpoint function with proper error handling
3. Add to FastAPI app with appropriate router
4. Update test suite with new endpoint tests
5. Update documentation

### Error Handling
All endpoints return standardized responses:
```json
{
  "success": true/false,
  "data": {...},
  "message": "Description",
  "timestamp": "2025-12-08T17:00:00.000Z"
}
```

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_endpoint.txt .
RUN pip install -r requirements_endpoint.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_endpoint_app.py", "--prod"]
```

### Production Considerations
- **CORS**: Configure appropriate origins for production
- **Authentication**: Add API key or JWT authentication
- **Rate Limiting**: Implement request rate limits
- **Logging**: Configure structured logging for monitoring
- **HTTPS**: Use reverse proxy (nginx/caddy) for SSL

## Troubleshooting

### Common Issues
1. **Port already in use**: Change port with `--port` argument
2. **Missing dependencies**: Run `pip install -r requirements_endpoint.txt`
3. **Configuration errors**: Check `python conjecture config`
4. **Provider connection**: Verify LLM provider URLs and API keys

### Debug Mode
```bash
# Enable debug logging and auto-reload
python run_endpoint_app.py --dev --log-level debug
```

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Get detailed statistics
curl http://localhost:8000/stats
```

## License

This EndPoint App is part of the Conjecture project and follows the same licensing terms.