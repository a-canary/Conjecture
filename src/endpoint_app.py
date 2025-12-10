#!/usr/bin/env python3
"""
FastAPI EndPoint App for Conjecture
Lightweight, minimal overhead design with direct ProcessingInterface integration
Provides clean REST API access to Conjecture's processing capabilities
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Import Conjecture components
from conjecture import Conjecture
from src.config.unified_config import get_config
from src.interfaces.processing_interface import (
    ProcessingInterface,
    EvaluationResult,
    ToolResult,
    Context as ProcessingContext,
    Session,
    SessionState,
    ProcessingEvent,
    EventType
)
from src.core.models import Claim, ClaimFilter, ClaimState
from src.processing.simplified_llm_manager import SimplifiedLLMManager
from src.processing.enhanced_llm_router import get_enhanced_llm_router

# API Request/Response Models
class ClaimCreateRequest(BaseModel):
    """Request model for claim creation"""
    content: str = Field(..., min_length=5, max_length=2000, description="Claim content text")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Initial confidence score")
    tags: Optional[List[str]] = Field(None, description="Optional tags for categorization")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class ClaimUpdateRequest(BaseModel):
    """Request model for claim updates"""
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to update")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class ClaimEvaluateRequest(BaseModel):
    """Request model for claim evaluation"""
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class ToolExecuteRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str = Field(..., description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class ContextRequest(BaseModel):
    """Request model for context building"""
    claim_ids: List[str] = Field(..., description="List of claim IDs to build context for")
    max_skills: int = Field(5, ge=1, le=20, description="Maximum number of skills to include")
    max_samples: int = Field(10, ge=1, le=50, description="Maximum number of samples to include")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class SessionCreateRequest(BaseModel):
    """Request model for session creation"""
    user_data: Optional[Dict[str, Any]] = Field(None, description="Optional user-specific data")

class BatchClaimsRequest(BaseModel):
    """Request model for batch claim creation"""
    claims_data: List[Dict[str, Any]] = Field(..., description="List of claim creation data")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class BatchEvaluateRequest(BaseModel):
    """Request model for batch claim evaluation"""
    claim_ids: List[str] = Field(..., description="List of claim IDs to evaluate")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")

class ProviderManagementRequest(BaseModel):
    """Request model for provider management"""
    action: str = Field(..., description="Action: enable, disable, reset_metrics, set_strategy")
    provider_name: Optional[str] = Field(None, description="Provider name for enable/disable actions")
    routing_strategy: Optional[str] = Field(None, description="Routing strategy (priority, round_robin, load_balanced, fastest_response)")

class ProviderTestRequest(BaseModel):
    """Request model for provider testing"""
    prompt: str = Field(default="Hello, please respond with a brief greeting.", description="Test prompt")
    provider_name: Optional[str] = Field(None, description="Specific provider to test (optional)")

# Response Models
class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(None, description="Response data")
    message: str = Field("", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Request success status")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Global ProcessingInterface instance
processing_interface: Optional[ProcessingInterface] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global processing_interface
    
    # Startup
    try:
        print("Starting Conjecture EndPoint App...")
        
        # Initialize configuration
        config = get_config()
        print(f"Configuration loaded: {len(config.providers)} providers")
        
        # Initialize ProcessingInterface with simplified manager for single provider
        processing_interface = Conjecture(config)
        # Force use of simplified manager for single provider configuration
        processing_interface._initialize_llm_bridge(use_enhanced=False)
        await processing_interface.start_services()
        
        print("Conjecture EndPoint App started successfully")
        
        yield
        
    except Exception as e:
        print(f"Failed to start EndPoint App: {e}")
        raise
    
    finally:
        # Shutdown
        if processing_interface:
            print("Stopping Conjecture EndPoint App...")
            await processing_interface.stop_services()
            print("EndPoint App stopped")

# Create FastAPI app
app = FastAPI(
    title="Conjecture EndPoint API",
    description="Lightweight FastAPI interface for Conjecture AI-Powered Evidence-Based Reasoning System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections manager
class WebSocketManager:
    """Manages WebSocket connections for real-time event streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.event_subscriptions: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, subscription_id: Optional[str] = None):
        """Accept and register WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if subscription_id:
            if subscription_id not in self.event_subscriptions:
                self.event_subscriptions[subscription_id] = []
            self.event_subscriptions[subscription_id].append(websocket)
        
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from event subscriptions
        for sub_id, connections in self.event_subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
                if not connections:
                    del self.event_subscriptions[sub_id]
                break
        
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_event(self, event: ProcessingEvent):
        """Broadcast event to all connected clients"""
        if not self.active_connections:
            return
        
        event_data = event.to_dict()
        message = {"type": "event", "data": event_data}
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"⚠️ WebSocket send error: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

# Global WebSocket manager
websocket_manager = WebSocketManager()

# Helper functions
def get_processing_interface() -> ProcessingInterface:
    """Get the global ProcessingInterface instance"""
    global processing_interface
    if not processing_interface:
        raise HTTPException(status_code=503, detail="ProcessingInterface not initialized")
    return processing_interface

def create_api_response(success: bool, data: Any = None, message: str = "") -> APIResponse:
    """Create standardized API response"""
    return APIResponse(success=success, data=data, message=message)

def create_error_response(error: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(error=error, details=details)

# Event streaming subscription management
event_subscriptions: Dict[str, str] = {}

async def event_callback(event: ProcessingEvent):
    """Callback for ProcessingInterface events"""
    # Broadcast to WebSocket connections
    await websocket_manager.broadcast_event(event)

# Core API Endpoints

@app.post("/claims", response_model=APIResponse)
async def create_claim(request: ClaimCreateRequest, background_tasks: BackgroundTasks):
    """
    Create a new claim with automatic evaluation.
    
    - **content**: Claim content text (required, 5-2000 chars)
    - **confidence**: Initial confidence score (optional, 0.0-1.0)
    - **tags**: Optional tags for categorization
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        claim = await interface.create_claim(
            content=request.content,
            confidence=request.confidence,
            tags=request.tags,
            session_id=request.session_id
        )
        
        return create_api_response(
            success=True,
            data=claim.to_dict(),
            message=f"Claim {claim.id} created successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/claims/{claim_id}", response_model=APIResponse)
async def get_claim(claim_id: str, session_id: Optional[str] = None):
    """
    Retrieve a specific claim by ID.
    
    - **claim_id**: ID of claim to retrieve
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        claim = await interface.get_claim(claim_id, session_id=session_id)
        
        return create_api_response(
            success=True,
            data=claim.to_dict(),
            message=f"Claim {claim_id} retrieved successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/claims/{claim_id}", response_model=APIResponse)
async def update_claim(claim_id: str, request: ClaimUpdateRequest):
    """
    Update an existing claim.
    
    - **claim_id**: ID of claim to update
    - **updates**: Dictionary of fields to update
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        updated_claim = await interface.update_claim(
            claim_id=claim_id,
            updates=request.updates,
            session_id=request.session_id
        )
        
        return create_api_response(
            success=True,
            data=updated_claim.to_dict(),
            message=f"Claim {claim_id} updated successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/claims/{claim_id}", response_model=APIResponse)
async def delete_claim(claim_id: str, session_id: Optional[str] = None):
    """
    Delete a claim by ID.
    
    - **claim_id**: ID of claim to delete
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        success = await interface.delete_claim(claim_id, session_id=session_id)
        
        if success:
            return create_api_response(
                success=True,
                message=f"Claim {claim_id} deleted successfully"
            )
        else:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/claims/search", response_model=APIResponse)
async def search_claims(
    query: str,
    tags: Optional[List[str]] = None,
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
    limit: int = 20,
    offset: int = 0,
    session_id: Optional[str] = None
):
    """
    Search for claims using semantic search and filters.
    
    - **query**: Search query text (required)
    - **tags**: Filter by tags
    - **confidence_min**: Minimum confidence filter
    - **confidence_max**: Maximum confidence filter
    - **limit**: Maximum results to return (default: 20)
    - **offset**: Results offset for pagination (default: 0)
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        # Build filters
        filters = ClaimFilter(
            tags=tags,
            confidence_min=confidence_min,
            confidence_max=confidence_max,
            limit=limit,
            offset=offset
        )
        
        claims = await interface.search_claims(query, filters, session_id=session_id)
        
        return create_api_response(
            success=True,
            data=[claim.to_dict() for claim in claims],
            message=f"Found {len(claims)} claims for query: {query}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=APIResponse)
async def evaluate_claim_endpoint(request: ClaimEvaluateRequest, claim_id: str):
    """
    Evaluate a claim using LLM and context.
    
    - **claim_id**: ID of claim to evaluate (path parameter)
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        result = await interface.evaluate_claim(claim_id, session_id=request.session_id)
        
        return create_api_response(
            success=True,
            data=result.to_dict(),
            message=f"Claim {claim_id} evaluated successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context", response_model=APIResponse)
async def get_context_endpoint(request: ContextRequest):
    """
    Build context for specified claims.
    
    - **claim_ids**: List of claim IDs to build context for
    - **max_skills**: Maximum number of skills to include (default: 5)
    - **max_samples**: Maximum number of samples to include (default: 10)
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        context = await interface.get_context(
            claim_ids=request.claim_ids,
            max_skills=request.max_skills,
            max_samples=request.max_samples,
            session_id=request.session_id
        )
        
        return create_api_response(
            success=True,
            data=context.to_dict(),
            message="Context built successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Session Management Endpoints

@app.post("/sessions", response_model=APIResponse)
async def create_session_endpoint(request: SessionCreateRequest):
    """
    Create a new processing session.
    
    - **user_data**: Optional user-specific data
    """
    try:
        interface = get_processing_interface()
        
        session = await interface.create_session(user_data=request.user_data)
        
        return create_api_response(
            success=True,
            data=session.to_dict(),
            message=f"Session {session.session_id} created successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/resume", response_model=APIResponse)
async def resume_session_endpoint(session_id: str):
    """
    Resume an existing session.
    
    - **session_id**: ID of session to resume
    """
    try:
        interface = get_processing_interface()
        
        session = await interface.resume_session(session_id)
        
        return create_api_response(
            success=True,
            data=session.to_dict(),
            message=f"Session {session_id} resumed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Tool Management Endpoints

@app.get("/tools", response_model=APIResponse)
async def get_available_tools(session_id: Optional[str] = None):
    """
    Get list of available tools.
    
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        tools = await interface.get_available_tools(session_id=session_id)
        
        return create_api_response(
            success=True,
            data=tools,
            message=f"Found {len(tools)} available tools"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute", response_model=APIResponse)
async def execute_tool_endpoint(request: ToolExecuteRequest):
    """
    Execute a tool with specified parameters.
    
    - **tool_name**: Name of tool to execute
    - **parameters**: Tool parameters
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        result = await interface.execute_tool(
            tool_name=request.tool_name,
            parameters=request.parameters,
            session_id=request.session_id
        )
        
        return create_api_response(
            success=True,
            data=result.to_dict(),
            message=f"Tool {request.tool_name} executed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch Processing Endpoints

@app.post("/claims/batch", response_model=APIResponse)
async def batch_create_claims_endpoint(request: BatchClaimsRequest):
    """
    Create multiple claims in batch for better performance.
    
    - **claims_data**: List of claim creation data
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        claims = await interface.batch_create_claims(request.claims_data, session_id=request.session_id)
        
        return create_api_response(
            success=True,
            data=[claim.to_dict() for claim in claims],
            message=f"Created {len(claims)} claims in batch"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/claims/batch/evaluate", response_model=APIResponse)
async def batch_evaluate_claims_endpoint(request: BatchEvaluateRequest):
    """
    Evaluate multiple claims in batch.
    
    - **claim_ids**: List of claim IDs to evaluate
    - **session_id**: Optional session ID for context
    """
    try:
        interface = get_processing_interface()
        
        results = await interface.batch_evaluate_claims(request.claim_ids, session_id=request.session_id)
        
        return create_api_response(
            success=True,
            data=[result.to_dict() for result in results],
            message=f"Evaluation completed for {len(results)} claims"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Event Streaming Endpoints

@app.get("/events/stream")
async def stream_events(
    session_id: Optional[str] = None,
    event_types: Optional[str] = None,
    since: Optional[str] = None
):
    """
    Stream processing events in real-time using Server-Sent Events.
    
    - **session_id**: Optional session ID to filter events
    - **event_types**: Comma-separated event types to filter
    - **since**: Optional timestamp to filter events (ISO format)
    """
    try:
        interface = get_processing_interface()
        
        # Parse event types
        event_type_list = None
        if event_types:
            event_type_list = [EventType(t.strip()) for t in event_types.split(",")]
        
        # Parse since timestamp
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
        
        async def event_generator():
            """Generate Server-Sent Events"""
            try:
                async for event in interface.stream_events(
                    session_id=session_id,
                    event_types=event_type_list,
                    since=since_dt
                ):
                    yield f"data: {event.to_dict()}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {{'error': '{str(e)}'}}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/events/ws")
async def websocket_events_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Connect with optional query parameters:
    - session_id: Optional session ID to filter events
    - event_types: Comma-separated event types to filter
    """
    try:
        interface = get_processing_interface()
        
        # Get query parameters
        query_params = websocket.query_params
        session_id = query_params.get("session_id") if query_params else None
        event_types_str = query_params.get("event_types") if query_params else None
        
        # Parse event types
        event_type_list = None
        if event_types_str:
            event_type_list = [EventType(t.strip()) for t in event_types_str.split(",")]
        
        # Subscribe to events
        subscription_id = await interface.subscribe_to_events(
            callback=event_callback,
            session_id=session_id,
            event_types=event_type_list
        )
        
        # Connect WebSocket
        await websocket_manager.connect(websocket, subscription_id)
        
        # Keep connection alive
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        
        # Cleanup
        await interface.unsubscribe_from_events(subscription_id)
        websocket_manager.disconnect(websocket)
        
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket.client_state != "disconnected":
            await websocket.close(code=1000, reason=str(e))

# Health and Status Endpoints

@app.get("/health", response_model=APIResponse)
async def health_check():
    """
    Get health status of processing layer.
    """
    try:
        interface = get_processing_interface()
        
        health_status = await interface.get_health_status()
        
        return create_api_response(
            success=health_status.get("healthy", False),
            data=health_status,
            message="System healthy" if health_status.get("healthy") else "System unhealthy"
        )
        
    except Exception as e:
        return create_api_response(
            success=False,
            data={"error": str(e)},
            message="Health check failed"
        )

@app.get("/stats", response_model=APIResponse)
async def get_statistics(session_id: Optional[str] = None):
    """
    Get processing statistics.
    
    - **session_id**: Optional session ID for session-specific stats
    """
    try:
        interface = get_processing_interface()
        
        stats = await interface.get_interface_statistics(session_id=session_id)
        
        return create_api_response(
            success=True,
            data=stats,
            message="Statistics retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration Endpoints

@app.get("/config", response_model=APIResponse)
async def get_config_info():
    """
    Get current configuration information.
    """
    try:
        config = get_config()
        
        config_info = {
            "providers": config.get_providers(),
            "primary_provider": config.get_primary_provider(),
            "confidence_threshold": config.confidence_threshold,
            "max_context_size": config.max_context_size,
            "workspace": config.workspace,
            "user": config.user,
            "team": config.team,
            "is_workspace_config": config.is_workspace_config()
        }
        
        return create_api_response(
            success=True,
            data=config_info,
            message="Configuration retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Provider Management Endpoints

@app.get("/providers/status", response_model=APIResponse)
async def get_providers_status():
    """
    Get comprehensive status of all LLM providers including health, metrics, and routing information.
    """
    try:
        router = SimplifiedLLMManager()
        status = router.get_provider_status()
        
        return create_api_response(
            success=True,
            data=status,
            message="Provider status retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers/metrics", response_model=APIResponse)
async def get_providers_metrics():
    """
    Get detailed metrics for all LLM providers including performance statistics.
    """
    try:
        router = SimplifiedLLMManager()
        metrics = router.get_provider_metrics()
        
        return create_api_response(
            success=True,
            data=metrics,
            message="Provider metrics retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/manage", response_model=APIResponse)
async def manage_providers(request: ProviderManagementRequest):
    """
    Manage LLM providers (enable, disable, reset metrics, set routing strategy).
    
    - **action**: Action to perform (enable, disable, reset_metrics, set_strategy)
    - **provider_name**: Provider name for enable/disable actions
    - **routing_strategy**: Routing strategy for set_strategy action
    """
    try:
        router = SimplifiedLLMManager()
        
        if request.action == "enable":
            if not request.provider_name:
                raise HTTPException(status_code=400, detail="Provider name required for enable action")
            
            success = router.enable_provider(request.provider_name)
            message = f"Provider {request.provider_name} enabled" if success else f"Failed to enable provider {request.provider_name}"
            
        elif request.action == "disable":
            if not request.provider_name:
                raise HTTPException(status_code=400, detail="Provider name required for disable action")
            
            success = router.disable_provider(request.provider_name)
            message = f"Provider {request.provider_name} disabled" if success else f"Failed to disable provider {request.provider_name}"
            
        elif request.action == "reset_metrics":
            if request.provider_name:
                router.reset_provider_metrics(request.provider_name)
                message = f"Metrics reset for provider {request.provider_name}"
            else:
                router.reset_provider_metrics()
                message = "Metrics reset for all providers"
            
        elif request.action == "set_strategy":
            if not request.routing_strategy:
                raise HTTPException(status_code=400, detail="Routing strategy required for set_strategy action")
            
            try:
                strategy = RoutingStrategy(request.routing_strategy)
                router.set_routing_strategy(strategy)
                message = f"Routing strategy set to {request.routing_strategy}"
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid routing strategy: {request.routing_strategy}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        return create_api_response(
            success=True,
            data={"action": request.action, "provider": request.provider_name},
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/test", response_model=APIResponse)
async def test_provider(request: ProviderTestRequest):
    """
    Test a specific provider or all providers with a simple prompt.
    
    - **prompt**: Test prompt (default: simple greeting)
    - **provider_name**: Specific provider to test (optional, tests all if not specified)
    """
    try:
        router = SimplifiedLLMManager()
        
        # Get current status before test
        initial_status = router.get_provider_status()
        
        # Test provider(s)
        if request.provider_name:
            # Test specific provider
            result = await router.generate_response(
                prompt=request.prompt,
                preferred_provider=request.provider_name
            )
            
            test_results = {
                request.provider_name: {
                    "success": result.success,
                    "response_time": result.processing_time,
                    "tokens_used": result.tokens_used,
                    "model_used": result.model_used,
                    "errors": result.errors if result.errors else None,
                    "content_preview": result.content[:200] + "..." if result.content and len(result.content) > 200 else result.content
                }
            }
        else:
            # Test all available providers
            test_results = {}
            for provider_name in router.providers.keys():
                try:
                    result = await router.generate_response(
                        prompt=request.prompt,
                        preferred_provider=provider_name
                    )
                    
                    test_results[provider_name] = {
                        "success": result.success,
                        "response_time": result.processing_time,
                        "tokens_used": result.tokens_used,
                        "model_used": result.model_used,
                        "errors": result.errors if result.errors else None,
                        "content_preview": result.content[:200] + "..." if result.content and len(result.content) > 200 else result.content
                    }
                except Exception as e:
                    test_results[provider_name] = {
                        "success": False,
                        "error": str(e)
                    }
        
        # Get final status after test
        final_status = router.get_provider_status()
        
        return create_api_response(
            success=True,
            data={
                "test_prompt": request.prompt,
                "test_results": test_results,
                "initial_status": initial_status,
                "final_status": final_status,
                "timestamp": datetime.utcnow().isoformat()
            },
            message="Provider test completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers/health", response_model=APIResponse)
async def check_providers_health():
    """
    Perform comprehensive health check on all providers and return detailed results.
    """
    try:
        router = SimplifiedLLMManager()
        
        # Trigger health checks
        await router._perform_health_checks()
        
        # Get updated status
        status = router.get_provider_status()
        
        # Calculate overall health summary
        total_providers = status["total_providers"]
        healthy_providers = status["healthy_providers"]
        
        if healthy_providers == 0:
            overall_health = "unhealthy"
            health_score = 0.0
        elif healthy_providers == total_providers:
            overall_health = "healthy"
            health_score = 1.0
        else:
            overall_health = "degraded"
            health_score = healthy_providers / total_providers
        
        health_summary = {
            "overall_health": overall_health,
            "health_score": health_score,
            "healthy_providers": healthy_providers,
            "total_providers": total_providers,
            "health_percentage": round(health_score * 100, 1),
            "timestamp": datetime.utcnow(),
            "detailed_status": status
        }
        
        return create_api_response(
            success=True,
            data=health_summary,
            message=f"Health check completed: {overall_health} ({health_score:.1%})"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/", response_model=APIResponse)
async def root():
    """
    Root endpoint with API information.
    """
    return create_api_response(
        success=True,
        data={
            "name": "Conjecture EndPoint API",
            "version": "1.0.0",
            "description": "Lightweight FastAPI interface for Conjecture AI-Powered Evidence-Based Reasoning System",
            "docs": "/docs",
            "health": "/health",
            "events": {
                "sse": "/events/stream",
                "websocket": "/events/ws"
            },
            "endpoints": {
                "claims": "/claims",
                "search": "/claims/search",
                "evaluate": "/evaluate",
                "context": "/context",
                "tools": "/tools",
                "sessions": "/sessions",
                "stats": "/stats",
                "config": "/config",
                "providers": {
                    "status": "/providers/status",
                    "metrics": "/providers/metrics",
                    "manage": "/providers/manage",
                    "test": "/providers/test",
                    "health": "/providers/health"
                }
            }
        },
        message="Conjecture EndPoint API is running"
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    error_response = create_error_response(
        error=exc.detail,
        details={"status_code": exc.status_code}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    error_response = create_error_response(
        error="Internal server error",
        details={"type": type(exc).__name__}
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Conjecture EndPoint API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"Starting Conjecture EndPoint API on {args.host}:{args.port}")
    print(f"Documentation available at http://{args.host}:{args.port}/docs")
    
    # Run server
    uvicorn.run(
        "src.endpoint_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )