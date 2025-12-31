"""
ConjectureEndpoint - Public API Interface

This module contains the ConjectureEndpoint class which serves as the public
API interface for the Conjecture system. It handles external requests,
validates input, routes to appropriate processing components, and formats
responses.

The Endpoint layer is part of the 4-layer architecture:
Presentation → Endpoint → Process → Data
"""

from typing import Any, Dict, Optional
import logging

from ..processing.process_context_builder import ProcessContextBuilder
from ..processing.process_llm_processor import ProcessLLMProcessor
from ..data.data_manager import DataManager

logger = logging.getLogger(__name__)

class ConjectureEndpoint:
    """
    Main endpoint class for the Conjecture system.
    
    The ConjectureEndpoint serves as the public API interface, handling
    external requests and routing them to the appropriate processing
    components. It provides a clean separation between the presentation
    layer and the business logic.
    
    This class will eventually replace the functionality currently in
    src/conjecture.py and src/endpoint_app.py as part of the 4-layer
    architecture migration.
    
    Attributes:
        config: Configuration settings for the endpoint
        processor: Reference to the processing layer component
        data_layer: Reference to the data layer component
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ConjectureEndpoint.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.data_layer = None  # Will be connected to data layer
        self.context_builder = None  # Process Layer component
        self.llm_processor = None  # Process Layer component
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming request.
        
        This method handles the complete request lifecycle:
        1. Validate input
        2. Route to appropriate processor
        3. Format response
        
        Args:
            request: The incoming request dictionary
            
        Returns:
            Dictionary containing the response
            
        Raises:
            ValidationError: If request validation fails
            ProcessingError: If processing encounters an error
        """
        try:
            # Validate request
            operation = request.get("operation")
            if not operation:
                return {
                    "status": "error",
                    "message": "Missing 'operation' field in request"
                }
            
            # Route to appropriate processor
            if operation == "create_claim":
                return await self._handle_create_claim(request)
            elif operation == "analyze_claim":
                return await self._handle_analyze_claim(request)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "status": "error",
                "message": f"Processing failed: {str(e)}"
            }
    
    async def connect_layers(self, data_layer: DataManager) -> None:
        """
        Connect the endpoint to the processing and data layers.
        
        This method instantiates Process Layer components and connects
        them to the data layer following the 4-layer architecture.
        
        Args:
            data_layer: The data layer component (DataManager)
        """
        self.data_layer = data_layer
        
        # Initialize Process Layer components
        self.context_builder = ProcessContextBuilder(data_layer)
        await self.context_builder.initialize()
        
        self.llm_processor = ProcessLLMProcessor()
        await self.llm_processor.initialize()
        
        logger.info("Endpoint connected to Process and Data layers")
    
    async def _handle_create_claim(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle create_claim operation through Process Layer.
        
        Args:
            request: Request containing 'content', 'confidence', and optional 'tags'
            
        Returns:
            Response dictionary with created claim information
        """
        try:
            # Extract request data
            content = request.get("content")
            confidence = request.get("confidence")
            tags = request.get("tags", [])
            
            if not content or confidence is None:
                return {
                    "status": "error",
                    "message": "Missing required fields: 'content' and 'confidence'"
                }
            
            # Build context using ProcessContextBuilder
            context = await self.context_builder.build_context_for_claim_creation(
                content=content,
                confidence=confidence,
                tags=tags
            )
            
            # Process using ProcessLLMProcessor
            processing_result = await self.llm_processor.process_claim_creation(context)
            
            if processing_result.success and hasattr(processing_result, '_metadata'):
                # Extract claim from processing result
                claim_data = processing_result._metadata.get("claim")
                if claim_data:
                    # Save claim to data layer
                    created_claim = await self.data_layer.create_claim(
                        content=claim_data["content"],
                        confidence=claim_data["confidence"],
                        tags=claim_data["tags"],
                        claim_id=claim_data["id"]
                    )
                    
                    return {
                        "status": "success",
                        "message": "Claim created successfully",
                        "claim": {
                            "id": created_claim.id,
                            "content": created_claim.content,
                            "confidence": created_claim.confidence,
                            "tags": created_claim.tags,
                            "state": created_claim.state.value,
                            "created": created_claim.created.isoformat()
                        },
                        "processing_time": processing_result.execution_time,
                        "provider_used": processing_result._metadata.get("provider_used", "unknown")
                    }
            
            # If processing failed
            return {
                "status": "error",
                "message": "Failed to process claim creation",
                "errors": processing_result.errors,
                "processing_time": processing_result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Error in create_claim handler: {e}")
            return {
                "status": "error",
                "message": f"Create claim failed: {str(e)}"
            }
    
    async def _handle_analyze_claim(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle analyze_claim operation through Process Layer.
        
        Args:
            request: Request containing 'claim_id'
            
        Returns:
            Response dictionary with claim analysis
        """
        try:
            claim_id = request.get("claim_id")
            if not claim_id:
                return {
                    "status": "error",
                    "message": "Missing required field: 'claim_id'"
                }
            
            # Build context using ProcessContextBuilder
            context = await self.context_builder.build_context_for_claim_analysis(claim_id)
            
            # Process using ProcessLLMProcessor
            processing_result = await self.llm_processor.process_claim_analysis(context)
            
            if processing_result.success:
                return {
                    "status": "success",
                    "message": "Claim analyzed successfully",
                    "claim_id": claim_id,
                    "analysis": processing_result._metadata.get("analysis", {}),
                    "processing_time": processing_result.execution_time,
                    "provider_used": processing_result._metadata.get("provider_used", "unknown")
                }
            
            # If processing failed
            return {
                "status": "error",
                "message": "Failed to analyze claim",
                "errors": processing_result.errors,
                "processing_time": processing_result.execution_time
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_claim handler: {e}")
            return {
                "status": "error",
                "message": f"Analyze claim failed: {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the endpoint and connected layers.
        
        Returns:
            Dictionary containing health status information
        """
        endpoint_health = {
            "status": "healthy",
            "endpoint": "operational"
        }
        
        # Check Process Layer components
        if self.context_builder and self.llm_processor:
            llm_health = self.llm_processor.health_check()
            endpoint_health["process_layer"] = {
                "context_builder": "operational",
                "llm_processor": llm_health["status"]
            }
        else:
            endpoint_health["process_layer"] = {
                "status": "disconnected",
                "message": "Process Layer components not initialized"
            }
        
        # Check Data Layer
        if self.data_layer:
            endpoint_health["data_layer"] = "connected"
        else:
            endpoint_health["data_layer"] = "disconnected"
        
        return endpoint_health