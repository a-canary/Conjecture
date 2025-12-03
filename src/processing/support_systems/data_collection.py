"""
Data Collection System for Agent Harness
Gathers, validates, and preprocesses data from various sources
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
import logging
import re
import yaml

from .models import (
    DataSource, DataItem, DataSchema, ValidationResult, ProcessedData, CacheEntry
)
from utils.id_generator import generate_context_id


logger = logging.getLogger(__name__)


class DataCollector:
    """
    Comprehensive data collection from multiple sources with validation and caching
    """

    def __init__(self, cache_ttl_seconds: int = 300, max_cache_size: int = 1000):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        
        # Storage
        self.data_cache: Dict[str, CacheEntry] = {}
        
        # Data collectors by source type
        self.source_collectors: Dict[str, Callable] = {}
        
        # Data validators
        self.data_validators: Dict[str, Callable] = {}
        
        # Preprocessors
        self.preprocessors: List[Callable] = []
        
        # Statistics
        self.collection_stats = {
            'total_collections': 0,
            'source_usage': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_failures': 0
        }
        
        # Initialize built-in collectors
        self._initialize_builtin_collectors()

    def register_source_collector(self, source: DataSource, collector: Callable) -> None:
        """
        Register a collector for a specific data source
        
        Args:
            source: Data source type
            collector: Collector function
        """
        self.source_collectors[source.value] = collector
        logger.info(f"Registered collector for source: {source}")

    def register_data_validator(self, schema_name: str, validator: Callable) -> None:
        """
        Register a data validator
        
        Args:
            schema_name: Schema name
            validator: Validator function
        """
        self.data_validators[schema_name] = validator
        logger.info(f"Registered validator for schema: {schema_name}")

    def register_preprocessor(self, preprocessor: Callable) -> None:
        """
        Register a data preprocessor
        
        Args:
            preprocessor: Preprocessor function
        """
        self.preprocessors.append(preprocessor)
        logger.info(f"Registered preprocessor: {preprocessor.__name__}")

    async def collect_from_source(self, source: DataSource, query: str,
                                parameters: Optional[Dict[str, Any]] = None,
                                cache_key: Optional[str] = None) -> List[DataItem]:
        """
        Collect data from a specific source
        
        Args:
            source: Data source type
            query: Query string
            parameters: Optional parameters
            cache_key: Optional cache key
            
        Returns:
            List of collected data items
        """
        start_time = time.time()
        parameters = parameters or {}
        
        try:
            # Generate cache key if not provided
            if cache_key is None:
                cache_key = self._generate_cache_key(source, query, parameters)
            
            # Check cache first
            cached_data = await self.get_cached_data(cache_key)
            if cached_data is not None:
                self.collection_stats['cache_hits'] += 1
                return cached_data
            
            self.collection_stats['cache_misses'] += 1
            
            # Get appropriate collector
            collector = self.source_collectors.get(source.value)
            if not collector:
                logger.warning(f"No collector registered for source: {source}")
                return []
            
            # Collect data
            if asyncio.iscoroutinefunction(collector):
                raw_data = await collector(query, parameters)
            else:
                raw_data = collector(query, parameters)
            
            if not raw_data:
                return []
            
            # Convert to DataItem objects
            data_items = await self._convert_to_data_items(raw_data, source, query)
            
            # Store in cache
            await self.cache_data(cache_key, data_items)
            
            # Update statistics
            self.collection_stats['total_collections'] += 1
            self.collection_stats['source_usage'][source.value] = (
                self.collection_stats['source_usage'].get(source.value, 0) + 1
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Collected {len(data_items)} items from {source} in {processing_time}ms")
            
            return data_items

        except Exception as e:
            logger.error(f"Failed to collect from {source}: {e}")
            return []

    async def validate_data(self, data: Any, schema: DataSchema) -> ValidationResult:
        """
        Validate data against a schema
        
        Args:
            data: Data to validate
            schema: Data schema
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        transformed_data = None
        
        try:
            if isinstance(data, dict):
                # Check required fields
                for field in schema.required_fields:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")
                
                # Check field types
                for field, expected_type in schema.field_types.items():
                    if field in data:
                        actual_value = data[field]
                        if not self._validate_field_type(actual_value, expected_type):
                            errors.append(f"Field {field} has wrong type. Expected: {expected_type}, Got: {type(actual_value).__name__}")
                
                # Run custom validators
                for field, validator_name in schema.validators.items():
                    if field in data:
                        validator = self.data_validators.get(validator_name)
                        if validator:
                            try:
                                if asyncio.iscoroutinefunction(validator):
                                    validation_result = await validator(data[field])
                                else:
                                    validation_result = validator(data[field])
                                
                                if isinstance(validation_result, ValidationResult):
                                    errors.extend(validation_result.errors)
                                    warnings.extend(validation_result.warnings)
                                elif not validation_result:
                                    errors.append(f"Validation failed for field: {field}")
                                    
                            except Exception as e:
                                errors.append(f"Validator error for field {field}: {e}")
                
                # Apply custom validation rule
                if schema.custom_validation:
                    validator = self.data_validators.get(schema.custom_validation)
                    if validator:
                        try:
                            if asyncio.iscoroutinefunction(validator):
                                custom_result = await validator(data)
                            else:
                                custom_result = validator(data)
                            
                            if isinstance(custom_result, ValidationResult):
                                errors.extend(custom_result.errors)
                                warnings.extend(custom_result.warnings)
                                if custom_result.transformed_data:
                                    transformed_data = custom_result.transformed_data
                                
                        except Exception as e:
                            errors.append(f"Custom validation error: {e}")
                
            else:
                errors.append("Data must be a dictionary")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                transformed_data = transformed_data or data
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                transformed_data=transformed_data
            )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {e}"],
                warnings=warnings,
                transformed_data=None
            )

    async def preprocess_data(self, data: List[DataItem]) -> List[ProcessedData]:
        """
        Apply preprocessing to collected data
        
        Args:
            data: List of data items to preprocess
            
        Returns:
            List of processed data items
        """
        processed_items = []
        
        for item in data:
            try:
                # Apply each preprocessor
                processed_content = item.content
                processing_operations = []
                
                for preprocessor in self.preprocessors:
                    try:
                        if asyncio.iscoroutinefunction(preprocessor):
                            result = await preprocessor(processed_content, item.metadata)
                        else:
                            result = preprocessor(processed_content, item.metadata)
                        
                        if result is not None:
                            processed_content = result
                            processing_operations.append(preprocessor.__name__)
                    
                    except Exception as e:
                        logger.warning(f"Preprocessor {preprocessor.__name__} failed: {e}")
                        continue
                
                # Validate processed data with default schema
                if isinstance(processed_content, dict):
                    default_schema = DataSchema()
                    validation_result = await self.validate_data(processed_content, default_schema)
                else:
                    validation_result = ValidationResult(is_valid=True, errors=[], warnings=[])
                
                processed_item = ProcessedData(
                    original_item=item,
                    validated_data=processed_content,
                    processing_operations=processing_operations,
                    validation_result=validation_result
                )
                
                processed_items.append(processed_item)
                
            except Exception as e:
                logger.error(f"Failed to process data item {item.id}: {e}")
                # Add original item as processed with error
                processed_items.append(ProcessedData(
                    original_item=item,
                    validated_data=item.content,
                    processing_operations=["error"],
                    validation_result=ValidationResult(
                        is_valid=False,
                        errors=[f"Processing failed: {e}"]
                    )
                ))
        
        return processed_items

    async def cache_data(self, key: str, data: List[DataItem], ttl: Optional[int] = None) -> bool:
        """
        Cache data items
        
        Args:
            key: Cache key
            data: Data items to cache
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            if ttl is None:
                ttl = self.cache_ttl_seconds
            
            # Serialize data
            serialized_data = [item.dict() for item in data]
            data_bytes = json.dumps(serialized_data).encode('utf-8')
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=key,
                data=serialized_data,
                ttl_seconds=ttl,
                size_bytes=len(data_bytes)
            )
            
            # Store in cache
            self.data_cache[key] = cache_entry
            
            # Clean up old entries if needed
            await self._cleanup_cache()
            
            return True

        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            return False

    async def get_cached_data(self, key: str) -> Optional[List[DataItem]]:
        """
        Get cached data
        
        Args:
            key: Cache key
            
        Returns:
            Cached data items or None
        """
        try:
            cache_entry = self.data_cache.get(key)
            if not cache_entry:
                return None
            
            # Check if expired
            if cache_entry.is_expired():
                del self.data_cache[key]
                return None
            
            # Increment hit count
            cache_entry.increment_hit()
            
            # Deserialize data
            data_items = []
            for item_data in cache_entry.data:
                if isinstance(item_data, dict):
                    # Convert timestamp back to datetime if it's a string
                    if 'timestamp' in item_data and isinstance(item_data['timestamp'], str):
                        item_data['timestamp'] = datetime.fromisoformat(item_data['timestamp'])
                    data_items.append(DataItem(**item_data))
            
            return data_items

        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return None

    async def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Optional pattern to match keys
            
        Returns:
            Number of cleared entries
        """
        try:
            cleared_count = 0
            
            if pattern:
                keys_to_remove = [key for key in self.data_cache.keys() if pattern in key]
                for key in keys_to_remove:
                    del self.data_cache[key]
                    cleared_count += 1
            else:
                cleared_count = len(self.data_cache)
                self.data_cache.clear()
            
            logger.info(f"Cleared {cleared_count} cache entries")
            return cleared_count

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Returns:
            Collection statistics
        """
        cache_stats = {
            'total_entries': len(self.data_cache),
            'cache_hits': self.collection_stats['cache_hits'],
            'cache_misses': self.collection_stats['cache_misses'],
            'hit_rate': (
                self.collection_stats['cache_hits'] / 
                (self.collection_stats['cache_hits'] + self.collection_stats['cache_misses'])
                if (self.collection_stats['cache_hits'] + self.collection_stats['cache_misses']) > 0
                else 0.0
            )
        }
        
        return {
            **self.collection_stats,
            'cache': cache_stats,
            'registered_collectors': list(self.source_collectors.keys()),
            'registered_validators': list(self.data_validators.keys()),
            'preprocessors_count': len(self.preprocessors)
        }

    def _generate_cache_key(self, source: DataSource, query: str, 
                          parameters: Dict[str, Any]) -> str:
        """Generate cache key for data"""
        key_data = f"{source.value}:{query}:{json.dumps(sorted(parameters.items()), sort_keys=True)}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict,
            'any': type(None)  # Any type is accepted
        }
        
        expected_python_type = type_mapping.get(expected_type, expected_type)
        if expected_python_type == type(None):  # any type
            return True
        
        return isinstance(value, expected_python_type)

    async def _convert_to_data_items(self, raw_data: Any, source: DataSource, 
                                   query: str) -> List[DataItem]:
        """Convert raw data to DataItem objects"""
        data_items = []
        
        try:
            if isinstance(raw_data, list):
                # Handle list of items
                for i, item in enumerate(raw_data):
                    data_item = DataItem(
                        id=generate_context_id(),
                        source=source,
                        content=item,
                        metadata={
                            'query': query,
                            'item_index': i,
                            'collection_time': datetime.utcnow().isoformat()
                        }
                    )
                    data_items.append(data_item)
            
            elif isinstance(raw_data, dict):
                # Handle single item
                data_item = DataItem(
                    id=generate_context_id(),
                    source=source,
                    content=raw_data,
                    metadata={
                        'query': query,
                        'collection_time': datetime.utcnow().isoformat()
                    }
                )
                data_items.append(data_item)
            
            else:
                # Handle primitive types
                data_item = DataItem(
                    id=generate_context_id(),
                    source=source,
                    content=raw_data,
                    metadata={
                        'query': query,
                        'data_type': type(raw_data).__name__,
                        'collection_time': datetime.utcnow().isoformat()
                    }
                )
                data_items.append(data_item)
            
            return data_items

        except Exception as e:
            logger.error(f"Failed to convert raw data to DataItems: {e}")
            # Return empty list on failure
            return []

    async def _cleanup_cache(self) -> None:
        """Clean up expired and oversized cache entries"""
        try:
            # Remove expired entries
            expired_keys = [
                key for key, entry in self.data_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.data_cache[key]
            
            # Remove oldest entries if cache is too large
            if len(self.data_cache) > self.max_cache_size:
                # Sort by timestamp (oldest first)
                sorted_entries = sorted(
                    self.data_cache.items(),
                    key=lambda x: x[1].timestamp
                )
                
                # Remove oldest entries
                excess_count = len(self.data_cache) - self.max_cache_size
                for key, _ in sorted_entries[:excess_count]:
                    del self.data_cache[key]
            
            logger.debug(f"Cache cleanup completed. Current size: {len(self.data_cache)}")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def _initialize_builtin_collectors(self) -> None:
        """Initialize built-in data collectors"""
        
        async def user_input_collector(query: str, parameters: Dict[str, Any]) -> Any:
            """Collect user input data"""
            return {
                'user_request': query,
                'user_parameters': parameters,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        async def existing_claims_collector(query: str, parameters: Dict[str, Any]) -> Any:
            """Collect existing claims (placeholder)"""
            # This would integrate with the actual data manager
            return []  # Return empty for now
        
        async def tool_result_collector(query: str, parameters: Dict[str, Any]) -> Any:
            """Collect tool execution results"""
            tool_name = parameters.get('tool_name')
            tool_result = parameters.get('tool_result', {})
            
            return {
                'tool_name': tool_name,
                'result': tool_result,
                'query': query,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Register built-in collectors
        self.register_source_collector(DataSource.USER_INPUT, user_input_collector)
        self.register_source_collector(DataSource.EXISTING_CLAIMS, existing_claims_collector)
        self.register_source_collector(DataSource.TOOL_RESULT, tool_result_collector)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on data collector
        
        Returns:
            Health check results
        """
        try:
            # Test basic collection
            test_data = await self.collect_from_source(
                DataSource.USER_INPUT, 
                "health_check_test",
                {"test": True}
            )
            
            # Test cache
            cache_key = "health_check"
            test_cache_data = [DataItem(id="test", source=DataSource.USER_INPUT, content="test")]
            cache_works = await self.cache_data(cache_key, test_cache_data)
            retrieved_data = await self.get_cached_data(cache_key)
            
            return {
                'healthy': len(test_data) > 0 and cache_works and len(retrieved_data) > 0,
                'cache_size': len(self.data_cache),
                'active_collectors': len(self.source_collectors),
                'active_validators': len(self.data_validators),
                'preprocessors': len(self.preprocessors),
                'test_collection_works': len(test_data) > 0,
                'test_cache_works': cache_works and len(retrieved_data) > 0
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }