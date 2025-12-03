"""
Persistence Layer for Agent Harness
Reliable storage, retrieval, and management of system data
"""

import asyncio
import json
import pickle
import shutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid

from ...core.models import Claim, ClaimBatch
from .models import CacheEntry


logger = logging.getLogger(__name__)


class StorageBackend:
    """Abstract storage backend base class"""
    
    async def store(self, key: str, data: Any) -> bool:
        """Store data with key"""
        raise NotImplementedError
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key"""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete data by key"""
        raise NotImplementedError
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys or keys matching pattern"""
        raise NotImplementedError
    
    async def backup(self, backup_location: str) -> str:
        """Create backup and return backup ID"""
        raise NotImplementedError
    
    async def restore(self, backup_id: str) -> bool:
        """Restore from backup"""
        raise NotImplementedError


class FileSystemBackend(StorageBackend):
    """File system-based storage backend"""
    
    def __init__(self, storage_directory: str, compression: bool = True):
        self.storage_directory = Path(storage_directory)
        self.compression = compression
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_directory / "claims").mkdir(exist_ok=True)
        (self.storage_directory / "sessions").mkdir(exist_ok=True)
        (self.storage_directory / "workflows").mkdir(exist_ok=True)
        (self.storage_directory / "backups").mkdir(exist_ok=True)
        (self.storage_directory / "temp").mkdir(exist_ok=True)
    
    async def store(self, key: str, data: Any) -> bool:
        """Store data in file system"""
        try:
            file_path = self._get_file_path(key)
            
            # Determine storage format
            if isinstance(data, (dict, list)):
                content = json.dumps(data, default=str, indent=2)
                file_path = file_path.with_suffix('.json')
            else:
                content = pickle.dumps(data)
                file_path = file_path.with_suffix('.pkl')
            
            # Write to file
            file_path.write_bytes(content.encode() if isinstance(content, str) else content)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to store {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from file system"""
        try:
            # Try JSON first
            json_path = self._get_file_path(key).with_suffix('.json')
            if json_path.exists():
                content = json_path.read_text()
                return json.loads(content)
            
            # Try pickle
            pkl_path = self._get_file_path(key).with_suffix('.pkl')
            if pkl_path.exists():
                content = pkl_path.read_bytes()
                return pickle.loads(content)
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to retrieve {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete data from file system"""
        try:
            deleted = False
            
            # Try JSON
            json_path = self._get_file_path(key).with_suffix('.json')
            if json_path.exists():
                json_path.unlink()
                deleted = True
            
            # Try pickle
            pkl_path = self._get_file_path(key).with_suffix('.pkl')
            if pkl_path.exists():
                pkl_path.unlink()
                deleted = True
            
            return deleted
        
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False
    
    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List keys in storage"""
        try:
            keys = []
            
            for file_path in self.storage_directory.rglob("*.json"):
                key = file_path.relative_to(self.storage_directory).with_suffix('')
                if pattern is None or pattern in str(key):
                    keys.append(str(key))
            
            for file_path in self.storage_directory.rglob("*.pkl"):
                key = file_path.relative_to(self.storage_directory).with_suffix('')
                if pattern is None or pattern in str(key):
                    keys.append(str(key))
            
            return sorted(keys)
        
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []
    
    async def backup(self, backup_location: str) -> str:
        """Create backup of storage directory"""
        try:
            backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            backup_path = Path(backup_location) / backup_id
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all data
            for item in self.storage_directory.iterdir():
                if item.name != "backups":  # Don't backup backups
                    dest = backup_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            
            # Store backup metadata
            backup_metadata = {
                'backup_id': backup_id,
                'created_at': datetime.utcnow().isoformat(),
                'original_directory': str(self.storage_directory),
                'key_count': len(await self.list_keys())
            }
            
            metadata_file = backup_path / "metadata.json"
            metadata_file.write_text(json.dumps(backup_metadata, indent=2))
            
            logger.info(f"Created backup: {backup_id}")
            return backup_id
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    async def restore(self, backup_id: str) -> bool:
        """Restore from backup"""
        try:
            backup_path = self.storage_directory.parent / "backups" / backup_id
            
            if not backup_path.exists():
                logger.error(f"Backup {backup_id} not found")
                return False
            
            # Verify backup metadata
            metadata_file = backup_path / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"Backup metadata not found for {backup_id}")
                return False
            
            metadata = json.loads(metadata_file.read_text())
            
            # Backup current data
            current_backup_id = await self.backup(str(self.storage_directory.parent / "backups"))
            logger.info(f"Backed up current data to {current_backup_id}")
            
            # Remove current data
            for item in self.storage_directory.glob("*"):
                if item.name != "backups":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Restore from backup
            for item in backup_path.iterdir():
                if item.name != "metadata.json":
                    dest = self.storage_directory / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            
            logger.info(f"Restored from backup {backup_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_id}: {e}")
            return False
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for key"""
        # Hash long keys to avoid deep directory structures
        if len(key) > 100:
            import hashlib
            key_hash = hashlib.md5(key.encode()).hexdigest()
            return self.storage_directory / f"hashed_{key_hash}"
        
        return self.storage_directory / key


class PersistenceLayer:
    """
    Reliable persistence layer with backup, recovery, and consistency
    """

    def __init__(self, storage_backend: StorageBackend, 
                 backup_interval_hours: int = 24,
                 max_backups: int = 10):
        self.storage_backend = storage_backend
        self.backup_interval_hours = backup_interval_hours
        self.max_backups = max_backups
        
        # Performance tracking
        self.operation_counts = {
            'stores': 0,
            'retrieves': 0,
            'deletes': 0,
            'backups': 0
        }
        
        # Background tasks
        self._backup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize persistence layer"""
        # Start backup task
        self._backup_task = asyncio.create_task(self._backup_loop())
        logger.info("Persistence layer initialized")

    async def shutdown(self) -> None:
        """Shutdown persistence layer"""
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        logger.info("Persistence layer shutdown")

    async def store_claim(self, claim: Claim) -> str:
        """Store a claim"""
        try:
            # Increment operation count
            self.operation_counts['stores'] += 1
            
            # Store claim data
            key = f"claims/{claim.id}"
            success = await self.storage_backend.store(key, claim.dict())
            
            if success:
                # Update indexes
                await self._update_claim_indexes(claim)
                logger.debug(f"Stored claim: {claim.id}")
                return claim.id
            else:
                raise RuntimeError(f"Failed to store claim {claim.id}")
        
        except Exception as e:
            logger.error(f"Failed to store claim: {e}")
            raise

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID"""
        try:
            # Increment operation count
            self.operation_counts['retrieves'] += 1
            
            # Retrieve claim data
            key = f"claims/{claim_id}"
            claim_data = await self.storage_backend.retrieve(key)
            
            if claim_data:
                return Claim(**claim_data)
            else:
                return None
        
        except Exception as e:
            logger.error(f"Failed to retrieve claim {claim_id}: {e}")
            return None

    async def update_claim(self, claim_id: str, updates: Dict[str, Any]) -> bool:
        """Update a claim"""
        try:
            # Get existing claim
            claim = await self.get_claim(claim_id)
            if not claim:
                return False
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(claim, field):
                    setattr(claim, field, value)
            
            # Update timestamp
            claim.updated = datetime.utcnow()
            
            # Store updated claim
            return await self.storage_backend.store(f"claims/{claim_id}", claim.dict())
        
        except Exception as e:
            logger.error(f"Failed to update claim {claim_id}: {e}")
            return False

    async def delete_claim(self, claim_id: str) -> bool:
        """Delete a claim"""
        try:
            # Increment operation count
            self.operation_counts['deletes'] += 1
            
            # Retrieve claim for index cleanup
            claim = await self.get_claim(claim_id)
            if not claim:
                return False
            
            # Delete claim
            success = await self.storage_backend.delete(f"claims/{claim_id}")
            
            if success:
                # Clean up indexes
                await self._cleanup_claim_indexes(claim)
                logger.debug(f"Deleted claim: {claim_id}")
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Failed to delete claim {claim_id}: {e}")
            return False

    async def search_claims(self, query: str, filters: Optional[Dict[str, Any]] = None,
                          limit: int = 50) -> List[Claim]:
        """Search claims"""
        try:
            # For now, simple text-based search
            # In a full implementation, this would use proper indexing
            all_keys = await self.storage_backend.list_keys("claims/")
            
            claims = []
            query_lower = query.lower()
            
            for key in all_keys[:limit * 2]:  # Get more than needed for filtering
                claim_data = await self.storage_backend.retrieve(key)
                if claim_data:
                    claim = Claim(**claim_data)
                    
                    # Simple text search
                    content_match = query_lower in claim.content.lower()
                    
                    # Apply filters
                    matches = content_match
                    if filters:
                        if 'type' in filters:
                            claim_types = [t.value for t in claim.type]
                            matches = matches and any(t in claim_types for t in filters['type'])
                        
                        if 'tags' in filters:
                            matches = matches and any(tag in claim.tags for tag in filters['tags'])
                        
                        if 'confidence_min' in filters:
                            matches = matches and claim.confidence >= filters['confidence_min']
                    
                    if matches:
                        claims.append(claim)
                        
                        if len(claims) >= limit:
                            break
            
            return claims
        
        except Exception as e:
            logger.error(f"Failed to search claims: {e}")
            return []

    async def store_batch(self, batch: ClaimBatch) -> List[str]:
        """Store a batch of claims"""
        try:
            claim_ids = []
            
            for claim in batch.claims:
                claim_id = await self.store_claim(claim)
                claim_ids.append(claim_id)
            
            return claim_ids
        
        except Exception as e:
            logger.error(f"Failed to store claim batch: {e}")
            raise

    async def backup_data(self, backup_location: str) -> str:
        """Create backup of all data"""
        try:
            # Increment operation count
            self.operation_counts['backups'] += 1
            
            backup_id = await self.storage_backend.backup(backup_location)
            
            # Clean up old backups
            await self._cleanup_old_backups(backup_location)
            
            return backup_id
        
        except Exception as e:
            logger.error(f"Failed to backup data: {e}")
            raise

    async def restore_data(self, backup_id: str) -> bool:
        """Restore data from backup"""
        try:
            return await self.storage_backend.restore(backup_id)
        
        except Exception as e:
            logger.error(f"Failed to restore data from backup {backup_id}: {e}")
            return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Count items by type
            all_keys = await self.storage_backend.list_keys()
            
            key_counts = {
                'total': len(all_keys),
                'claims': len([k for k in all_keys if k.startswith('claims/')]),
                'sessions': len([k for k in all_keys if k.startswith('sessions/')]),
                'workflows': len([k for k in all_keys if k.startswith('workflows/')])
            }
            
            return {
                'key_counts': key_counts,
                'operation_counts': self.operation_counts,
                'backup_interval_hours': self.backup_interval_hours,
                'max_backups': self.max_backups
            }
        
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    async def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify data integrity"""
        try:
            all_keys = await self.storage_backend.list_keys()
            
            integrity_issues = []
            verified_count = 0
            
            for key in all_keys:
                try:
                    data = await self.storage_backend.retrieve(key)
                    if data is None:
                        integrity_issues.append(f"Could not retrieve: {key}")
                    else:
                        # Basic validation
                        if key.startswith('claims/'):
                            if 'id' not in data or 'content' not in data:
                                integrity_issues.append(f"Invalid claim structure: {key}")
                        elif key.startswith('sessions/'):
                            if 'user_id' not in data:
                                integrity_issues.append(f"Invalid session structure: {key}")
                        
                        verified_count += 1
                except Exception as e:
                    integrity_issues.append(f"Error verifying {key}: {e}")
            
            return {
                'total_keys': len(all_keys),
                'verified_count': verified_count,
                'integrity_issues': integrity_issues,
                'integrity_score': 1.0 - (len(integrity_issues) / len(all_keys)) if all_keys else 1.0
            }
        
        except Exception as e:
            logger.error(f"Failed to verify data integrity: {e}")
            return {'error': str(e)}

    async def _update_claim_indexes(self, claim: Claim) -> None:
        """Update claim indexes for efficient queries"""
        try:
            # Type index
            type_key = f"indexes/by_type/{claim.type[0].value}"
            type_data = await self.storage_backend.retrieve(type_key) or []
            if claim.id not in type_data:
                type_data.append(claim.id)
                await self.storage_backend.store(type_key, type_data)
            
            # Tags index
            for tag in claim.tags:
                tag_key = f"indexes/by_tag/{tag}"
                tag_data = await self.storage_backend.retrieve(tag_key) or []
                if claim.id not in tag_data:
                    tag_data.append(claim.id)
                    await self.storage_backend.store(tag_key, tag_data)
            
            # Timestamp index (for recency queries)
            date_key = f"indexes/by_date/{claim.updated.strftime('%Y-%m-%d')}"
            date_data = await self.storage_backend.retrieve(date_key) or []
            if claim.id not in date_data:
                date_data.append(claim.id)
                await self.storage_backend.store(date_key, date_data)
        
        except Exception as e:
            logger.error(f"Failed to update claim indexes: {e}")

    async def _cleanup_claim_indexes(self, claim: Claim) -> None:
        """Clean up claim indexes after deletion"""
        try:
            # Remove from type index
            type_key = f"indexes/by_type/{claim.type[0].value}"
            type_data = await self.storage_backend.retrieve(type_key) or []
            if claim.id in type_data:
                type_data.remove(claim.id)
                await self.storage_backend.store(type_key, type_data)
            
            # Remove from tags index
            for tag in claim.tags:
                tag_key = f"indexes/by_tag/{tag}"
                tag_data = await self.storage_backend.retrieve(tag_key) or []
                if claim.id in tag_data:
                    tag_data.remove(claim.id)
                    await self.storage_backend.store(tag_key, tag_data)
            
            # Remove from date index
            date_key = f"indexes/by_date/{claim.updated.strftime('%Y-%m-%d')}"
            date_data = await self.storage_backend.retrieve(date_key) or []
            if claim.id in date_data:
                date_data.remove(claim.id)
                await self.storage_backend.store(date_key, date_data)
        
        except Exception as e:
            logger.error(f"Failed to cleanup claim indexes: {e}")

    async def _cleanup_old_backups(self, backup_location: str) -> None:
        """Clean up old backups"""
        try:
            backup_path = Path(backup_location)
            if not backup_path.exists():
                return
            
            # Get all backup directories
            backup_dirs = [d for d in backup_path.iterdir() 
                          if d.is_dir() and d.name.startswith('backup_')]
            
            # Sort by name (which includes timestamp)
            backup_dirs.sort(reverse=True)
            
            # Remove excess backups
            if len(backup_dirs) > self.max_backups:
                for backup_dir in backup_dirs[self.max_backups:]:
                    shutil.rmtree(backup_dir)
                    logger.info(f"Removed old backup: {backup_dir.name}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

    async def _backup_loop(self) -> None:
        """Background backup loop"""
        while True:
            try:
                await asyncio.sleep(self.backup_interval_hours * 3600)
                
                # Create automatic backup
                backup_location = str(Path(self.storage_backend.storage_directory).parent / "backups")
                backup_id = await self.backup_data(backup_location)
                logger.info(f"Automatic backup created: {backup_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Automatic backup failed: {e}")