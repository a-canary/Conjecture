#!/usr/bin/env python3
"""
Conjecture Data Layer CLI

A simple command-line interface for testing and interacting with the Conjecture data layer.
Provides commands for creating, searching, and managing claims and relationships.
"""

import asyncio
import argparse
import json
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_manager import DataManager, get_data_manager
from data.models import Claim, ClaimFilter, DataConfig


class ConjectureCLI:
    """Command-line interface for Conjecture data layer."""
    
    def __init__(self, config_path: str = None, use_mock: bool = True):
        """Initialize CLI with configuration."""
        self.config = self._load_config(config_path)
        self.use_mock = use_mock
        self.data_manager = None
    
    def _load_config(self, config_path: str) -> DataConfig:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return DataConfig(**config_data)
        return DataConfig()
    
    async def initialize(self):
        """Initialize the data manager."""
        self.data_manager = get_data_manager(self.config, self.use_mock)
        await self.data_manager.initialize()
        print(f"✅ Data layer initialized (mock={self.use_mock})")
        print(f"   SQLite: {self.config.sqlite_path}")
        print(f"   ChromaDB: {self.config.chroma_path}")
        print(f"   Embedding model: {self.config.embedding_model}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.data_manager:
            await self.data_manager.close()
            print("✅ Data layer closed")
    
    async def create_claim(self, content: str, user: str, confidence: float = 0.5, 
                          tags: List[str] = None, dirty: bool = True) -> Claim:
        """Create a new claim."""
        try:
            claim = await self.data_manager.create_claim(
                content=content,
                created_by=user,
                confidence=confidence,
                tags=tags or [],
                dirty=dirty
            )
            print(f"✅ Created claim: {claim.id}")
            print(f"   Content: {claim.content}")
            print(f"   Confidence: {claim.confidence}")
            print(f"   Tags: {', '.join(claim.tags) if claim.tags else 'None'}")
            print(f"   Dirty: {claim.dirty}")
            return claim
        except Exception as e:
            print(f"❌ Failed to create claim: {e}")
            raise
    
    async def get_claim(self, claim_id: str) -> Claim:
        """Retrieve a claim by ID."""
        try:
            claim = await self.data_manager.get_claim(claim_id)
            if claim:
                print(f"✅ Found claim: {claim.id}")
                print(f"   Content: {claim.content}")
                print(f"   Confidence: {claim.confidence}")
                print(f"   Tags: {', '.join(claim.tags) if claim.tags else 'None'}")
                print(f"   Dirty: {claim.dirty}")
                print(f"   Created: {claim.created_at}")
                print(f"   Creator: {claim.created_by}")
                return claim
            else:
                print(f"❌ Claim {claim_id} not found")
                return None
        except Exception as e:
            print(f"❌ Failed to get claim: {e}")
            raise
    
    async def search_claims(self, query: str, limit: int = 10) -> List[Claim]:
        """Search for similar claims."""
        try:
            claims = await self.data_manager.search_similar(query, limit=limit)
            print(f"✅ Found {len(claims)} similar claims for: '{query}'")
            
            for i, claim in enumerate(claims, 1):
                print(f"\n{i}. {claim.id}")
                print(f"   Content: {claim.content}")
                print(f"   Confidence: {claim.confidence}")
                print(f"   Tags: {', '.join(claim.tags) if claim.tags else 'None'}")
            
            return claims
        except Exception as e:
            print(f"❌ Failed to search claims: {e}")
            raise
    
    async def filter_claims(self, tags: List[str] = None, confidence_min: float = None,
                           confidence_max: float = None, dirty_only: bool = None,
                           created_by: str = None, limit: int = 20) -> List[Claim]:
        """Filter claims based on criteria."""
        try:
            filter_obj = ClaimFilter(
                tags=tags,
                confidence_min=confidence_min,
                confidence_max=confidence_max,
                dirty_only=dirty_only,
                created_by=created_by,
                limit=limit
            )
            
            claims = await self.data_manager.filter_claims(filter_obj)
            print(f"✅ Found {len(claims)} claims matching filters")
            
            for i, claim in enumerate(claims, 1):
                print(f"\n{i}. {claim.id}")
                print(f"   Content: {claim.content}")
                print(f"   Confidence: {claim.confidence}")
                print(f"   Tags: {', '.join(claim.tags) if claim.tags else 'None'}")
                print(f"   Dirty: {claim.dirty}")
                print(f"   Creator: {claim.created_by}")
            
            return claims
        except Exception as e:
            print(f"❌ Failed to filter claims: {e}")
            raise
    
    async def add_relationship(self, supporter_id: str, supported_id: str,
                              relationship_type: str = "supports", user: str = None) -> int:
        """Add relationship between claims."""
        try:
            rel_id = await self.data_manager.add_relationship(
                supporter_id=supporter_id,
                supported_id=supported_id,
                relationship_type=relationship_type,
                created_by=user
            )
            print(f"✅ Added relationship: {supporter_id} -> {supported_id} ({relationship_type})")
            print(f"   Relationship ID: {rel_id}")
            return rel_id
        except Exception as e:
            print(f"❌ Failed to add relationship: {e}")
            raise
    
    async def get_relationships(self, claim_id: str) -> List[Dict]:
        """Get relationships for a claim."""
        try:
            relationships = await self.data_manager.get_relationships(claim_id)
            print(f"✅ Found {len(relationships)} relationships for claim {claim_id}")
            
            for i, rel in enumerate(relationships, 1):
                direction = "supports" if rel['supporter_id'] == claim_id else "supported by"
                other_id = rel['supported_id'] if rel['supporter_id'] == claim_id else rel['supporter_id']
                print(f"\n{i}. {claim_id} {direction} {other_id}")
                print(f"   Type: {rel['relationship_type']}")
                print(f"   Created: {rel['created_at']}")
            
            return relationships
        except Exception as e:
            print(f"❌ Failed to get relationships: {e}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get data layer statistics."""
        try:
            stats = await self.data_manager.get_stats()
            print("✅ Data Layer Statistics")
            print(f"   Total claims: {stats['total_claims']}")
            print(f"   Dirty claims: {stats['dirty_claims']}")
            print(f"   Clean claims: {stats['clean_claims']}")
            print(f"   ChromaDB claims: {stats['chroma_stats']['total_claims']}")
            print(f"   Embedding model: {stats['embedding_model']['model_name']}")
            print(f"   Embedding dimension: {stats['embedding_model']['embedding_dimension']}")
            return stats
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            raise
    
    async def interactive_mode(self):
        """Run CLI in interactive mode."""
        print("\n🎯 Conjecture Data Layer - Interactive Mode")
        print("Type 'help' for available commands or 'quit' to exit.\n")
        
        while True:
            try:
                command = input("conjecture> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if command.lower() == 'help':
                    self._show_help()
                    continue
                
                # Parse and execute command
                await self._execute_command(command)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
📖 Available Commands:

📝 Create Claim:
   create "<content>" --user <username> [--confidence <0.0-1.0>] [--tags tag1,tag2] [--dirty true/false]

🔍 Get Claim:
   get <claim_id>

🔎 Search Claims:
   search "<query>" [--limit <number>]

🎯 Filter Claims:
   filter [--tags tag1,tag2] [--confidence-min <0.0-1.0>] [--confidence-max <0.0-1.0>] 
          [--dirty-only true/false] [--created-by <username>] [--limit <number>]

🔗 Add Relationship:
   relate <supporter_id> <supported_id> [--type supports|contradicts|extends|clarifies] [--user <username>]

📊 Get Relationships:
   relationships <claim_id>

📈 Statistics:
   stats

💾 Reset Database:
   reset (WARNING: deletes all data!)

Examples:
   create "Machine learning is a subset of AI" --user alice --confidence 0.8 --tags ml,ai
   get c0000001
   search "neural networks" --limit 5
   filter --tags ml --confidence-min 0.7
   relate c0000001 c0000002 --type supports --user alice
   relationships c0000001
   stats
        """
        print(help_text)
    
    async def _execute_command(self, command: str):
        """Execute a parsed command."""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        if cmd == 'create':
            await self._cmd_create(parts[1:])
        elif cmd == 'get':
            await self._cmd_get(parts[1:])
        elif cmd == 'search':
            await self._cmd_search(parts[1:])
        elif cmd == 'filter':
            await self._cmd_filter(parts[1:])
        elif cmd == 'relate':
            await self._cmd_relate(parts[1:])
        elif cmd == 'relationships':
            await self._cmd_relationships(parts[1:])
        elif cmd == 'stats':
            await self._cmd_stats()
        elif cmd == 'reset':
            await self._cmd_reset()
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type 'help' for available commands.")
    
    async def _cmd_create(self, args: List[str]):
        """Handle create command."""
        if not args:
            print("❌ Usage: create \"<content>\" --user <username> [options]")
            return
        
        # Parse arguments (simplified parsing)
        content = args[0].strip('"\'')
        user = None
        confidence = 0.5
        tags = []
        dirty = True
        
        i = 1
        while i < len(args):
            if args[i] == '--user' and i + 1 < len(args):
                user = args[i + 1]
                i += 2
            elif args[i] == '--confidence' and i + 1 < len(args):
                confidence = float(args[i + 1])
                i += 2
            elif args[i] == '--tags' and i + 1 < len(args):
                tags = args[i + 1].split(',')
                i += 2
            elif args[i] == '--dirty' and i + 1 < len(args):
                dirty = args[i + 1].lower() == 'true'
                i += 2
            else:
                i += 1
        
        if not user:
            print("❌ --user is required")
            return
        
        await self.create_claim(content, user, confidence, tags, dirty)
    
    async def _cmd_get(self, args: List[str]):
        """Handle get command."""
        if not args:
            print("❌ Usage: get <claim_id>")
            return
        
        await self.get_claim(args[0])
    
    async def _cmd_search(self, args: List[str]):
        """Handle search command."""
        if not args:
            print("❌ Usage: search \"<query>\" [--limit <number>]")
            return
        
        query = args[0].strip('"\'')
        limit = 10
        
        i = 1
        while i < len(args):
            if args[i] == '--limit' and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        await self.search_claims(query, limit)
    
    async def _cmd_filter(self, args: List[str]):
        """Handle filter command."""
        tags = None
        confidence_min = None
        confidence_max = None
        dirty_only = None
        created_by = None
        limit = 20
        
        i = 0
        while i < len(args):
            if args[i] == '--tags' and i + 1 < len(args):
                tags = args[i + 1].split(',')
                i += 2
            elif args[i] == '--confidence-min' and i + 1 < len(args):
                confidence_min = float(args[i + 1])
                i += 2
            elif args[i] == '--confidence-max' and i + 1 < len(args):
                confidence_max = float(args[i + 1])
                i += 2
            elif args[i] == '--dirty-only' and i + 1 < len(args):
                dirty_only = args[i + 1].lower() == 'true'
                i += 2
            elif args[i] == '--created-by' and i + 1 < len(args):
                created_by = args[i + 1]
                i += 2
            elif args[i] == '--limit' and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        await self.filter_claims(tags, confidence_min, confidence_max, dirty_only, created_by, limit)
    
    async def _cmd_relate(self, args: List[str]):
        """Handle relate command."""
        if len(args) < 2:
            print("❌ Usage: relate <supporter_id> <supported_id> [--type <type>] [--user <username>]")
            return
        
        supporter_id = args[0]
        supported_id = args[1]
        relationship_type = "supports"
        user = None
        
        i = 2
        while i < len(args):
            if args[i] == '--type' and i + 1 < len(args):
                relationship_type = args[i + 1]
                i += 2
            elif args[i] == '--user' and i + 1 < len(args):
                user = args[i + 1]
                i += 2
            else:
                i += 1
        
        await self.add_relationship(supporter_id, supported_id, relationship_type, user)
    
    async def _cmd_relationships(self, args: List[str]):
        """Handle relationships command."""
        if not args:
            print("❌ Usage: relationships <claim_id>")
            return
        
        await self.get_relationships(args[0])
    
    async def _cmd_stats(self):
        """Handle stats command."""
        await self.get_stats()
    
    async def _cmd_reset(self):
        """Handle reset command."""
        confirm = input("⚠️  This will delete ALL data. Are you sure? (yes/no): ")
        if confirm.lower() == 'yes':
            await self.data_manager.reset_database()
            print("✅ Database reset complete")
        else:
            print("❌ Reset cancelled")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Conjecture Data Layer CLI")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-mock", action="store_true", help="Use real embeddings instead of mock")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    # Command arguments
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new claim")
    create_parser.add_argument("content", help="Claim content")
    create_parser.add_argument("--user", required=True, help="Creator username")
    create_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence score")
    create_parser.add_argument("--tags", help="Comma-separated tags")
    create_parser.add_argument("--dirty", type=bool, default=True, help="Dirty flag")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get a claim by ID")
    get_parser.add_argument("claim_id", help="Claim ID")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search similar claims")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Result limit")
    
    # Stats command
    subparsers.add_parser("stats", help="Show statistics")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = ConjectureCLI(args.config, use_mock=not args.no_mock)
    
    try:
        await cli.initialize()
        
        if args.interactive or not args.command:
            await cli.interactive_mode()
        else:
            # Execute single command
            if args.command == "create":
                tags = args.tags.split(',') if args.tags else []
                await cli.create_claim(args.content, args.user, args.confidence, tags, args.dirty)
            elif args.command == "get":
                await cli.get_claim(args.claim_id)
            elif args.command == "search":
                await cli.search_claims(args.query, args.limit)
            elif args.command == "stats":
                await cli.get_stats()
            else:
                print(f"❌ Unknown command: {args.command}")
                parser.print_help()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        await cli.cleanup()


if __name__ == "__main__":
    asyncio.run(main())