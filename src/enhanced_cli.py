#!/usr/bin/env python3
"""
Enhanced Conjecture CLI with LLM Integration
Comprehensive command-line interface with LLM-powered features and rubric evaluation
"""

import asyncio
import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_manager import DataManager, get_data_manager
from data.models import Claim, ClaimFilter, DataConfig
from processing.llm.llm_manager import LLMManager
from config.simple_config import Config


class EnhancedConjectureCLI:
    """Enhanced CLI with LLM integration and comprehensive features."""

    def __init__(self, config_path: str = None, use_mock: bool = True):
        """Initialize CLI with configuration."""
        self.config = Config()  # Use the new config system
        self.use_mock = use_mock
        self.data_manager = None
        self.llm_manager = None

    async def initialize(self):
        """Initialize all components."""
        print("🚀 Initializing Enhanced Conjecture CLI...")
        
        # Initialize data manager
        self.data_manager = get_data_manager(
            DataConfig(
                sqlite_path=self.config.db_path,
                chroma_path=self.config.chroma_path,
                embedding_model=self.config.embedding_model
            ), 
            self.use_mock
        )
        await self.data_manager.initialize()
        print(f"✅ Data layer initialized (mock={self.use_mock})")
        
        # Initialize LLM manager
        try:
            self.llm_manager = LLMManager(self.config)
            providers = self.llm_manager.get_available_providers()
            print(f"✅ LLM Manager initialized with providers: {providers}")
            
            # Health check
            health = self.llm_manager.health_check()
            print(f"   Health status: {health['overall_status']}")
            
        except Exception as e:
            print(f"⚠️  LLM Manager initialization failed: {e}")
            self.llm_manager = None

    async def cleanup(self):
        """Clean up resources."""
        if self.data_manager:
            await self.data_manager.close()
            print("✅ Data layer closed")

    # === Core Claim Operations ===

    async def create_claim(self, content: str, user: str, confidence: float = 0.5,
                          tags: List[str] = None, dirty: bool = True, 
                          analyze_with_llm: bool = False) -> Claim:
        """Create a new claim with optional LLM analysis."""
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
            
            # Optional LLM analysis
            if analyze_with_llm and self.llm_manager:
                print("🤖 Analyzing with LLM...")
                await self._analyze_claim_with_llm(claim)
            
            return claim
            
        except Exception as e:
            print(f"❌ Failed to create claim: {e}")
            raise

    async def get_claim(self, claim_id: str) -> Claim:
        """Retrieve a claim by ID with enhanced display."""
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

    async def search_claims(self, query: str, limit: int = 10, 
                           include_llm_insights: bool = False) -> List[Claim]:
        """Search for similar claims with optional LLM insights."""
        try:
            claims = await self.data_manager.search_similar(query, limit=limit)
            print(f"✅ Found {len(claims)} similar claims for: '{query}'")

            for i, claim in enumerate(claims, 1):
                print(f"\n{i}. {claim.id}")
                print(f"   Content: {claim.content}")
                print(f"   Confidence: {claim.confidence}")
                print(f"   Tags: {', '.join(claim.tags) if claim.tags else 'None'}")

            # Optional LLM insights
            if include_llm_insights and self.llm_manager and claims:
                print("\n🤖 LLM Insights:")
                await self._generate_search_insights(query, claims)

            return claims
            
        except Exception as e:
            print(f"❌ Failed to search claims: {e}")
            raise

    # === LLM-Powered Features ===

    async def analyze_claims(self, claim_ids: List[str], task: str = "analyze") -> Dict[str, Any]:
        """Analyze claims using LLM."""
        if not self.llm_manager:
            print("❌ LLM Manager not available")
            return {}

        try:
            # Get claims
            claims = []
            for claim_id in claim_ids:
                claim = await self.data_manager.get_claim(claim_id)
                if claim:
                    claims.append(claim)

            if not claims:
                print("❌ No valid claims found")
                return {}

            print(f"🤖 Analyzing {len(claims)} claims with LLM...")
            
            # Convert to BasicClaim format for LLM processing
            from core.basic_models import BasicClaim, ClaimState, ClaimType
            basic_claims = []
            for claim in claims:
                basic_claim = BasicClaim(
                    id=claim.id,
                    content=claim.content,
                    confidence=claim.confidence,
                    type=[ClaimType.CONCEPT],  # Default type
                    state=ClaimState.EXPLORE,
                    created_by=claim.created_by,
                    created_at=claim.created_at
                )
                basic_claims.append(basic_claim)

            # Process with LLM
            result = self.llm_manager.process_claims(basic_claims, task)
            
            if result.success:
                print(f"✅ LLM analysis completed")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Tokens used: {result.tokens_used}")
                print(f"   Generated {len(result.processed_claims)} insights:")
                
                for i, insight in enumerate(result.processed_claims, 1):
                    print(f"\n{i}. {insight.content}")
                    print(f"   Confidence: {insight.confidence}")
                
                return {
                    "success": True,
                    "insights": result.processed_claims,
                    "stats": {
                        "processing_time": result.processing_time,
                        "tokens_used": result.tokens_used,
                        "model": result.model_used
                    }
                }
            else:
                print(f"❌ LLM analysis failed: {result.errors}")
                return {"success": False, "errors": result.errors}
                
        except Exception as e:
            print(f"❌ Failed to analyze claims: {e}")
            return {"success": False, "errors": [str(e)]}

    async def generate_response(self, prompt: str, context_claims: List[str] = None) -> str:
        """Generate response using LLM with optional claim context."""
        if not self.llm_manager:
            print("❌ LLM Manager not available")
            return ""

        try:
            # Build context if provided
            if context_claims:
                context_parts = ["Context from claims:"]
                for claim_id in context_claims:
                    claim = await self.data_manager.get_claim(claim_id)
                    if claim:
                        context_parts.append(f"- {claim.content}")
                full_prompt = f"{' '.join(context_parts)}\n\nQuestion: {prompt}"
            else:
                full_prompt = prompt

            print(f"🤖 Generating response...")
            
            result = self.llm_manager.generate_response(full_prompt)
            
            if result.success:
                response = result.processed_claims[0].content
                print(f"✅ Response generated")
                print(f"   Processing time: {result.processing_time:.2f}s")
                print(f"   Tokens used: {result.tokens_used}")
                print(f"\n📝 Response:\n{response}")
                return response
            else:
                print(f"❌ Generation failed: {result.errors}")
                return ""
                
        except Exception as e:
            print(f"❌ Failed to generate response: {e}")
            return ""

    async def explore_topic(self, topic: str, max_claims: int = 10, 
                           generate_insights: bool = True) -> Dict[str, Any]:
        """Explore a topic with comprehensive analysis."""
        print(f"🔍 Exploring topic: '{topic}'")
        
        # Search for existing claims
        claims = await self.search_claims(topic, max_claims, include_llm_insights=False)
        
        result = {
            "topic": topic,
            "existing_claims": claims,
            "claim_count": len(claims),
            "insights": None,
            "recommendations": []
        }

        # Generate LLM insights if requested and available
        if generate_insights and self.llm_manager and claims:
            print("\n🤖 Generating topic insights...")
            insight_result = await self.analyze_claims([c.id for c in claims], f"analyze the topic of {topic}")
            
            if insight_result.get("success"):
                result["insights"] = insight_result["insights"]
                
                # Generate recommendations
                recommendations_prompt = f"""
                Based on the existing claims about {topic} and the analysis provided, 
                suggest 3-5 new claims that would enhance understanding of this topic.
                Format each as a clear, concise statement.
                """
                recommendations = await self.generate_response(recommendations_prompt)
                result["recommendations"] = recommendations.split('\n') if recommendations else []

        return result

    # === Statistics and Health ===

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Data layer stats
            data_stats = await self.data_manager.get_stats()
            
            print("📊 Comprehensive System Statistics")
            print(f"   Data Layer:")
            print(f"     Total claims: {data_stats['total_claims']}")
            print(f"     Dirty claims: {data_stats['dirty_claims']}")
            print(f"     Clean claims: {data_stats['clean_claims']}")
            print(f"     ChromaDB claims: {data_stats['chroma_stats']['total_claims']}")
            
            # LLM stats
            llm_stats = {}
            if self.llm_manager:
                llm_stats = self.llm_manager.get_combined_stats()
                print(f"   LLM Layer:")
                print(f"     Available providers: {llm_stats['total_providers']}")
                print(f"     Primary provider: {llm_stats['primary_provider']}")
                print(f"     Total requests: {llm_stats['total_requests']}")
                print(f"     Success rate: {llm_stats['overall_success_rate']:.2%}")
                print(f"     Total tokens: {llm_stats['total_tokens']}")
                
                # Provider-specific stats
                for provider, stats in llm_stats['providers'].items():
                    print(f"     {provider}: {stats['total_requests']} requests, {stats['success_rate']:.2%} success")
            else:
                print(f"   LLM Layer: Not available")

            return {
                "data_stats": data_stats,
                "llm_stats": llm_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            raise

    # === Helper Methods ===

    async def _analyze_claim_with_llm(self, claim: Claim):
        """Analyze a single claim with LLM."""
        try:
            from core.basic_models import BasicClaim, ClaimState, ClaimType
            
            basic_claim = BasicClaim(
                id=claim.id,
                content=claim.content,
                confidence=claim.confidence,
                type=[ClaimType.CONCEPT],
                state=ClaimState.EXPLORE,
                created_by=claim.created_by,
                created_at=claim.created_at
            )
            
            result = self.llm_manager.process_claims([basic_claim], "analyze this claim")
            
            if result.success:
                print(f"   🤖 LLM Analysis: {result.processed_claims[0].content}")
            else:
                print(f"   ⚠️  LLM analysis failed: {result.errors}")
                
        except Exception as e:
            print(f"   ⚠️  LLM analysis error: {e}")

    async def _generate_search_insights(self, query: str, claims: List[Claim]):
        """Generate insights for search results."""
        try:
            claim_summaries = [f"- {c.content} (confidence: {c.confidence})" for c in claims[:5]]
            context = f"Search query: '{query}'\nFound claims:\n" + "\n".join(claim_summaries)
            
            insights_prompt = f"""
            Analyze these search results and provide key insights:
            {context}
            
            What patterns do you notice? What are the key themes?
            """
            
            insights = await self.generate_response(insights_prompt)
            if insights:
                print(f"   {insights}")
                
        except Exception as e:
            print(f"   ⚠️  Could not generate insights: {e}")

    # === Interactive Mode ===

    async def interactive_mode(self):
        """Run enhanced CLI in interactive mode."""
        print("\n🎯 Enhanced Conjecture CLI - Interactive Mode")
        print("Features: LLM integration, comprehensive analysis, intelligent search")
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
                    self._show_enhanced_help()
                    continue

                # Parse and execute command
                await self._execute_enhanced_command(command)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_enhanced_help(self):
        """Show enhanced help information."""
        help_text = """
📖 Enhanced Conjecture CLI Commands:

📝 Create Claim:
   create "<content>" --user <username> [--confidence <0.0-1.0>] [--tags tag1,tag2] [--analyze]

🔍 Get Claim:
   get <claim_id>

🔎 Search Claims:
   search "<query>" [--limit <number>] [--insights]

🤖 Analyze Claims:
   analyze <claim_id1,claim_id2,...> [--task <task>]

💬 Generate Response:
   ask "<question>" [--context <claim_id1,claim_id2,...>]

🔍 Explore Topic:
   explore "<topic>" [--max-claims <number>] [--insights]

📊 Comprehensive Stats:
   stats

🎯 Interactive Mode:
   (no arguments) - Run in interactive mode

Examples:
   create "Machine learning enables pattern recognition" --user alice --confidence 0.8 --analyze
   search "neural networks" --limit 5 --insights
   analyze c0000001,c0000002 --task "find relationships"
   ask "What are the limitations of deep learning?" --context c0000001,c0000003
   explore "artificial intelligence" --max-claims 10 --insights
        """
        print(help_text)

    async def _execute_enhanced_command(self, command: str):
        """Execute enhanced command."""
        parts = command.split()
        if not parts:
            return

        cmd = parts[0].lower()

        if cmd == 'create':
            await self._cmd_create_enhanced(parts[1:])
        elif cmd == 'get':
            await self._cmd_get(parts[1:])
        elif cmd == 'search':
            await self._cmd_search_enhanced(parts[1:])
        elif cmd == 'analyze':
            await self._cmd_analyze(parts[1:])
        elif cmd == 'ask':
            await self._cmd_ask(parts[1:])
        elif cmd == 'explore':
            await self._cmd_explore(parts[1:])
        elif cmd == 'stats':
            await self._cmd_stats_enhanced()
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type 'help' for available commands.")

    # Enhanced command implementations would go here...
    # For brevity, I'll implement the key ones

    async def _cmd_create_enhanced(self, args: List[str]):
        """Handle enhanced create command."""
        if not args:
            print("❌ Usage: create \"<content>\" --user <username> [options]")
            return

        content = args[0].strip('"\'')
        user = None
        confidence = 0.5
        tags = []
        dirty = True
        analyze = False

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
            elif args[i] == '--analyze':
                analyze = True
                i += 1
            else:
                i += 1

        if not user:
            print("❌ --user is required")
            return

        await self.create_claim(content, user, confidence, tags, dirty, analyze)

    async def _cmd_search_enhanced(self, args: List[str]):
        """Handle enhanced search command."""
        if not args:
            print("❌ Usage: search \"<query>\" [--limit <number>] [--insights]")
            return

        query = args[0].strip('"\'')
        limit = 10
        insights = False

        i = 1
        while i < len(args):
            if args[i] == '--limit' and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            elif args[i] == '--insights':
                insights = True
                i += 1
            else:
                i += 1

        await self.search_claims(query, limit, insights)

    async def _cmd_analyze(self, args: List[str]):
        """Handle analyze command."""
        if not args:
            print("❌ Usage: analyze <claim_id1,claim_id2,...> [--task <task>]")
            return

        claim_ids = args[0].split(',')
        task = "analyze"

        i = 1
        while i < len(args):
            if args[i] == '--task' and i + 1 < len(args):
                task = args[i + 1]
                i += 2
            else:
                i += 1

        await self.analyze_claims(claim_ids, task)

    async def _cmd_ask(self, args: List[str]):
        """Handle ask command."""
        if not args:
            print("❌ Usage: ask \"<question>\" [--context <claim_id1,claim_id2,...>]")
            return

        question = args[0].strip('"\'')
        context_claims = []

        i = 1
        while i < len(args):
            if args[i] == '--context' and i + 1 < len(args):
                context_claims = args[i + 1].split(',')
                i += 2
            else:
                i += 1

        await self.generate_response(question, context_claims)

    async def _cmd_explore(self, args: List[str]):
        """Handle explore command."""
        if not args:
            print("❌ Usage: explore \"<topic>\" [--max-claims <number>] [--insights]")
            return

        topic = args[0].strip('"\'')
        max_claims = 10
        insights = True

        i = 1
        while i < len(args):
            if args[i] == '--max-claims' and i + 1 < len(args):
                max_claims = int(args[i + 1])
                i += 2
            elif args[i] == '--no-insights':
                insights = False
                i += 1
            else:
                i += 1

        result = await self.explore_topic(topic, max_claims, insights)
        
        print(f"\n📋 Exploration Summary:")
        print(f"   Topic: {result['topic']}")
        print(f"   Claims found: {result['claim_count']}")
        
        if result['insights']:
            print(f"   Insights generated: {len(result['insights'])}")
        
        if result['recommendations']:
            print(f"   Recommendations: {len(result['recommendations'])}")

    async def _cmd_stats_enhanced(self):
        """Handle enhanced stats command."""
        await self.get_comprehensive_stats()

    async def _cmd_get(self, args: List[str]):
        """Handle get command."""
        if not args:
            print("❌ Usage: get <claim_id>")
            return

        await self.get_claim(args[0])


async def main():
    """Main enhanced CLI entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Conjecture CLI with LLM Integration")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-mock", action="store_true", help="Use real embeddings instead of mock")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    # Enhanced command arguments
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new claim")
    create_parser.add_argument("content", help="Claim content")
    create_parser.add_argument("--user", required=True, help="Creator username")
    create_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence score")
    create_parser.add_argument("--tags", help="Comma-separated tags")
    create_parser.add_argument("--analyze", action="store_true", help="Analyze with LLM")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search similar claims")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Result limit")
    search_parser.add_argument("--insights", action="store_true", help="Include LLM insights")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze claims with LLM")
    analyze_parser.add_argument("claims", help="Comma-separated claim IDs")
    analyze_parser.add_argument("--task", default="analyze", help="Analysis task")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask LLM a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--context", help="Comma-separated claim IDs for context")

    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Explore a topic")
    explore_parser.add_argument("topic", help="Topic to explore")
    explore_parser.add_argument("--max-claims", type=int, default=10, help="Maximum claims to find")
    explore_parser.add_argument("--no-insights", action="store_true", help="Skip LLM insights")

    # Stats command
    subparsers.add_parser("stats", help="Show comprehensive statistics")

    args = parser.parse_args()

    # Initialize enhanced CLI
    cli = EnhancedConjectureCLI(args.config, use_mock=not args.no_mock)

    try:
        await cli.initialize()

        if args.interactive or not args.command:
            await cli.interactive_mode()
        else:
            # Execute single command
            if args.command == "create":
                tags = args.tags.split(',') if args.tags else []
                await cli.create_claim(args.content, args.user, args.confidence, tags, True, args.analyze)
            elif args.command == "search":
                await cli.search_claims(args.query, args.limit, args.insights)
            elif args.command == "analyze":
                claim_ids = args.claims.split(',')
                await cli.analyze_claims(claim_ids, args.task)
            elif args.command == "ask":
                context = args.context.split(',') if args.context else None
                await cli.generate_response(args.question, context)
            elif args.command == "explore":
                await cli.explore_topic(args.topic, args.max_claims, not args.no_insights)
            elif args.command == "stats":
                await cli.get_comprehensive_stats()
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