"""
Comprehensive Benchmarking Suite for Conjecture
Provides real metrics on LLM performance improvements and system capabilities
"""

import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class BenchmarkTask:
    """Individual benchmark task definition"""
    id: str
    name: str
    domain: str
    prompt: str
    expected_response_type: str
    difficulty: str
    max_tokens: int = 500
    temperature: float = 0.7

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    task_id: str
    task_name: str
    success: bool
    response_time: float
    response_length: int
    token_count: int
    provider: str
    model: str
    error: Optional[str] = None
    response_preview: str = ""
    quality_score: float = 0.0

@dataclass
class BenchmarkReport:
    """Complete benchmark report"""
    timestamp: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_response_time: float
    avg_tokens_used: int
    total_tokens: int
    cost_estimate: float
    provider_performance: Dict[str, Dict[str, Any]]
    task_results: List[BenchmarkResult]
    system_metrics: Dict[str, Any]

class ComprehensiveBenchmark:
    """Comprehensive benchmarking system for Conjecture"""

    def __init__(self):
        self.tasks = []
        self.results = []
        self.start_time = None

        # Define benchmark tasks across different domains
        self._define_benchmark_tasks()

    def _define_benchmark_tasks(self):
        """Define comprehensive benchmark tasks"""

        # Basic LLM capabilities
        self.tasks.extend([
            BenchmarkTask(
                id="basic_qa_1",
                name="Simple Q&A",
                domain="general",
                prompt="What is the capital of France? Answer in one word.",
                expected_response_type="text",
                difficulty="easy"
            ),
            BenchmarkTask(
                id="basic_qa_2",
                name="Math Problem",
                domain="mathematics",
                prompt="Calculate: 25 × 4 + 17 = ?",
                expected_response_type="text",
                difficulty="easy"
            ),
            BenchmarkTask(
                id="basic_qa_3",
                name="Logic Puzzle",
                domain="reasoning",
                prompt="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                expected_response_type="text",
                difficulty="medium"
            )
        ])

        # Code generation tasks
        self.tasks.extend([
            BenchmarkTask(
                id="code_1",
                name="Hello World Function",
                domain="programming",
                prompt="Write a Python function called hello_world that prints 'Hello, World!' and returns None.",
                expected_response_type="code",
                difficulty="easy"
            ),
            BenchmarkTask(
                id="code_2",
                name="Prime Number Checker",
                domain="programming",
                prompt="Write a Python function is_prime(n) that returns True if n is prime, False otherwise.",
                expected_response_type="code",
                difficulty="medium"
            ),
            BenchmarkTask(
                id="code_3",
                name="List Comprehension",
                domain="programming",
                prompt="Write a Python one-liner using list comprehension to create a list of squares from 1 to 10.",
                expected_response_type="code",
                difficulty="medium"
            )
        ])

        # Text analysis tasks
        self.tasks.extend([
            BenchmarkTask(
                id="text_1",
                name="Sentiment Analysis",
                domain="nlp",
                prompt="Analyze the sentiment of this sentence: 'I absolutely love this new restaurant, the food is amazing!'",
                expected_response_type="analysis",
                difficulty="easy"
            ),
            BenchmarkTask(
                id="text_2",
                name="Summarization",
                domain="nlp",
                prompt="Summarize this text in one sentence: 'The scientific method involves making observations, forming hypotheses, conducting experiments, analyzing results, and drawing conclusions.'",
                expected_response_type="summary",
                difficulty="medium"
            ),
            BenchmarkTask(
                id="text_3",
                name="Entity Recognition",
                domain="nlp",
                prompt="Extract all named entities from: 'Apple Inc. announced the iPhone 15 in Cupertino, California on September 12, 2023.'",
                expected_response_type="entities",
                difficulty="medium"
            )
        ])

        # Problem-solving tasks
        self.tasks.extend([
            BenchmarkTask(
                id="problem_1",
                name="Word Problem",
                domain="mathematics",
                prompt="A train travels 300 miles in 4 hours. What is its average speed in mph?",
                expected_response_type="calculation",
                difficulty="medium"
            ),
            BenchmarkTask(
                id="problem_2",
                name="Logic Sequence",
                domain="reasoning",
                prompt="What comes next in the sequence: 2, 4, 8, 16, ?",
                expected_response_type="reasoning",
                difficulty="easy"
            ),
            BenchmarkTask(
                id="problem_3",
                name="Scenario Analysis",
                domain="reasoning",
                prompt="You have 10 apples. You give away 3, then buy 5 more. How many apples do you have now?",
                expected_response_type="reasoning",
                difficulty="easy"
            )
        ])

    async def run_benchmark(self) -> BenchmarkReport:
        """Run comprehensive benchmark suite"""
        print("🚀 STARTING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)

        self.start_time = time.time()

        try:
            # Initialize LLM components
            from processing.simplified_llm_manager import get_simplified_llm_manager
            from processing.unified_bridge import UnifiedLLMBridge, LLMRequest

            print("📡 Initializing LLM Manager...")
            llm_manager = get_simplified_llm_manager()

            print("🔗 Setting up Unified Bridge...")
            bridge = UnifiedLLMBridge(llm_manager=llm_manager)

            # Check provider availability
            available_providers = bridge.get_available_providers()
            print(f"✅ Available Providers: {available_providers}")

            if not available_providers:
                raise Exception("No LLM providers available")

            # Run benchmark tasks
            print(f"\n🎯 Running {len(self.tasks)} benchmark tasks...")
            print("-" * 60)

            for i, task in enumerate(self.tasks, 1):
                print(f"[{i:2d}/{len(self.tasks)}] {task.name} ({task.domain})")
                result = await self._run_single_task(task, bridge)
                self.results.append(result)

                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"    {status} - {result.response_time:.2f}s - {result.provider}")
                if not result.success:
                    print(f"    Error: {result.error}")

            # Generate report
            report = self._generate_report()
            self._save_report(report)

            return report

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise

    async def _run_single_task(self, task: BenchmarkTask, bridge) -> BenchmarkResult:
        """Run a single benchmark task"""
        try:
            # Create LLM request
            request = LLMRequest(
                prompt=task.prompt,
                task_type=task.domain,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                top_p=0.9
            )

            # Time the request
            start_time = time.time()
            response = bridge.process(request)
            response_time = time.time() - start_time

            if response.success:
                return BenchmarkResult(
                    task_id=task.id,
                    task_name=task.name,
                    success=True,
                    response_time=response_time,
                    response_length=len(response.content),
                    token_count=response.tokens_used,
                    provider=response.metadata.get('provider', 'unknown'),
                    model=response.metadata.get('model', 'unknown'),
                    response_preview=response.content[:100] + "..." if len(response.content) > 100 else response.content,
                    quality_score=self._assess_response_quality(task, response.content)
                )
            else:
                return BenchmarkResult(
                    task_id=task.id,
                    task_name=task.name,
                    success=False,
                    response_time=response_time,
                    response_length=0,
                    token_count=0,
                    provider='failed',
                    model='failed',
                    error=str(response.errors) if response.errors else "Unknown error"
                )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.id,
                task_name=task.name,
                success=False,
                response_time=0.0,
                response_length=0,
                token_count=0,
                provider='error',
                model='error',
                error=str(e)
            )

    def _assess_response_quality(self, task: BenchmarkTask, response: str) -> float:
        """Assess response quality (0-1 scale)"""
        if not response:
            return 0.0

        score = 0.0

        # Basic content quality (length, relevance)
        if len(response.strip()) > 0:
            score += 0.3

        # Expected response type matching
        if task.expected_response_type == "code" and ("def " in response or "function" in response):
            score += 0.4
        elif task.expected_response_type == "calculation" and any(c.isdigit() for c in response):
            score += 0.4
        elif task.expected_response_type in ["text", "analysis", "summary", "entities", "reasoning"]:
            score += 0.4

        # Reasonable length (not too short, not too long)
        if 10 <= len(response) <= 500:
            score += 0.3

        return min(score, 1.0)

    def _generate_report(self) -> BenchmarkReport:
        """Generate comprehensive benchmark report"""
        successful_tasks = [r for r in self.results if r.success]
        failed_tasks = [r for r in self.results if not r.success]

        # Calculate metrics
        success_rate = len(successful_tasks) / len(self.results) if self.results else 0
        avg_response_time = statistics.mean([r.response_time for r in successful_tasks]) if successful_tasks else 0
        avg_tokens_used = statistics.mean([r.token_count for r in successful_tasks]) if successful_tasks else 0
        total_tokens = sum(r.token_count for r in successful_tasks)

        # Provider performance
        provider_stats = {}
        for result in successful_tasks:
            provider = result.provider
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'tasks': 0,
                    'total_time': 0,
                    'total_tokens': 0,
                    'avg_time': 0,
                    'avg_tokens': 0,
                    'avg_quality': 0
                }

            provider_stats[provider]['tasks'] += 1
            provider_stats[provider]['total_time'] += result.response_time
            provider_stats[provider]['total_tokens'] += result.token_count
            provider_stats[provider]['avg_quality'] += result.quality_score

        # Calculate provider averages
        for provider, stats in provider_stats.items():
            if stats['tasks'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['tasks']
                stats['avg_tokens'] = stats['total_tokens'] / stats['tasks']
                stats['avg_quality'] = stats['avg_quality'] / stats['tasks']

        # System metrics
        import psutil
        system_metrics = {
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage_percent': psutil.cpu_percent(),
            'total_benchmark_time': time.time() - self.start_time if self.start_time else 0
        }

        # Cost estimation (rough: $0.002 per 1K tokens)
        cost_estimate = (total_tokens / 1000) * 0.002

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_tasks=len(self.results),
            successful_tasks=len(successful_tasks),
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            avg_tokens_used=int(avg_tokens_used),
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            provider_performance=provider_stats,
            task_results=self.results,
            system_metrics=system_metrics
        )

    def _save_report(self, report: BenchmarkReport):
        """Save benchmark report to files"""
        # Save detailed JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"benchmark_report_{timestamp}.json"

        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save human-readable report
        md_file = f"benchmark_report_{timestamp}.md"
        self._save_markdown_report(report, md_file)

        print(f"\n📊 Benchmark reports saved:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📝 Markdown: {md_file}")

    def _save_markdown_report(self, report: BenchmarkReport, filename: str):
        """Save human-readable markdown report"""
        with open(filename, 'w') as f:
            f.write(f"# Conjecture Benchmark Report\n\n")
            f.write(f"**Generated**: {report.timestamp}\n")
            f.write(f"**Total Tasks**: {report.total_tasks}\n")
            f.write(f"**Successful**: {report.successful_tasks}\n")
            f.write(f"**Success Rate**: {report.success_rate:.1%}\n\n")

            f.write("## 📊 Performance Metrics\n\n")
            f.write(f"- **Average Response Time**: {report.avg_response_time:.2f}s\n")
            f.write(f"- **Average Tokens Used**: {report.avg_tokens_used:,}\n")
            f.write(f"- **Total Tokens**: {report.total_tokens:,}\n")
            f.write(f"- **Estimated Cost**: ${report.cost_estimate:.4f}\n\n")

            f.write("## 🏆 Provider Performance\n\n")
            for provider, stats in report.provider_performance.items():
                f.write(f"### {provider}\n")
                f.write(f"- **Tasks Completed**: {stats['tasks']}\n")
                f.write(f"- **Average Time**: {stats['avg_time']:.2f}s\n")
                f.write(f"- **Average Tokens**: {int(stats['avg_tokens']):,}\n")
                f.write(f"- **Average Quality**: {stats['avg_quality']:.2f}/1.0\n\n")

            f.write("## 💻 System Metrics\n\n")
            f.write(f"- **Memory Usage**: {report.system_metrics['memory_usage_mb']:.1f}MB\n")
            f.write(f"- **CPU Usage**: {report.system_metrics['cpu_usage_percent']:.1f}%\n")
            f.write(f"- **Total Benchmark Time**: {report.system_metrics['total_benchmark_time']:.2f}s\n\n")

            f.write("## 📋 Detailed Results\n\n")
            f.write("| Task | Status | Time | Tokens | Provider | Quality |\n")
            f.write("|------|--------|------|--------|----------|----------|\n")

            for result in report.task_results:
                status = "✅" if result.success else "❌"
                time_str = f"{result.response_time:.2f}s" if result.success else "N/A"
                tokens_str = f"{result.token_count:,}" if result.success else "N/A"
                quality_str = f"{result.quality_score:.2f}" if result.success else "N/A"

                f.write(f"| {result.task_name} | {status} | {time_str} | {tokens_str} | {result.provider} | {quality_str} |\n")

async def main():
    """Run comprehensive benchmark"""
    benchmark = ComprehensiveBenchmark()
    report = await benchmark.run_benchmark()

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Average Response Time: {report.avg_response_time:.2f}s")
    print(f"Total Tokens Used: {report.total_tokens:,}")
    print(f"Estimated Cost: ${report.cost_estimate:.4f}")
    print(f"Providers Tested: {len(report.provider_performance)}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())