"""
Scaling Analysis Test - Phase 1: Problem Analysis & Research
Comprehensive analysis of current system scaling capabilities and limitations
"""

import asyncio
import time
import psutil
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from src.conjecture import Conjecture
from src.processing.simplified_llm_manager import SimplifiedLLMManager, get_simplified_llm_manager
from src.processing.async_eval import AsyncClaimEvaluationService
from src.monitoring import get_performance_monitor

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scaling_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScalingAnalyzer:
    """Comprehensive scaling analysis tool"""

    def __init__(self):
        self.results = {
            "test_start": datetime.utcnow().isoformat(),
            "system_info": self._get_system_info(),
            "current_limitations": {},
            "bottlenecks": {},
            "resource_utilization": {},
            "concurrent_capability": {},
            "provider_scaling": {},
            "database_isolation": {},
            "recommendations": []
        }
        self.performance_monitor = get_performance_monitor()
        self.monitoring_active = False

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            "python_version": os.sys.version,
            "platform": os.name
        }

    def _monitor_resources(self, duration: int = 10) -> Dict[str, Any]:
        """Monitor resource usage for specified duration"""
        logger.info(f"Monitoring resources for {duration} seconds...")

        cpu_samples = []
        memory_samples = []
        disk_samples = []

        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            memory = psutil.virtual_memory()
            memory_samples.append(memory.percent)
            disk_samples.append(psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent)
            time.sleep(0.5)

        return {
            "cpu": {
                "average": sum(cpu_samples) / len(cpu_samples),
                "max": max(cpu_samples),
                "min": min(cpu_samples)
            },
            "memory": {
                "average": sum(memory_samples) / len(memory_samples),
                "max": max(memory_samples),
                "min": min(memory_samples)
            },
            "disk": {
                "average": sum(disk_samples) / len(disk_samples),
                "max": max(disk_samples),
                "min": min(disk_samples)
            }
        }

    async def analyze_provider_scaling(self) -> Dict[str, Any]:
        """Analyze multi-provider scaling capabilities"""
        logger.info("Analyzing provider scaling capabilities...")

        llm_manager = get_simplified_llm_manager()

        # Test provider availability and configuration
        provider_info = llm_manager.get_provider_info()

        # Test concurrent provider access
        concurrent_results = {}

        for provider_name in provider_info["available_providers"]:
            logger.info(f"Testing concurrent access to {provider_name}...")

            # Test simultaneous requests to the same provider
            start_time = time.time()
            tasks = []

            # Create 5 concurrent requests
            for i in range(5):
                task = asyncio.create_task(
                    self._test_provider_request(llm_manager, provider_name, f"Test request {i}")
                )
                tasks.append(task)

            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            concurrent_results[provider_name] = {
                "concurrent_requests": 5,
                "successful_requests": successful,
                "total_time": end_time - start_time,
                "average_time_per_request": (end_time - start_time) / 5,
                "success_rate": successful / 5,
                "errors": [str(r) for r in results if isinstance(r, Exception)]
            }

        return {
            "provider_configuration": provider_info,
            "concurrent_access_test": concurrent_results,
            "scaling_recommendations": self._analyze_provider_scaling_results(concurrent_results)
        }

    async def _test_provider_request(self, llm_manager: SimplifiedLLMManager, provider: str, prompt: str) -> str:
        """Test a single provider request"""
        try:
            response = llm_manager.generate_response(
                prompt=prompt,
                provider=provider,
                max_tokens=100,
                temperature=0.1
            )
            return response[:100] if response else "Empty response"
        except Exception as e:
            raise e

    def _analyze_provider_scaling_results(self, results: Dict[str, Any]) -> List[str]:
        """Analyze provider scaling results and generate recommendations"""
        recommendations = []

        for provider, data in results.items():
            success_rate = data["success_rate"]
            avg_time = data["average_time_per_request"]

            if success_rate < 0.8:
                recommendations.append(
                    f"{provider}: Low success rate ({success_rate:.1%}) - Consider rate limiting or fallback strategies"
                )

            if avg_time > 5.0:
                recommendations.append(
                    f"{provider}: High response time ({avg_time:.1f}s) - Consider optimization or timeout adjustments"
                )

        return recommendations

    async def analyze_concurrent_evaluation(self) -> Dict[str, Any]:
        """Analyze concurrent claim evaluation capabilities"""
        logger.info("Analyzing concurrent claim evaluation...")

        conjecture = Conjecture()
        await conjecture.start_services()

        try:
            # Create test claims
            test_claims = []
            for i in range(20):
                claim = await conjecture.add_claim(
                    content=f"Test claim {i+1} about artificial intelligence and machine learning capabilities",
                    confidence=0.5,
                    tags=["test", "concurrent_analysis", f"batch_{i//5+1}"]
                )
                test_claims.append(claim)

            # Measure concurrent submission
            submission_start = time.time()
            submission_tasks = []

            for claim in test_claims:
                task = asyncio.create_task(
                    conjecture.async_evaluation.submit_claim(claim)
                )
                submission_tasks.append(task)

            # Wait for all submissions
            await asyncio.gather(*submission_tasks, return_exceptions=True)
            submission_time = time.time() - submission_start

            # Wait for some evaluations to complete
            await asyncio.sleep(10)

            # Get evaluation statistics
            eval_stats = conjecture.async_evaluation.get_statistics()

            return {
                "claims_submitted": len(test_claims),
                "submission_time": submission_time,
                "average_submission_time": submission_time / len(test_claims),
                "evaluation_statistics": eval_stats,
                "concurrent_capacity": {
                    "max_concurrent_evaluations": conjecture.async_evaluation.max_concurrent_evaluations,
                    "active_evaluations": eval_stats.get("active_evaluations", 0),
                    "queue_depth": eval_stats.get("queue_depth", 0)
                },
                "scaling_recommendations": self._analyze_concurrent_eval_results(eval_stats)
            }

        finally:
            await conjecture.stop_services()

    def _analyze_concurrent_eval_results(self, stats: Dict[str, Any]) -> List[str]:
        """Analyze concurrent evaluation results"""
        recommendations = []

        active = stats.get("active_evaluations", 0)
        max_concurrent = stats.get("max_concurrent_evaluations", 5)
        queue_depth = stats.get("queue_depth", 0)

        if active >= max_concurrent * 0.8:
            recommendations.append(
                f"High concurrent load ({active}/{max_concurrent}) - Consider increasing max_concurrent_evaluations"
            )

        if queue_depth > 10:
            recommendations.append(
                f"Deep evaluation queue ({queue_depth} items) - Consider batch processing or priority optimization"
            )

        avg_time = stats.get("average_evaluation_time", 0)
        if avg_time > 30:
            recommendations.append(
                f"Long evaluation times ({avg_time:.1f}s) - Consider optimization or caching"
            )

        return recommendations

    async def analyze_database_isolation(self) -> Dict[str, Any]:
        """Analyze database isolation under concurrent load"""
        logger.info("Analyzing database isolation...")

        conjecture = Conjecture()
        await conjecture.start_services()

        try:
            # Test concurrent database operations
            isolation_results = {}

            # Test 1: Concurrent claim creation
            logger.info("Testing concurrent claim creation...")
            creation_start = time.time()

            creation_tasks = []
            for i in range(50):
                task = asyncio.create_task(
                    conjecture.add_claim(
                        content=f"Isolation test claim {i+1} for database concurrent access testing",
                        confidence=0.6,
                        tags=["isolation_test", "database", f"batch_{i//10+1}"]
                    )
                )
                creation_tasks.append(task)

            creation_results = await asyncio.gather(*creation_tasks, return_exceptions=True)
            creation_time = time.time() - creation_start

            successful_creations = sum(1 for r in creation_results if not isinstance(r, Exception))

            # Test 2: Concurrent claim retrieval
            if successful_creations > 0:
                logger.info("Testing concurrent claim retrieval...")
                retrieval_start = time.time()

                retrieval_tasks = []
                for result in creation_results:
                    if not isinstance(result, Exception):
                        task = asyncio.create_task(
                            conjecture.claim_repository.get_by_id(result.id)
                        )
                        retrieval_tasks.append(task)

                retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
                retrieval_time = time.time() - retrieval_start

                successful_retrievals = sum(1 for r in retrieval_results if r is not None)
            else:
                retrieval_time = 0
                successful_retrievals = 0

            isolation_results = {
                "concurrent_creation": {
                    "attempts": len(creation_tasks),
                    "successful": successful_creations,
                    "time": creation_time,
                    "average_time": creation_time / len(creation_tasks),
                    "success_rate": successful_creations / len(creation_tasks)
                },
                "concurrent_retrieval": {
                    "attempts": successful_creations,
                    "successful": successful_retrievals,
                    "time": retrieval_time,
                    "average_time": retrieval_time / max(1, successful_creations),
                    "success_rate": successful_retrievals / max(1, successful_creations)
                }
            }

            return {
                "isolation_test_results": isolation_results,
                "database_recommendations": self._analyze_database_isolation(isolation_results)
            }

        finally:
            await conjecture.stop_services()

    def _analyze_database_isolation(self, results: Dict[str, Any]) -> List[str]:
        """Analyze database isolation results"""
        recommendations = []

        creation_success = results["concurrent_creation"]["success_rate"]
        retrieval_success = results["concurrent_retrieval"]["success_rate"]
        creation_time = results["concurrent_creation"]["average_time"]

        if creation_success < 0.9:
            recommendations.append(
                f"Low concurrent creation success rate ({creation_success:.1%}) - Check database isolation"
            )

        if retrieval_success < 0.9:
            recommendations.append(
                f"Low concurrent retrieval success rate ({retrieval_success:.1%}) - Check database locking"
            )

        if creation_time > 0.5:
            recommendations.append(
                f"Slow claim creation ({creation_time:.2f}s) - Consider database optimization"
            )

        return recommendations

    async def identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify system bottlenecks under various loads"""
        logger.info("Identifying system bottlenecks...")

        bottleneck_results = {}

        # Test 1: LLM processing bottleneck
        logger.info("Testing LLM processing bottleneck...")
        llm_manager = get_simplified_llm_manager()

        llm_start = time.time()
        llm_tasks = []

        for i in range(10):
            task = asyncio.create_task(
                self._test_llm_processing(llm_manager)
            )
            llm_tasks.append(task)

        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        llm_time = time.time() - llm_start

        bottleneck_results["llm_processing"] = {
            "concurrent_requests": 10,
            "successful": sum(1 for r in llm_results if not isinstance(r, Exception)),
            "total_time": llm_time,
            "average_time": llm_time / 10,
            "throughput": 10 / llm_time
        }

        # Test 2: Memory bottleneck
        logger.info("Testing memory usage patterns...")
        initial_memory = psutil.virtual_memory().percent

        # Create memory pressure
        conjecture = Conjecture()
        await conjecture.start_services()

        # Generate many operations
        memory_test_start = time.time()
        for i in range(100):
            await conjecture.add_claim(
                content=f"Memory test claim {i+1} with substantial content to test memory usage patterns",
                confidence=0.7,
                tags=["memory_test", "pressure_test"]
            )

        peak_memory = psutil.virtual_memory().percent
        memory_test_time = time.time() - memory_test_start

        await conjecture.stop_services()

        bottleneck_results["memory_usage"] = {
            "initial_memory_percent": initial_memory,
            "peak_memory_percent": peak_memory,
            "memory_increase": peak_memory - initial_memory,
            "operations_per_second": 100 / memory_test_time
        }

        return {
            "bottleneck_analysis": bottleneck_results,
            "primary_bottlenecks": self._identify_primary_bottlenecks(bottleneck_results),
            "optimization_recommendations": self._generate_optimization_recommendations(bottleneck_results)
        }

    async def _test_llm_processing(self, llm_manager: SimplifiedLLMManager) -> str:
        """Test LLM processing performance"""
        response = llm_manager.generate_response(
            prompt="Briefly test LLM processing performance",
            max_tokens=50,
            temperature=0.1
        )
        return "success" if response else "failed"

    def _identify_primary_bottlenecks(self, results: Dict[str, Any]) -> List[str]:
        """Identify primary bottlenecks from test results"""
        bottlenecks = []

        llm_throughput = results["llm_processing"]["throughput"]
        if llm_throughput < 2:  # Less than 2 requests per second
            bottlenecks.append(f"LLM processing throughput ({llm_throughput:.1f} req/s)")

        memory_increase = results["memory_usage"]["memory_increase"]
        if memory_increase > 20:  # More than 20% memory increase
            bottlenecks.append(f"Memory usage increase ({memory_increase:.1f}%)")

        return bottlenecks

    def _generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        llm_success = results["llm_processing"]["successful"] / results["llm_processing"]["concurrent_requests"]
        if llm_success < 0.8:
            recommendations.append("Implement LLM request pooling and rate limiting")

        ops_per_sec = results["memory_usage"]["operations_per_second"]
        if ops_per_sec < 50:
            recommendations.append("Optimize memory usage and implement better garbage collection")

        return recommendations

    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive scaling analysis"""
        logger.info("Starting comprehensive scaling analysis...")

        # Start resource monitoring
        self.monitoring_active = True
        monitor_thread = threading.Thread(
            target=self._continuous_monitoring,
            daemon=True
        )
        monitor_thread.start()

        try:
            # 1. Analyze provider scaling
            provider_results = await self.analyze_provider_scaling()
            self.results["provider_scaling"] = provider_results

            # 2. Analyze concurrent evaluation
            eval_results = await self.analyze_concurrent_evaluation()
            self.results["concurrent_capability"] = eval_results

            # 3. Analyze database isolation
            db_results = await self.analyze_database_isolation()
            self.results["database_isolation"] = db_results

            # 4. Identify bottlenecks
            bottleneck_results = await self.identify_bottlenecks()
            self.results["bottlenecks"] = bottleneck_results

            # 5. Resource utilization baseline
            baseline_resources = self._monitor_resources(duration=5)
            self.results["resource_utilization"]["baseline"] = baseline_resources

            # Compile recommendations
            self.results["recommendations"] = self._compile_all_recommendations()

            self.results["test_complete"] = datetime.utcnow().isoformat()

            return self.results

        finally:
            self.monitoring_active = False

    def _continuous_monitoring(self):
        """Continuous resource monitoring in background thread"""
        while self.monitoring_active:
            # Collect resource metrics
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Store results (simplified - in production would use proper monitoring)
            time.sleep(2)

    def _compile_all_recommendations(self) -> List[str]:
        """Compile recommendations from all analyses"""
        all_recommendations = []

        # Collect from all sections
        if "scaling_recommendations" in self.results.get("provider_scaling", {}):
            all_recommendations.extend(
                self.results["provider_scaling"]["scaling_recommendations"]
            )

        if "scaling_recommendations" in self.results.get("concurrent_capability", {}):
            all_recommendations.extend(
                self.results["concurrent_capability"]["scaling_recommendations"]
            )

        if "database_recommendations" in self.results.get("database_isolation", {}):
            all_recommendations.extend(
                self.results["database_isolation"]["database_recommendations"]
            )

        if "optimization_recommendations" in self.results.get("bottlenecks", {}):
            all_recommendations.extend(
                self.results["bottlenecks"]["optimization_recommendations"]
            )

        return list(set(all_recommendations))  # Remove duplicates

    def save_results(self, filename: str = "scaling_analysis_results.json"):
        """Save analysis results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {filename}")


async def main():
    """Main scaling analysis execution"""
    logger.info("=" * 60)
    logger.info("SCALING ANALYSIS - PHASE 1: PROBLEM ANALYSIS & RESEARCH")
    logger.info("=" * 60)

    analyzer = ScalingAnalyzer()

    try:
        # Run comprehensive analysis
        results = await analyzer.run_comprehensive_analysis()

        # Display summary
        print("\n" + "=" * 60)
        print("SCALING ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nSystem Information:")
        print(f"- CPU Cores: {results['system_info']['cpu_count']}")
        print(f"- Memory: {results['system_info']['memory_total_gb']:.1f} GB")
        print(f"- Platform: {results['system_info']['platform']}")

        print(f"\nPrimary Bottlenecks:")
        for bottleneck in results.get("bottlenecks", {}).get("primary_bottlenecks", []):
            print(f"- {bottleneck}")

        print(f"\nKey Recommendations:")
        for i, rec in enumerate(results.get("recommendations", [])[:10], 1):
            print(f"{i}. {rec}")

        # Save detailed results
        analyzer.save_results()

        print(f"\nDetailed results saved to: scaling_analysis_results.json")
        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())